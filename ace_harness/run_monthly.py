"""
Runs the self-improvement structures against engine_monthly.py — your
"fair portbench" script's exact protocol (same-close execution,
calendar-month rebalance, no warmup, equal-weight start, 15bps fees,
$1M default capital).

The Solver here uses prompt_style="monthly": its system prompt is
byte-for-byte the same as your pasted DECISION_FUNCTION (same wording,
same Rules, same "ReAct format" header, same invalid-allocation nudge
text) — see agents.py's Solver._build_prompt(). Both `get_market_screener`
and `get_portfolio_status` are available by default, matching every
version of your script (they always exposed both tools).

Default system is dual_permanent with --risk_harness --adaptive_risk —
the diagram's closed loop: Solver proposes -> Debater reviews once -> if
"revise", Solver gets exactly one more turn using that feedback -> that
Result, plus the Debater's own lessons AND a reflection on the previous
task's realized outcome, both go to the Consolidator -> updates the
permanent .txt Experience Memory -> which feeds the Solver's prompt next
task AND (via RiskTuner) can adjust the risk harness's own parameters.

Usage (from the directory containing ace_harness/):

    python -m ace_harness.run_monthly \
        --tickers AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX \
        --start 2024-01-01 --end 2024-12-31 --output_dir ./results
"""
import argparse
import os

from ace_harness.market import MarketUniverse
from ace_harness.engine_monthly import run_backtest
from ace_harness.memory import ExperienceMemory
from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
from ace_harness.risk_harness import (
    GPTInstitutionalRiskHarness, SimpleTrailingRiskHarness, SimpleMomentumHarness, ConvictionHarness,
)
from ace_harness import harnesses

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "QCOM", "TXN",
    "AMAT", "MU", "AMZN", "TSLA", "PEP", "COST", "HON", "GOOGL", "META", "NFLX",
]

_MEMORY_SYSTEMS = ("inter", "dual", "memory_only", "dual_permanent")

_RISK_HARNESS_CLASSES = {
    "gpt": GPTInstitutionalRiskHarness,
    "simple": SimpleTrailingRiskHarness,
    "momentum": SimpleMomentumHarness,
    "conviction": ConvictionHarness,
}


def build_universe(tickers, start, end):
    universe = MarketUniverse(tickers)
    universe.prefetch(start, end)
    return universe


def run_system(system, tickers, start, end, output_dir, initial_capital=1_000_000.0,
                rebalance_days=20, max_rounds=3, use_risk_harness=True, risk_harness_type="conviction",
                adaptive_risk=True, include_screener_tool=True, fallback_mode="equal_weight",
                truncate_context=False, max_tokens=300):
    universe = build_universe(tickers, start, end)
    solver = Solver(universe, include_screener_tool=include_screener_tool, fallback_mode=fallback_mode,
                     max_tokens=max_tokens, truncate_context=truncate_context, prompt_style="monthly")
    debater = Debater()

    is_adaptive = use_risk_harness and adaptive_risk and system in _MEMORY_SYSTEMS

    tag = f"monthly_{system}"
    if not include_screener_tool:
        tag += "_noscreener"
    if use_risk_harness:
        tag += f"_{risk_harness_type}riskharness"
        tag += "_adaptive" if is_adaptive else ""
    output_file = os.path.join(output_dir, f"{tag}_results.json")

    risk_harness_cls = _RISK_HARNESS_CLASSES[risk_harness_type]
    risk_harness = risk_harness_cls(universe) if use_risk_harness else None
    risk_params_path = os.path.join(output_dir, f"{tag}_riskparams.json") if use_risk_harness else None
    risk_tuner = RiskTuner() if is_adaptive else None

    if system == "baseline":
        decision_fn = harnesses.make_baseline(universe, solver)
        if risk_harness is not None:
            decision_fn = harnesses.wrap_with_risk_harness(decision_fn, risk_harness)

    elif system == "intra":
        decision_fn = harnesses.make_intra_task(universe, solver, debater, max_rounds=max_rounds)
        if risk_harness is not None:
            decision_fn = harnesses.wrap_with_risk_harness(decision_fn, risk_harness)

    elif system == "inter":
        memory = ExperienceMemory()
        memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
        if os.path.exists(memory_path):
            memory = ExperienceMemory.load(memory_path)
        memory.seed_with_principles()
        decision_fn = harnesses.make_inter_task(
            universe, solver, debater, Consolidator(), memory, memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system == "dual":
        memory = ExperienceMemory()
        memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
        if os.path.exists(memory_path):
            memory = ExperienceMemory.load(memory_path)
        memory.seed_with_principles()
        decision_fn = harnesses.make_dual_timescale(
            universe, solver, debater, Consolidator(), memory, memory_path, max_rounds=max_rounds,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system == "dual_session":
        decision_fn = harnesses.make_dual_session(
            universe, solver, debater, Consolidator(), max_rounds=max_rounds, risk_harness=risk_harness,
        )

    elif system == "memory_only":
        memory = ExperienceMemory()
        memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
        if os.path.exists(memory_path):
            memory = ExperienceMemory.load(memory_path)
        memory.seed_with_principles()
        decision_fn = harnesses.make_memory_only(
            universe, solver, debater, Consolidator(), memory, memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system == "dual_permanent":
        memory = ExperienceMemory()
        memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
        if os.path.exists(memory_path):
            memory = ExperienceMemory.load(memory_path)
        memory.seed_with_principles()
        decision_fn = harnesses.make_dual_permanent(
            universe, solver, debater, Consolidator(), memory, memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    else:
        raise ValueError(f"Unknown system: {system}")

    run_backtest(universe, decision_fn, start, end, initial_capital, output_file, rebalance_days)
    print(f"[{tag}] done -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run self-improvement backtests on the monthly-rebalance engine")
    parser.add_argument("--systems", nargs="+",
                         choices=["baseline", "intra", "inter", "dual", "dual_session", "memory_only", "dual_permanent"],
                         default=["dual_permanent"])
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--initial_capital", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-days", type=int, default=20, dest="rebalance_days",
                         help="only affects the Solver's 'decision horizon' prompt text — the engine's actual "
                              "rebalance trigger is always the first trading day of each new calendar month")
    parser.add_argument("--max_rounds", type=int, default=3,
                         help="max Solver<->Debater rounds per task (intra/dual/dual_session only — "
                              "dual_permanent always does exactly one review + at most one revision)")
    parser.add_argument("--no_risk_harness", action="store_true", help="disable the risk-management layer")
    parser.add_argument("--risk_harness_type", choices=["gpt", "simple", "momentum", "conviction"],
                         default="conviction",
                         help="'conviction' (default) matches your latest script's apply_institutional_harness")
    parser.add_argument("--no_adaptive_risk", action="store_true",
                         help="disable memory tuning the risk harness's parameters")
    parser.add_argument("--no_screener", action="store_true",
                         help="drop the get_market_screener tool — OFF by default because every version of "
                              "your script always exposes both get_market_screener and get_portfolio_status")
    parser.add_argument("--fallback_mode", choices=["cash", "equal_weight"], default="equal_weight",
                         help="what the Solver falls back to if parsing ever fails (matches this script's "
                              "equal-weight fallback by default)")
    parser.add_argument("--truncate_context", action="store_true",
                         help="resend only [system, user, last_2_messages] each turn instead of full history "
                              "(off by default, matching your latest script, which sends full history)")
    parser.add_argument("--max_tokens", type=int, default=600,
                         help="per-turn output budget for the Solver (increased from 300 to give Solver room to reason)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for system in args.systems:
        run_system(
            system, args.tickers, args.start, args.end, args.output_dir,
            initial_capital=args.initial_capital, rebalance_days=args.rebalance_days, max_rounds=args.max_rounds,
            use_risk_harness=not args.no_risk_harness, risk_harness_type=args.risk_harness_type,
            adaptive_risk=not args.no_adaptive_risk, include_screener_tool=not args.no_screener,
            fallback_mode=args.fallback_mode, truncate_context=args.truncate_context, max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
