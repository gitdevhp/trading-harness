"""
Run any/all self-improvement structures against the same universe/date
range, so results are directly comparable.

Usage (from the directory containing ace_harness/):

    python -m ace_harness.run_all --systems dual_permanent \
        --tickers GLD LLY KRE TSLA GOOGL TLT XLU XLE XOM NVDA \
        --start 2023-03-15 --end 2026-04-01 --output_dir ./results \
        --risk_harness --risk_harness_type gpt --adaptive_risk

Systems: intra, inter, dual, dual_session, dual_permanent — see
harnesses.py for what each one does. dual_permanent (one Debater round
per task, memory persisted forever to a human-editable .txt playbook) is
the primary configuration; the others exist for comparison.

--risk_harness_type picks which post-processing filter --risk_harness
applies: "gpt" (GPTInstitutionalRiskHarness, matches yesharnessgpt.py) or
"simple" (SimpleTrailingRiskHarness, matches yesharness.py). --no_screener
drops the get_market_screener tool from the Solver, matching
yesharness.py's leaner tool set (get_portfolio_status only).
"""
import argparse
import os

from ace_harness.market import MarketUniverse
from ace_harness.engine import run_backtest
from ace_harness.memory import ExperienceMemory
from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
from ace_harness.risk_harness import GPTInstitutionalRiskHarness, SimpleTrailingRiskHarness
from ace_harness import harnesses

DEFAULT_UNIVERSE = ["GLD", "LLY", "KRE", "TSLA", "GOOGL", "TLT", "XLU", "XLE", "XOM", "NVDA"]

# systems with a persistent, cross-task Experience Memory file
_MEMORY_SYSTEMS = ("inter", "dual", "memory_only", "dual_permanent")

_RISK_HARNESS_CLASSES = {
    "gpt": GPTInstitutionalRiskHarness,
    "simple": SimpleTrailingRiskHarness,
}


def build_universe(tickers, start, end):
    universe = MarketUniverse(tickers)
    universe.prefetch(start, end)
    return universe


def run_system(system, tickers, start, end, output_dir, initial_capital=100000.0,
                rebalance_days=20, warmup_days=60, max_rounds=3,
                use_risk_harness=False, risk_harness_type="gpt", adaptive_risk=False,
                include_screener_tool=True):
    universe = build_universe(tickers, start, end)
    solver = Solver(universe, include_screener_tool=include_screener_tool)
    debater = Debater()

    # adaptive tuning only makes sense where there's persistent memory to
    # drive it — silently falls back to a static risk harness otherwise
    # (intra and dual_session have no cross-task memory by design).
    is_adaptive = use_risk_harness and adaptive_risk and system in _MEMORY_SYSTEMS

    tag = system
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
        decision_fn = harnesses.make_inter_task(
            universe, solver, debater, Consolidator(), memory, memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system == "dual":
        memory = ExperienceMemory()
        memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
        decision_fn = harnesses.make_dual_timescale(
            universe, solver, debater, Consolidator(), memory, memory_path, max_rounds=max_rounds,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system == "dual_session":
        # no memory_path: session memory is deliberately never written to disk
        decision_fn = harnesses.make_dual_session(
            universe, solver, debater, Consolidator(), max_rounds=max_rounds, risk_harness=risk_harness,
        )

    elif system == "memory_only":
        memory = ExperienceMemory()
        memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
        decision_fn = harnesses.make_memory_only(
            universe, solver, debater, Consolidator(), memory, memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system == "dual_permanent":
        memory = ExperienceMemory()
        memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
        decision_fn = harnesses.make_dual_permanent(
            universe, solver, debater, Consolidator(), memory, memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    else:
        raise ValueError(f"Unknown system: {system}")

    run_backtest(universe, decision_fn, start, end, initial_capital, output_file, rebalance_days, warmup_days)
    print(f"[{tag}] done -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run self-improvement backtests")
    parser.add_argument("--systems", nargs="+",
                         choices=["baseline", "intra", "inter", "dual", "dual_session", "memory_only", "dual_permanent"],
                         default=["dual_permanent"])
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2023-03-15")
    parser.add_argument("--end", type=str, default="2026-04-01")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--rebalance-days", type=int, default=20, dest="rebalance_days")
    parser.add_argument("--warmup-days", type=int, default=60, dest="warmup_days")
    parser.add_argument("--max_rounds", type=int, default=3,
                         help="max Solver<->Debater rounds per task (intra/dual/dual_session only — "
                              "dual_permanent always uses exactly 1)")
    parser.add_argument("--risk_harness", action="store_true",
                         help="apply a risk-management post-processing layer on top of each system")
    parser.add_argument("--risk_harness_type", choices=["gpt", "simple"], default="gpt",
                         help="which risk harness --risk_harness applies: 'gpt' (yesharnessgpt.py-style, "
                              "5 overlays) or 'simple' (yesharness.py-style, 3-rule trailing stop)")
    parser.add_argument("--adaptive_risk", action="store_true",
                         help="let Experience Memory tune the risk harness's parameters "
                              "(inter/dual/dual_permanent only; requires --risk_harness)")
    parser.add_argument("--no_screener", action="store_true",
                         help="drop the get_market_screener tool from the Solver, matching yesharness.py's "
                              "leaner tool set (get_portfolio_status only)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for system in args.systems:
        run_system(
            system, args.tickers, args.start, args.end, args.output_dir,
            rebalance_days=args.rebalance_days, warmup_days=args.warmup_days, max_rounds=args.max_rounds,
            use_risk_harness=args.risk_harness, risk_harness_type=args.risk_harness_type,
            adaptive_risk=args.adaptive_risk, include_screener_tool=not args.no_screener,
        )


if __name__ == "__main__":
    main()
