"""
Runs the dual_permanent system with AdaReMo (Adaptive Reward Modeling)
added as a purely additive signal. Same engine, same risk harness, same
monthly-rebalance protocol as run_monthly.py — the only addition is the
Ridge-regression reward model that accumulates allocation→return pairs
across months and injects a weak prior into the Solver's prompt.

Intended for side-by-side comparison with dual_permanent (no AdaReMo)
to isolate the reward model's effect.

Usage (from the directory containing ace_harness/):

    python -m ace_harness.run_monthly_adamo \
        --tickers AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX \
        --start 2024-01-01 --end 2024-12-31 --output_dir ./results
"""
import argparse
import os

from ace_harness.market import MarketUniverse
from ace_harness.engine_monthly import run_backtest
from ace_harness.memory import ExperienceMemory
from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
from ace_harness.reward_model import AdaptiveRewardModel
from ace_harness.risk_harness import (
    GPTInstitutionalRiskHarness, SimpleTrailingRiskHarness, SimpleMomentumHarness, ConvictionHarness,
)
from ace_harness import harnesses

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "QCOM", "TXN",
    "AMAT", "MU", "AMZN", "TSLA", "PEP", "COST", "HON", "GOOGL", "META", "NFLX",
]

_RISK_HARNESS_CLASSES = {
    "gpt": GPTInstitutionalRiskHarness,
    "simple": SimpleTrailingRiskHarness,
    "momentum": SimpleMomentumHarness,
    "conviction": ConvictionHarness,
}


def run_adamo(tickers, start, end, output_dir, initial_capital=1_000_000.0,
              rebalance_days=20, use_risk_harness=True, risk_harness_type="conviction",
              adaptive_risk=True, include_screener_tool=True, fallback_mode="equal_weight",
              truncate_context=False, max_tokens=800, adamo_alpha=1.0, adamo_min_samples=3):
    universe = MarketUniverse(tickers)
    universe.prefetch(start, end)

    solver = Solver(universe, include_screener_tool=include_screener_tool, fallback_mode=fallback_mode,
                    max_tokens=max_tokens, truncate_context=truncate_context, prompt_style="monthly")
    debater = Debater()

    is_adaptive = use_risk_harness and adaptive_risk
    tag = "monthly_dual_permanent_adamo"
    if not include_screener_tool:
        tag += "_noscreener"
    if use_risk_harness:
        tag += f"_{risk_harness_type}riskharness"
        tag += "_adaptive" if is_adaptive else ""

    output_file = os.path.join(output_dir, f"{tag}_results.json")
    memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
    risk_params_path = os.path.join(output_dir, f"{tag}_riskparams.json") if use_risk_harness else None

    memory = ExperienceMemory()
    if os.path.exists(memory_path):
        memory = ExperienceMemory.load(memory_path)

    risk_harness_cls = _RISK_HARNESS_CLASSES[risk_harness_type]
    risk_harness = risk_harness_cls(universe) if use_risk_harness else None
    risk_tuner = RiskTuner() if is_adaptive else None
    reward_model = AdaptiveRewardModel(alpha=adamo_alpha, min_samples=adamo_min_samples)

    decision_fn = harnesses.make_dual_permanent_adamo(
        universe, solver, debater, Consolidator(), memory, memory_path,
        reward_model=reward_model,
        risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
    )

    run_backtest(universe, decision_fn, start, end, initial_capital, output_file, rebalance_days)
    print(f"[{tag}] done -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run dual_permanent + AdaReMo backtest")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--initial_capital", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-days", type=int, default=20, dest="rebalance_days")
    parser.add_argument("--no_risk_harness", action="store_true")
    parser.add_argument("--risk_harness_type", choices=["gpt", "simple", "momentum", "conviction"],
                        default="conviction")
    parser.add_argument("--no_adaptive_risk", action="store_true")
    parser.add_argument("--no_screener", action="store_true")
    parser.add_argument("--fallback_mode", choices=["cash", "equal_weight"], default="equal_weight")
    parser.add_argument("--truncate_context", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--adamo_alpha", type=float, default=1.0,
                        help="Ridge regression L2 regularization strength")
    parser.add_argument("--adamo_min_samples", type=int, default=3,
                        help="minimum periods before AdaReMo signal activates")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_adamo(
        args.tickers, args.start, args.end, args.output_dir,
        initial_capital=args.initial_capital, rebalance_days=args.rebalance_days,
        use_risk_harness=not args.no_risk_harness, risk_harness_type=args.risk_harness_type,
        adaptive_risk=not args.no_adaptive_risk, include_screener_tool=not args.no_screener,
        fallback_mode=args.fallback_mode, truncate_context=args.truncate_context,
        max_tokens=args.max_tokens, adamo_alpha=args.adamo_alpha,
        adamo_min_samples=args.adamo_min_samples,
    )


if __name__ == "__main__":
    main()
