"""
Runs the harness + ReMo system: conviction risk harness + plain reward model
(raw returns, no risk-adjustment). No memory, no debater.

Ablation between HARNESS+ADAMO (risk-adjusted signal) and HARNESS+REMO (raw
return signal) isolates the value of Sharpe/Sortino/drawdown scoring in AdaReMo.

Usage (from the directory containing ace_harness/):

    python -m ace_harness.run_monthly_baseline_remo \
        --tickers AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX \
        --start 2024-01-01 --end 2024-12-31 --output_dir ./results
"""
import argparse
import os

from ace_harness.market import MarketUniverse
from ace_harness.engine_monthly import run_backtest
from ace_harness.agents import Solver
from ace_harness.reward_model import SimpleRewardModel
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


def run_baseline_remo(tickers, start, end, output_dir, initial_capital=1_000_000.0,
                      rebalance_days=20, use_risk_harness=True, risk_harness_type="conviction",
                      include_screener_tool=True, fallback_mode="equal_weight",
                      truncate_context=False, max_tokens=800,
                      remo_alpha=1.0, remo_min_samples=2):
    universe = MarketUniverse(tickers)
    universe.prefetch(start, end)

    solver = Solver(universe, include_screener_tool=include_screener_tool, fallback_mode=fallback_mode,
                    max_tokens=max_tokens, truncate_context=truncate_context, prompt_style="monthly")

    tag = "monthly_baseline_remo"
    if not include_screener_tool:
        tag += "_noscreener"
    if use_risk_harness:
        tag += f"_{risk_harness_type}riskharness"

    output_file = os.path.join(output_dir, f"{tag}_results.json")

    risk_harness_cls = _RISK_HARNESS_CLASSES[risk_harness_type]
    risk_harness = risk_harness_cls(universe) if use_risk_harness else None
    reward_model = SimpleRewardModel(alpha=remo_alpha, min_samples=remo_min_samples)

    decision_fn = harnesses.make_baseline_remo(
        universe, solver, reward_model, risk_harness=risk_harness,
    )

    run_backtest(universe, decision_fn, start, end, initial_capital, output_file, rebalance_days)
    print(f"[{tag}] done -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run harness + ReMo backtest (raw returns, no risk-adjustment)")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--initial_capital", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-days", type=int, default=20, dest="rebalance_days")
    parser.add_argument("--no_risk_harness", action="store_true")
    parser.add_argument("--risk_harness_type", choices=["gpt", "simple", "momentum", "conviction"],
                        default="conviction")
    parser.add_argument("--no_screener", action="store_true")
    parser.add_argument("--fallback_mode", choices=["cash", "equal_weight"], default="equal_weight")
    parser.add_argument("--truncate_context", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--remo_alpha", type=float, default=1.0)
    parser.add_argument("--remo_min_samples", type=int, default=2,
                        help="minimum periods before ReMo signal activates (default 2 = fast activation)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_baseline_remo(
        args.tickers, args.start, args.end, args.output_dir,
        initial_capital=args.initial_capital, rebalance_days=args.rebalance_days,
        use_risk_harness=not args.no_risk_harness, risk_harness_type=args.risk_harness_type,
        include_screener_tool=not args.no_screener, fallback_mode=args.fallback_mode,
        truncate_context=args.truncate_context, max_tokens=args.max_tokens,
        remo_alpha=args.remo_alpha, remo_min_samples=args.remo_min_samples,
    )


if __name__ == "__main__":
    main()
