"""
AutoGovern backtest: a single unified dual-timescale system where memory
is ALWAYS on and the critic governs how many refinement rounds happen and
whether to consolidate (AdaReMo Algorithm 2).

This is NOT a mode router. At every rebalance:
  1. Slow loop: post_task_reflect on realized outcomes → memory update
  2. Fast loop (K=3 rounds max): Solver (conditioned on memory) →
     Debater review → break on admitted OR g_ref=False
  3. Consolidate only if: admitted AND g_sto AND not SATURATED(M)

Usage:
    python -m ace_harness.run_monthly_adaptive \
        --tickers AAPL MSFT NVDA ... \
        --start 2024-01-01 --end 2024-12-31 --output_dir ./results
"""
import argparse
import os

from ace_harness.market import MarketUniverse
from ace_harness.engine_monthly import run_backtest
from ace_harness.memory import ExperienceMemory
from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
from ace_harness.risk_harness import ConvictionHarness
from ace_harness import harnesses

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "QCOM", "TXN",
    "AMAT", "MU", "AMZN", "TSLA", "PEP", "COST", "HON", "GOOGL", "META", "NFLX",
]


def run_adaptive(tickers, start, end, output_dir,
                 initial_capital=1_000_000.0, rebalance_days=20,
                 include_screener_tool=True, fallback_mode="equal_weight",
                 truncate_context=False, max_tokens=800):

    universe = MarketUniverse(tickers)
    universe.prefetch(start, end)

    solver = Solver(universe, include_screener_tool=include_screener_tool,
                    fallback_mode=fallback_mode, max_tokens=max_tokens,
                    truncate_context=truncate_context, prompt_style="monthly")
    debater = Debater()
    consolidator = Consolidator()

    tag = "monthly_adaptive_autogover"
    memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")
    memory = ExperienceMemory.load(memory_path) if os.path.exists(memory_path) else ExperienceMemory()

    risk_harness = ConvictionHarness(universe)
    risk_params_path = os.path.join(output_dir, f"{tag}_riskparams.json")
    risk_tuner = RiskTuner()

    # Single unified AutoGovern system — no routing between modes
    decision_fn = harnesses.make_autogover(
        universe, solver, debater, consolidator, memory, memory_path,
        max_rounds=3,
        risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
    )

    output_file = os.path.join(output_dir, f"{tag}_results.json")
    run_backtest(universe, decision_fn, start, end, initial_capital, output_file, rebalance_days)
    print(f"[adaptive] done -> {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="AutoGovern (AdaReMo Algorithm 2) backtest")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--initial_capital", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-days", type=int, default=20, dest="rebalance_days")
    parser.add_argument("--no_screener", action="store_true")
    parser.add_argument("--fallback_mode", choices=["cash", "equal_weight"], default="equal_weight")
    parser.add_argument("--truncate_context", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=800)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_adaptive(
        args.tickers, args.start, args.end, args.output_dir,
        initial_capital=args.initial_capital, rebalance_days=args.rebalance_days,
        include_screener_tool=not args.no_screener, fallback_mode=args.fallback_mode,
        truncate_context=args.truncate_context, max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
