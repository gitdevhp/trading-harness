"""Run ACE self-improvement systems on the fixed-day execution engine.

Unlike run_monthly.py, this wrapper uses engine.py: rebalances occur every
N trading days, orders execute at the next day's open, and warmup_days are
simulated before the official output period.

Example:
    python -m ace_harness.run_weekly \
        --tickers AAPL MSFT NVDA \
        --start 2024-01-01 --end 2024-12-31 \
        --output_dir ./results --rebalance-days 5 --warmup-days 60 \
        --risk_harness --risk_harness_type gpt --adaptive_risk
"""
import argparse
import os

from ace_harness.run_all import DEFAULT_UNIVERSE, run_system


_SYSTEMS = [
    "baseline", "intra", "inter", "dual", "dual_session", "memory_only", "dual_permanent",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run ACE systems with fixed trading-day rebalances and next-open execution."
    )
    parser.add_argument("--systems", nargs="+", choices=_SYSTEMS, default=["dual_permanent"])
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--initial_capital", type=float, default=1_000_000.0)
    parser.add_argument(
        "--rebalance-days", type=int, default=5, dest="rebalance_days",
        help="rebalance every N trading days; 5 is approximately weekly",
    )
    parser.add_argument("--warmup-days", type=int, default=60, dest="warmup_days")
    parser.add_argument(
        "--max_rounds", type=int, default=3,
        help="maximum Solver/Debater rounds for intra, dual, and dual_session",
    )
    parser.add_argument("--risk_harness", action="store_true")
    parser.add_argument("--risk_harness_type", choices=["gpt", "simple"], default="gpt")
    parser.add_argument(
        "--adaptive_risk", action="store_true",
        help="let persistent memory tune risk parameters",
    )
    parser.add_argument("--no_screener", action="store_true")
    args = parser.parse_args()

    if args.rebalance_days < 1:
        parser.error("--rebalance-days must be at least 1")
    if args.warmup_days < 0:
        parser.error("--warmup-days cannot be negative")

    os.makedirs(args.output_dir, exist_ok=True)
    for system in args.systems:
        run_system(
            system,
            args.tickers,
            args.start,
            args.end,
            args.output_dir,
            initial_capital=args.initial_capital,
            rebalance_days=args.rebalance_days,
            warmup_days=args.warmup_days,
            max_rounds=args.max_rounds,
            use_risk_harness=args.risk_harness,
            risk_harness_type=args.risk_harness_type,
            adaptive_risk=args.adaptive_risk,
            include_screener_tool=not args.no_screener,
        )


if __name__ == "__main__":
    main()
