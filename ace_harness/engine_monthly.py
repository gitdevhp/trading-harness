"""
Alternate execution engine — matches your pasted "fair portbench" script's
protocol exactly:
  - executes at the SAME day's close (not next-day open, unlike engine.py)
  - rebalances on the first trading day of each new calendar MONTH
    (not a fixed day-count interval, unlike engine.py)
  - no warmup period; the initial portfolio is equal-weight, not cash
  - fee rate and default initial capital match that script (15bps, $1M)

This is a genuinely different backtest protocol from engine.py's, so it
lives in its own module rather than being bolted onto engine.py with a
pile of conditionals — engine.py is already relied on by five other
systems and well worth not touching.

decision_fn has the EXACT SAME signature as engine.py's:
    decision_fn(current_date, portfolio_state, decision_log, rebalance_days)
        -> (raw_allocations, meta)
so every harness in harnesses.py — make_intra_task, make_inter_task,
make_dual_timescale, make_dual_session, make_dual_permanent — plugs into
this engine completely unchanged. Only the execution mechanics differ,
never the self-improvement structure wired around them.
"""
import json
import pandas as pd

FEE_RATE = 0.0015


def run_backtest(
    universe,
    decision_fn,
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000.0,
    output_file: str = "results.json",
    rebalance_days: int = 20,  # only used for the Solver's "decision horizon" prompt text
):
    trading_days = universe.common_trading_days(start_date, end_date)
    if not trading_days:
        raise ValueError(f"No trading data found between {start_date} and {end_date}.")

    first_prices = universe.close_prices(trading_days[0])
    n = len(universe.anon_universe)
    cash = 0.0
    holdings = {a: (initial_capital / n) / first_prices[a] for a in universe.anon_universe}

    target_allocs = {a: 100.0 / n for a in universe.anon_universe}
    target_allocs["CASH"] = 0.0

    results = []
    decision_log = {}

    for idx, current_date in enumerate(trading_days):
        prices = universe.close_prices(current_date)
        value = cash + sum(holdings[a] * prices[a] for a in universe.anon_universe)

        is_rebalance = idx > 0 and (
            pd.to_datetime(current_date).month != pd.to_datetime(trading_days[idx - 1]).month
        )

        if is_rebalance:
            allocations_pct = {
                a: ((holdings[a] * prices[a]) / value * 100.0 if value > 0 else 0.0)
                for a in universe.anon_universe
            }
            allocations_pct["CASH"] = cash / value * 100.0 if value > 0 else 0.0
            portfolio_state = {
                "cash": cash, "cash_pct": allocations_pct["CASH"],
                "portfolio_value": value, "allocations_pct": allocations_pct,
            }

            raw_target_allocs, meta = decision_fn(current_date, portfolio_state, decision_log, rebalance_days)

            # execute immediately at TODAY's close — no next-day lag
            target_w = {k: v / 100.0 for k, v in raw_target_allocs.items()}
            current_w = {
                a: ((holdings[a] * prices[a]) / value if value > 0 else 0.0)
                for a in universe.anon_universe
            }
            turnover = sum(abs(target_w.get(a, 0.0) - current_w.get(a, 0.0)) for a in universe.anon_universe)
            fee = value * turnover * FEE_RATE
            exec_value = max(value - fee, 0.0)

            cash = exec_value * target_w.get("CASH", 0.0)
            for a in universe.anon_universe:
                p = prices[a]
                holdings[a] = exec_value * target_w.get(a, 0.0) / p if p > 0 else 0.0

            target_allocs = raw_target_allocs
            decision_log[current_date] = {
                "targets": raw_target_allocs, "close_prices": prices, "meta": meta,
                "screener": universe.get_market_screener(current_date),
            }

        new_value = cash + sum(holdings[a] * prices[a] for a in universe.anon_universe)
        entry = {
            "date": current_date,
            "prices": {universe.reverse_map[a]: prices[a] for a in universe.anon_universe},
            "portfolio_value": round(new_value, 2),
            "allocations": {universe.reverse_map.get(k, k): v for k, v in target_allocs.items()},
        }
        if is_rebalance and meta:
            # Save adaptive routing info and any other lightweight meta fields;
            # skip large nested structures (rounds logs, traces) to keep the file small.
            saved_meta = {}
            if "adaptive" in meta:
                saved_meta["adaptive"] = meta["adaptive"]
            if saved_meta:
                entry["meta"] = saved_meta
        results.append(entry)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return results
