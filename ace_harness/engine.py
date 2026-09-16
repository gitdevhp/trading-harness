"""Execution engine — same mechanics as run_backtest() in your no-harness
script: fixed rebalance_days schedule, execution at the next day's open,
turnover-based fees (FEE_RATE), a warmup period that's simulated but
rebased away at the official start so early decisions don't get "free"
history. Decision-making is pulled out into an injectable decision_fn
so the three self-improvement harnesses can each plug in differently.
"""
import json

FEE_RATE = 0.0010


def run_backtest(
    universe,
    decision_fn,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    output_file: str = "results.json",
    rebalance_days: int = 20,
    warmup_days: int = 60,
):
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be at least 1.")
    if warmup_days < 0:
        raise ValueError("warmup_days cannot be negative.")

    all_trading_days = universe.common_trading_days()
    start_position = next((i for i, d in enumerate(all_trading_days) if d >= start_date), None)
    if start_position is None:
        raise ValueError(f"No trading data found on or after {start_date}.")

    simulation_start_position = max(0, start_position - warmup_days)
    trading_days = all_trading_days[simulation_start_position:]
    official_start_idx = start_position - simulation_start_position

    cash = float(initial_capital)
    holdings = {t: 0.0 for t in universe.anon_universe}
    results = []
    decision_log = {}  # date -> {"targets", "close_prices", "meta"} — read-only history for harnesses

    target_allocs = {t: 0.0 for t in universe.anon_universe}
    target_allocs["CASH"] = 100.0

    for idx, current_date in enumerate(trading_days):
        close_prices = universe.close_prices(current_date)
        total_value = cash + sum(holdings[t] * close_prices[t] for t in universe.anon_universe)

        is_rebalance_day = (idx == 0) or (idx % rebalance_days == 0)
        if is_rebalance_day:
            allocations_pct = {
                t: ((holdings[t] * close_prices[t]) / total_value * 100.0 if total_value > 0 else 0.0)
                for t in universe.anon_universe
            }
            allocations_pct["CASH"] = cash / total_value * 100.0 if total_value > 0 else 100.0

            portfolio_state = {
                "cash": cash, "cash_pct": allocations_pct["CASH"],
                "portfolio_value": total_value, "allocations_pct": allocations_pct,
            }
            target_allocs, meta = decision_fn(current_date, portfolio_state, decision_log, rebalance_days)
            decision_log[current_date] = {
                "targets": target_allocs, "close_prices": close_prices, "meta": meta,
                "screener": universe.get_market_screener(current_date),
            }

        is_execution_day = idx > 0 and ((idx - 1) % rebalance_days == 0)

        if idx == official_start_idx and warmup_days > 0:
            rebase_value = cash + sum(holdings[t] * close_prices[t] for t in universe.anon_universe)
            if rebase_value > 0:
                scale = initial_capital / rebase_value
                cash *= scale
                holdings = {t: holdings[t] * scale for t in universe.anon_universe}

        if is_execution_day:
            execution_prices = universe.open_prices(current_date)
            execution_value = cash + sum(holdings[t] * execution_prices[t] for t in universe.anon_universe)

            total_alloc_sum = sum(target_allocs.values())
            norm_targets = (
                {k: v / total_alloc_sum for k, v in target_allocs.items()} if total_alloc_sum > 0
                else {**{t: 0.0 for t in universe.anon_universe}, "CASH": 1.0}
            )

            curr_weights = {
                t: (holdings[t] * execution_prices[t]) / execution_value if execution_value > 0 else 0.0
                for t in universe.anon_universe
            }
            turnover = sum(abs(norm_targets.get(t, 0.0) - curr_weights[t]) for t in universe.anon_universe)
            fee = execution_value * turnover * FEE_RATE
            net_exec_value = execution_value - fee

            cash = net_exec_value * norm_targets.get("CASH", 0.0)
            for t in universe.anon_universe:
                p = execution_prices[t]
                holdings[t] = (net_exec_value * norm_targets.get(t, 0.0)) / p if p > 0 else 0.0

        new_value = cash + sum(holdings[t] * close_prices[t] for t in universe.anon_universe)

        if idx < official_start_idx:
            continue  # warmup days never appear in the output

        real_executed = {universe.reverse_map.get(k, k): v for k, v in target_allocs.items()}
        real_prices = {universe.reverse_map[k]: v for k, v in close_prices.items()}
        results.append({
            "date": current_date, "prices": real_prices,
            "portfolio_value": round(new_value, 2), "allocations": real_executed,
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return results
