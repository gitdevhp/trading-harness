import argparse
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yf_cache"
yf.set_tz_cache_location("/tmp/yf_tz_cache")

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"

DEFAULT_UNIVERSE = ["GLD", "LLY", "KRE", "TSLA", "GOOGL", "TLT", "XLU", "XLE", "XOM", "NVDA"]

RAW_UNIVERSE = []
ANONYMOUS_MAP = {}
REVERSE_MAP = {}
ANONYMOUS_UNIVERSE = []
GLOBAL_DATA_CACHE = {}

def setup_universe(tickers: list):
    global RAW_UNIVERSE, ANONYMOUS_MAP, REVERSE_MAP, ANONYMOUS_UNIVERSE, GLOBAL_DATA_CACHE
    RAW_UNIVERSE = [t.upper() for t in tickers]
    ANONYMOUS_MAP = {ticker: f"ASSET_{chr(65+i)}" for i, ticker in enumerate(RAW_UNIVERSE)}
    REVERSE_MAP = {v: k for k, v in ANONYMOUS_MAP.items()}
    ANONYMOUS_UNIVERSE = list(ANONYMOUS_MAP.values())
    GLOBAL_DATA_CACHE.clear()

def prefetch_data(start_date: str, end_date: str):
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    for ticker in RAW_UNIVERSE:
        df = yf.download(ticker, start=lookback_start, end=end_date, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.ffill()
        required_cols = [c for c in ("Open", "Close") if c in df.columns]
        if required_cols:
            df = df.dropna(subset=required_cols)
        GLOBAL_DATA_CACHE[ANONYMOUS_MAP[ticker]] = df

def run_raw_qwen_agent(current_date: str, holdings_prices: dict) -> dict:
    price_context = ", ".join([f"{a}: ${holdings_prices[a]:.2f}" for a in ANONYMOUS_UNIVERSE])

    prompt = f"""Date: {current_date}
Asset Prices: {price_context}

Provide percentage target allocations for: {ANONYMOUS_UNIVERSE} + CASH.
Percentages must sum to exactly 100.
Format output strictly as JSON: {{"ASSET_A": 10, "ASSET_B": 10, ..., "CASH": 0}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )

    reply = response.choices[0].message.content.strip()
    parsed = re.findall(r'["\']?([A-Za-z0-9_]+)["\']?\s*:\s*([\d\.-]+)', reply)

    if parsed:
        allocs = {k: float(v) for k, v in parsed}
    else:
        allocs = {a: 10.0 for a in ANONYMOUS_UNIVERSE}
        allocs["CASH"] = 0.0

    total = sum(allocs.values())
    return {k: round((v / total * 100.0), 2) if total > 0 else 0.0 for k, v in allocs.items()}

def run_backtest(
    start_date: str = "2023-03-15",
    end_date: str = "2026-04-01",
    initial_capital: float = 100000.0,
    tickers: list = None,
    output_file: str = "qwen_raw_results.json"
):
    if tickers is None or len(tickers) == 0:
        tickers = DEFAULT_UNIVERSE

    setup_universe(tickers)
    prefetch_data(start_date, end_date)

    trading_days = [
        d.strftime("%Y-%m-%d")
        for d in GLOBAL_DATA_CACHE[ANONYMOUS_UNIVERSE[0]].index
        if d.strftime("%Y-%m-%d") >= start_date
    ]

    cash = float(initial_capital)
    holdings = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
    backtest_results = []
    target_allocs = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
    target_allocs["CASH"] = 100.0

    for idx, current_date in enumerate(trading_days):
        prices = {
            t: float(GLOBAL_DATA_CACHE[t].loc[current_date]["Close"])
            for t in ANONYMOUS_UNIVERSE
        }
        total_value = cash + sum(holdings[t] * prices[t] for t in ANONYMOUS_UNIVERSE)

        # Signal is generated from today's close.
        # Execution occurs at the NEXT trading day's OPEN.
        if idx % 5 == 0 or idx == 0:
            target_allocs = run_raw_qwen_agent(current_date, prices)

        if idx > 0 and ((idx - 1) % 5 == 0 or idx == 1):
            prior_date = trading_days[idx - 1]
            execution_prices = {
                t: float(GLOBAL_DATA_CACHE[t].loc[current_date]["Open"])
                for t in ANONYMOUS_UNIVERSE
            }

            total_exec_value = cash + sum(
                holdings[t] * execution_prices[t]
                for t in ANONYMOUS_UNIVERSE
            )

            total_alloc_sum = sum(target_allocs.values())
            if total_alloc_sum > 0:
                norm_targets = {
                    k: (v / total_alloc_sum)
                    for k, v in target_allocs.items()
                }
            else:
                norm_targets = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
                norm_targets["CASH"] = 1.0

            cash = total_exec_value * norm_targets.get("CASH", 0.0)

            for t in ANONYMOUS_UNIVERSE:
                p = execution_prices[t]
                holdings[t] = (
                    (total_exec_value * norm_targets.get(t, 0.0)) / p
                    if p > 0 else 0.0
                )

        new_value = cash + sum(
            holdings[t] * prices[t] for t in ANONYMOUS_UNIVERSE
        )

        real_executed = {
            REVERSE_MAP.get(k, k): v for k, v in target_allocs.items()
        }
        real_prices = {REVERSE_MAP[k]: v for k, v in prices.items()}

        backtest_results.append({
            "date": current_date,
            "prices": real_prices,
            "portfolio_value": round(new_value, 2),
            "harnessed_allocations": real_executed
        })

    with open(output_file, "w") as f:
        json.dump(backtest_results, f, indent=4)

    print(
        f"Raw Qwen Backtest Complete ({len(tickers)} Assets) -> Saved to {output_file}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Raw Qwen Backtest with a custom portfolio."
    )
    parser.add_argument("--tickers", nargs="+", help="List of stock tickers", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2023-03-15", help="Backtest start date")
    parser.add_argument("--end", type=str, default="2026-04-01", help="Backtest end date")
    parser.add_argument("--output", type=str, default="qwen_raw_results.json", help="Output JSON filename")

    args = parser.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        tickers=args.tickers,
        output_file=args.output
    )