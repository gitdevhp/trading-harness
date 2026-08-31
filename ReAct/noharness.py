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

def get_market_screener(current_date: str) -> str:
    screener = []
    for asset in ANONYMOUS_UNIVERSE:
        closes = GLOBAL_DATA_CACHE[asset].loc[:current_date]["Close"]
        cp = float(closes.iloc[-1])
        sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
        mom120 = ((cp - float(closes.iloc[-120])) / float(closes.iloc[-120])) * 100.0 if len(closes) >= 120 else 0.0
        screener.append(
            f"{asset}: Price=${cp:.2f} | 200d-SMA=${sma200:.2f} | 120d-Mom={mom120:.1f}%"
        )
    return "\n".join(screener)

def run_react_agent(current_date: str, portfolio_state: dict) -> dict:
    def get_portfolio_status(arg: str = "") -> str:
        alloc_str = ", ".join(
            [f"{k}: {v:.1f}%" for k, v in portfolio_state["allocations_pct"].items()]
        )
        return (
            f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
            f"Cash: {portfolio_state['cash_pct']:.1f}%\n"
            f"Allocations: {alloc_str}"
        )

    def tool_screener(arg: str = "") -> str:
        return get_market_screener(current_date)

    available_tools = {
        "get_market_screener": tool_screener,
        "get_portfolio_status": get_portfolio_status,
    }

    system_prompt = f"""You are an autonomous ReAct Portfolio Manager on {current_date}.
Assets: {ANONYMOUS_UNIVERSE} + CASH

Tools:
- get_market_screener[]
- get_portfolio_status[]

Format:
Thought: <Reasoning step>
Action: <tool_name>[]
Observation: <tool response>
...
Thought: <Final allocation decision>
Action: Target_Allocations[{{\"ASSET_A\": 15, \"ASSET_B\": 15, ..., \"CASH\": 10}}]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Date: {current_date}. Analyze market conditions and set target allocations.",
        },
    ]

    raw_decision = {"CASH": 100.0}

    for step in range(4):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=600,
            stop=["Observation:"],
        )
        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})

        action_match = re.search(
            r"Action:\s*(\w+)\[(.*?)\]",
            reply,
            re.DOTALL,
        )

        if action_match:
            action_name = action_match.group(1)
            action_arg = action_match.group(2).strip()

            if action_name == "Target_Allocations":
                parsed = re.findall(
                    r'["\']?([A-Za-z0-9_]+)["\']?\s*:\s*([\d\.-]+)',
                    action_arg,
                )
                if parsed:
                    raw_decision = {k: float(v) for k, v in parsed}
                break

            if action_name in available_tools:
                obs_text = (
                    f"Observation: {available_tools[action_name](action_arg)}"
                )
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."

            messages.append({"role": "user", "content": obs_text})

    total = sum(raw_decision.values())

    return {
        k: round((v / total * 100.0), 2) if total > 0 else 0.0
        for k, v in raw_decision.items()
    }

def run_backtest(
    start_date: str = "2023-03-15",
    end_date: str = "2026-04-01",
    initial_capital: float = 100000.0,
    tickers: list = None,
    output_file: str = "react_no_harness_results.json"
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
        close_prices = {
            t: float(GLOBAL_DATA_CACHE[t].loc[current_date]["Close"])
            for t in ANONYMOUS_UNIVERSE
        }

        total_value = cash + sum(
            holdings[t] * close_prices[t]
            for t in ANONYMOUS_UNIVERSE
        )

        # Signal is generated from the close of the current day.
        # It is NOT applied until the next trading day's OPEN.
        if idx % 5 == 0 or idx == 0:
            allocations_pct = {
                t: (
                    (holdings[t] * close_prices[t]) / total_value * 100.0
                    if total_value > 0 else 0.0
                )
                for t in ANONYMOUS_UNIVERSE
            }
            allocations_pct["CASH"] = (
                cash / total_value * 100.0
                if total_value > 0 else 100.0
            )

            portfolio_state = {
                "cash": cash,
                "cash_pct": allocations_pct["CASH"],
                "portfolio_value": total_value,
                "allocations_pct": allocations_pct,
            }

            target_allocs = run_react_agent(
                current_date,
                portfolio_state,
            )

        # The target generated on day idx-1 is executed at today's OPEN.
        if idx > 0 and (idx - 1) % 5 == 0:
            execution_prices = {
                t: float(GLOBAL_DATA_CACHE[t].loc[current_date]["Open"])
                for t in ANONYMOUS_UNIVERSE
            }

            execution_value = cash + sum(
                holdings[t] * execution_prices[t]
                for t in ANONYMOUS_UNIVERSE
            )

            total_alloc_sum = sum(target_allocs.values())

            if total_alloc_sum > 0:
                norm_targets = {
                    k: v / total_alloc_sum
                    for k, v in target_allocs.items()
                }
            else:
                norm_targets = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
                norm_targets["CASH"] = 1.0

            cash = execution_value * norm_targets.get("CASH", 0.0)

            for t in ANONYMOUS_UNIVERSE:
                p = execution_prices[t]
                holdings[t] = (
                    execution_value * norm_targets.get(t, 0.0) / p
                    if p > 0 else 0.0
                )

        new_value = cash + sum(
            holdings[t] * close_prices[t]
            for t in ANONYMOUS_UNIVERSE
        )

        real_executed = {
            REVERSE_MAP.get(k, k): v
            for k, v in target_allocs.items()
        }
        real_prices = {
            REVERSE_MAP[k]: v
            for k, v in close_prices.items()
        }

        backtest_results.append({
            "date": current_date,
            "prices": real_prices,
            "portfolio_value": round(new_value, 2),
            "harnessed_allocations": real_executed
        })

    with open(output_file, "w") as f:
        json.dump(backtest_results, f, indent=4)

    print(
        f"ReAct No Harness Backtest Complete ({len(tickers)} Assets) -> Saved to {output_file}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ReAct (No Harness) Backtest with a custom portfolio."
    )
    parser.add_argument("--tickers", nargs="+", help="List of stock tickers", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2023-03-15", help="Backtest start date")
    parser.add_argument("--end", type=str, default="2026-04-01", help="Backtest end date")
    parser.add_argument("--output", type=str, default="react_no_harness_results.json", help="Output JSON filename")

    args = parser.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        tickers=args.tickers,
        output_file=args.output
    )