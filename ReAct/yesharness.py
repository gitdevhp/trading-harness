import argparse
import json
import os
import random
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

# Global containers dynamically populated at runtime
RAW_UNIVERSE = []
ANONYMOUS_MAP = {}
REVERSE_MAP = {}
ANONYMOUS_UNIVERSE = []
GLOBAL_DATA_CACHE = {}
POSITION_PEAKS = {}

PRICE_INDEX_BASE = {}
DISPLAY_MAP = {}
DISPLAY_REVERSE = {}

def setup_universe(tickers: list):
    """Dynamically sets up asset mapping and resets global caches."""
    global RAW_UNIVERSE, ANONYMOUS_MAP, REVERSE_MAP, ANONYMOUS_UNIVERSE, GLOBAL_DATA_CACHE, POSITION_PEAKS
    global PRICE_INDEX_BASE, DISPLAY_MAP, DISPLAY_REVERSE

    RAW_UNIVERSE = [t.upper() for t in tickers]
    ANONYMOUS_MAP = {ticker: f"ASSET_{chr(65+i)}" for i, ticker in enumerate(RAW_UNIVERSE)}
    REVERSE_MAP = {v: k for k, v in ANONYMOUS_MAP.items()}
    ANONYMOUS_UNIVERSE = list(ANONYMOUS_MAP.values())

    GLOBAL_DATA_CACHE.clear()
    POSITION_PEAKS.clear()
    PRICE_INDEX_BASE.clear()
    DISPLAY_MAP.clear()
    DISPLAY_REVERSE.clear()

def reshuffle_display_map():
    """Shuffle which display label each internal asset gets this rebalance period."""
    global DISPLAY_MAP, DISPLAY_REVERSE
    n = len(ANONYMOUS_UNIVERSE)
    labels = [f"ASSET_{chr(65+i)}" for i in range(n)]
    random.shuffle(labels)
    DISPLAY_MAP = {internal: display for internal, display in zip(ANONYMOUS_UNIVERSE, labels)}
    DISPLAY_REVERSE = {v: k for k, v in DISPLAY_MAP.items()}

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

def calculate_technical_indicators(past_df: pd.DataFrame) -> dict:
    closes = past_df["Close"]
    current_p = float(closes.iloc[-1])

    sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else current_p
    sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else current_p
    mom120 = ((current_p - float(closes.iloc[-120])) / float(closes.iloc[-120])) * 100.0 if len(closes) >= 120 else 0.0

    daily_returns = closes.pct_change().tail(60).dropna()
    total_vol = float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.20

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = (
        float(np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252))
        if len(downside_returns) > 5 else max(total_vol * 0.7, 0.05)
    )

    return {
        "price": round(current_p, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "momentum120_pct": round(mom120, 2),
        "annual_volatility": max(total_vol, 0.05),
        "downside_volatility": max(downside_std, 0.03),
    }

def apply_institutional_risk_harness(
    raw_allocations: dict,
    current_allocations: dict,
    current_date: str,
    holdings_prices: dict,
    drift_threshold: float = 2.0,
    max_asset_cap_pct: float = 20.0,
) -> dict:
    stopped_out = set()
    tech_cache = {}

    for asset in ANONYMOUS_UNIVERSE:
        past_df = GLOBAL_DATA_CACHE[asset].loc[:current_date]
        tech_cache[asset] = calculate_technical_indicators(past_df)

    # 1. Trailing Stop-Loss Check (8% - 15% dynamic stop)
    for asset in ANONYMOUS_UNIVERSE:
        price = holdings_prices[asset]
        curr_hold_pct = current_allocations.get(asset, 0.0)
        tech = tech_cache[asset]
        
        dynamic_stop_loss = float(np.clip(1.2 * tech["annual_volatility"], 0.08, 0.15))

        if curr_hold_pct > 0.5:
            if asset not in POSITION_PEAKS or price > POSITION_PEAKS[asset]:
                POSITION_PEAKS[asset] = price
            peak = POSITION_PEAKS[asset]
            drawdown = (price - peak) / peak if peak > 0 else 0.0
            if drawdown <= -dynamic_stop_loss:
                stopped_out.add(asset)
                POSITION_PEAKS.pop(asset, None)
        else:
            POSITION_PEAKS.pop(asset, None)

    # 2. Portfolio Market Breadth Regime Scale
    healthy_assets = sum(1 for a in ANONYMOUS_UNIVERSE if tech_cache[a]["price"] >= tech_cache[a]["sma200"])
    breadth_ratio = healthy_assets / len(ANONYMOUS_UNIVERSE) if ANONYMOUS_UNIVERSE else 1.0
    
    macro_exposure_scale = 1.0 if breadth_ratio >= 0.6 else max(0.2, breadth_ratio)

    # 3. Filter Individual Asset Allocations
    filtered_targets = {}
    for asset in ANONYMOUS_UNIVERSE:
        requested_w = float(raw_allocations.get(asset, 0.0))

        if asset in stopped_out or requested_w <= 0:
            filtered_targets[asset] = 0.0
            continue

        tech = tech_cache[asset]
        w = min(requested_w, max_asset_cap_pct)

        if tech["price"] < tech["sma200"]:
            w *= 0.5
        if tech["momentum120_pct"] < 0:
            w *= 0.5

        w *= macro_exposure_scale
        filtered_targets[asset] = round(w, 2)

    # 4. Strict Cash Diversion
    allocated_equity = sum(filtered_targets.values())
    final_targets = filtered_targets.copy()

    if allocated_equity > 100.0:
        norm_factor = 100.0 / allocated_equity
        final_targets = {k: round(v * norm_factor, 2) for k, v in final_targets.items()}
        final_targets["CASH"] = 0.0
    else:
        final_targets["CASH"] = round(max(0.0, 100.0 - allocated_equity), 2)

    # 5. Rebalance Drift Filter
    if not stopped_out:
        max_drift = max([abs(final_targets.get(a, 0.0) - current_allocations.get(a, 0.0)) for a in ANONYMOUS_UNIVERSE])
        if max_drift < drift_threshold:
            return current_allocations

    return final_targets

def get_market_screener(current_date: str, price_index: dict) -> str:
    screener = []
    for asset in ANONYMOUS_UNIVERSE:
        display = DISPLAY_MAP[asset]
        closes = GLOBAL_DATA_CACHE[asset].loc[:current_date]["Close"]
        cp = float(closes.iloc[-1])
        sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
        above_200d = cp >= sma200
        mom120 = ((cp - float(closes.iloc[-120])) / float(closes.iloc[-120])) * 100.0 if len(closes) >= 120 else 0.0
        idx = price_index.get(asset, 100.0)
        screener.append(f"{display}: Idx={idx:.1f} | Above200d={above_200d} | 120d-Mom={mom120:.1f}%")
    return "\n".join(screener)

def run_react_agent(current_date: str, portfolio_state: dict, price_index: dict) -> dict:
    display_universe = [DISPLAY_MAP[a] for a in ANONYMOUS_UNIVERSE]

    def get_portfolio_status(arg: str = "") -> str:
        alloc_str = ", ".join([f"{k}: {v:.1f}%" for k, v in portfolio_state["allocations_pct"].items()])
        return f"Portfolio Value (normalized) | Cash: {portfolio_state['cash_pct']:.1f}%\nAllocations: {alloc_str}"

    def tool_screener(arg: str = "") -> str:
        return get_market_screener(current_date, price_index)

    available_tools = {"get_market_screener": tool_screener, "get_portfolio_status": get_portfolio_status}

    first_display = display_universe[0] if display_universe else "ASSET_A"
    system_prompt = f"""You are an autonomous ReAct Portfolio Manager on {current_date}.
Assets: {display_universe} + CASH

Tools:
- get_market_screener[]
- get_portfolio_status[]

Format:
Thought: <Reasoning step>
Action: <tool_name>[]
Observation: <tool response>
...
Thought: <Final allocation decision>
Action: Target_Allocations[{{"{first_display}": 15, ..., "CASH": 10}}]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Date: {current_date}. Analyze market conditions and set target allocations."}
    ]

    raw_decision = {"CASH": 100.0}

    for step in range(4):
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=600, stop=["Observation:"]
        )
        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})

        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, re.DOTALL)
        if action_match:
            action_name, action_arg = action_match.group(1), action_match.group(2).strip()

            if action_name == "Target_Allocations":
                parsed = re.findall(r'["\']?([A-Za-z0-9_]+)["\']?\s*:\s*([\d\.-]+)', action_arg)
                if parsed:
                    display_allocs = {k: float(v) for k, v in parsed}
                    raw_decision = {}
                    for k, v in display_allocs.items():
                        internal = DISPLAY_REVERSE.get(k, k)
                        raw_decision[internal] = v
                break

            if action_name in available_tools:
                obs_text = f"Observation: {available_tools[action_name](action_arg)}"
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."
            messages.append({"role": "user", "content": obs_text})

    total = sum(raw_decision.values())
    return {k: round((v / total * 100.0), 2) if total > 0 else 0.0 for k, v in raw_decision.items()}

def run_backtest(
    start_date: str = "2023-03-15",
    end_date: str = "2026-04-01",
    initial_capital: float = 100000.0,
    tickers: list = None,
    output_file: str = "react_harness_results.json"
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

    last_raw_allocs = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
    last_raw_allocs["CASH"] = 100.0
    pending_harnessed_targets = {"CASH": 100.0}

    for idx, current_date in enumerate(trading_days):
        close_prices = {
            t: float(GLOBAL_DATA_CACHE[t].loc[current_date]["Close"])
            for t in ANONYMOUS_UNIVERSE
        }

        total_value = cash + sum(
            holdings[t] * close_prices[t]
            for t in ANONYMOUS_UNIVERSE
        )

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

        # Initialise base prices at the first observation
        if idx == 0:
            for a in ANONYMOUS_UNIVERSE:
                PRICE_INDEX_BASE[a] = close_prices[a]

        price_index = {
            a: close_prices[a] / PRICE_INDEX_BASE[a] * 100.0
            for a in ANONYMOUS_UNIVERSE
        }

        # Generate the signal from the current day's CLOSE.
        # The signal is deliberately NOT executed at this same close.
        if idx % 5 == 0 or idx == 0:
            reshuffle_display_map()
            portfolio_state = {
                "cash": cash,
                "cash_pct": allocations_pct["CASH"],
                "portfolio_value": total_value,
                "allocations_pct": allocations_pct,
            }
            raw_target_allocs = run_react_agent(
                current_date,
                portfolio_state,
                price_index,
            )
            last_raw_allocs = raw_target_allocs
        else:
            raw_target_allocs = last_raw_allocs

        # Harness the signal using information available at today's close.
        # Execution happens only on the NEXT trading day's OPEN.
        # At every close we compute the target for the next session.
        # Keep it pending until the next OPEN.
        if idx % 5 == 0 or idx == 0:
            pending_harnessed_targets = apply_institutional_risk_harness(
                raw_target_allocs,
                allocations_pct,
                current_date,
                close_prices,
            )

        harnessed_targets = pending_harnessed_targets

        if idx > 0 and (idx - 1) % 5 == 0:
            execution_prices = {
                t: float(GLOBAL_DATA_CACHE[t].loc[current_date]["Open"])
                for t in ANONYMOUS_UNIVERSE
            }

            execution_value = cash + sum(
                holdings[t] * execution_prices[t]
                for t in ANONYMOUS_UNIVERSE
            )

            total_harnessed_sum = sum(harnessed_targets.values())

            if total_harnessed_sum > 0:
                norm_targets = {
                    k: v / total_harnessed_sum
                    for k, v in harnessed_targets.items()
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
            for k, v in harnessed_targets.items()
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
        f"ReAct + Risk Harness Backtest Complete ({len(tickers)} Assets) -> Saved to {output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReAct + Risk Harness Backtest with a custom portfolio.")
    parser.add_argument("--tickers", nargs="+", help="List of stock tickers (e.g. AAPL NVDA MSFT AMZN)", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2023-03-15", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-04-01", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="react_harness_results.json", help="Output JSON filename")

    args = parser.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        tickers=args.tickers,
        output_file=args.output
    ) 