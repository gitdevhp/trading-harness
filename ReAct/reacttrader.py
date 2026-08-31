import json
import os
import re
import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yf_cache"
yf.set_tz_cache_location("/tmp/yf_tz_cache")

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"

# Single Stock Configuration
RAW_TICKER = "INTC"
ANONYMOUS_MAP = {RAW_TICKER: "ASSET_A"}
REVERSE_MAP = {"ASSET_A": RAW_TICKER}
ANONYMOUS_UNIVERSE = ["ASSET_A"]

GLOBAL_DATA_CACHE = {}
POSITION_PEAKS = {}  # Tracking highest asset price while position is open


# ==========================================
# 1. DATA PREFETCHING & TECHNICAL INDICATORS
# ==========================================
def prefetch_data(start_date: str, end_date: str):
    lookback_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=365)
    ).strftime("%Y-%m-%d")
    print(
        f"Prefetching single stock ({RAW_TICKER}) data from {lookback_start} to {end_date}..."
    )
    df = yf.download(
        RAW_TICKER,
        start=lookback_start,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.ffill().bfill()
    GLOBAL_DATA_CACHE["ASSET_A"] = df


def calculate_technical_indicators(past_df: pd.DataFrame) -> dict:
    closes = past_df["Close"]
    highs = past_df["High"] if "High" in past_df.columns else closes
    lows = past_df["Low"] if "Low" in past_df.columns else closes

    current_p = float(closes.iloc[-1])

    sma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else current_p
    sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else current_p

    std20 = float(closes.tail(20).std()) if len(closes) >= 20 else 0.0
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)

    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).tail(14).mean()
    loss = (-delta.where(delta < 0, 0)).tail(14).mean()
    if loss == 0 or np.isnan(loss):
        rsi = 100.0 if gain > 0 else 50.0
    else:
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)))

    supp_20d = float(lows.tail(20).min()) if len(lows) >= 20 else current_p
    res_20d = float(highs.tail(20).max()) if len(highs) >= 20 else current_p

    dist_to_support_pct = (
        ((current_p - bb_lower) / bb_lower) * 100.0 if bb_lower > 0 else 0.0
    )
    dist_to_resistance_pct = (
        ((bb_upper - current_p) / current_p) * 100.0 if current_p > 0 else 0.0
    )

    trend = "BULLISH" if sma20 >= sma200 else "BEARISH"

    return {
        "price": round(current_p, 2),
        "sma20": round(sma20, 2),
        "sma200": round(sma200, 2),
        "trend": trend,
        "support_bb_lower": round(bb_lower, 2),
        "resistance_bb_upper": round(bb_upper, 2),
        "support_20d_low": round(supp_20d, 2),
        "resistance_20d_high": round(res_20d, 2),
        "rsi14": round(rsi, 1),
        "dist_to_support_pct": round(dist_to_support_pct, 2),
        "dist_to_resistance_pct": round(dist_to_resistance_pct, 2),
    }


# ==========================================
# 2. QUANTITATIVE RISK HARNESS
# ==========================================
def apply_institutional_risk_harness(
    raw_allocations: dict,
    current_allocations: dict,
    current_date: str,
    holdings_prices: dict,
    drift_threshold: float = 2.0,  # Lowered from 10.0 to execute swing signals responsively
    stop_loss_pct: float = 0.08,  # Tightened trailing stop from 10% to 8%
) -> dict:

    asset = "ASSET_A"
    price = holdings_prices[asset]
    curr_holdings_pct = current_allocations.get(asset, 0.0)

    # Manage trailing stop peak
    if curr_holdings_pct > 5.0:
        if asset not in POSITION_PEAKS or price > POSITION_PEAKS[asset]:
            POSITION_PEAKS[asset] = price
        peak = POSITION_PEAKS[asset]
        drawdown = (price - peak) / peak if peak > 0 else 0.0

        if drawdown <= -stop_loss_pct:
            POSITION_PEAKS.pop(asset, None)
            return {"ASSET_A": 0.0, "CASH": 100.0}
    else:
        POSITION_PEAKS.pop(asset, None)

    past_df = GLOBAL_DATA_CACHE[asset].loc[:current_date]
    tech = calculate_technical_indicators(past_df)
    requested_asset_w = float(raw_allocations.get(asset, 0.0))

    # Overbought or near resistance exit filter
    if tech["rsi14"] > 62.0 or tech["dist_to_resistance_pct"] < 1.0:
        return {"ASSET_A": 0.0, "CASH": 100.0}

    # Trend-guided Mean Reversion filter
    if tech["trend"] == "BEARISH":
        if tech["rsi14"] < 32.0 or tech["dist_to_support_pct"] < 1.0:
            target_asset_w = max(requested_asset_w, 70.0)
        elif tech["rsi14"] > 52.0:
            return {"ASSET_A": 0.0, "CASH": 100.0}
        else:
            target_asset_w = requested_asset_w * 0.5
    else:
        if tech["rsi14"] < 42.0 or tech["dist_to_support_pct"] < 2.5:
            target_asset_w = max(requested_asset_w, 95.0)
        else:
            target_asset_w = requested_asset_w

    target_asset_w = min(target_asset_w, 95.0)
    target_cash_w = round(100.0 - target_asset_w, 2)

    final_targets = {"ASSET_A": target_asset_w, "CASH": target_cash_w}

    drift = abs(final_targets["ASSET_A"] - curr_holdings_pct)
    if drift < drift_threshold:
        return current_allocations

    return final_targets


# ==========================================
# 3. REACT DAILY AGENT
# ==========================================
def run_daily_agent(current_date: str, portfolio_state: dict) -> dict:
    def get_market_screener(arg: str = "") -> str:
        past_df = GLOBAL_DATA_CACHE["ASSET_A"].loc[:current_date]
        tech = calculate_technical_indicators(past_df)
        return (
            f"ASSET_A: Price=${tech['price']} | Trend={tech['trend']} | Supp(BB_Lower)=${tech['support_bb_lower']} | "
            f"Res(BB_Upper)=${tech['resistance_bb_upper']} | RSI(14)={tech['rsi14']} | "
            f"Dist_to_Supp={tech['dist_to_support_pct']}% | Dist_to_Res={tech['dist_to_resistance_pct']}%"
        )

    def get_portfolio_status(arg: str = "") -> str:
        alloc_str = ", ".join(
            [
                f"{k}: {v:.1f}%"
                for k, v in portfolio_state["allocations_pct"].items()
            ]
        )
        return f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | Cash: {portfolio_state['cash_pct']:.1f}%\nAllocations: {alloc_str}"

    available_tools = {
        "get_market_screener": get_market_screener,
        "get_portfolio_status": get_portfolio_status,
    }

    react_system_prompt = f"""You are an autonomous Support & Resistance Swing Trader operating on {current_date}.

Available Assets: ['ASSET_A', 'CASH']

Tools:
- get_market_screener[]
- get_portfolio_status[]

TRADING RULES:
1. BUY / HOLD: Allocate up to 95% to ASSET_A when near Support (Dist_to_Supp < 2.5% or RSI < 42).
2. SELL / CASH: Move to CASH when near Resistance (Dist_to_Res < 1% or RSI > 62).
3. BEARISH TREND: Scale back exposure if Trend=BEARISH unless deeply oversold (RSI < 32).
4. Target allocations MUST sum to 100%.

Format:
Thought: <Analyze trend, S&R levels, and RSI>
Action: <tool_name>[]
Observation: <result>
...
Thought: <Final allocation decision summing to 100>
Action: Target_Allocations[{{\"ASSET_A\": 95, \"CASH\": 5}}]"""

    messages = [
        {"role": "system", "content": react_system_prompt},
        {
            "role": "user",
            "content": f"Date: {current_date}. Evaluate S&R levels and select target allocation.",
        },
    ]

    trajectory_log = []
    raw_decision = {"ASSET_A": 0.0, "CASH": 100.0}

    for step in range(4):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=350,
            stop=["Observation:"],
        )
        reply = response.choices[0].message.content.strip()
        trajectory_log.append(reply)
        messages.append({"role": "assistant", "content": reply})

        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, re.DOTALL)
        if action_match:
            action_name, action_arg = (
                action_match.group(1),
                action_match.group(2).strip(),
            )

            if action_name == "Target_Allocations":
                try:
                    raw_decision = {
                        k: float(v) for k, v in json.loads(action_arg).items()
                    }
                except Exception:
                    pass
                break

            if action_name in available_tools:
                obs_text = (
                    f"Observation: {available_tools[action_name](action_arg)}"
                )
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."

            trajectory_log.append(obs_text)
            messages.append({"role": "user", "content": obs_text})

    return {
        "date": current_date,
        "raw_allocations": raw_decision,
        "trajectory": "\n".join(trajectory_log),
    }


# ==========================================
# 4. BACKTEST ENGINE & PERFORMANCE EVALUATION
# ==========================================
def run_backtest(
    start_date: str, end_date: str, initial_capital: float = 100000.0
):
    global POSITION_PEAKS
    POSITION_PEAKS = {}

    prefetch_data(start_date, end_date)
    trading_days = [
        d.strftime("%Y-%m-%d")
        for d in GLOBAL_DATA_CACHE["ASSET_A"].index
        if d.strftime("%Y-%m-%d") >= start_date
    ]

    cash = float(initial_capital)
    holdings = {"ASSET_A": 0.0}
    output_filename = "react_results_single_stock.json"
    backtest_results = []
    history = []

    for current_date in trading_days:
        price = float(
            GLOBAL_DATA_CACHE["ASSET_A"].loc[current_date]["Close"]
        )
        prices = {"ASSET_A": price}

        total_value = cash + (holdings["ASSET_A"] * price)

        allocations_pct = {
            "ASSET_A": (
                (holdings["ASSET_A"] * price / total_value * 100.0)
                if total_value > 0
                else 0.0
            ),
            "CASH": (
                (cash / total_value * 100.0) if total_value > 0 else 100.0
            ),
        }

        portfolio_state = {
            "cash": cash,
            "cash_pct": allocations_pct["CASH"],
            "portfolio_value": total_value,
            "allocations_pct": allocations_pct,
        }

        result = run_daily_agent(current_date, portfolio_state)

        harnessed_targets = apply_institutional_risk_harness(
            result["raw_allocations"], allocations_pct, current_date, prices
        )

        norm_targets = {k: (v / 100.0) for k, v in harnessed_targets.items()}
        cash = total_value * norm_targets.get("CASH", 0.05)
        holdings["ASSET_A"] = (
            (total_value * norm_targets.get("ASSET_A", 0.0)) / price
            if price > 0
            else 0.0
        )

        new_value = cash + (holdings["ASSET_A"] * price)
        real_executed = {
            REVERSE_MAP.get(k, k): v for k, v in harnessed_targets.items()
        }

        # Determine trade action tag for plotting
        prev_asset_pct = allocations_pct["ASSET_A"]
        new_asset_pct = harnessed_targets.get("ASSET_A", 0.0)
        if new_asset_pct > prev_asset_pct + 5.0:
            trade_action = "BUY"
        elif new_asset_pct < prev_asset_pct - 5.0:
            trade_action = "SELL"
        else:
            trade_action = "HOLD"

        # Export full dataset for plot.py schema auto-detection
        result.update(
            {
                "price": price,
                "prices": {RAW_TICKER: price},
                "ai_portfolio_value": round(new_value, 2),
                "portfolio_value": round(new_value, 2),
                "harnessed_allocations": real_executed,
                "trade_executed": trade_action,
            }
        )
        backtest_results.append(result)
        history.append({"date": current_date, "value": new_value})

        if (
            len(backtest_results) % 10 == 0
            or current_date == trading_days[-1]
        ):
            with open(output_filename, "w") as f:
                json.dump(backtest_results, f, indent=4)

        print(
            f"[{current_date}] Portfolio Value: ${new_value:,.2f} | Price: ${price:.2f} | Executed: {real_executed}"
        )

    # Metrics computation
    values = pd.Series([h["value"] for h in history])
    total_return = (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
    daily_rets = values.pct_change().dropna()
    sharpe = (
        (daily_rets.mean() / daily_rets.std() * np.sqrt(252))
        if daily_rets.std() > 0
        else 0.0
    )
    peaks = values.cummax()
    max_dd = float(((values - peaks) / peaks).min())

    print("\n" + "=" * 50)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Initial Capital: ${values.iloc[0]:,.2f}")
    print(f"Final Value:     ${values.iloc[-1]:,.2f}")
    print(f"Total Return:    {total_return * 100:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {max_dd * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_backtest(start_date="2023-03-15", end_date="2026-04-01")