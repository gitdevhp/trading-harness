import json
import os
import re
import pandas as pd
import yfinance as yf
from openai import OpenAI

# Connect to the background vLLM server launched by Slurm
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"

UNIVERSE = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "JNJ"]
GLOBAL_DATA_CACHE = {}

# ==========================================
# 1. DATA PREFETCHING & TECHNICAL ANALYSIS
# ==========================================
def prefetch_data(start_date: str, end_date: str):
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    print(f"Prefetching universe data from {lookback_start} to {end_date}...")
    for ticker in UNIVERSE:
        df = yf.download(ticker, start=lookback_start, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        GLOBAL_DATA_CACHE[ticker] = df

def calculate_technical_indicators(past_df: pd.DataFrame) -> dict:
    closes = past_df['Close']
    current_p = float(closes.iloc[-1])
    sma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else current_p
    sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else current_p
    mom14 = ((current_p - float(closes.iloc[-14])) / float(closes.iloc[-14])) * 100 if len(closes) >= 14 else 0.0
    vol20 = (closes.tail(20).std() / sma20) * 100 if len(closes) >= 20 else 0.0

    if len(closes) >= 15:
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rsi = 100.0 if loss == 0 else 100.0 - (100.0 / (1.0 + (gain / loss)))
    else:
        rsi = 50.0

    return {
        "price": round(current_p, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "rsi14": round(rsi, 1),
        "momentum14_pct": round(mom14, 2),
        "volatility20_pct": round(vol20, 2)
    }

# ==========================================
# 2. STRATEGY HARNESS RISK INTERCEPTOR
# ==========================================
def apply_portfolio_harness_rules(raw_allocations: dict, current_allocations: dict, current_date: str) -> dict:
    """Strategy Harness enforcing concentration limits, oversold protections, and cash floors."""
    sanitized = {}

    # Rule 1: Anti-Panic Oversold Guardrail (If RSI < 32, prevent dumping holdings)
    for ticker in UNIVERSE:
        raw_weight = float(raw_allocations.get(ticker, 0.0))
        current_weight = float(current_allocations.get(ticker, 0.0))
        past_df = GLOBAL_DATA_CACHE[ticker].loc[:current_date]
        tech = calculate_technical_indicators(past_df)

        if tech['rsi14'] < 32.0 and raw_weight < current_weight:
            sanitized[ticker] = current_weight  # Block bottom-selling
        else:
            sanitized[ticker] = raw_weight

    # Rule 2: Single-Asset Concentration Cap (Max 30% per stock)
    for ticker in UNIVERSE:
        sanitized[ticker] = min(30.0, sanitized[ticker])

    # Rule 3: Minimum Cash Floor (Enforce 15% Cash minimum)
    total_stock_weight = sum(sanitized[t] for t in UNIVERSE)
    if total_stock_weight > 85.0:
        scale_factor = 85.0 / total_stock_weight
        for t in UNIVERSE:
            sanitized[t] *= scale_factor
        sanitized["CASH"] = 15.0
    else:
        sanitized["CASH"] = max(15.0, 100.0 - total_stock_weight)

    # Normalize to 100%
    total = sum(sanitized.values())
    return {k: round((v / total) * 100.0, 2) for k, v in sanitized.items()}

# ==========================================
# 3. DAILY ReAct AGENT WITH HARNESS
# ==========================================
def run_daily_agent(current_date: str, portfolio_state: dict) -> dict:
    
    def get_market_screener(arg: str = "") -> str:
        screener = []
        for ticker in UNIVERSE:
            past_df = GLOBAL_DATA_CACHE[ticker].loc[:current_date]
            tech = calculate_technical_indicators(past_df)
            screener.append(
                f"{ticker}: Price=${tech['price']} | 20d-SMA=${tech['sma20']} | "
                f"RSI={tech['rsi14']} | 14d-Mom={tech['momentum14_pct']}% | Vol={tech['volatility20_pct']}%"
            )
        return "\n".join(screener)

    def get_portfolio_status(arg: str = "") -> str:
        alloc_str = ", ".join([f"{k}: {v:.1f}%" for k, v in portfolio_state['allocations_pct'].items()])
        return (f"Total Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                f"Cash: ${portfolio_state['cash']:,.2f} ({portfolio_state['cash_pct']:.1f}%)\n"
                f"Current Equity Allocations: {alloc_str}")

    available_tools = {
        "get_market_screener": get_market_screener,
        "get_portfolio_status": get_portfolio_status
    }

    react_system_prompt = f"""You are an autonomous quantitative portfolio manager evaluating market opportunities on {current_date}.

Available Universe: {UNIVERSE} + CASH

Available Tools:
- get_market_screener[]: Get technicals (RSI, 14d Momentum, SMAs, Volatility) for all tickers in the universe.
- get_portfolio_status[]: Fetch current cash, portfolio value, and asset exposure percentages.

Goal: Maximize long-term risk-adjusted returns by shifting capital into higher-opportunity assets while managing downside risks.

Guidelines for Reasoning:
- Compare cross-sectional relative strength: allocate higher weights to assets with positive momentum and healthy setups.
- Reduce allocation to weakening or high-volatility assets.
- Total portfolio allocations across selected stocks and CASH MUST sum strictly to 100%.

Output Format:
Thought: <Analyze cross-sectional indicators and explain portfolio weighting shifts>
Action: <tool_name>[]
Observation: <result>
... (repeat Thought/Action/Observation as needed)
Thought: <Declare final optimal portfolio target weights>
Action: Target_Allocations[{{\"AAPL\": 10, \"NVDA\": 25, \"MSFT\": 15, \"AMZN\": 0, \"GOOGL\": 0, \"META\": 15, \"TSLA\": 0, \"JPM\": 10, \"XOM\": 10, \"JNJ\": 0, \"CASH\": 15}}]"""

    messages = [
        {"role": "system", "content": react_system_prompt},
        {"role": "user", "content": f"Today is {current_date}. Evaluate the market universe and output your target portfolio allocation."}
    ]

    trajectory_log = []
    raw_decision = {"CASH": 100.0}

    for step in range(5):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=450,
            stop=["Observation:"]
        )
        reply = response.choices[0].message.content.strip()
        trajectory_log.append(reply)
        messages.append({"role": "assistant", "content": reply})

        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, re.DOTALL)
        if action_match:
            action_name = action_match.group(1)
            action_arg = action_match.group(2).strip()

            if action_name == "Target_Allocations":
                try:
                    parsed = json.loads(action_arg)
                    raw_decision = {k: float(v) for k, v in parsed.items()}
                except Exception:
                    pass
                break

            if action_name in available_tools:
                obs_text = f"Observation: {available_tools[action_name](action_arg)}"
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."

            trajectory_log.append(obs_text)
            messages.append({"role": "user", "content": obs_text})

    return {"date": current_date, "raw_allocations": raw_decision, "trajectory": "\n".join(trajectory_log)}

# ==========================================
# 4. MULTI-ASSET BACKTEST ENGINE
# ==========================================
def run_backtest(start_date: str, end_date: str, initial_capital: float = 100000.0):
    prefetch_data(start_date, end_date)
    trading_days = [d.strftime('%Y-%m-%d') for d in GLOBAL_DATA_CACHE[UNIVERSE[0]].index if d.strftime('%Y-%m-%d') >= start_date]

    cash = initial_capital
    holdings = {t: 0.0 for t in UNIVERSE}
    output_filename = "react_results_harness_portfolio.json"
    backtest_results = []

    for current_date in trading_days:
        prices = {t: float(GLOBAL_DATA_CACHE[t].loc[current_date]['Close']) for t in UNIVERSE}
        total_portfolio_value = cash + sum(holdings[t] * prices[t] for t in UNIVERSE)

        allocations_pct = {t: (holdings[t] * prices[t] / total_portfolio_value * 100.0) for t in UNIVERSE}
        cash_pct = (cash / total_portfolio_value) * 100.0

        portfolio_state = {
            "cash": cash,
            "cash_pct": cash_pct,
            "portfolio_value": total_portfolio_value,
            "allocations_pct": allocations_pct
        }

        result = run_daily_agent(current_date, portfolio_state)
        raw_targets = result["raw_allocations"]

        # Intercept raw targets using Strategy Harness
        harnessed_targets = apply_portfolio_harness_rules(raw_targets, allocations_pct, current_date)

        # Portfolio Rebalance Execution
        norm_targets = {k: (v / 100.0) for k, v in harnessed_targets.items()}
        cash = total_portfolio_value * norm_targets.get("CASH", 0.0)

        for t in UNIVERSE:
            target_cash_for_asset = total_portfolio_value * norm_targets.get(t, 0.0)
            holdings[t] = target_cash_for_asset / prices[t]

        new_portfolio_value = cash + sum(holdings[t] * prices[t] for t in UNIVERSE)

        result.update({
            "portfolio_value": round(new_portfolio_value, 2),
            "harnessed_allocations": harnessed_targets
        })
        backtest_results.append(result)

        if len(backtest_results) % 10 == 0 or current_date == trading_days[-1]:
            with open(output_filename, "w") as f:
                json.dump(backtest_results, f, indent=4)

        print(f"[{current_date}] Portfolio Value: ${new_portfolio_value:,.2f} | Harnessed Allocations: {harnessed_targets}")

if __name__ == "__main__":
    run_backtest(start_date="2023-03-15", end_date="2026-04-01")