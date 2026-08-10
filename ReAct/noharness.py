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
# 1. DATA PREFETCHING
# ==========================================
def prefetch_data(start_date: str, end_date: str):
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    print(f"Prefetching universe data from {lookback_start} to {end_date}...")
    for ticker in UNIVERSE:
        df = yf.download(ticker, start=lookback_start, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        GLOBAL_DATA_CACHE[ticker] = df

# ==========================================
# 2. UNRESTRICTED PLAIN ReAct AGENT
# ==========================================
def run_daily_agent(current_date: str, portfolio_state: dict) -> dict:
    
    def get_market_prices(arg: str = "") -> str:
        """Returns raw closing prices for all assets in the universe."""
        prices = {}
        for ticker in UNIVERSE:
            past_df = GLOBAL_DATA_CACHE[ticker].loc[:current_date]
            prices[ticker] = round(float(past_df['Close'].iloc[-1]), 2)
        return json.dumps(prices)

    def get_price_history(ticker: str) -> str:
        """Returns recent 10-day raw closing price history for a given asset."""
        ticker = ticker.strip().upper()
        if ticker not in UNIVERSE:
            return f"Ticker {ticker} not in universe."
        past_df = GLOBAL_DATA_CACHE[ticker].loc[:current_date].tail(10)
        history = {d.strftime("%Y-%m-%d"): round(float(p), 2) for d, p in past_df['Close'].items()}
        return json.dumps(history)

    def get_portfolio_status(arg: str = "") -> str:
        alloc_str = ", ".join([f"{k}: {v:.1f}%" for k, v in portfolio_state['allocations_pct'].items()])
        return (f"Total Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                f"Cash: ${portfolio_state['cash']:,.2f} ({portfolio_state['cash_pct']:.1f}%)\n"
                f"Current Equity Allocations: {alloc_str}")

    available_tools = {
        "get_market_prices": get_market_prices,
        "get_price_history": get_price_history,
        "get_portfolio_status": get_portfolio_status
    }

    react_system_prompt = f"""You are an autonomous trading agent evaluating the market on {current_date}.

Available Assets: {UNIVERSE} + CASH

Available Tools:
- get_market_prices[]: Fetch latest closing prices for all universe assets.
- get_price_history[ticker]: Fetch raw price history for the last 10 trading days of a ticker.
- get_portfolio_status[]: Fetch current cash, total valuation, and exposure percentages.

Goal: Analyze price action directly through reasoning and allocate portfolio capital to maximize returns.

Format Requirements:
Thought: <Analyze market prices and reason through asset selection>
Action: <tool_name>[<optional_argument>]
Observation: <tool output>
... (repeat Thought/Action/Observation as needed)
Thought: <Declare final allocation weights>
Action: Target_Allocations[{{\"AAPL\": 10, \"NVDA\": 20, \"MSFT\": 15, \"AMZN\": 0, \"GOOGL\": 10, \"META\": 15, \"TSLA\": 0, \"JPM\": 10, \"XOM\": 10, \"JNJ\": 0, \"CASH\": 10}}]

Note: Target allocations across all assets and CASH MUST sum to 100."""

    messages = [
        {"role": "system", "content": react_system_prompt},
        {"role": "user", "content": f"Today is {current_date}. Evaluate price action and rebalance the portfolio."}
    ]

    trajectory_log = []
    final_decision = {"CASH": 100.0}

    for step in range(5):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=400,
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
                    final_decision = {k: float(v) for k, v in parsed.items()}
                except Exception:
                    pass
                break

            if action_name in available_tools:
                obs_text = f"Observation: {available_tools[action_name](action_arg)}"
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."

            trajectory_log.append(obs_text)
            messages.append({"role": "user", "content": obs_text})

    return {"date": current_date, "allocations": final_decision, "trajectory": "\n".join(trajectory_log)}

# ==========================================
# 3. BACKTEST ENGINE
# ==========================================
def run_backtest(start_date: str, end_date: str, initial_capital: float = 100000.0):
    prefetch_data(start_date, end_date)
    trading_days = [d.strftime('%Y-%m-%d') for d in GLOBAL_DATA_CACHE[UNIVERSE[0]].index if d.strftime('%Y-%m-%d') >= start_date]

    cash = initial_capital
    holdings = {t: 0.0 for t in UNIVERSE}
    output_filename = "react_results_plain_portfolio.json"
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
        target_pcts = result["allocations"]

        # Directly execute trades to match target percentages
        total_target_pct = sum(target_pcts.values()) if sum(target_pcts.values()) > 0 else 100.0
        norm_targets = {k: (v / total_target_pct) for k, v in target_pcts.items()}

        cash = total_portfolio_value * norm_targets.get("CASH", 0.0)
        for t in UNIVERSE:
            target_cash_for_asset = total_portfolio_value * norm_targets.get(t, 0.0)
            holdings[t] = target_cash_for_asset / prices[t]

        new_portfolio_value = cash + sum(holdings[t] * prices[t] for t in UNIVERSE)

        result.update({
            "portfolio_value": round(new_portfolio_value, 2),
            "executed_allocations": {k: round(v * 100, 2) for k, v in norm_targets.items()}
        })
        backtest_results.append(result)

        if len(backtest_results) % 10 == 0 or current_date == trading_days[-1]:
            with open(output_filename, "w") as f:
                json.dump(backtest_results, f, indent=4)

        print(f"[{current_date}] Portfolio Value: ${new_portfolio_value:,.2f} | Target Allocations: {result['executed_allocations']}")

if __name__ == "__main__":
    run_backtest(start_date="2023-03-15", end_date="2026-04-01")