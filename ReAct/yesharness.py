import json
import os
import re
import pandas as pd
import yfinance as yf
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"

RAW_UNIVERSE = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "JNJ"]

# Anonymization Mapping to Eliminate Hindsight Bias
ANONYMOUS_MAP = {ticker: f"ASSET_{chr(65+i)}" for i, ticker in enumerate(RAW_UNIVERSE)}
REVERSE_MAP = {v: k for k, v in ANONYMOUS_MAP.items()}
ANONYMOUS_UNIVERSE = list(ANONYMOUS_MAP.values())

GLOBAL_DATA_CACHE = {}

def prefetch_data(start_date: str, end_date: str):
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    for ticker in RAW_UNIVERSE:
        df = yf.download(ticker, start=lookback_start, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        GLOBAL_DATA_CACHE[ANONYMOUS_MAP[ticker]] = df

def calculate_technical_indicators(past_df: pd.DataFrame) -> dict:
    closes = past_df['Close']
    current_p = float(closes.iloc[-1])
    sma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else current_p
    sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else current_p
    mom14 = ((current_p - float(closes.iloc[-14])) / float(closes.iloc[-14])) * 100 if len(closes) >= 14 else 0.0
    vol20 = (closes.tail(20).std() / sma20) * 100 if len(closes) >= 20 else 0.0

    return {
        "price": round(current_p, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "momentum14_pct": round(mom14, 2),
        "volatility20_pct": round(vol20, 2)
    }

# Dynamic Risk Interceptor
def apply_portfolio_harness_rules(raw_allocations: dict, current_allocations: dict, current_date: str) -> dict:
    sanitized = {}
    bullish_count = 0

    # Rule 1: Trailing Stop & Trend Override
    for asset in ANONYMOUS_UNIVERSE:
        past_df = GLOBAL_DATA_CACHE[asset].loc[:current_date]
        tech = calculate_technical_indicators(past_df)
        raw_weight = float(raw_allocations.get(asset, 0.0))
        
        # Track market regime
        if tech['price'] > tech['sma20']:
            bullish_count += 1

        # Hard Risk Override: Force allocation to 0% if asset breaks below 50d SMA and has negative momentum
        if tech['price'] < tech['sma50'] and tech['momentum14_pct'] < -3.0:
            sanitized[asset] = 0.0
        else:
            sanitized[asset] = raw_weight

    # Rule 2: Regime-Adaptive Cash Floor (0% in strong bull markets, up to 25% in bear markets)
    market_health_ratio = bullish_count / len(ANONYMOUS_UNIVERSE)
    min_cash_floor = 0.0 if market_health_ratio >= 0.6 else (25.0 * (1.0 - market_health_ratio))

    # Rule 3: Single Asset Concentration Cap (Max 35%)
    for asset in ANONYMOUS_UNIVERSE:
        sanitized[asset] = min(35.0, sanitized[asset])

    # Rule 4: Churn Buffer (Ignore tiny reallocation adjustments under 3%)
    for asset in ANONYMOUS_UNIVERSE:
        curr_w = float(current_allocations.get(asset, 0.0))
        if abs(sanitized[asset] - curr_w) < 3.0:
            sanitized[asset] = curr_w

    total_stock_weight = sum(sanitized[t] for t in ANONYMOUS_UNIVERSE)
    max_equity_allowed = 100.0 - min_cash_floor

    if total_stock_weight > max_equity_allowed and total_stock_weight > 0:
        scale_factor = max_equity_allowed / total_stock_weight
        for t in ANONYMOUS_UNIVERSE:
            sanitized[t] *= scale_factor
        sanitized["CASH"] = min_cash_floor
    else:
        sanitized["CASH"] = 100.0 - sum(sanitized[t] for t in ANONYMOUS_UNIVERSE)

    total = sum(sanitized.values())
    return {k: round((v / total) * 100.0, 2) for k, v in sanitized.items()}

def run_daily_agent(current_date: str, portfolio_state: dict) -> dict:
    def get_market_screener(arg: str = "") -> str:
        screener = []
        for asset in ANONYMOUS_UNIVERSE:
            past_df = GLOBAL_DATA_CACHE[asset].loc[:current_date]
            tech = calculate_technical_indicators(past_df)
            screener.append(f"{asset}: Price=${tech['price']} | 20d-SMA=${tech['sma20']} | 50d-SMA=${tech['sma50']} | 14d-Mom={tech['momentum14_pct']}%")
        return "\n".join(screener)

    def get_portfolio_status(arg: str = "") -> str:
        alloc_str = ", ".join([f"{k}: {v:.1f}%" for k, v in portfolio_state['allocations_pct'].items()])
        return f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | Cash: {portfolio_state['cash_pct']:.1f}%\nAllocations: {alloc_str}"

    available_tools = {"get_market_screener": get_market_screener, "get_portfolio_status": get_portfolio_status}

    react_system_prompt = f"""You are an autonomous quantitative portfolio manager evaluating market opportunities on {current_date}.

Available Assets: {ANONYMOUS_UNIVERSE} + CASH

Available Tools:
- get_market_screener[]: Fetch quantitative metrics (Prices, SMAs, Momentum) for all anonymized assets.
- get_portfolio_status[]: Fetch current cash and asset allocations.

Output Format:
Thought: <Analyze cross-sectional momentum and market health>
Action: <tool_name>[]
Observation: <result>
... (repeat Thought/Action/Observation as needed)
Thought: <Declare final target weights>
Action: Target_Allocations[{{\"ASSET_A\": 15, \"ASSET_B\": 25, \"ASSET_C\": 15, \"ASSET_D\": 10, \"ASSET_E\": 10, \"ASSET_F\": 10, \"ASSET_G\": 0, \"ASSET_H\": 5, \"ASSET_I\": 5, \"ASSET_J\": 0, \"CASH\": 5}}]"""

    messages = [{"role": "system", "content": react_system_prompt}, {"role": "user", "content": f"Date: {current_date}. Output your target allocations."}]
    trajectory_log, raw_decision = [], {"CASH": 100.0}

    for step in range(5):
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=400, stop=["Observation:"])
        reply = response.choices[0].message.content.strip()
        trajectory_log.append(reply)
        messages.append({"role": "assistant", "content": reply})

        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, re.DOTALL)
        if action_match:
            action_name, action_arg = action_match.group(1), action_match.group(2).strip()
            if action_name == "Target_Allocations":
                try:
                    raw_decision = {k: float(v) for k, v in json.loads(action_arg).items()}
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

def run_backtest(start_date: str, end_date: str, initial_capital: float = 100000.0):
    prefetch_data(start_date, end_date)
    trading_days = [d.strftime('%Y-%m-%d') for d in GLOBAL_DATA_CACHE[ANONYMOUS_UNIVERSE[0]].index if d.strftime('%Y-%m-%d') >= start_date]

    cash = initial_capital
    holdings = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
    backtest_results = []

    for current_date in trading_days:
        prices = {t: float(GLOBAL_DATA_CACHE[t].loc[current_date]['Close']) for t in ANONYMOUS_UNIVERSE}
        total_value = cash + sum(holdings[t] * prices[t] for t in ANONYMOUS_UNIVERSE)
        allocations_pct = {t: (holdings[t] * prices[t] / total_value * 100.0) for t in ANONYMOUS_UNIVERSE}

        portfolio_state = {"cash": cash, "cash_pct": (cash / total_value) * 100.0, "portfolio_value": total_value, "allocations_pct": allocations_pct}
        
        result = run_daily_agent(current_date, portfolio_state)
        harnessed_targets = apply_portfolio_harness_rules(result["raw_allocations"], allocations_pct, current_date)

        norm_targets = {k: (v / 100.0) for k, v in harnessed_targets.items()}
        cash = total_value * norm_targets.get("CASH", 0.0)

        for t in ANONYMOUS_UNIVERSE:
            holdings[t] = (total_value * norm_targets.get(t, 0.0)) / prices[t]

        new_value = cash + sum(holdings[t] * prices[t] for t in ANONYMOUS_UNIVERSE)
        
        # Convert anonymous tags back to real tickers for output log transparency
        real_executed = {REVERSE_MAP.get(k, k): v for k, v in harnessed_targets.items()}
        
        result.update({"portfolio_value": round(new_value, 2), "executed_allocations": real_executed})
        backtest_results.append(result)

        print(f"[{current_date}] Portfolio Value: ${new_value:,.2f} | Real Executed Allocations: {real_executed}")

if __name__ == "__main__":
    run_backtest(start_date="2023-03-15", end_date="2026-04-01")