import json
import os
import re
import pandas as pd
import yfinance as yf
from openai import OpenAI

# Connect to the background vLLM server launched by Slurm
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"

GLOBAL_DATA_CACHE = {}

def prefetch_data(ticker: str, start_date: str, end_date: str):
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    print(f"Prefetching data for {ticker} from {lookback_start} to {end_date}...")
    df = yf.download(ticker, start=lookback_start, end=end_date, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    GLOBAL_DATA_CACHE[ticker] = df

def calculate_technical_indicators(past_df: pd.DataFrame) -> dict:
    closes = past_df['Close']
    current_p = float(closes.iloc[-1])
    sma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else current_p
    sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else current_p
    
    if len(closes) >= 15:
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rsi = 100.0 if loss == 0 else 100.0 - (100.0 / (1.0 + (gain / loss)))
    else:
        rsi = 50.0

    if current_p > sma20 and sma20 > sma50:
        trend = "STRONG_UPTREND"
    elif current_p < sma20 and sma20 < sma50:
        trend = "STRONG_DOWNTREND"
    else:
        trend = "NEUTRAL"

    return {
        "current_price": round(current_p, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "rsi14": round(rsi, 1),
        "trend": trend
    }

def run_daily_agent(ticker: str, current_date: str, portfolio_state: dict) -> dict:
    df = GLOBAL_DATA_CACHE.get(ticker)
    past_data = df.loc[:current_date]

    def get_technical_analysis(arg: str = "") -> str:
        tech = calculate_technical_indicators(past_data)
        return f"Price: ${tech['current_price']} | 20-SMA: ${tech['sma20']} | 50-SMA: ${tech['sma50']} | RSI: {tech['rsi14']} | Trend: {tech['trend']}"

    def get_portfolio_status(arg: str = "") -> str:
        return f"Cash: ${portfolio_state['cash']:,.2f} | Shares: {portfolio_state['shares']:.2f} | PnL: {portfolio_state['unrealized_pnl_pct']:+.2f}%"

    available_tools = {
        "get_technical_analysis": get_technical_analysis,
        "get_portfolio_status": get_portfolio_status
    }

    # REACT SYSTEM PROMPT WITH MEAN REVERSION & POSITION SCALING
    react_system_prompt = f"""You are a quantitative trend-following trader evaluating {ticker} on {current_date}.

Available Tools:
- get_technical_analysis[]: Fetch 20-SMA, 50-SMA, RSI, and Trend Regime.
- get_portfolio_status[]: Fetch current cash, shares held, and unrealized PnL.

Strategy Mandate:
1. BULLISH UPTREND: BUY_1.0 (deploy 100% available cash).
2. NEUTRAL REGIME: Maintain current position. EXCEPTION: If RSI < 42 (oversold), issue BUY_0.5 (deploy 50% available cash).
3. BEARISH DOWNTREND: SELL_1.0 (exit 100% position to cash).

You MUST use the following format:
Thought: <reasoning about what tool to call or what decision to make>
Action: <tool_name>[]
Observation: <result of tool call>
... (repeat Thought/Action/Observation as needed)
Thought: <final summary reasoning>
Action: Final_Decision[BUY_1.0] OR Final_Decision[BUY_0.5] OR Final_Decision[SELL_1.0] OR Final_Decision[SELL_0.5] OR Final_Decision[HOLD]"""

    messages = [
        {"role": "system", "content": react_system_prompt},
        {"role": "user", "content": f"Analyze {ticker} for {current_date} and issue your trading decision."}
    ]

    trajectory_log = []
    final_decision = "HOLD"

    for step in range(5):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=300,
            stop=["Observation:"]
        )
        reply = response.choices[0].message.content.strip()
        trajectory_log.append(reply)
        messages.append({"role": "assistant", "content": reply})

        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, re.DOTALL)
        if action_match:
            action_name = action_match.group(1)
            action_arg = action_match.group(2).strip()

            if action_name == "Final_Decision":
                final_decision = action_arg.upper()
                break

            if action_name in available_tools:
                obs_text = f"Observation: {available_tools[action_name](action_arg)}"
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."

            trajectory_log.append(obs_text)
            messages.append({"role": "user", "content": obs_text})

    return {"date": current_date, "decision": final_decision, "trajectory": "\n".join(trajectory_log)}

def run_backtest(ticker: str, start_date: str, end_date: str, initial_capital: float = 100000.0):
    prefetch_data(ticker, start_date, end_date)
    df = GLOBAL_DATA_CACHE[ticker]
    trading_days = [d.strftime('%Y-%m-%d') for d in df.index if d.strftime('%Y-%m-%d') >= start_date]

    ai_cash, ai_shares, cost_basis = initial_capital, 0.0, 0.0
    baseline_shares = initial_capital / float(df.loc[trading_days[0]]['Close'])
    
    output_filename = "react_results_high_growth_INTC.json"
    backtest_results = []
    prev_trend = None

    for current_date in trading_days:
        past_df = df.loc[:current_date]
        current_price = float(past_df.iloc[-1]['Close'])
        tech = calculate_technical_indicators(past_df)
        
        unrealized_pnl_pct = (((current_price - cost_basis) / cost_basis) * 100.0) if ai_shares > 0 else 0.0
        portfolio_state = {
            "cash": ai_cash,
            "shares": ai_shares,
            "cost_basis": cost_basis,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "portfolio_value": ai_cash + (ai_shares * current_price)
        }

        is_uptrend = tech["trend"] == "STRONG_UPTREND"
        is_downtrend = tech["trend"] == "STRONG_DOWNTREND"

        # Smart trigger including NEUTRAL oversold dip check
        should_call_llm = (
            current_date == trading_days[0]
            or tech["trend"] != prev_trend
            or (ai_cash > 10 and is_uptrend)
            or (ai_shares > 0.001 and is_downtrend)
            or (ai_cash > 10 and tech["trend"] == "NEUTRAL" and tech["rsi14"] < 42)
        )

        if should_call_llm:
            result = run_daily_agent(ticker, current_date, portfolio_state)
            decision = result["decision"]
            prev_trend = tech["trend"]
        else:
            decision = "HOLD"
            result = {"date": current_date, "decision": "HOLD", "trajectory": "Skipped LLM (Position aligned with trend)"}

        # Dynamic trade execution with scaled position sizing
        trade_action = "HOLD"
        if "BUY" in decision and ai_cash > 10:
            fraction = 0.5 if "0.5" in decision else 1.0
            cash_to_spend = ai_cash * fraction
            shares_bought = cash_to_spend / current_price
            
            # Recalculate weighted cost basis
            total_cost = (ai_shares * cost_basis) + cash_to_spend
            ai_shares += shares_bought
            ai_cash -= cash_to_spend
            cost_basis = total_cost / ai_shares if ai_shares > 0 else 0.0
            
            trade_action = f"BOUGHT {shares_bought:.2f} shares ({fraction*100:.0f}% cash)"

        elif "SELL" in decision and ai_shares > 0.001:
            fraction = 0.5 if "0.5" in decision else 1.0
            shares_to_sell = ai_shares * fraction
            ai_cash += shares_to_sell * current_price
            ai_shares -= shares_to_sell
            
            if ai_shares <= 0.001:
                cost_basis = 0.0
                
            trade_action = f"SOLD {shares_to_sell:.2f} shares ({fraction*100:.0f}% position)"

        ai_portfolio_value = ai_cash + (ai_shares * current_price)
        result.update({
            "price": current_price,
            "trade_executed": trade_action,
            "ai_portfolio_value": round(ai_portfolio_value, 2),
            "baseline_value": round(baseline_shares * current_price, 2)
        })
        backtest_results.append(result)

        # Efficient file saving (every 20 days or on the final day)
        if len(backtest_results) % 20 == 0 or current_date == trading_days[-1]:
            with open(output_filename, "w") as f:
                json.dump(backtest_results, f, indent=4)

        print(f"[{current_date}] Portfolio: ${ai_portfolio_value:,.2f} | Decision: {decision} | Action: {trade_action}")

if __name__ == "__main__":
    run_backtest("INTC", start_date="2023-03-15", end_date="2026-04-01")