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
    
    # 14-Day Price Momentum %
    mom14 = ((current_p - float(closes.iloc[-14])) / float(closes.iloc[-14])) * 100 if len(closes) >= 14 else 0.0
    
    # 20-Day Volatility (Standard Deviation %)
    vol20 = (closes.tail(20).std() / sma20) * 100 if len(closes) >= 20 else 0.0

    # Drawdown metrics
    high_20d = float(closes.tail(20).max()) if len(closes) >= 20 else current_p
    dd_from_20d_peak = ((current_p - high_20d) / high_20d) * 100.0
    
    high_52w = float(closes.tail(252).max()) if len(closes) >= 252 else current_p
    dd_from_52w_high = ((current_p - high_52w) / high_52w) * 100.0

    # Trend Regime Classification
    if current_p < sma20 and current_p < sma50:
        trend_regime = "BEARISH (Price below 20 & 50 SMAs)"
    elif current_p > sma20 and current_p > sma50:
        trend_regime = "BULLISH (Price above 20 & 50 SMAs)"
    else:
        trend_regime = "NEUTRAL / MIXED"

    # RSI (14-day)
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
        "volatility20_pct": round(vol20, 2),
        "dd_from_20d_peak_pct": round(dd_from_20d_peak, 2),
        "dd_from_52w_high_pct": round(dd_from_52w_high, 2),
        "trend_regime": trend_regime
    }

def run_daily_agent(ticker: str, current_date: str, portfolio_state: dict) -> dict:
    df = GLOBAL_DATA_CACHE.get(ticker)
    past_data = df.loc[:current_date]

    def get_technical_analysis(arg: str = "") -> str:
        tech = calculate_technical_indicators(past_data)
        return (f"Price: ${tech['price']} | Trend: {tech['trend_regime']} | "
                f"20-SMA: ${tech['sma20']} | 50-SMA: ${tech['sma50']} | "
                f"RSI: {tech['rsi14']} | 14d Mom: {tech['momentum14_pct']}% | "
                f"20d Vol: {tech['volatility20_pct']}% | "
                f"20d Peak DD: {tech['dd_from_20d_peak_pct']}% | 52w High DD: {tech['dd_from_52w_high_pct']}%")

    def get_portfolio_status(arg: str = "") -> str:
        total_val = portfolio_state['portfolio_value']
        exposure_pct = (portfolio_state['shares'] * portfolio_state['current_price'] / total_val * 100.0) if total_val > 0 else 0.0
        return (f"Cash: ${portfolio_state['cash']:,.2f} | Shares: {portfolio_state['shares']:.2f} | "
                f"Total Val: ${total_val:,.2f} | Current Exposure: {exposure_pct:.1f}% | "
                f"Portfolio DD from Peak: {portfolio_state['portfolio_drawdown_pct']:.2f}%")

    available_tools = {
        "get_technical_analysis": get_technical_analysis,
        "get_portfolio_status": get_portfolio_status
    }

    react_system_prompt = f"""You are an autonomous quantitative portfolio manager evaluating {ticker} on {current_date}.

Available Tools:
- get_technical_analysis[]: Fetch Price, Trend Regime, SMAs, RSI, Momentum, Volatility, and Drawdowns.
- get_portfolio_status[]: Fetch Cash, Shares, Total Valuation, Exposure %, and Portfolio Drawdown from Peak.

Goal: Maximize upside participation while strictly limiting drawdown during stock decline.

STRICT RISK & ALLOCATION RULES:
1. BEARISH REGIME (Price < 50-SMA AND Momentum < 0%): Target allocation MUST NOT exceed 25%. Set allocation to 0% if 14d Momentum < -5% or Portfolio Drawdown > 5%.
2. BULLISH REGIME (Price > 20-SMA and Price > 50-SMA): Target allocation can scale up to 75% - 100%.
3. NEUTRAL / RECOVERY REGIME: Scale gradually (25% to 50%). Do not jump directly to 100%.
4. DO NOT hold high equity exposure (50%+) hoping for a turnaround while in a Bearish trend. De-risk aggressively to protect capital.

You MUST follow this output format:
Thought: <Analyze trend regime, momentum, drawdowns, and risk level>
Action: <tool_name>[]
Observation: <result>
... (repeat Thought/Action/Observation as needed)
Thought: <Declare final risk assessment and exact target exposure>
Action: Target_Allocation[X%] (where X is an integer between 0 and 100)"""

    messages = [
        {"role": "system", "content": react_system_prompt},
        {"role": "user", "content": f"Analyze {ticker} for {current_date} and issue your target portfolio allocation."}
    ]

    trajectory_log = []
    final_decision = "Target_Allocation[0%]"

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

            if action_name == "Target_Allocation":
                final_decision = f"Target_Allocation[{action_arg}]"
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
    peak_portfolio_value = initial_capital
    baseline_shares = initial_capital / float(df.loc[trading_days[0]]['Close'])
    
    output_filename = f"react_results_{ticker}.json"
    backtest_results = []

    for current_date in trading_days:
        past_df = df.loc[:current_date]
        current_price = float(past_df.iloc[-1]['Close'])
        
        ai_portfolio_value = ai_cash + (ai_shares * current_price)
        peak_portfolio_value = max(peak_portfolio_value, ai_portfolio_value)
        portfolio_drawdown_pct = ((ai_portfolio_value - peak_portfolio_value) / peak_portfolio_value) * 100.0
        
        unrealized_pnl_pct = (((current_price - cost_basis) / cost_basis) * 100.0) if ai_shares > 0 else 0.0
        
        portfolio_state = {
            "cash": ai_cash,
            "shares": ai_shares,
            "cost_basis": cost_basis,
            "current_price": current_price,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "portfolio_value": ai_portfolio_value,
            "portfolio_drawdown_pct": portfolio_drawdown_pct
        }

        # Run LLM daily to maintain continuous portfolio optimization
        result = run_daily_agent(ticker, current_date, portfolio_state)
        decision = result["decision"]

        # Parse target allocation percentage
        alloc_match = re.search(r"Target_Allocation\[(\d+)%?\]", decision, re.IGNORECASE)
        if alloc_match:
            target_pct = float(alloc_match.group(1)) / 100.0
        else:
            target_pct = (ai_shares * current_price / ai_portfolio_value) if ai_portfolio_value > 0 else 0.0

        # Continuous Portfolio Rebalancing
        target_equity_value = ai_portfolio_value * target_pct
        current_equity_value = ai_shares * current_price
        trade_diff_dollars = target_equity_value - current_equity_value

        trade_action = "HOLD"
        
        # Execute Buy Rebalance
        if trade_diff_dollars > 200 and ai_cash > 10:
            spend_cash = min(trade_diff_dollars, ai_cash)
            shares_bought = spend_cash / current_price
            
            total_cost = (ai_shares * cost_basis) + spend_cash
            ai_shares += shares_bought
            ai_cash -= spend_cash
            cost_basis = total_cost / ai_shares if ai_shares > 0 else 0.0
            
            trade_action = f"BOUGHT {shares_bought:.2f} shs (Target Exposure: {target_pct*100:.0f}%)"

        # Execute Sell Rebalance
        elif trade_diff_dollars < -200 and ai_shares > 0.001:
            sell_value = min(abs(trade_diff_dollars), current_equity_value)
            shares_to_sell = sell_value / current_price
            
            ai_cash += shares_to_sell * current_price
            ai_shares -= shares_to_sell
            
            if ai_shares <= 0.001:
                cost_basis = 0.0
                
            trade_action = f"SOLD {shares_to_sell:.2f} shs (Target Exposure: {target_pct*100:.0f}%)"

        ai_portfolio_value = ai_cash + (ai_shares * current_price)
        result.update({
            "price": current_price,
            "trade_executed": trade_action,
            "ai_portfolio_value": round(ai_portfolio_value, 2),
            "baseline_value": round(baseline_shares * current_price, 2)
        })
        backtest_results.append(result)

        if len(backtest_results) % 20 == 0 or current_date == trading_days[-1]:
            with open(output_filename, "w") as f:
                json.dump(backtest_results, f, indent=4)

        print(f"[{ticker} | {current_date}] Value: ${ai_portfolio_value:,.2f} | Decision: {decision} | Action: {trade_action}")

if __name__ == "__main__":
    for ticker in ["INTC", "META"]:
        run_backtest(ticker, start_date="2023-03-15", end_date="2026-04-01")