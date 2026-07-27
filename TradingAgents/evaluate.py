import time
import json
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Force all sub-modules to direct requests to local vLLM server
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

# --- CONFIGURATION ---
TICKER = "META"
START_DATE = "2025-10-01"
END_DATE = "2026-01-01"  # ~60 trading days
INITIAL_CAPITAL = 100000.0

# MASTER HARNESS SWITCH
USE_HARNESS = False  # Set to False to run Vanilla mode

# FIX 1: Define output filename BEFORE starting the backtest loop
output_filename = "multi_agent_backtest_results_harness.json" if USE_HARNESS else "multi_agent_backtest_results_vanilla.json"

# 1. Fetch exact trading days and prices
print(f"Prefetching historical data for {TICKER} from {START_DATE} to {END_DATE}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
trading_days = [d.strftime('%Y-%m-%d') for d in df.index]

if len(trading_days) < 6:
    raise ValueError("Not enough trading days in this window to run a test.")

# Base configuration for local vLLM
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai_compatible"
config["llm_backend_url"] = "http://localhost:8000/v1"
config["deep_think_llm"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
config["quick_think_llm"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"

# Portfolio Initialization
ai_cash = INITIAL_CAPITAL
ai_shares = 0.0

first_trade_date = trading_days[5]
first_day_price = float(df.loc[first_trade_date]['Close'])
baseline_shares = INITIAL_CAPITAL / first_day_price

backtest_results = []
print(f"\nStarting evaluation for {TICKER} across {len(trading_days[5:])} trading days...")

# --- THE DAILY LOOP ---
for date in trading_days[5:]:
    print(f"\n--- Evaluating Date: {date} ---")
    current_price = float(df.loc[date]['Close'])
    
    decision_str = "HOLD"
    daily_config = config.copy()
    
    if USE_HARNESS:
        try:
            with open("../prompts/multiagent_harness.md", "r", encoding="utf-8") as f:
                harness_template = f.read()
            
            harness_rules = (
                f"{harness_template}\n\n"
                f"--- LIVE RUNTIME METRICS FOR {date} ---\n"
                f"Current Cash: ${ai_cash:,.2f}\n"
                f"Current Shares Held: {ai_shares:,.2f}\n"
                f"Asset Current Price: ${current_price:,.2f}\n"
            )
            daily_config["system_prompt_extension"] = harness_rules
        except FileNotFoundError:
            daily_config["system_prompt_extension"] = "--- MANDATE ---\nTrade portfolio-aware and conservatively."
    else:
        daily_config["system_prompt_extension"] = ""

    # Instantiate agent graph with current runtime rules
    ta = TradingAgentsGraph(debug=False, config=daily_config)
    
    try:
        raw_output, decision = ta.propagate(TICKER, date)
        
        raw_decision = str(decision).upper()
        if "BUY" in raw_decision:
            decision_str = "BUY"
        elif "SELL" in raw_decision:
            decision_str = "SELL"
            
    except Exception as e:
        print(f"❌ ERROR evaluating {date}: {type(e).__name__} - {e}, defaulting to HOLD")
        decision = "HOLD"

    # Execute Trade
    trade_action = "NONE"
    if decision_str == "BUY" and ai_cash > 0:
        shares_bought = ai_cash / current_price
        ai_shares += shares_bought
        ai_cash = 0.0
        trade_action = f"BOUGHT {shares_bought:.2f} shares @ ${current_price:.2f}"
        
    elif decision_str == "SELL" and ai_shares > 0:
        ai_cash += ai_shares * current_price
        trade_action = f"SOLD {ai_shares:.2f} shares @ ${current_price:.2f}"
        ai_shares = 0.0
        
    ai_portfolio_value = ai_cash + (ai_shares * current_price)
    baseline_value = baseline_shares * current_price
    
    backtest_results.append({
        "date": date,
        "decision": decision_str,
        "price": current_price,
        "trade_executed": trade_action,
        "ai_portfolio_value": round(ai_portfolio_value, 2),
        "baseline_value": round(baseline_value, 2),
        "trajectory": f"Multi-Agent Final Output: {str(decision)}" 
    })

    # Incremental save
    with open(output_filename, "w") as f:
        json.dump({
            "metrics": {
                "harness_active": USE_HARNESS,
                "days_completed": len(backtest_results)
            },
            "daily_logs": backtest_results
        }, f, indent=4)

    print(f"Decision: {decision_str} | Portfolio: ${ai_portfolio_value:,.2f} | Action: {trade_action}")

# --- FINAL PnL CALCULATION ---
final_price = float(df.loc[trading_days[-1]]['Close'])
final_ai_value = ai_cash + (ai_shares * final_price)
final_baseline_value = baseline_shares * final_price

ai_return_pct = ((final_ai_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
baseline_return_pct = ((final_baseline_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print("\n" + "="*40)
print(f"🏁 MULTI-AGENT BACKTEST COMPLETE ({'HARNESS ENABLED' if USE_HARNESS else 'VANILLA'}) 🏁")
print("="*40)
print(f"Initial Capital:   ${INITIAL_CAPITAL:,.2f}")
print(f"AI Final Value:    ${final_ai_value:,.2f} ({ai_return_pct:+.2f}%)")
print(f"Baseline Value:    ${final_baseline_value:,.2f} ({baseline_return_pct:+.2f}%)")

with open(output_filename, "w") as f:
    json.dump({
        "metrics": {
            "harness_active": USE_HARNESS,
            "ai_return_pct": round(ai_return_pct, 2),
            "baseline_return_pct": round(baseline_return_pct, 2),
            "beat_market": final_ai_value > final_baseline_value
        },
        "daily_logs": backtest_results
    }, f, indent=4)
    
print(f"Saved execution results to {output_filename}")