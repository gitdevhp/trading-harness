import time
import json
import pandas as pd
import yfinance as yf
from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# --- CONFIGURATION ---
TICKER = "NVDA"
START_DATE = "2026-01-01"
END_DATE = "2026-01-15"
INITIAL_CAPITAL = 100000.0

# MASTER HARNESS SWITCH
USE_HARNESS = False  # Set to False to run Vanilla mode

# 1. Fetch exact trading days and prices to match ReAct environment
print(f"Prefetching historical data for {TICKER} from {START_DATE} to {END_DATE}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
trading_days = [d.strftime('%Y-%m-%d') for d in df.index]

if len(trading_days) < 6:
    raise ValueError("Not enough trading days in this window to run a test.")

# Initialize the Multi-Agent system base configuration
config = DEFAULT_CONFIG.copy()

# Force framework to use your local vLLM A100 server
config["llm_provider"] = "openai_compatible"
config["llm_backend_url"] = "http://localhost:8000/v1"
config["deep_think_llm"] = "Qwen/Qwen2.5-32B-Instruct"
config["quick_think_llm"] = "Qwen/Qwen2.5-32B-Instruct"

# --- PORTFOLIO INITIALIZATION ---
ai_cash = INITIAL_CAPITAL
ai_shares = 0

# Start on day 6 so agents have 5 days of history, identical to ReAct baseline
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
    
    # DYNAMIC HARNESS INJECTION
    # Re-initialize or update graph configuration to inject new portfolio metrics every day
    daily_config = config.copy()
    
    if USE_HARNESS:
        try:
            with open("../prompts/multiagent_harness.md", "r", encoding="utf-8") as f:
                harness_template = f.read()
            
            # Format the harness rules with live portfolio values
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
        # Vanilla mode removes all extra guidelines
        daily_config["system_prompt_extension"] = ""

    # Instantiate graph with today's context configurations
    ta = TradingAgentsGraph(debug=False, config=daily_config)
    
    try:
        # Propagate runs the entire agent debate/decision process
        _, decision = ta.propagate(TICKER, date)
        
        # Normalize the decision output to BUY/SELL/HOLD
        raw_decision = str(decision).upper()
        if "BUY" in raw_decision:
            decision_str = "BUY"
        elif "SELL" in raw_decision:
            decision_str = "SELL"
            
    except Exception as e:
        print(f"Error evaluating {date}: {e}, defaulting to HOLD")
    
    # Execute Trade
    trade_action = "NONE"
    if decision_str == "BUY" and ai_cash > 0:
        shares_bought = ai_cash / current_price
        ai_shares += shares_bought
        ai_cash = 0
        trade_action = f"BOUGHT {shares_bought:.2f} shares @ ${current_price:.2f}"
        
    elif decision_str == "SELL" and ai_shares > 0:
        ai_cash += ai_shares * current_price
        trade_action = f"SOLD {ai_shares:.2f} shares @ ${current_price:.2f}"
        ai_shares = 0
        
    # Calculate daily portfolio value
    ai_portfolio_value = ai_cash + (ai_shares * current_price)
    baseline_value = baseline_shares * current_price
    
    # Log the financial result
    backtest_results.append({
        "date": date,
        "decision": decision_str,
        "price": current_price,
        "trade_executed": trade_action,
        "ai_portfolio_value": round(ai_portfolio_value, 2),
        "baseline_value": round(baseline_value, 2),
        "trajectory": f"Multi-Agent Final Output: {str(decision)}" 
    })
    
    print(f"Decision: {decision_str} | Portfolio: ${ai_portfolio_value:,.2f} | Action: {trade_action}")
    time.sleep(1) # Let vLLM breathe

# --- FINAL PnL CALCULATION ---
final_price = float(df.loc[trading_days[-1]]['Close'])
final_ai_value = ai_cash + (ai_shares * final_price)
final_baseline_value = baseline_shares * final_price

ai_return_pct = ((final_ai_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
baseline_return_pct = ((final_baseline_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print("\n" + "="*40)
print(f"🏁 MULTI-AGENT BACKTEST COMPLETE ({'HARNESS ENABLED' if USE_HARNESS else 'VANILLA'}) 🏁")
print("="*40)
print(f"Initial Capital:  ${INITIAL_CAPITAL:,.2f}")
print(f"AI Final Value:   ${final_ai_value:,.2f} ({ai_return_pct:+.2f}%)")
print(f"Baseline Value:   ${final_baseline_value:,.2f} ({baseline_return_pct:+.2f}%)")

# Determine output filename based on current mode
output_filename = "multi_agent_backtest_results_harness.json" if USE_HARNESS else "multi_agent_backtest_results_vanilla.json"

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