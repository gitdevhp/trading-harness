import os

# ==========================================
# 1. ENVIRONMENT & LLM ROUTING OVERRIDES
# (Must be declared before importing TradingAgents)
# ==========================================
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"

# Force framework-wide model overrides to prevent gpt-4o-mini fallbacks
os.environ["TRADINGAGENTS_LLM_PROVIDER"] = "openai_compatible"
os.environ["TRADINGAGENTS_BACKEND_URL"] = "http://127.0.0.1:8000/v1"
os.environ["TRADINGAGENTS_DEEP_THINK_LLM"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
os.environ["TRADINGAGENTS_QUICK_THINK_LLM"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
os.environ["TRADINGAGENTS_DEFAULT_MODEL"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"

# API keys for data tools
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "dummy_key")
os.environ["FINNHUB_API_KEY"] = os.getenv("FINNHUB_API_KEY", "dummy_key")

# ==========================================
# 2. IMPORTS & CONFIGURATION
# ==========================================
import json
import pandas as pd
import yfinance as yf
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

TICKER = "META"
START_DATE = "2023-03-15"
END_DATE = "2026-04-01"
INITIAL_CAPITAL = 100000.0
OUTPUT_FILE = "trading_agents_backtest.json"

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai_compatible"
config["backend_url"] = "http://127.0.0.1:8000/v1"
config["deep_think_llm"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
config["quick_think_llm"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
config["default_model"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
config["model"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
config["max_debate_rounds"] = 2

# ==========================================
# 3. DATA PREFETCH
# ==========================================
df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

trading_days = [d.strftime("%Y-%m-%d") for d in df.index]

ai_cash = INITIAL_CAPITAL
ai_shares = 0.0
baseline_shares = INITIAL_CAPITAL / float(df.loc[trading_days[0], "Close"])
backtest_results = []

def extract_decision(decision_obj) -> str:
    """Safely parses dictionaries, Pydantic objects, or raw strings."""
    if isinstance(decision_obj, dict):
        action = str(decision_obj.get("action", decision_obj.get("decision", ""))).upper()
    else:
        action = str(decision_obj).upper()

    if "BUY" in action and "SELL" not in action:
        return "BUY"
    elif "SELL" in action and "BUY" not in action:
        return "SELL"
    return "HOLD"

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
print(f"Starting evaluation across {len(trading_days)} trading days...")
ta = TradingAgentsGraph(debug=False, config=config)

for date in trading_days:
    current_price = float(df.loc[date, "Close"])
    
    try:
        _, raw_decision = ta.propagate(TICKER, date)
        decision = extract_decision(raw_decision)
    except Exception as e:
        print(f"[{date}] Error running graph: {e}")
        decision = "HOLD"

    trade_executed = "HOLD"
    if decision == "BUY" and ai_cash > 10:
        ai_shares = ai_cash / current_price
        trade_executed = f"BOUGHT {ai_shares:.2f} shares @ ${current_price:.2f}"
        ai_cash = 0.0
    elif decision == "SELL" and ai_shares > 0.001:
        ai_cash = ai_shares * current_price
        trade_executed = f"SOLD {ai_shares:.2f} shares @ ${current_price:.2f}"
        ai_shares = 0.0

    ai_portfolio_value = ai_cash + (ai_shares * current_price)
    baseline_value = baseline_shares * current_price

    log_entry = {
        "date": date,
        "decision": decision,
        "price": current_price,
        "trade_executed": trade_executed,
        "ai_cash": round(ai_cash, 2),
        "ai_shares": round(ai_shares, 2),
        "ai_portfolio_value": round(ai_portfolio_value, 2),
        "baseline_value": round(baseline_value, 2)
    }
    backtest_results.append(log_entry)

    print(f"[{date}] Decision: {decision} | Action: {trade_executed} | Value: ${ai_portfolio_value:,.2f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(backtest_results, f, indent=4)

print("Backtest finished successfully!")