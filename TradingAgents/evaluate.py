import time
from datetime import datetime, timedelta
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# --- CONFIGURATION ---
TICKER = "NVDA"
START_DATE = "2026-01-01"
END_DATE = "2026-01-15"
DECISION_INTERVAL_DAYS = 7  # Evaluate weekly to save on API token costs

# Initialize the Multi-Agent system
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"  # or 'anthropic', 'deepseek', etc.
# Pro-tip: Use cheaper mini models for testing so you don't burn API budget
config["deep_think_llm"] = "gpt-4o-mini" 

ta = TradingAgentsGraph(debug=False, config=config)

# Generate list of evaluation dates
current_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
dates_to_test = []

while current_dt <= end_dt:
    dates_to_test.append(current_dt.strftime("%Y-%m-%d"))
    current_dt += timedelta(days=DECISION_INTERVAL_DAYS)

print(f"Starting evaluation for {TICKER} across {len(dates_to_test)} intervals...")

# Run the evaluation loop
results = {}
for date in dates_to_test:
    print(f"\n--- Evaluating Date: {date} ---")
    try:
        # Propagate runs the entire agent debate/decision process for that historical slice
        _, decision = ta.propagate(TICKER, date)
        
        # Capture the portfolio manager's final rating (Buy, Sell, Hold, etc.)
        results[date] = decision
        print(f"Decision for {date}: {decision}")
        
    except Exception as e:
        print(f"Error evaluating {date}: {e}")
    
    # Rest a bit to respect rate limits
    time.sleep(2)

print("\n=== Evaluation Complete ===")
for date, decision in results.items():
    print(f"{date}: {decision}")