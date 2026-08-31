import json
import os
import re
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. ENVIRONMENT & LLM ROUTING OVERRIDES
# ==========================================
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"

os.environ["TRADINGAGENTS_LLM_PROVIDER"] = "openai_compatible"
os.environ["TRADINGAGENTS_BACKEND_URL"] = "http://127.0.0.1:8000/v1"
os.environ["TRADINGAGENTS_DEEP_THINK_LLM"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
os.environ["TRADINGAGENTS_QUICK_THINK_LLM"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
os.environ["TRADINGAGENTS_DEFAULT_MODEL"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"

os.environ["FRED_API_KEY"] = "4f0a230f9d2fc1c79e382a4cab851119"
os.environ["FINNHUB_API_KEY"] = os.getenv(
    "FINNHUB_API_KEY", "d9uidqhr01qs9cmcuuagd9uidqhr01qs9cmcuub0"
)
os.environ["TAVILY_API_KEY"] = os.getenv(
    "TAVILY_API_KEY", "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"
)
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TradingAgents/1.0"
)

# ==========================================
# 2. IMPORTS & CONFIGURATION
# ==========================================
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
config["max_debate_rounds"] = 1

# ==========================================
# 3. DATA PREFETCH & DECISION PARSER
# ==========================================
df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

trading_days = [d.strftime("%Y-%m-%d") for d in df.index]

ai_cash = INITIAL_CAPITAL
ai_shares = 0.0
baseline_shares = INITIAL_CAPITAL / float(df.loc[trading_days[0], "Close"])
backtest_results = []
history = []


def extract_decision_robust(
    final_state, raw_decision_obj, current_shares: float
) -> str:
    combined_text = (str(final_state) + " " + str(raw_decision_obj)).upper()

    is_bearish = any(
        term in combined_text
        for term in ["BEARISH", "SELL", "LIQUIDATE", "SHORT", "DOWNWARD"]
    )
    is_bullish = any(
        term in combined_text
        for term in [
            "BULLISH",
            "BUY",
            "ACCUMULATE",
            "LONG",
            "OUTPERFORM",
            "POSITIVE",
        ]
    )

    if is_bearish:
        return "SELL"

    # Force initial capital deployment: If holding 0 shares and market isn't explicitly bearish, BUY
    if current_shares == 0 or is_bullish:
        return "BUY"

    return "HOLD"


# ==========================================
# 4. EXECUTION LOOP
# ==========================================
print(f"Starting evaluation across {len(trading_days)} trading days...")
ta = TradingAgentsGraph(debug=False, config=config)

for date in trading_days:
    current_price = float(df.loc[date, "Close"])

    final_state, raw_decision = ta.propagate(TICKER, date)
    decision = extract_decision_robust(final_state, raw_decision, ai_shares)

    # Execution Engine Logic
    trade_executed = "HOLD"
    if decision == "BUY" and ai_cash > 10:
        ai_shares = ai_cash / current_price
        trade_executed = (
            f"BOUGHT {ai_shares:.2f} shares @ ${current_price:.2f}"
        )
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
        "baseline_value": round(baseline_value, 2),
    }
    backtest_results.append(log_entry)
    history.append({"date": date, "value": ai_portfolio_value})

    print(
        f"[{date}] Decision: {decision:4s} | Action: {trade_executed:35s} | Value: ${ai_portfolio_value:,.2f}"
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(backtest_results, f, indent=4)

# ==========================================
# 5. PERFORMANCE METRICS EVALUATION
# ==========================================
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
print("TRADING AGENTS BACKTEST PERFORMANCE SUMMARY")
print("=" * 50)
print(f"Initial Capital: ${values.iloc[0]:,.2f}")
print(f"Final Value:     ${values.iloc[-1]:,.2f}")
print(f"Total Return:    {total_return * 100:.2f}%")
print(f"Sharpe Ratio:    {sharpe:.2f}")
print(f"Max Drawdown:    {max_dd * 100:.2f}%")
print("=" * 50)