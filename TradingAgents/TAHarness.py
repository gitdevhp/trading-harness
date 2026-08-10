import json
import os
import re
import pandas as pd
import yfinance as yf
from datetime import datetime

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Force all sub-modules to direct requests to local vLLM server
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

# --- CONFIGURATION ---
TICKER = "META"
START_DATE = "2025-10-01"
END_DATE = "2026-01-01"  # ~60 trading days
INITIAL_CAPITAL = 100000.0

OUTPUT_FILENAME = "multi_agent_backtest_results_harness.json"

# --- HARNESS SUBAGENT WEIGHTS & THRESHOLDS ---
HARNESS_CONFIG = {
    "subagent_weights": {
        "technical": 1.3,      # Overweighting technicals in current regime
        "fundamental": 0.8,    # Standard fundamental weight
        "sentiment": 0.6,      # Lower weight for news noise
        "risk_manager": 1.5    # Heavy veto weight for risk manager
    },
    "buy_consensus_threshold": 0.35,   # Minimum score required to approve/trigger BUY
    "veto_consensus_threshold": -0.25  # Score below which BUYs are vetoed
}


# --- META-ARBITRATOR HARNESS CLASS ---
class MetaArbitratorHarness:
    def __init__(self, config: dict):
        self.weights = config["subagent_weights"]
        self.buy_thresh = config["buy_consensus_threshold"]
        self.veto_thresh = config["veto_consensus_threshold"]

    def parse_signals_from_state(self, raw_state) -> dict:
        """Parses analyst stances (-1.0 Bearish, 0.0 Neutral, +1.0 Bullish) from TradingAgents state."""
        state_str = str(raw_state).lower()

        # Stance parsing rules based on key outputs
        tech_stance = 1.0 if "technical analysis: bullish" in state_str or "technical: buy" in state_str else (
            -1.0 if "technical analysis: bearish" in state_str or "technical: sell" in state_str else 0.0
        )
        
        fund_stance = 1.0 if "fundamental analysis: bullish" in state_str or "fundamental: buy" in state_str else (
            -1.0 if "fundamental analysis: bearish" in state_str or "fundamental: sell" in state_str else 0.0
        )
        
        sent_stance = 1.0 if "sentiment: positive" in state_str or "bullish sentiment" in state_str else (
            -1.0 if "sentiment: negative" in state_str or "bearish sentiment" in state_str else 0.0
        )
        
        risk_stance = -1.0 if ("high risk" in state_str or "extreme volatility" in state_str or "reject" in state_str) else 1.0

        return {
            "technical": tech_stance,
            "fundamental": fund_stance,
            "sentiment": sent_stance,
            "risk_manager": risk_stance
        }

    def compute_weighted_score(self, stance_dict: dict) -> float:
        """Calculates normalized consensus score (-1.0 to +1.0)."""
        total_score = sum(stance_dict[agent] * self.weights.get(agent, 1.0) for agent in stance_dict)
        max_possible_weight = sum(self.weights.values())
        return total_score / max_possible_weight if max_possible_weight > 0 else 0.0

    def arbitrate_decision(self, raw_decision: str, raw_state) -> tuple[str, str, float, dict]:
        """Applies reweighting and guardrails to TradingAgents raw decision."""
        # 1. Parse underlying stances
        stances = self.parse_signals_from_state(raw_state)
        
        # 2. Compute weighted score
        consensus_score = self.compute_weighted_score(stances)
        
        # 3. Extract baseline proposed action
        proposed_action = self._extract_action(raw_decision)
        
        # 4. Harness Arbitration Logic
        if proposed_action == "BUY" and consensus_score < self.veto_thresh:
            final_action = "HOLD"
            reasoning = f"HARNESS VETO: Subagent consensus negative ({consensus_score:.2f} < {self.veto_thresh})"
        elif proposed_action == "HOLD" and consensus_score > self.buy_thresh:
            final_action = "BUY"
            reasoning = f"HARNESS OVERRIDE: Strong subagent consensus ({consensus_score:.2f} > {self.buy_thresh})"
        else:
            final_action = proposed_action
            reasoning = f"HARNESS APPROVED: Proposal '{proposed_action}' matches consensus ({consensus_score:.2f})"

        return final_action, reasoning, round(consensus_score, 3), stances

    def _extract_action(self, text: str) -> str:
        """Parses explicit action directive."""
        text_upper = text.upper()
        match = re.search(r'DECISION:\s*(BUY|SELL|HOLD)', text_upper)
        if match:
            return match.group(1)
        
        tail = text_upper[-150:]
        if "BUY" in tail and "SELL" not in tail:
            return "BUY"
        if "SELL" in tail and "BUY" not in tail:
            return "SELL"
        return "HOLD"


# --- MAIN BACKTEST EXECUTION ---
if __name__ == "__main__":
    print(f"Prefetching historical data for {TICKER} from {START_DATE} to {END_DATE}...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    trading_days = [d.strftime('%Y-%m-%d') for d in df.index]

    if len(trading_days) < 6:
        raise ValueError("Not enough trading days in this window to run a test.")

    # Base configuration for local vLLM
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "openai"
    config["backend_url"] = "http://localhost:8000/v1"
    config["deep_think_llm"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    config["quick_think_llm"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"

    # Instantiate Meta-Arbitrator Harness
    harness = MetaArbitratorHarness(HARNESS_CONFIG)

    # Portfolio Initialization
    ai_cash = INITIAL_CAPITAL
    ai_shares = 0.0

    first_trade_date = trading_days[5]
    first_day_price = float(df.loc[first_trade_date]['Close'])
    baseline_shares = INITIAL_CAPITAL / first_day_price

    backtest_results = []
    completed_dates = set()

    # --- CHECKPOINT RESUME LOGIC ---
    if os.path.exists(OUTPUT_FILENAME):
        try:
            with open(OUTPUT_FILENAME, "r") as f:
                existing_data = json.load(f)
                backtest_results = existing_data.get("daily_logs", [])
                completed_dates = {log["date"] for log in backtest_results}
                
                if backtest_results:
                    last_log = backtest_results[-1]
                    ai_cash = last_log.get("ai_cash", ai_cash)
                    ai_shares = last_log.get("ai_shares", ai_shares)
                    print(f"🔄 Checkpoint found! Resuming from day {len(completed_dates)+1}. (Cash: ${ai_cash:,.2f}, Shares: {ai_shares:.2f})")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint file: {e}. Starting fresh.")

    # Initialize Graph
    ta = TradingAgentsGraph(debug=False, config=config)
    print(f"\nStarting HARNESS BACKTEST for {TICKER} across {len(trading_days[5:])} trading days...")

    # --- THE HARNESS DAILY LOOP ---
    for date in trading_days[5:]:
        if date in completed_dates:
            continue

        print(f"\n--- Harness Evaluating Date: {date} ---")
        current_price = float(df.loc[date]['Close'])

        try:
            # Step 1: Execute original TradingAgents graph
            raw_state, raw_decision = ta.propagate(TICKER, date)
            
            # Step 2: Pass through Meta-Arbitrator Harness
            decision_str, harness_reasoning, weighted_score, stances = harness.arbitrate_decision(
                raw_decision=str(raw_decision),
                raw_state=raw_state
            )
                
        except Exception as e:
            print(f"❌ ERROR evaluating {date}: {type(e).__name__} - {e}, defaulting to HOLD")
            decision_str = "HOLD"
            harness_reasoning = f"ERROR: {type(e).__name__}"
            weighted_score = 0.0
            stances = {}

        # Step 3: Portfolio Execution
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
        
        # Step 4: Record Structured Log
        backtest_results.append({
            "date": date,
            "decision": decision_str,
            "price": current_price,
            "trade_executed": trade_action,
            "ai_cash": ai_cash,
            "ai_shares": ai_shares,
            "ai_portfolio_value": round(ai_portfolio_value, 2),
            "baseline_value": round(baseline_value, 2),
            "harness_score": weighted_score,
            "harness_reasoning": harness_reasoning,
            "analyst_stances": stances
        })

        # Incremental Save
        with open(OUTPUT_FILENAME, "w") as f:
            json.dump({
                "metrics": {
                    "harness_active": True,
                    "days_completed": len(backtest_results)
                },
                "daily_logs": backtest_results
            }, f, indent=4)

        print(f"Harness Decision: {decision_str} (Score: {weighted_score:+.2f}) | Portfolio: ${ai_portfolio_value:,.2f}")
        print(f"Reasoning: {harness_reasoning}")

    # --- FINAL PnL CALCULATION ---
    final_price = float(df.loc[trading_days[-1]]['Close'])
    final_ai_value = ai_cash + (ai_shares * final_price)
    final_baseline_value = baseline_shares * final_price

    ai_return_pct = ((final_ai_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    baseline_return_pct = ((final_baseline_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    print("\n" + "="*50)
    print("🏁 HARNESS MULTI-AGENT BACKTEST COMPLETE 🏁")
    print("="*50)
    print(f"Initial Capital:   ${INITIAL_CAPITAL:,.2f}")
    print(f"Harness AI Value:  ${final_ai_value:,.2f} ({ai_return_pct:+.2f}%)")
    print(f"Baseline Value:    ${final_baseline_value:,.2f} ({baseline_return_pct:+.2f}%)")

    with open(OUTPUT_FILENAME, "w") as f:
        json.dump({
            "metrics": {
                "harness_active": True,
                "ai_return_pct": round(ai_return_pct, 2),
                "baseline_return_pct": round(baseline_return_pct, 2),
                "beat_market": final_ai_value > final_baseline_value
            },
            "daily_logs": backtest_results
        }, f, indent=4)
        
    print(f"Saved execution results to {OUTPUT_FILENAME}")