import time
import json
import re
import pandas as pd
import yfinance as yf
from vllm import LLM, SamplingParams

# ==========================================
# 1. GLOBAL DATA CACHE 
# ==========================================
GLOBAL_DATA_CACHE = {}

def prefetch_data(ticker: str, start_date: str, end_date: str):
    print(f"Prefetching historical data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    GLOBAL_DATA_CACHE[ticker] = df
    print("Pre-fetch complete!")

# ==========================================
# 2. THE DAILY ReAct EXECUTION LOOP
# ==========================================
def run_daily_agent(
    ticker: str, 
    current_date: str, 
    llm: LLM, 
    playbook: str, 
    current_cash: float, 
    current_shares: float, 
    cost_basis: float,
    recent_history: list,
    use_harness: bool
) -> dict:
    
    def date_aware_price(arg_ticker: str) -> str:
        try:
            df = GLOBAL_DATA_CACHE.get(ticker)
            if df is None or df.empty:
                return f"No pre-fetched data found for {ticker}."
            past_data = df.loc[:current_date]
            if past_data.empty:
                return f"No price data found before {current_date}."
            
            # 20-day historical window for medium-term trend
            last_20 = past_data.tail(20)
            closes = [round(float(p), 2) for p in last_20['Close'].values]
            sma_20 = round(sum(closes) / len(closes), 2)
            current_p = closes[-1]
            
            return (
                f"Recent 20-Day Closes for {ticker} (leading to {current_date}): {closes[-5:]} "
                f"(Showing last 5 of 20 days)\n"
                f"Current Price: ${current_p:.2f} | 20-Day Moving Average: ${sma_20:.2f}\n"
                f"Trend Signal: {'ABOVE 20-MA (Bullish)' if current_p > sma_20 else 'BELOW 20-MA (Bearish)'}"
            )
        except Exception as e:
            return f"Error fetching price: {str(e)}"

    def date_aware_news(arg_ticker: str) -> str:
        return f"Historical news unavailable for {current_date}. Focus your reasoning strictly on price action."

    available_tools = {
        "get_stock_price": date_aware_price,
        "get_news": date_aware_news
    }

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["Observation:"])

    # Base ReAct system instructions
    base_engine_rules = """TRADING DIRECTIVES:
1. Goal: Maximize risk-adjusted return. Active capital deployment is expected when technicals align; do not default to HOLD out of hesitation.
2. Sizing Protocol: Specify position sizing using percentage allocations:
   - Final_Decision[BUY, 25%], Final_Decision[BUY, 50%], Final_Decision[BUY, 100%]
   - Final_Decision[SELL, 50%], Final_Decision[SELL, 100%]
   - Final_Decision[HOLD]
3. ReAct Structure:
   Thought: [Brief trend analysis + position sizing rationale]
   Action: tool_name[input]
   Observation: [System response]

For your final decision, use strictly:
Action: Final_Decision[BUY, <pct>] or Final_Decision[SELL, <pct>] or Final_Decision[HOLD]"""

    # --- TOGGLE LOGIC FOR HARNESS ---
    if use_harness:
        # Calculate dynamic position metrics for prompt injection
        position_value = current_shares * (float(GLOBAL_DATA_CACHE[ticker].loc[current_date]['Close']) if ticker in GLOBAL_DATA_CACHE else 0.0)
        total_value = current_cash + position_value
        
        if current_shares > 0:
            current_p = float(GLOBAL_DATA_CACHE[ticker].loc[current_date]['Close'])
            unrealized_pnl = (current_p - cost_basis) * current_shares
            unrealized_str = f"${unrealized_pnl:+,.2f} ({((current_p - cost_basis) / cost_basis) * 100:+.2f}%)"
            cost_basis_str = f"${cost_basis:.2f}"
        else:
            unrealized_str = "N/A (No open position)"
            cost_basis_str = "N/A"

        # Format 5-day rolling history string
        if recent_history:
            history_str = "\n".join([
                f"- [{h['date']}] Price: ${h['price']:.2f} | Action: {h['action']} | Portfolio: ${h['value']:,.2f}"
                for h in recent_history
            ])
        else:
            history_str = "No prior daily actions logged."

        harness_context = f"""--- PORTFOLIO-AWARE TRADING HARNESS ---

[PORTFOLIO STATE]
- Cash Available: ${current_cash:,.2f}
- Shares Held: {current_shares:,.2f} (${position_value:,.2f})
- Total Portfolio Value: ${total_value:,.2f}
- Avg Entry Price (Cost Basis): {cost_basis_str}
- Unrealized P&L: {unrealized_str}

[RECENT EXECUTION HISTORY (LAST 5 DAYS)]
{history_str}

[STRATEGIC PLAYBOOK MANDATES]
{playbook}"""

        master_system_prompt = f"{harness_context}\n\n{base_engine_rules}"
        user_prompt = f"Today is {current_date}. Analyze {ticker} using available tools and decide whether to BUY, SELL, or HOLD today."
    
    else:
        master_system_prompt = f"You are an autonomous stock trader managing a portfolio on {current_date}.\n\n{base_engine_rules}"
        user_prompt = f"Today is {current_date}. Analyze {ticker} and decide whether to BUY, SELL, or HOLD today."
    
    messages = [
        {"role": "system", "content": master_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    trajectory_log = []
    final_decision = "HOLD" 
    
    mode_str = "HARNESS" if use_harness else "VANILLA"
    print(f"\n[{current_date}] 🤖 Running AI logic [{mode_str}] (Cash: ${current_cash:,.2f} | Shares: {current_shares:,.2f})...")
    
    for step in range(8):
        outputs = llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=False)
        reply = outputs[0].outputs[0].text.strip()
        
        trajectory_log.append(reply)
        messages.append({"role": "assistant", "content": reply})
        
        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, re.DOTALL)
        if action_match:
            action_name = action_match.group(1)
            action_arg = action_match.group(2).strip()
            
            if action_name == "Final_Decision":
                raw_arg = action_arg.upper()
                pct_match = re.search(r"(\d+)%", raw_arg)
                trade_pct = float(pct_match.group(1)) / 100.0 if pct_match else 1.0
                
                if "BUY" in raw_arg:
                    final_decision = f"BUY_{trade_pct}"
                elif "SELL" in raw_arg:
                    final_decision = f"SELL_{trade_pct}"
                else:
                    final_decision = "HOLD"
                break
            
            if action_name in available_tools:
                obs_text = f"Observation: {available_tools[action_name](action_arg)}"
                trajectory_log.append(obs_text)
                messages.append({"role": "user", "content": obs_text})
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."
                trajectory_log.append(obs_text)
                messages.append({"role": "user", "content": obs_text})
        else:
            obs_text = "Observation: Invalid format. You must provide an Action."
            trajectory_log.append(obs_text)
            messages.append({"role": "user", "content": obs_text})

    print(f"[{current_date}] ✅ DECISION: {final_decision}")
    
    return {
        "date": current_date,
        "decision": final_decision,
        "trajectory": "\n".join(trajectory_log)
    }

# ==========================================
# 3. THE HISTORICAL BACKTEST ENGINE 
# ==========================================
def run_backtest(ticker: str, start_date: str, end_date: str, initial_capital: float = 100000.0, use_harness: bool = True):
    
    prefetch_data(ticker, start_date, end_date)
    df = GLOBAL_DATA_CACHE[ticker]
    trading_days = [d.strftime('%Y-%m-%d') for d in df.index]
    
    playbook_prompt = ""
    if use_harness:
        try:
            with open("../prompts/react_harness.md", "r", encoding="utf-8") as f:
                playbook_prompt = f.read()
        except FileNotFoundError:
            playbook_prompt = "Maintain disciplined risk control. Allocate based on strong setup signals."
            print("⚠️ react_harness.md not found, using basic default prompt rules.")

    print("Loading Qwen 32B AWQ model directly into in-process vLLM engine...")
    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ", 
        max_model_len=4096,
        gpu_memory_utilization=0.85
    )
    
    ai_cash = initial_capital
    ai_shares = 0.0
    cost_basis = 0.0
    recent_history = []  # 5-day rolling execution queue
    
    output_filename = "react_backtest_results_harness.json" if use_harness else "react_backtest_results_vanilla.json"
    
    first_trade_date = trading_days[5] 
    first_day_price = float(df.loc[first_trade_date]['Close'])
    baseline_shares = initial_capital / first_day_price
    
    backtest_results = []
    
    for current_date in trading_days[5:]: 
        current_price = float(df.loc[current_date]['Close'])
        
        result = run_daily_agent(
            ticker=ticker, 
            current_date=current_date, 
            llm=llm, 
            playbook=playbook_prompt, 
            current_cash=ai_cash, 
            current_shares=ai_shares, 
            cost_basis=cost_basis,
            recent_history=recent_history,
            use_harness=use_harness
        )
        
        decision = result["decision"]
        trade_action = "NONE"
        
        # BUY Execution Logic
        if decision.startswith("BUY") and ai_cash > 10:
            trade_pct = float(decision.split("_")[1]) if "_" in decision else 1.0
            capital_to_use = ai_cash * trade_pct
            shares_bought = capital_to_use / current_price
            
            # Update weighted average cost basis
            total_cost = (ai_shares * cost_basis) + capital_to_use
            ai_shares += shares_bought
            ai_cash -= capital_to_use
            cost_basis = total_cost / ai_shares
            
            trade_action = f"BOUGHT {shares_bought:.2f} shares ({trade_pct*100:.0f}% cash) @ ${current_price:.2f}"
            
        # SELL Execution Logic
        elif decision.startswith("SELL") and ai_shares > 0.001:
            trade_pct = float(decision.split("_")[1]) if "_" in decision else 1.0
            shares_to_sell = ai_shares * trade_pct
            cash_received = shares_to_sell * current_price
            
            ai_cash += cash_received
            ai_shares -= shares_to_sell
            
            if ai_shares < 0.001:
                ai_shares = 0.0
                cost_basis = 0.0
                
            trade_action = f"SOLD {shares_to_sell:.2f} shares ({trade_pct*100:.0f}% holdings) @ ${current_price:.2f}"

        ai_portfolio_value = ai_cash + (ai_shares * current_price)
        baseline_value = baseline_shares * current_price
        
        result["price"] = current_price
        result["trade_executed"] = trade_action
        result["ai_portfolio_value"] = round(ai_portfolio_value, 2)
        result["baseline_value"] = round(baseline_value, 2)
        backtest_results.append(result)
        
        # Update 5-day rolling window memory
        recent_history.append({
            "date": current_date,
            "price": current_price,
            "action": decision,
            "value": round(ai_portfolio_value, 2)
        })
        if len(recent_history) > 5:
            recent_history.pop(0)

        # INCREMENTAL SAVE (Prevents data loss on HPC/Slurm timeouts)
        with open(output_filename, "w") as f:
            json.dump({
                "metrics": {
                    "harness_active": use_harness,
                    "days_completed": len(backtest_results),
                    "current_ai_value": round(ai_portfolio_value, 2)
                },
                "daily_logs": backtest_results
            }, f, indent=4)
        
        print(f"[{current_date}] Portfolio: ${ai_portfolio_value:,.2f} | Action: {trade_action}")
        
    # --- FINAL SUMMARY REPORTING ---
    final_price = float(df.loc[trading_days[-1]]['Close'])
    final_ai_value = ai_cash + (ai_shares * final_price)
    final_baseline_value = baseline_shares * final_price
    
    ai_return_pct = ((final_ai_value - initial_capital) / initial_capital) * 100
    baseline_return_pct = ((final_baseline_value - initial_capital) / initial_capital) * 100

    print("\n" + "="*40)
    print(f"🏁 ReAct BACKTEST COMPLETE ({'HARNESS ENABLED' if use_harness else 'VANILLA'}) 🏁")
    print("="*40)
    print(f"Initial Capital:   ${initial_capital:,.2f}")
    print(f"AI Final Value:    ${final_ai_value:,.2f} ({ai_return_pct:+.2f}%)")
    print(f"Baseline Value:    ${final_baseline_value:,.2f} ({baseline_return_pct:+.2f}%)")
    
    with open(output_filename, "w") as f:
        json.dump({
            "metrics": {
                "harness_active": use_harness,
                "ai_return_pct": round(ai_return_pct, 2),
                "baseline_return_pct": round(baseline_return_pct, 2),
                "beat_market": final_ai_value > final_baseline_value
            },
            "daily_logs": backtest_results
        }, f, indent=4)
        
    print(f"Saved trajectories and financial PnL to {output_filename}")

if __name__ == "__main__":
    USE_HARNESS = True 
    
    # Running across a backtest window
    run_backtest("META", start_date="2023-01-01", end_date="2026-04-01", use_harness=USE_HARNESS)