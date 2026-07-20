import yfinance as yf
import pandas as pd
import re
from vllm import LLM, SamplingParams
import json

# ==========================================
# 1. GLOBAL DATA CACHE 
# ==========================================
GLOBAL_DATA_CACHE = {}

def prefetch_data(ticker: str, start_date: str, end_date: str):
    print(f"Prefetching historical data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    GLOBAL_DATA_CACHE[ticker] = df
    print("Pre-fetch complete!")

# ==========================================
# 2. THE DAILY ReAct EXECUTION LOOP
# ==========================================
def run_daily_agent(ticker: str, current_date: str, llm: LLM, playbook: str, current_cash: float, current_shares: float, use_harness: bool) -> dict:
    
    def date_aware_price(arg_ticker: str) -> str:
        try:
            df = GLOBAL_DATA_CACHE.get(arg_ticker)
            if df is None or df.empty:
                return f"No pre-fetched data found for {arg_ticker}."
            past_data = df.loc[:current_date]
            if past_data.empty:
                return f"No price data found before {current_date}."
            last_5 = past_data.tail(5)
            closes = [round(price, 2) for price in last_5['Close'].tolist()]
            return f"Closes leading up to {current_date}: {closes}"
        except Exception as e:
            return f"Error fetching price: {str(e)}"

    def date_aware_news(arg_ticker: str) -> str:
        return f"Historical news unavailable for {current_date}. Focus your reasoning strictly on price action."

    available_tools = {
        "get_stock_price": date_aware_price,
        "get_news": date_aware_news
    }

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512, stop=["Observation:"])

    engine_prompt = f"""You are an expert trading assistant simulating a trading session on {current_date}. 
You must answer questions by interleaving Thought, Action, and Observation steps.
Available tools: [get_stock_price, get_news, Final_Decision]

Format your response exactly like this:
Thought: [Your reasoning for what to do next based on observations and portfolio state]
Action: tool_name[input_for_tool]
Observation: [Provided by the system]"""

    # --- TOGGLE LOGIC FOR HARNESS ---
    if use_harness:
        master_system_prompt = f"{engine_prompt}\n\n{playbook}"
        # Inject precise portfolio state and constraints
        user_prompt = (
            f"Today is {current_date}.\n"
            f"PORTFOLIO STATUS: You have ${current_cash:,.2f} in cash and own {current_shares:,.2f} shares of {ticker}.\n"
            f"Analyze {ticker} and decide whether to BUY, SELL, or HOLD today."
        )
    else:
        # Vanilla mode: No guardrails, no memory of cash/shares limits
        master_system_prompt = engine_prompt
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
        
        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply)
        if action_match:
            action_name = action_match.group(1)
            action_arg = action_match.group(2)
            
            if action_name == "Final_Decision":
                final_decision = action_arg
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
    
    # Load playbook only if harness is active
    playbook_prompt = ""
    if use_harness:
        try:
            with open("../prompts/react_harness.md", "r", encoding="utf-8") as f:
                playbook_prompt = f.read()
        except FileNotFoundError:
            playbook_prompt = "--- CURRENT TRADING MANDATE ---\nTrade conservatively. Do not exceed cash balance."
            print("⚠️ react_harness.md not found, using basic default constraints.")

    print("Loading Qwen into vLLM engine...")
    llm = LLM(
        model="Qwen/Qwen2.5-Coder-7B-Instruct", 
        max_model_len=4096,
        gpu_memory_utilization=0.90
    )
    
    ai_cash = initial_capital
    ai_shares = 0
    
    first_trade_date = trading_days[5] 
    first_day_price = float(df.loc[first_trade_date]['Close'])
    baseline_shares = initial_capital / first_day_price
    
    backtest_results = []
    
    for current_date in trading_days[5:]: 
        # Pass the use_harness flag down into the agent
        result = run_daily_agent(ticker, current_date, llm, playbook_prompt, ai_cash, ai_shares, use_harness)
        decision = result["decision"]
        current_price = float(df.loc[current_date]['Close'])
        
        trade_action = "NONE"
        if decision == "BUY" and ai_cash > 0:
            shares_bought = ai_cash / current_price
            ai_shares += shares_bought
            ai_cash = 0
            trade_action = f"BOUGHT {shares_bought:.2f} shares @ ${current_price:.2f}"
            
        elif decision == "SELL" and ai_shares > 0:
            ai_cash += ai_shares * current_price
            trade_action = f"SOLD {ai_shares:.2f} shares @ ${current_price:.2f}"
            ai_shares = 0
            
        ai_portfolio_value = ai_cash + (ai_shares * current_price)
        baseline_value = baseline_shares * current_price
        
        result["price"] = current_price
        result["trade_executed"] = trade_action
        result["ai_portfolio_value"] = round(ai_portfolio_value, 2)
        result["baseline_value"] = round(baseline_value, 2)
        backtest_results.append(result)
        
        print(f"[{current_date}] Portfolio: ${ai_portfolio_value:,.2f} | Action: {trade_action}")
        
    final_price = float(df.loc[trading_days[-1]]['Close'])
    final_ai_value = ai_cash + (ai_shares * final_price)
    final_baseline_value = baseline_shares * final_price
    
    ai_return_pct = ((final_ai_value - initial_capital) / initial_capital) * 100
    baseline_return_pct = ((final_baseline_value - initial_capital) / initial_capital) * 100

    print("\n" + "="*40)
    print(f"🏁 ReAct BACKTEST COMPLETE ({'HARNESS ENABLED' if use_harness else 'VANILLA'}) 🏁")
    print("="*40)
    print(f"Initial Capital:  ${initial_capital:,.2f}")
    print(f"AI Final Value:   ${final_ai_value:,.2f} ({ai_return_pct:+.2f}%)")
    print(f"Baseline Value:   ${final_baseline_value:,.2f} ({baseline_return_pct:+.2f}%)")
    
    # Save to dedicated files based on mode
    output_filename = "react_backtest_results_harness.json" if use_harness else "react_backtest_results_vanilla.json"
    
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
    # EASY TOGGLE RIGHT HERE BEFORE RUNNING
    USE_HARNESS = False 
    
    run_backtest("NVDA", start_date="2023-01-01", end_date="2026-04-01", use_harness=USE_HARNESS)