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
    
    # Safe MultiIndex column flattening
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    GLOBAL_DATA_CACHE[ticker] = df
    print("Pre-fetch complete!")

# ==========================================
# 2. THE DAILY ReAct EXECUTION LOOP (VANILLA)
# ==========================================
def run_daily_agent(
    ticker: str, 
    current_date: str, 
    current_price: float,
    llm: LLM
) -> dict:
    
    def date_aware_price(arg_ticker: str) -> str:
        try:
            df = GLOBAL_DATA_CACHE.get(ticker)
            if df is None or df.empty:
                return f"No pre-fetched data found for {ticker}."
            past_data = df.loc[:current_date]
            if past_data.empty:
                return f"No price data found before {current_date}."
            
            last_5 = past_data.tail(5)
            closes = [round(float(p), 2) for p in last_5['Close'].values]
            current_p = closes[-1]
            
            return (
                f"Recent 5-Day Closing Prices for {ticker} leading to {current_date}: {closes}\n"
                f"Current Price: ${current_p:.2f}"
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

    base_engine_rules = """TRADING DIRECTIVES & POSITION SIZING:
1. Goal: Maximize risk-adjusted return through disciplined, deliberate capital deployment.
2. Position Sizing Rules:
   - You can select ANY integer percentage from 1% to 100% based on conviction (e.g., 10%, 15%, 33%, 45%, 70%, 85%).
   - BUY <pct>%: Specifies the percentage of available cash to deploy.
   - SELL <pct>%: Specifies the percentage of open shares to liquidate.
   - HOLD: Keeps position unchanged.

3. ReAct Execution Structure:
   Thought: [Analyze technical setup + detail rationale for position size percentage]
   Action: tool_name[input]
   Observation: [System response]

For your final decision, output strictly:
Action: Final_Decision[BUY, <pct>%] or Final_Decision[SELL, <pct>%] or Final_Decision[HOLD]

Format Examples:
   Action: Final_Decision[BUY, 15%]
   Action: Final_Decision[BUY, 35%]
   Action: Final_Decision[SELL, 60%]
   Action: Final_Decision[HOLD]"""

    master_system_prompt = f"You are an autonomous stock trader managing a portfolio on {current_date}.\n\n{base_engine_rules}"
    user_prompt = f"Today is {current_date}. Analyze {ticker} using available tools and decide whether to BUY, SELL, or HOLD today."
    
    messages = [
        {"role": "system", "content": master_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    trajectory_log = []
    final_decision = "HOLD" 
    
    print(f"\n[{current_date}] 🤖 Running AI logic [VANILLA]...")
    
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
                
                pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", raw_arg)
                if pct_match:
                    trade_pct = float(pct_match.group(1)) / 100.0
                else:
                    num_match = re.search(r"(\d+(?:\.\d+)?)", raw_arg)
                    if num_match:
                        val = float(num_match.group(1))
                        trade_pct = val / 100.0 if val > 1.0 else val
                    else:
                        trade_pct = 0.25
                
                trade_pct = max(0.01, min(1.0, trade_pct))
                
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
# 3. HISTORICAL BACKTEST ENGINE (VANILLA)
# ==========================================
def run_backtest(ticker: str, start_date: str, end_date: str, initial_capital: float = 100000.0):
    
    prefetch_data(ticker, start_date, end_date)
    df = GLOBAL_DATA_CACHE[ticker]
    trading_days = [d.strftime('%Y-%m-%d') for d in df.index]

    print("Loading Qwen 32B AWQ model directly into in-process vLLM engine...")
    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ", 
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1  # Set to 1 GPU
    )
    
    ai_cash = initial_capital
    ai_shares = 0.0
    cost_basis = 0.0
    
    output_filename = "react_backtest_results_vanilla.json"
    
    first_trade_date = trading_days[5] 
    first_day_price = float(df.loc[first_trade_date]['Close'])
    baseline_shares = initial_capital / first_day_price
    
    backtest_results = []
    
    for current_date in trading_days[5:]: 
        current_price = float(df.loc[current_date]['Close'])
        
        result = run_daily_agent(
            ticker=ticker, 
            current_date=current_date, 
            current_price=current_price,
            llm=llm
        )
        
        decision = result["decision"]
        trade_action = "NONE"
        
        if decision.startswith("BUY") and ai_cash > 10:
            trade_pct = float(decision.split("_")[1]) if "_" in decision else 0.25
            capital_to_use = ai_cash * trade_pct
            shares_bought = capital_to_use / current_price
            
            total_cost = (ai_shares * cost_basis) + capital_to_use
            ai_shares += shares_bought
            ai_cash -= capital_to_use
            cost_basis = total_cost / ai_shares
            
            trade_action = f"BOUGHT {shares_bought:.2f} shares ({trade_pct*100:.1f}% cash deployed) @ ${current_price:.2f}"
            
        elif decision.startswith("SELL") and ai_shares > 0.001:
            trade_pct = float(decision.split("_")[1]) if "_" in decision else 0.25
            shares_to_sell = ai_shares * trade_pct
            cash_received = shares_to_sell * current_price
            
            ai_cash += cash_received
            ai_shares -= shares_to_sell
            
            if ai_shares < 0.001:
                ai_shares = 0.0
                cost_basis = 0.0
                
            trade_action = f"SOLD {shares_to_sell:.2f} shares ({trade_pct*100:.1f}% position liquidated) @ ${current_price:.2f}"

        ai_portfolio_value = ai_cash + (ai_shares * current_price)
        baseline_value = baseline_shares * current_price
        
        result["price"] = current_price
        result["trade_executed"] = trade_action
        result["ai_portfolio_value"] = round(ai_portfolio_value, 2)
        result["baseline_value"] = round(baseline_value, 2)
        backtest_results.append(result)

        with open(output_filename, "w") as f:
            json.dump({
                "metrics": {
                    "harness_active": False,
                    "days_completed": len(backtest_results),
                    "current_ai_value": round(ai_portfolio_value, 2)
                },
                "daily_logs": backtest_results
            }, f, indent=4)
        
        print(f"[{current_date}] Portfolio: ${ai_portfolio_value:,.2f} | Action: {trade_action}")
        
    final_price = float(df.loc[trading_days[-1]]['Close'])
    final_ai_value = ai_cash + (ai_shares * final_price)
    final_baseline_value = baseline_shares * final_price
    
    ai_return_pct = ((final_ai_value - initial_capital) / initial_capital) * 100
    baseline_return_pct = ((final_baseline_value - initial_capital) / initial_capital) * 100

    print("\n" + "="*40)
    print("🏁 ReAct BACKTEST COMPLETE (VANILLA MODE) 🏁")
    print("="*40)
    print(f"Initial Capital:   ${initial_capital:,.2f}")
    print(f"AI Final Value:    ${final_ai_value:,.2f} ({ai_return_pct:+.2f}%)")
    print(f"Baseline Value:    ${final_baseline_value:,.2f} ({baseline_return_pct:+.2f}%)")
    
    with open(output_filename, "w") as f:
        json.dump({
            "metrics": {
                "harness_active": False,
                "ai_return_pct": round(ai_return_pct, 2),
                "baseline_return_pct": round(baseline_return_pct, 2),
                "beat_market": final_ai_value > final_baseline_value
            },
            "daily_logs": backtest_results
        }, f, indent=4)

if __name__ == "__main__":
    run_backtest("INTC", start_date="2023-03-15", end_date="2026-04-01")