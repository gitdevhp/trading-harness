import json
import os
import re
import copy
import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

# Environment Setup
os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yf_cache"
yf.set_tz_cache_location("/tmp/yf_tz_cache")

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"

# Single Stock Configuration
RAW_TICKER = "NVDA"
ANONYMOUS_MAP = {RAW_TICKER: "ASSET_A"}
REVERSE_MAP = {"ASSET_A": RAW_TICKER}
ANONYMOUS_UNIVERSE = ["ASSET_A"]

GLOBAL_DATA_CACHE = {}
POSITION_PEAKS = {}

TRAIN_START, TRAIN_END = "2021-01-01", "2024-06-30"  # In-Sample Training
TEST_START, TEST_END   = "2024-07-01", "2026-04-01"  # Out-of-Sample Testing

INITIAL_STRATEGY = {
    "strategy_directives": "Focus on high momentum and dip-buying near SMA20. Avoid holding during sharp momentum decelerations.",
    "stop_loss_pct": 0.10,
    "drift_threshold": 8.0,
    "min_mom20": -2.0
}

# ==========================================
# 1. DATA PREFETCHING & TECHNICAL INDICATORS
# ==========================================
def prefetch_data(start_date: str, end_date: str):
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    if "ASSET_A" not in GLOBAL_DATA_CACHE:
        df = yf.download(RAW_TICKER, start=lookback_start, end=end_date, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Sanitize data to prevent propagation of NaNs
        df = df.ffill().bfill()
        GLOBAL_DATA_CACHE["ASSET_A"] = df

def calculate_technical_indicators(past_df: pd.DataFrame) -> dict:
    closes = past_df['Close']
    current_p = float(closes.iloc[-1])
    
    sma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else current_p
    sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else current_p
    mom20 = ((current_p - float(closes.iloc[-20])) / float(closes.iloc[-20])) * 100 if len(closes) >= 20 else 0.0
    
    log_returns = np.log(closes / closes.shift(1)).tail(20)
    vol20 = float(log_returns.std() * np.sqrt(252)) if len(log_returns) >= 20 and not np.isnan(log_returns.std()) else 0.25

    return {
        "price": round(current_p, 2),
        "sma20": round(sma20, 2),
        "sma200": round(sma200, 2),
        "momentum20_pct": round(mom20, 2),
        "annual_volatility": max(vol20, 0.05)
    }

# ==========================================
# 2. QUANT RISK HARNESS
# ==========================================
def apply_institutional_risk_harness(
    raw_allocations: dict, 
    current_allocations: dict, 
    current_date: str, 
    holdings_prices: dict,
    strategy_config: dict
) -> tuple[dict, bool]:
    stop_loss_pct = strategy_config.get("stop_loss_pct", 0.10)
    drift_threshold = strategy_config.get("drift_threshold", 8.0)
    min_mom20 = strategy_config.get("min_mom20", -2.0)

    asset = "ASSET_A"
    price = holdings_prices[asset]

    # Step 1: Trailing Stop Check
    if asset not in POSITION_PEAKS or price > POSITION_PEAKS[asset]:
        POSITION_PEAKS[asset] = price
        
    peak = POSITION_PEAKS[asset]
    drawdown = (price - peak) / peak if peak > 0 else 0.0
    
    if drawdown <= -stop_loss_pct and current_allocations.get(asset, 0.0) > 0.0:
        POSITION_PEAKS[asset] = price
        return {"ASSET_A": 0.0, "CASH": 100.0}, True

    # Step 2: Technical Regime Filter
    past_df = GLOBAL_DATA_CACHE[asset].loc[:current_date]
    tech = calculate_technical_indicators(past_df)
    
    requested_asset_w = float(raw_allocations.get(asset, 0.0))

    if tech['momentum20_pct'] < min_mom20 or tech['price'] < tech['sma200']:
        target_asset_w = 0.0
    else:
        target_asset_w = min(requested_asset_w, 95.0)

    target_cash_w = round(100.0 - target_asset_w, 2)
    final_targets = {"ASSET_A": target_asset_w, "CASH": target_cash_w}

    # Step 3: Drift Suppression
    drift = abs(final_targets["ASSET_A"] - current_allocations.get("ASSET_A", 0.0))
    if drift < drift_threshold:
        return current_allocations, False

    return final_targets, False

# ==========================================
# 3. REACT DAILY AGENT
# ==========================================
def run_daily_agent(current_date: str, portfolio_state: dict, strategy_directives: str) -> dict:
    def get_market_screener(arg: str = "") -> str:
        past_df = GLOBAL_DATA_CACHE["ASSET_A"].loc[:current_date]
        tech = calculate_technical_indicators(past_df)
        return (
            f"ASSET_A: Price=${tech['price']} | 20d-SMA=${tech['sma20']} | 200d-SMA=${tech['sma200']} | "
            f"20d-Mom={tech['momentum20_pct']}% | Vol={tech['annual_volatility']*100:.1f}%"
        )

    def get_portfolio_status(arg: str = "") -> str:
        alloc_str = ", ".join([f"{k}: {v:.1f}%" for k, v in portfolio_state['allocations_pct'].items()])
        return f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | Cash: {portfolio_state['cash_pct']:.1f}%\nAllocations: {alloc_str}"

    available_tools = {
        "get_market_screener": get_market_screener,
        "get_portfolio_status": get_portfolio_status
    }

    react_system_prompt = f"""You are an autonomous trading agent operating on {current_date}.

STRATEGY DIRECTIVE:
{strategy_directives}

Available Assets: ['ASSET_A', 'CASH']

Tools:
- get_market_screener[]
- get_portfolio_status[]

Format:
Thought: <Analyze metrics and trends>
Action: <tool_name>[]
Observation: <result>
...
Thought: <Final allocation decision summing to 100>
Action: Target_Allocations[{{\"ASSET_A\": 95, \"CASH\": 5}}]"""

    messages = [
        {"role": "system", "content": react_system_prompt},
        {"role": "user", "content": f"Date: {current_date}. Select target allocation."}
    ]
    
    trajectory_log = []
    raw_decision = {"ASSET_A": 0.0, "CASH": 100.0}

    for step in range(4):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=400,
            stop=["Observation:"]
        )
        reply = response.choices[0].message.content.strip()
        trajectory_log.append(reply)
        messages.append({"role": "assistant", "content": reply})

        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, re.DOTALL)
        if action_match:
            action_name, action_arg = action_match.group(1), action_match.group(2).strip()
            
            if action_name == "Target_Allocations":
                try:
                    raw_decision = {k: float(v) for k, v in json.loads(action_arg).items()}
                except Exception:
                    pass
                break
                
            if action_name in available_tools:
                obs_text = f"Observation: {available_tools[action_name](action_arg)}"
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."
                
            trajectory_log.append(obs_text)
            messages.append({"role": "user", "content": obs_text})

    return {"date": current_date, "raw_allocations": raw_decision, "trajectory": "\n".join(trajectory_log)}

# ==========================================
# 4. BACKTEST RUNNER
# ==========================================
def run_backtest_pass(start_date: str, end_date: str, strategy: dict, initial_capital: float = 100000.0) -> dict:
    global POSITION_PEAKS
    POSITION_PEAKS = {}

    prefetch_data(start_date, end_date)
    trading_days = [d.strftime('%Y-%m-%d') for d in GLOBAL_DATA_CACHE["ASSET_A"].index if d.strftime('%Y-%m-%d') >= start_date]

    cash = initial_capital
    holdings = {"ASSET_A": 0.0}
    history = []
    
    stop_loss_count = 0
    cash_days_count = 0

    for current_date in trading_days:
        prices = {"ASSET_A": float(GLOBAL_DATA_CACHE["ASSET_A"].loc[current_date]['Close'])}
        total_value = cash + (holdings["ASSET_A"] * prices["ASSET_A"])
        allocations_pct = {"ASSET_A": (holdings["ASSET_A"] * prices["ASSET_A"] / total_value * 100.0) if total_value > 0 else 0.0}

        portfolio_state = {
            "cash": cash,
            "cash_pct": (cash / total_value) * 100.0 if total_value > 0 else 100.0,
            "portfolio_value": total_value,
            "allocations_pct": allocations_pct
        }
        
        result = run_daily_agent(current_date, portfolio_state, strategy["strategy_directives"])
        
        harnessed_targets, sl_triggered = apply_institutional_risk_harness(
            result["raw_allocations"], 
            allocations_pct, 
            current_date, 
            prices,
            strategy
        )

        if sl_triggered:
            stop_loss_count += 1
            
        if harnessed_targets.get("CASH", 0.0) >= 50.0:
            cash_days_count += 1

        norm_targets = {k: (v / 100.0) for k, v in harnessed_targets.items()}
        cash = total_value * norm_targets.get("CASH", 0.05)
        holdings["ASSET_A"] = (total_value * norm_targets.get("ASSET_A", 0.0)) / prices["ASSET_A"]

        new_value = cash + (holdings["ASSET_A"] * prices["ASSET_A"])
        history.append({"date": current_date, "value": new_value})

    values = [h["value"] for h in history]
    total_return = (values[-1] - values[0]) / values[0]
    daily_rets = pd.Series(values).pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0.0
    peaks = pd.Series(values).cummax()
    max_dd = float(((pd.Series(values) - peaks) / peaks).min())

    composite_score = (sharpe * 2.5) + (total_return * 1.5) - (abs(max_dd) * 3.5)

    return {
        "score": round(composite_score, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "final_value": round(values[-1], 2),
        "stop_loss_count": stop_loss_count,
        "cash_days_pct": round((cash_days_count / len(trading_days)) * 100.0, 1)
    }

# ==========================================
# 5. REFINER ENGINE (MUTATION NODE)
# ==========================================
def refiner_engine_mutate(metrics: dict, current_strategy: dict) -> dict:
    refiner_prompt = f"""You are an Expert Single-Asset Strategy Refiner.
Analyze the backtest results below and mutate the strategy instructions and risk metrics to boost Sharpe Ratio and limit Drawdown.

PERFORMANCE METRICS:
- Total Return: {metrics['total_return_pct']}%
- Sharpe Ratio: {metrics['sharpe_ratio']}
- Max Drawdown: {metrics['max_drawdown_pct']}%
- Stop-Loss Triggers: {metrics.get('stop_loss_count', 0)}
- Time Spent in Cash: {metrics.get('cash_days_pct', 0.0)}%

CURRENT CONFIGURATION:
{json.dumps(current_strategy, indent=2)}

DIAGNOSTIC ADVICE:
- If Stop-Loss Triggers > 5, consider widening `stop_loss_pct` slightly.
- If Time Spent in Cash > 40% during a bull trend, lower `min_mom20` or relax entry directives.

STRICT RULES:
1. `stop_loss_pct` MUST be between 0.05 and 0.20.
2. `drift_threshold` MUST be between 2.0 and 15.0.
3. `min_mom20` MUST be between -10.0 and 10.0.
4. Output ONLY a valid JSON block inside ```json ... ```.

JSON SCHEMA:
```json
{{
    "strategy_directives": "Updated trading guidelines...",
    "stop_loss_pct": 0.08,
    "drift_threshold": 5.0,
    "min_mom20": 0.0
}}
```"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": refiner_prompt}],
        temperature=0.3
    )

    content = response.choices[0].message.content
    match = re.search(r'(\{[\s\S]*\})', content)
    raw_json = match.group(1) if match else content
    
    try:
        mutated = json.loads(raw_json)
        # Bounding Safeguards
        mutated["stop_loss_pct"] = float(np.clip(mutated.get("stop_loss_pct", 0.10), 0.05, 0.20))
        mutated["drift_threshold"] = float(np.clip(mutated.get("drift_threshold", 8.0), 2.0, 15.0))
        mutated["min_mom20"] = float(np.clip(mutated.get("min_mom20", -2.0), -10.0, 10.0))
        return mutated
    except Exception:
        return current_strategy

# ==========================================
# 6. CONTINUOUS OPTIMIZATION EXECUTION
# ==========================================
def run_continuous_optimization(iterations: int = 5):
    current_strategy = copy.deepcopy(INITIAL_STRATEGY)
    champion_strategy = copy.deepcopy(INITIAL_STRATEGY)
    best_score = -9999.0

    print("=" * 70)
    print("SINGLE STOCK CONTINUOUS REFINEMENT LOOP (IN-SAMPLE)")
    print("=" * 70)

    for i in range(1, iterations + 1):
        print(f"\n--- [OPTIMIZATION ITERATION {i}/{iterations}] ---")
        metrics = run_backtest_pass(TRAIN_START, TRAIN_END, current_strategy)
        print(
            f"Results -> Score: {metrics['score']} | Return: {metrics['total_return_pct']}% | "
            f"Sharpe: {metrics['sharpe_ratio']} | MaxDD: {metrics['max_drawdown_pct']}% | "
            f"Stops Hit: {metrics['stop_loss_count']} | Cash Days: {metrics['cash_days_pct']}%"
        )

        if metrics['score'] > best_score:
            print(f" ⭐ New Champion Found! (Score: {best_score} -> {metrics['score']})")
            best_score = metrics['score']
            champion_strategy = copy.deepcopy(current_strategy)

        if i < iterations:
            print(" Refiner Node mutating based on best champion strategy...")
            # Mutate off champion strategy to avoid random-walk decay
            current_strategy = refiner_engine_mutate(metrics, champion_strategy)

    print("\n" + "=" * 70)
    print("RUNNING OUT-OF-SAMPLE TEST (FROZEN CHAMPION STRATEGY)")
    print("=" * 70)
    
    test_metrics = run_backtest_pass(TEST_START, TEST_END, champion_strategy)

    print("\nFINAL OUT-OF-SAMPLE TEST RESULTS (UNSEEN MARKET DATA):")
    print(f"- Return: {test_metrics['total_return_pct']}% | Sharpe: {test_metrics['sharpe_ratio']} | MaxDD: {test_metrics['max_drawdown_pct']}%")

if __name__ == "__main__":
    run_continuous_optimization(iterations=5)