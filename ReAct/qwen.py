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
    print(
        f"Prefetching historical data for {ticker} from {start_date} to {end_date}..."
    )
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    GLOBAL_DATA_CACHE[ticker] = df
    print("Pre-fetch complete!")


# ==========================================
# 2. DAILY DIRECT AGENT (NON-REACT BASELINE)
# ==========================================
def run_daily_agent_baseline(
    ticker: str, current_date: str, llm: LLM
) -> dict:
    df = GLOBAL_DATA_CACHE[ticker]
    past_data = df.loc[:current_date]
    last_5 = past_data.tail(5)
    closes = [round(float(p), 2) for p in last_5["Close"].values]
    current_price = closes[-1]

    # Exact parity directives without tool-loop overhead
    system_prompt = (
        "You are an autonomous stock trader managing a portfolio.\n\n"
        "TRADING DIRECTIVES & POSITION SIZING:\n"
        "1. Goal: Maximize risk-adjusted return through disciplined, deliberate capital deployment.\n"
        "2. Position Sizing Rules:\n"
        "   - You can select ANY integer percentage from 1% to 100% based on conviction (e.g., 10%, 15%, 33%, 45%, 70%, 85%).\n"
        "   - BUY <pct>%: Specifies the percentage of available cash to deploy.\n"
        "   - SELL <pct>%: Specifies the percentage of open shares to liquidate.\n"
        "   - HOLD: Keeps position unchanged.\n\n"
        "For your decision, output strictly in this exact format:\n"
        "Action: Final_Decision[BUY, <pct>%] or Final_Decision[SELL, <pct>%] or Final_Decision[HOLD]\n\n"
        "Format Examples:\n"
        "   Action: Final_Decision[BUY, 15%]\n"
        "   Action: Final_Decision[BUY, 35%]\n"
        "   Action: Final_Decision[SELL, 60%]\n"
        "   Action: Final_Decision[HOLD]"
    )

    user_prompt = (
        f"Today is {current_date}.\n"
        f"Ticker: {ticker}\n"
        f"Recent 5-Day Closing Prices leading to {current_date}: {closes}\n"
        f"Current Price: ${current_price:.2f}\n\n"
        f"Decide whether to BUY, SELL, or HOLD today."
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print(f"\n[{current_date}] 🤖 Running AI logic [DIRECT BASELINE]...")

    outputs = llm.chat(
        messages=messages, sampling_params=sampling_params, use_tqdm=False
    )
    reply = outputs[0].outputs[0].text.strip()

    final_decision = "HOLD"

    # Dynamic Percentage Parser matching the ReAct logic
    m = re.search(
        r"Final_Decision\[(BUY|SELL|HOLD)\s*,?\s*(\d+(?:\.\d+)?%)?\]",
        reply,
        re.IGNORECASE,
    )
    if m:
        action = m.group(1).upper()
        pct_str = m.group(2)
        if pct_str:
            trade_pct = float(pct_str.replace("%", "").strip()) / 100.0
        else:
            trade_pct = 0.25

        trade_pct = max(0.01, min(1.0, trade_pct))

        if action == "BUY":
            final_decision = f"BUY_{trade_pct}"
        elif action == "SELL":
            final_decision = f"SELL_{trade_pct}"
        else:
            final_decision = "HOLD"

    print(f"[{current_date}] ✅ DECISION: {final_decision}")

    return {
        "date": current_date,
        "decision": final_decision,
        "trajectory": reply,
    }


# ==========================================
# 3. HISTORICAL BACKTEST ENGINE (BASELINE)
# ==========================================
def run_backtest_baseline(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
):
    prefetch_data(ticker, start_date, end_date)
    df = GLOBAL_DATA_CACHE[ticker]
    trading_days = [d.strftime("%Y-%m-%d") for d in df.index]

    print(
        "Loading Qwen 32B AWQ model directly into in-process vLLM engine..."
    )
    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        max_model_len=16384,
        gpu_memory_utilization=0.85,
    )

    ai_cash = initial_capital
    ai_shares = 0.0
    cost_basis = 0.0

    output_filename = "qwen_baseline_results.json"

    first_trade_date = trading_days[5]
    first_day_price = float(df.loc[first_trade_date]["Close"])
    baseline_shares = initial_capital / first_day_price

    backtest_results = []

    for current_date in trading_days[5:]:
        current_price = float(df.loc[current_date]["Close"])

        result = run_daily_agent_baseline(
            ticker=ticker, current_date=current_date, llm=llm
        )

        decision = result["decision"]
        trade_action = "NONE"

        # BUY Execution Logic (Identical to ReAct runner)
        if decision.startswith("BUY") and ai_cash > 10:
            trade_pct = (
                float(decision.split("_")[1]) if "_" in decision else 0.25
            )
            capital_to_use = ai_cash * trade_pct
            shares_bought = capital_to_use / current_price

            total_cost = (ai_shares * cost_basis) + capital_to_use
            ai_shares += shares_bought
            ai_cash -= capital_to_use
            cost_basis = total_cost / ai_shares

            trade_action = f"BOUGHT {shares_bought:.2f} shares ({trade_pct*100:.1f}% cash deployed) @ ${current_price:.2f}"

        # SELL Execution Logic (Identical to ReAct runner)
        elif decision.startswith("SELL") and ai_shares > 0.001:
            trade_pct = (
                float(decision.split("_")[1]) if "_" in decision else 0.25
            )
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

        # INCREMENTAL SAVE
        with open(output_filename, "w") as f:
            json.dump(
                {
                    "metrics": {
                        "harness_active": False,
                        "days_completed": len(backtest_results),
                        "current_ai_value": round(ai_portfolio_value, 2),
                    },
                    "daily_logs": backtest_results,
                },
                f,
                indent=4,
            )

        print(
            f"[{current_date}] Portfolio: ${ai_portfolio_value:,.2f} | Action: {trade_action}"
        )

    # FINAL SUMMARY REPORTING
    final_price = float(df.loc[trading_days[-1]]["Close"])
    final_ai_value = ai_cash + (ai_shares * final_price)
    final_baseline_value = baseline_shares * final_price

    ai_return_pct = (
        (final_ai_value - initial_capital) / initial_capital
    ) * 100
    baseline_return_pct = (
        (final_baseline_value - initial_capital) / initial_capital
    ) * 100

    print("\n" + "=" * 40)
    print("🏁 DIRECT BASELINE BACKTEST COMPLETE 🏁")
    print("=" * 40)
    print(f"Initial Capital:   ${initial_capital:,.2f}")
    print(f"AI Final Value:    ${final_ai_value:,.2f} ({ai_return_pct:+.2f}%)")
    print(
        f"Baseline Value:    ${final_baseline_value:,.2f} ({baseline_return_pct:+.2f}%)"
    )

    with open(output_filename, "w") as f:
        json.dump(
            {
                "metrics": {
                    "harness_active": False,
                    "ai_return_pct": round(ai_return_pct, 2),
                    "baseline_return_pct": round(baseline_return_pct, 2),
                    "beat_market": final_ai_value > final_baseline_value,
                },
                "daily_logs": backtest_results,
            },
            f,
            indent=4,
        )

    print(f"Saved trajectories and financial PnL to {output_filename}")


if __name__ == "__main__":
    run_backtest_baseline("INTC", start_date="2023-03-15", end_date="2026-04-01")