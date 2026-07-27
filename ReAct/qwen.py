import json
import pandas as pd
import yfinance as yf
from vllm import LLM, SamplingParams


def prefetch_data(ticker: str, start_date: str, end_date: str):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def run_qwen_baseline(ticker: str, start_date: str, end_date: str, initial_capital: float = 100000.0):
    df = prefetch_data(ticker, start_date, end_date)
    trading_days = [d.strftime('%Y-%m-%d') for d in df.index]

    print("Loading Qwen model (baseline, no ReAct)...")
    llm = LLM(model="Qwen/Qwen2.5-32B-Instruct-AWQ", max_model_len=4096, gpu_memory_utilization=0.85)
    sampling = SamplingParams(temperature=0.0, max_tokens=128)

    cash = initial_capital
    shares = 0.0
    cost_basis = 0.0

    results = []

    # start after 20 days to have SMA
    for current_date in trading_days[20:]:
        window = df.loc[:current_date].tail(20)
        closes = [round(float(x), 2) for x in window['Close'].values]
        sma20 = round(sum(closes) / len(closes), 2)
        current_price = closes[-1]

        system = (
            "You are a concise trading assistant. Given recent price action, respond with a single line:\n"
            "Final_Decision[BUY, <pct>] or Final_Decision[SELL, <pct>] or Final_Decision[HOLD].\n"
            "Use percentage choices: 25%, 50%, 100%. No other text.\n"
        )

        user = (
            f"Ticker: {ticker} | Date: {current_date}\n"
            f"Last 20 closes: {closes}\n"
            f"Current Price: ${current_price:.2f}"
            "Decide:"
        )

        outputs = llm.chat(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], sampling_params=sampling, use_tqdm=False)
        reply = outputs[0].outputs[0].text.strip()

        decision = "HOLD"
        if "Final_Decision" in reply:
            import re
            m = re.search(r"Final_Decision\[(BUY|SELL|HOLD)\s*,?\s*(\d+%)?\]", reply, re.IGNORECASE)
            if m:
                action = m.group(1).upper()
                pct = m.group(2) or "100%"
                pct_val = float(pct.replace('%', '')) / 100.0
                if action == "BUY":
                    capital = cash * pct_val
                    if capital > 1:
                        shares_bought = capital / current_price
                        total_cost = (shares * cost_basis) + capital
                        shares += shares_bought
                        cash -= capital
                        cost_basis = total_cost / shares
                        decision = f"BUY_{pct_val}"
                elif action == "SELL":
                    shares_to_sell = shares * pct_val
                    cash += shares_to_sell * current_price
                    shares -= shares_to_sell
                    if shares < 1e-6:
                        shares = 0.0
                        cost_basis = 0.0
                    decision = f"SELL_{pct_val}"
                else:
                    decision = "HOLD"

        portfolio_value = cash + shares * current_price
        results.append({
            "date": current_date,
            "price": current_price,
            "decision": decision,
            "cash": round(cash, 2),
            "shares": round(shares, 6),
            "portfolio_value": round(portfolio_value, 2)
        })

        print(f"[{current_date}] Price ${current_price:.2f} | Decision: {decision} | Portfolio: ${portfolio_value:,.2f}")

    out_file = "qwen_baseline_results.json"
    with open(out_file, "w") as f:
        json.dump({"ticker": ticker, "results": results}, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    run_qwen_baseline("META", start_date="2023-01-01", end_date="2026-04-01")
