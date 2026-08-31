import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calculate_portfolio_metrics(portfolio_series: pd.Series, benchmark_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    df = pd.DataFrame({"portfolio": portfolio_series, "benchmark": benchmark_series}).dropna()
    p_returns = df["portfolio"].pct_change().dropna()
    b_returns = df["benchmark"].pct_change().dropna()

    total_days = (df.index[-1] - df.index[0]).days
    years = max(total_days / 365.25, 0.01)

    total_return = (df["portfolio"].iloc[-1] / df["portfolio"].iloc[0]) - 1.0
    cagr = ((df["portfolio"].iloc[-1] / df["portfolio"].iloc[0]) ** (1.0 / years)) - 1.0

    ann_vol = p_returns.std() * np.sqrt(252)
    downside_returns = p_returns[p_returns < 0]
    downside_vol = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 1e-6

    rolling_peak = df["portfolio"].cummax()
    drawdown_series = (df["portfolio"] - rolling_peak) / rolling_peak
    max_drawdown = abs(float(drawdown_series.min()))

    excess_cagr = cagr - risk_free_rate
    sharpe = (p_returns.mean() * 252 - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    sortino = excess_cagr / downside_vol if downside_vol > 0 else 0.0
    calmar = excess_cagr / max_drawdown if max_drawdown > 0 else 0.0

    if df["portfolio"].equals(df["benchmark"]):
        beta = 1.0
        alpha_ann = 0.0
    else:
        beta, alpha_daily = np.polyfit(b_returns, p_returns, 1)
        alpha_ann = alpha_daily * 252 * 100.0

    return {
        "Total Return (%)": round(total_return * 100.0, 2),
        "CAGR (%)": round(cagr * 100.0, 2),
        "Ann. Volatility (%)": round(ann_vol * 100.0, 2),
        "Max Drawdown (%)": round(max_drawdown * 100.0, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Calmar Ratio": round(calmar, 2),
        "Beta (Systematic Risk)": round(beta, 2),
        "Alpha (% p.a.)": round(alpha_ann, 2),
        "drawdown_series": drawdown_series
    }

def load_json_series(file_path: str):
    if not file_path or not os.path.exists(file_path):
        return None, None
    with open(file_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df["portfolio_value"], data

# --- QUANTITATIVE BASELINE WEIGHT SOLVERS ---

def get_risk_parity_weights(returns_window: pd.DataFrame) -> np.ndarray:
    vols = returns_window.std().values
    inv_vols = np.where(vols > 1e-8, 1.0 / vols, 0.0)
    s = np.sum(inv_vols)
    return inv_vols / s if s > 0 else np.ones(len(vols)) / len(vols)

def get_min_var_weights(returns_window: pd.DataFrame) -> np.ndarray:
    n = returns_window.shape[1]
    if len(returns_window) < 5:
        return np.ones(n) / n

    cov = returns_window.cov().values + np.eye(n) * 1e-6
    init_w = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(
        fun=lambda w: w.T @ cov @ w,
        x0=init_w,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )

    w = res.x if (res.success and res.x is not None) else init_w
    w = np.clip(w, 0, 1)
    total = np.sum(w)
    return w / total if total > 0 else np.ones(n) / n

def get_cov_risk_parity_weights(returns_window: pd.DataFrame) -> np.ndarray:
    n = returns_window.shape[1]
    if len(returns_window) < 5:
        return np.ones(n) / n

    cov = returns_window.cov().values + np.eye(n) * 1e-6
    init_y = np.ones(n)

    def spinu_erc_objective(y):
        return 0.5 * (y.T @ cov @ y) - (1.0 / n) * np.sum(np.log(np.maximum(y, 1e-8)))

    bounds = tuple((1e-4, None) for _ in range(n))
    res = minimize(
        fun=spinu_erc_objective,
        x0=init_y,
        method='L-BFGS-B',
        bounds=bounds
    )

    if res.x is not None:
        w = res.x / np.sum(res.x)
        return w
    return np.ones(n) / n

def build_baseline_portfolios(price_df: pd.DataFrame, initial_capital: float, rebalance_freq: int = 5, lookback: int = 126) -> dict:
    daily_returns = price_df.pct_change().fillna(0.0)
    assets = price_df.columns
    n_assets = len(assets)

    strategies = ["60/40", "RiskParity", "CovRiskParity", "MinVariance"]
    portfolio_values = {s: [initial_capital] for s in strategies}
    current_weights = {s: np.ones(n_assets) / n_assets for s in strategies}

    has_bond = "TLT" in assets
    bond_idx = list(assets).index("TLT") if has_bond else -1

    for t in range(1, len(price_df)):
        if t % rebalance_freq == 0 or t == 1:
            window = daily_returns.iloc[max(0, t - lookback):t]
            
            if len(window) >= 5:
                current_weights["RiskParity"] = get_risk_parity_weights(window)
                current_weights["MinVariance"] = get_min_var_weights(window)
                current_weights["CovRiskParity"] = get_cov_risk_parity_weights(window)
            
            w_6040 = np.zeros(n_assets)
            if has_bond:
                w_6040[bond_idx] = 0.40
                eq_indices = [i for i in range(n_assets) if i != bond_idx]
                w_6040[eq_indices] = 0.60 / len(eq_indices)
            else:
                w_6040 = (0.60 / n_assets) * np.ones(n_assets)
            current_weights["60/40"] = w_6040

        r_t = daily_returns.iloc[t].values
        for strat in strategies:
            strat_ret = np.dot(current_weights[strat], r_t)
            prev_val = portfolio_values[strat][-1]
            portfolio_values[strat].append(prev_val * (1.0 + strat_ret))

    return {strat: pd.Series(vals, index=price_df.index) for strat, vals in portfolio_values.items()}

def generate_evaluation_report(
    harness_file: str = "react_harness_results1.json",
    gpt_harness_file: str = None,
    no_harness_file: str = "react_no_harness_results1.json",
    raw_llm_file: str = "qwen_raw_results1.json",
    output_plot: str = "MultiAsset_Baseline_Performance1.png"
):
    harness_series, harness_raw = load_json_series(harness_file)
    gpt_harness_series, _ = load_json_series(gpt_harness_file)
    react_no_harness_series, _ = load_json_series(no_harness_file)
    raw_llm_series, _ = load_json_series(raw_llm_file)

    if harness_series is None:
        raise FileNotFoundError(f"Primary file '{harness_file}' not found.")

    traded_assets = list(harness_raw[0]["prices"].keys())
    price_dict = {t: [r["prices"][t] for r in harness_raw] for t in traded_assets}
    df_index = pd.to_datetime([r["date"] for r in harness_raw])
    price_df = pd.DataFrame(price_dict, index=df_index)
    
    initial_val = float(harness_series.iloc[0])
    eq_returns = price_df.pct_change().mean(axis=1)
    eq_portfolio = (1 + eq_returns.fillna(0)).cumprod() * initial_val

    # Generate Baseline Strategies
    baselines = build_baseline_portfolios(price_df, initial_capital=initial_val)

    # Master Systems Dict
    systems = {
        "ReAct + Risk Harness": harness_series,
        "ReAct + GPT Harness": gpt_harness_series,
        "Vanilla ReAct": react_no_harness_series,
        "Raw Direct LLM": raw_llm_series,
        "Equal-Weight (1/n)": eq_portfolio,
        "60/40 Allocation": baselines["60/40"],
        "Risk Parity": baselines["RiskParity"],
        "Covariance Risk Parity": baselines["CovRiskParity"],
        "Minimum Variance": baselines["MinVariance"],
    }

    metrics = {name: calculate_portfolio_metrics(series, eq_portfolio) for name, series in systems.items() if series is not None}

    print("\n" + "="*140)
    print("                                         COMPLETE SYSTEM & QUANT BASELINE SUMMARY")
    print("="*140)
    
    headers = list(metrics.keys())
    header_row = f"{'Metric':<24} | " + " | ".join([f"{h[:10]:<10}" for h in headers])
    print(header_row)
    print("-" * len(header_row))

    metric_keys = [
        "Total Return (%)", "CAGR (%)", "Ann. Volatility (%)", "Max Drawdown (%)", 
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Beta (Systematic Risk)", "Alpha (% p.a.)"
    ]
    for k in metric_keys:
        row = f"{k:<24} | " + " | ".join([f"{metrics[h][k]:<10}" for h in headers])
        print(row)
    print("="*140 + "\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    styles = {
        "ReAct + Risk Harness": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.2},
        "ReAct + GPT Harness": {"color": "#9467bd", "linestyle": "-", "linewidth": 2.2},
        "Vanilla ReAct": {"color": "#2ca02c", "linestyle": "--", "linewidth": 1.5},
        "Raw Direct LLM": {"color": "#d62728", "linestyle": "-.", "linewidth": 1.5},
        "Equal-Weight (1/n)": {"color": "#ff7f0e", "linestyle": ":", "linewidth": 1.5},
        "60/40 Allocation": {"color": "#8c564b", "linestyle": "--", "linewidth": 1.2},
        "Risk Parity": {"color": "#17becf", "linestyle": "-", "linewidth": 1.2},
        "Covariance Risk Parity": {"color": "#e377c2", "linestyle": "-", "linewidth": 1.2},
        "Minimum Variance": {"color": "#7f7f7f", "linestyle": "-.", "linewidth": 1.2},
    }

    for name, series in systems.items():
        if series is not None:
            st = styles[name]
            alpha_val = metrics[name]["Alpha (% p.a.)"]
            beta_val = metrics[name]["Beta (Systematic Risk)"]
            ax1.plot(series.index, series, label=f"{name} (α: {alpha_val}%, β: {beta_val})", **st)

    ax1.set_title("AI Agent vs. Quantitative Benchmark Baselines", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    for name, series in systems.items():
        if series is not None:
            st = styles[name]
            dd_series = metrics[name]["drawdown_series"] * 100
            ax2.plot(series.index, dd_series, label=name, **st)
            if "Harness" in name:
                ax2.fill_between(series.index, dd_series, 0, color=st["color"], alpha=0.08)

    ax2.set_title("Underwater Chart (Drawdown %)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Graph saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Agent vs Quantitative Portfolio Baselines")
    parser.add_argument("--harness-file", type=str, default="react_harness_results1.json", help="Path to ReAct + Harness JSON results")
    parser.add_argument("--gpt-harness-file", type=str, default="react_harness_results_gpt1.json", help="Path to GPT Harness JSON results")
    parser.add_argument("--no-harness-file", type=str, default="react_no_harness_results1.json", help="Path to Vanilla ReAct JSON results")
    parser.add_argument("--raw-llm-file", type=str, default="qwen_raw_results1.json", help="Path to Raw LLM JSON results")
    parser.add_argument("--output", "-o", type=str, default="MultiAsset_Baseline_Performance1.png", help="Filename for saved plot")

    args = parser.parse_args()

    generate_evaluation_report(
        harness_file=args.harness_file,
        gpt_harness_file=args.gpt_harness_file,
        no_harness_file=args.no_harness_file,
        raw_llm_file=args.raw_llm_file,
        output_plot=args.output
    )