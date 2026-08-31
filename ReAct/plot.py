import json
import os
import re
import sys
import matplotlib
import numpy as np
import pandas as pd

# Non-interactive backend for Slurm / HPC runners
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ANONYMOUS_MAP = {
    "ASSET_A": "AAPL",
    "ASSET_B": "NVDA",
    "ASSET_C": "MSFT",
    "ASSET_D": "AMZN",
    "ASSET_E": "GOOGL",
    "ASSET_F": "META",
    "ASSET_G": "TSLA",
    "ASSET_H": "JPM",
    "ASSET_I": "XOM",
    "ASSET_J": "JNJ",
}


def compute_drawdown(series: pd.Series) -> pd.Series:
    """Computes daily peak-to-trough percentage drawdown."""
    peak = series.cummax()
    return (series - peak) / peak * 100.0


def compute_cagr(series: pd.Series) -> float:
    """Computes Compound Annual Growth Rate (CAGR)."""
    n_days = len(series)
    if n_days < 2:
        return 0.0
    total_return = (series.iloc[-1] / series.iloc[0]) - 1.0
    return (1.0 + total_return) ** (252.0 / n_days) - 1.0


def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Computes Annualized Sharpe Ratio."""
    rf_daily = risk_free_rate / 252.0
    excess = daily_returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return np.sqrt(252.0) * (excess.mean() / std)


def compute_sortino(daily_returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Computes Annualized Sortino Ratio using downside deviation."""
    rf_daily = risk_free_rate / 252.0
    excess = daily_returns - rf_daily
    downside_returns = excess[excess < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(np.square(downside_returns)))
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return np.sqrt(252.0) * (excess.mean() / downside_std)


def compute_calmar(series: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Computes Calmar Ratio (CAGR / Absolute Max Drawdown)."""
    cagr = compute_cagr(series)
    mdd_pct = abs(compute_drawdown(series).min()) / 100.0
    if mdd_pct == 0 or np.isnan(mdd_pct):
        return 0.0
    return cagr / mdd_pct


def extract_asset_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts asset daily close prices from 'prices' dict or parses 'trajectory' logs."""
    price_records = []

    for idx, row in df.iterrows():
        prices = {}
        # 1. Direct prices payload
        if "prices" in row and isinstance(row["prices"], dict):
            prices = row["prices"]
        # 2. Extract from trajectory log text
        elif "trajectory" in row and isinstance(row["trajectory"], str):
            matches = re.findall(
                r"([A-Z0-9_]+):\s*Price=\$?([0-9\.]+)", row["trajectory"]
            )
            for asset, p_val in matches:
                ticker = ANONYMOUS_MAP.get(asset, asset)
                prices[ticker] = float(p_val)

        price_records.append(prices)

    prices_df = pd.DataFrame(price_records, index=df.index).apply(
        pd.to_numeric, errors="coerce"
    )
    valid_cols = [c for c in prices_df.columns if prices_df[c].notna().any()]
    if valid_cols:
        return prices_df[valid_cols].ffill().bfill()
    return pd.DataFrame(index=df.index)


def plot_comparative_performance(
    json_files: list, output_png: str = "comparative_performance.png"
):
    dfs = {}
    price_panels = {}

    for file_path in json_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File '{file_path}' not found. Skipping.")
            continue

        with open(file_path, "r") as f:
            raw_data = json.load(f)

        daily_logs = (
            raw_data
            if isinstance(raw_data, list)
            else raw_data.get("daily_logs", [])
        )
        if not daily_logs:
            continue

        df = pd.DataFrame(daily_logs)
        if "ai_portfolio_value" not in df.columns and "portfolio_value" in df.columns:
            df["ai_portfolio_value"] = df["portfolio_value"]

        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)

        strategy_label = (
            os.path.basename(file_path)
            .replace("react_results_", "")
            .replace(".json", "")
            .upper()
        )
        dfs[strategy_label] = df
        price_panels[strategy_label] = extract_asset_prices(df)

    if not dfs:
        print("❌ Error: No valid JSON performance log files loaded.")
        return

    # Base reference timeframe and baseline extraction
    primary_label = list(dfs.keys())[0]
    ref_df = dfs[primary_label]
    initial_val = ref_df["ai_portfolio_value"].iloc[0]

    prices_df = price_panels[primary_label]
    if not prices_df.empty:
        norm_prices = prices_df.div(prices_df.iloc[0], axis=1)
        benchmark_series = norm_prices.mean(axis=1) * initial_val
    else:
        benchmark_series = pd.Series(initial_val, index=ref_df.index)

    # Plot Configuration
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.1, 1.2]},
    )

    # PANEL 1: EQUITY CURVES & BACKGROUND ASSET TRAJECTORIES
    if not prices_df.empty:
        scaled_stock_prices = norm_prices * initial_val
        for col in scaled_stock_prices.columns:
            ax1.plot(
                scaled_stock_prices.index,
                scaled_stock_prices[col],
                color="gray",
                alpha=0.18,
                linewidth=0.9,
                linestyle="-.",
            )
        ax1.plot([], [], color="gray", alpha=0.4, linestyle="-.", label="Underlying Stock Trajectories")

    bench_ret = (
        (benchmark_series.iloc[-1] - benchmark_series.iloc[0])
        / benchmark_series.iloc[0]
    ) * 100.0
    ax1.plot(
        benchmark_series.index,
        benchmark_series,
        label=f"Equal-Weight Buy & Hold Benchmark [{bench_ret:+.2f}%]",
        color="#7f8c8d",
        linestyle="--",
        linewidth=2.0,
    )

    colors = ["#1f77b4", "#2ecc71", "#9b59b6", "#e67e22", "#e74c3c"]
    stats_records = []

    for i, (label, df) in enumerate(dfs.items()):
        series = df["ai_portfolio_value"]
        ret = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100.0
        color = colors[i % len(colors)]
        
        ax1.plot(
            df.index,
            series,
            label=f"Agent ({label}) [{ret:+.2f}%]",
            color=color,
            linewidth=2.2,
        )

        # Quantitative Performance Metrics
        daily_rets = series.pct_change().dropna()
        stats_records.append(
            {
                "Strategy": f"Agent ({label})",
                "Total Return (%)": f"{ret:+.2f}%",
                "CAGR (%)": f"{compute_cagr(series)*100:+.2f}%",
                "Sharpe Ratio": f"{compute_sharpe(daily_rets):.2f}",
                "Sortino Ratio": f"{compute_sortino(daily_rets):.2f}",
                "Max Drawdown (%)": f"{compute_drawdown(series).min():.2f}%",
                "Calmar Ratio": f"{compute_calmar(series):.2f}",
            }
        )

    # Benchmark Quantitative Metrics
    bench_daily = benchmark_series.pct_change().dropna()
    stats_records.append(
        {
            "Strategy": "Equal-Weight Benchmark",
            "Total Return (%)": f"{bench_ret:+.2f}%",
            "CAGR (%)": f"{compute_cagr(benchmark_series)*100:+.2f}%",
            "Sharpe Ratio": f"{compute_sharpe(bench_daily):.2f}",
            "Sortino Ratio": f"{compute_sortino(bench_daily):.2f}",
            "Max Drawdown (%)": f"{compute_drawdown(benchmark_series).min():.2f}%",
            "Calmar Ratio": f"{compute_calmar(benchmark_series):.2f}",
        }
    )

    ax1.set_title(
        "Multi-Strategy Comparative Analysis with Background Market Trajectories",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10, fontweight="bold")
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PANEL 2: ROLLING EXCESS RETURN VS BENCHMARK
    prim_df = list(dfs.values())[0]
    rolling_window = 21
    prim_rolling = prim_df["ai_portfolio_value"].pct_change(rolling_window) * 100.0
    bench_rolling = benchmark_series.pct_change(rolling_window) * 100.0
    excess_spread = prim_rolling - bench_rolling

    ax2.plot(excess_spread.index, excess_spread, color="#2c3e50", linewidth=1.0, alpha=0.8)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.fill_between(
        excess_spread.index,
        0,
        excess_spread,
        where=(excess_spread >= 0),
        color="#2ecc71",
        alpha=0.4,
        label="Outperforming",
    )
    ax2.fill_between(
        excess_spread.index,
        0,
        excess_spread,
        where=(excess_spread < 0),
        color="#e74c3c",
        alpha=0.4,
        label="Underperforming",
    )
    ax2.set_ylabel(f"Rolling {rolling_window}d Spread (%)", fontsize=10, fontweight="bold")
    ax2.set_title(f"Rolling Excess Return vs. Benchmark ({primary_label})", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # PANEL 3: NORMALIZED ASSET PRICE MOVEMENTS
    if not prices_df.empty:
        for col in prices_df.columns:
            ax3.plot(
                norm_prices.index,
                norm_prices[col],
                label=col,
                linewidth=1.2,
                alpha=0.75,
            )
        ax3.set_ylabel("Normalized Growth", fontsize=10, fontweight="bold")
        ax3.set_title("Underlying Stock Performance Normalized (Base=1.0)", fontsize=11, fontweight="bold")
        ax3.legend(loc="upper left", ncol=5, frameon=True, framealpha=0.9, fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.6)

    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"📊 Performance plot saved to: {output_png}")

    # Display Comparative Summary Table
    summary_df = pd.DataFrame(stats_records).set_index("Strategy")
    print("\n" + "=" * 85)
    print("                 MULTI-STRATEGY STATISTICAL PERFORMANCE SUMMARY")
    print("=" * 85)
    print(summary_df.to_string())
    print("=" * 85 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_inputs = [f for f in sys.argv[1:] if f.endswith(".json")]
        out_png = "comparative_performance.png"
        plot_comparative_performance(json_inputs, out_png)
    else:
        default_files = [
            "react_results_portfolio.json",
            "react_backtest_results_vanilla.json",
            "react_results_qwen_harness.json",
        ]
        plot_comparative_performance(default_files, "portfolio_comparison.png")