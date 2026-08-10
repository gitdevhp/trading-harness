import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
# Non-interactive backend for HPC / Slurm node environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

UNIVERSE = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "JNJ"]
ROLLING_WINDOW = 21  # 21 trading days (~1 month)
RISK_FREE_RATE = 0.04

def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return (series - peak) / peak * 100.0

def compute_sharpe(daily_returns: pd.Series) -> float:
    rf_daily = RISK_FREE_RATE / 252
    excess = daily_returns - rf_daily
    if excess.std() == 0 or np.isnan(excess.std()):
        return 0.0
    return np.sqrt(252) * (excess.mean() / excess.std())

def plot_multi_asset_portfolio(input_json: str, output_png: str):
    if not os.path.exists(input_json):
        print(f"❌ Error: Could not find '{input_json}'.")
        return

    with open(input_json, 'r') as f:
        raw_data = json.load(f)

    daily_logs = raw_data if isinstance(raw_data, list) else raw_data.get("daily_logs", raw_data.get("results", []))
    if not daily_logs:
        print(f"❌ Error: No valid daily entries in '{input_json}'.")
        return

    df = pd.DataFrame(daily_logs)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")
    initial_cap = float(df['portfolio_value'].iloc[0])

    # Fetch asset price data for benchmark & background stock paths
    print(f"Downloading underlying universe stock prices ({start_date} to {end_date})...")
    stock_data = yf.download(UNIVERSE, start=start_date, end=end_date, auto_adjust=True)["Close"]
    stock_data = stock_data.reindex(df.index).ffill().bfill()

    # Equal-weight buy & hold benchmark
    norm_stocks = stock_data / stock_data.iloc[0]
    df['baseline_value'] = norm_stocks.mean(axis=1) * initial_cap

    # Individual stock paths scaled to starting portfolio capital
    scaled_stocks = norm_stocks * initial_cap

    # Calculate short-term rolling returns & excess return delta
    df['ai_rolling'] = df['portfolio_value'].pct_change(ROLLING_WINDOW) * 100.0
    df['base_rolling'] = df['baseline_value'].pct_change(ROLLING_WINDOW) * 100.0
    df['rolling_excess'] = df['ai_rolling'] - df['base_rolling']

    # Returns & Drawdowns
    ai_return = ((df['portfolio_value'].iloc[-1] - initial_cap) / initial_cap) * 100.0
    base_return = ((df['baseline_value'].iloc[-1] - initial_cap) / initial_cap) * 100.0
    dd_ai = compute_drawdown(df['portfolio_value'])
    dd_base = compute_drawdown(df['baseline_value'])

    # Parse Executed Asset Allocations (%) over time
    alloc_data = []
    for d, row in df.iterrows():
        allocs = row.get("executed_allocations", row.get("allocations", {}))
        alloc_data.append(allocs if isinstance(allocs, dict) else {})
    alloc_df = pd.DataFrame(alloc_data, index=df.index).fillna(0.0)

    # Styling setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, 
        figsize=(14, 11), 
        sharex=True, 
        gridspec_kw={'height_ratios': [2.2, 1.0, 1.2]}
    )

    # ----------------------------------------------------
    # PANEL 1: EQUITY CURVE + FADED STOCK PRICE PATHS
    # ----------------------------------------------------
    # Plot faint individual asset trajectories
    first_stock = True
    for col in scaled_stocks.columns:
        ax1.plot(
            scaled_stocks.index, 
            scaled_stocks[col], 
            color='#95a5a6', 
            alpha=0.25, 
            linewidth=1.0, 
            linestyle='-', 
            label='Individual Stocks' if first_stock else ""
        )
        first_stock = False

    # Plot Buy & Hold Equal-Weight Benchmark
    ax1.plot(
        df.index, df['baseline_value'], 
        label=f'Equal-Weight Benchmark [{base_return:+.2f}%]', 
        color='#2c3e50', linestyle='--', linewidth=2.0, alpha=0.9
    )

    # Plot ReAct Portfolio Value
    ax1.plot(
        df.index, df['portfolio_value'], 
        label=f'Multi-Asset AI Agent [{ai_return:+.2f}%]', 
        color='#e74c3c', linewidth=2.5
    )

    # Outperformance Regime Overlay
    harness_winning = df['rolling_excess'] > 0
    ax1.fill_between(
        df.index, df['portfolio_value'].min(), df['portfolio_value'].max(), 
        where=harness_winning, color='#2ecc71', alpha=0.10, label=f'Short-Term Outperformance ({ROLLING_WINDOW}d)'
    )

    ax1.set_title("Multi-Asset ReAct Portfolio vs. Benchmark & Stock Universe Movement", fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10, fontweight='bold')
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # PANEL 2: SHORT-TERM ROLLING OUTPERFORMANCE DELTA
    # ----------------------------------------------------
    ax2.plot(df.index, df['rolling_excess'], color='#34495e', linewidth=1.0, alpha=0.8)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.fill_between(df.index, 0, df['rolling_excess'], where=(df['rolling_excess'] >= 0), color='#2ecc71', alpha=0.45, label='AI Outperforming')
    ax2.fill_between(df.index, 0, df['rolling_excess'], where=(df['rolling_excess'] < 0), color='#e74c3c', alpha=0.45, label='Benchmark Outperforming')

    ax2.set_ylabel(f"{ROLLING_WINDOW}d Excess Return (%)", fontsize=10, fontweight='bold')
    ax2.set_title(f"Rolling Short-Term Excess Return Spread ({ROLLING_WINDOW}-Day Window)", fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # PANEL 3: DYNAMIC ASSET ALLOCATION BREAKDOWN
    # ----------------------------------------------------
    if not alloc_df.empty:
        cols_to_plot = [c for c in alloc_df.columns if c in UNIVERSE or c == "CASH"]
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in np.linspace(0, 1, len(cols_to_plot))]
        
        ax3.stackplot(
            alloc_df.index, 
            [alloc_df[c] for c in cols_to_plot], 
            labels=cols_to_plot, 
            colors=colors, 
            alpha=0.85
        )
        ax3.set_ylabel("Allocation (%)", fontsize=10, fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.set_title("Dynamic Portfolio Exposure Shifting over Time", fontsize=11, fontweight='bold')
        ax3.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.6)

    # Formatting X-axis dates
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"📊 Portfolio performance plot saved to: {output_png}\n")

    # ----------------------------------------------------
    # STATISTICAL OUTPERFORMANCE SUMMARY
    # ----------------------------------------------------
    daily_ai = df['portfolio_value'].pct_change().dropna()
    daily_base = df['baseline_value'].pct_change().dropna()

    win_rate = (df['rolling_excess'] > 0).mean() * 100.0

    summary_df = pd.DataFrame({
        "Metric": ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"],
        "AI Multi-Asset Agent": [f"{ai_return:+.2f}%", f"{compute_sharpe(daily_ai):.2f}", f"{dd_ai.min():.2f}%"],
        "Buy & Hold Equal Weight": [f"{base_return:+.2f}%", f"{compute_sharpe(daily_base):.2f}", f"{dd_base.min():.2f}%"]
    }).set_index("Metric")

    print("=" * 60)
    print("        MULTI-ASSET PORTFOLIO STATISTICAL SUMMARY        ")
    print("=" * 60)
    print(summary_df.to_string())
    print("-" * 60)
    print(f"Short-Term ({ROLLING_WINDOW}-Day) Win Rate vs Benchmark: {win_rate:.1f}% of trading days")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    targets = [
        ("react_results_plain_portfolio.json", "MultiAsset_Plain_Performance.png"),
        # ("react_results_harness_portfolio.json", "MultiAsset_Harness_Performance.png")
    ]
    for json_file, png_file in targets:
        plot_multi_asset_portfolio(json_file, png_file)