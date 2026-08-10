import os
import json
import numpy as np
import pandas as pd
import matplotlib
# Non-interactive backend so it runs smoothly on HPC / Slurm nodes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return (series - peak) / peak * 100.0

def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    rf_daily = risk_free_rate / 252
    excess = daily_returns - rf_daily
    if excess.std() == 0 or np.isnan(excess.std()):
        return 0.0
    return np.sqrt(252) * (excess.mean() / excess.std())

def plot_scientific_performance(input_json: str, output_png: str, rolling_window: int = 21):
    # Fallback check for alternate file naming conventions
    if not os.path.exists(input_json):
        alt_json = input_json.replace("react_results_", "react_results_high_growth_")
        if os.path.exists(alt_json):
            input_json = alt_json
        else:
            print(f"❌ Error: Could not find '{input_json}'.")
            return

    # Load JSON output from backtester
    with open(input_json, 'r') as f:
        raw_data = json.load(f)

    # Parse JSON list structure directly or extract daily_logs if nested dict
    metrics = {}
    if isinstance(raw_data, list):
        daily_logs = raw_data
    elif isinstance(raw_data, dict):
        metrics = raw_data.get("metrics", {})
        daily_logs = (
            raw_data.get("daily_logs") 
            or raw_data.get("daily_log") 
            or raw_data.get("results") 
            or []
        )
    else:
        print(f"❌ Error: Unexpected JSON data type in '{input_json}'.")
        return

    if not daily_logs:
        print(f"❌ Error: No daily log rows found in '{input_json}'.")
        return

    df = pd.DataFrame(daily_logs)

    # Verify required columns exist
    required_columns = ['date', 'ai_portfolio_value']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"❌ Error: Missing required columns in daily logs: {missing}")
        return

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # Generate or extract baseline value if missing
    if 'baseline_value' not in df.columns:
        if 'price' in df.columns:
            df['baseline_value'] = (df['price'] / df['price'].iloc[0]) * df['ai_portfolio_value'].iloc[0]
        else:
            df['baseline_value'] = df['ai_portfolio_value'].iloc[0]

    # Calculate short-term rolling returns and outperformance delta
    df['ai_rolling_ret'] = df['ai_portfolio_value'].pct_change(rolling_window) * 100.0
    df['base_rolling_ret'] = df['baseline_value'].pct_change(rolling_window) * 100.0
    df['rolling_excess'] = df['ai_rolling_ret'] - df['base_rolling_ret']

    # Cumulative performance metrics
    initial_ai = df['ai_portfolio_value'].iloc[0]
    final_ai = df['ai_portfolio_value'].iloc[-1]
    ai_return = metrics.get('ai_return_pct', ((final_ai - initial_ai) / initial_ai) * 100.0)

    initial_base = df['baseline_value'].iloc[0]
    final_base = df['baseline_value'].iloc[-1]
    baseline_return = metrics.get('baseline_return_pct', ((final_base - initial_base) / initial_base) * 100.0)

    # Styling setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    has_price = 'price' in df.columns and 'trade_executed' in df.columns
    fig_rows = 3 if has_price else 2
    height_ratios = [2.2, 1.1, 1.1] if has_price else [2.5, 1.2]
    
    fig, axes = plt.subplots(
        fig_rows, 1, 
        figsize=(14, 10 if has_price else 8), 
        sharex=True, 
        gridspec_kw={'height_ratios': height_ratios}
    )
    
    if fig_rows == 2:
        ax1, ax2 = axes
        ax3 = None
    else:
        ax1, ax2, ax3 = axes

    ticker = input_json.replace("react_results_high_growth_", "").replace("react_results_harness_", "").replace("react_results_plain_", "").replace("react_results_", "").replace(".json", "")

    # ----------------------------------------------------
    # PANEL 1: EQUITY CURVES & OUTPERFORMANCE REGIMES
    # ----------------------------------------------------
    ax1.plot(
        df.index, 
        df['baseline_value'], 
        label=f'Buy & Hold Benchmark [{baseline_return:+.2f}%]', 
        color='#7f8c8d', 
        linestyle='--', 
        linewidth=1.8, 
        alpha=0.85
    )

    ax1.plot(
        df.index, 
        df['ai_portfolio_value'], 
        label=f'AI Agent / Harness [{ai_return:+.2f}%]', 
        color='#1f77b4', 
        linewidth=2.2
    )

    # Highlight short-term outperformance regimes in light green
    outperforming = df['rolling_excess'] > 0
    ax1.fill_between(
        df.index, 
        df['ai_portfolio_value'].min(), 
        df['ai_portfolio_value'].max(), 
        where=outperforming, 
        color='#2ecc71', 
        alpha=0.12, 
        label=f'Short-Term Outperformance ({rolling_window}d)'
    )

    ax1.set_title(f"Scientific Performance Analysis: AI Agent vs. Buy & Hold ({ticker.upper()})", fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10, fontweight='bold')
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # PANEL 2: ROLLING SHORT-TERM OUTPERFORMANCE SPREAD
    # ----------------------------------------------------
    ax2.plot(df.index, df['rolling_excess'], color='#2c3e50', linewidth=1.0, alpha=0.7)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    ax2.fill_between(
        df.index, 0, df['rolling_excess'], 
        where=(df['rolling_excess'] >= 0), color='#2ecc71', alpha=0.45, label='AI Outperforming'
    )
    ax2.fill_between(
        df.index, 0, df['rolling_excess'], 
        where=(df['rolling_excess'] < 0), color='#e74c3c', alpha=0.45, label='Buy & Hold Outperforming'
    )

    ax2.set_ylabel(f"Rolling {rolling_window}d Delta (%)", fontsize=10, fontweight='bold')
    ax2.set_title(f"Short-Term Excess Return Spread ({rolling_window}-Trading Day Window)", fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # PANEL 3: STOCK PRICE & TRADE EXECUTIONS (IF APPLICABLE)
    # ----------------------------------------------------
    if ax3 is not None:
        ax3.plot(
            df.index, 
            df['price'], 
            label='Stock Price', 
            color='#3498db', 
            linewidth=1.5
        )

        buys = df[df['trade_executed'].astype(str).str.startswith('BOUGHT', na=False)]
        sells = df[df['trade_executed'].astype(str).str.startswith('SOLD', na=False)]

        if not buys.empty:
            ax3.scatter(
                buys.index, buys['price'], 
                marker='^', color='#27ae60', s=100, label='BUY Action', zorder=5
            )

        if not sells.empty:
            ax3.scatter(
                sells.index, sells['price'], 
                marker='v', color='#c0392b', s=100, label='SELL Action', zorder=5
            )

        ax3.set_ylabel("Stock Price ($)", fontsize=10, fontweight='bold')
        ax3.yaxis.set_major_formatter('${x:,.2f}')
        ax3.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
        ax3.grid(True, linestyle=':', alpha=0.6)

    # Date formatting on bottom X-axis
    bottom_ax = ax3 if ax3 is not None else ax2
    bottom_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"📊 Performance plot saved to: {output_png}")

    # ----------------------------------------------------
    # STATISTICAL OUTPERFORMANCE TABLE
    # ----------------------------------------------------
    daily_ai_ret = df['ai_portfolio_value'].pct_change().dropna()
    daily_base_ret = df['baseline_value'].pct_change().dropna()
    
    sharpe_ai = compute_sharpe(daily_ai_ret)
    sharpe_base = compute_sharpe(daily_base_ret)
    
    max_dd_ai = compute_drawdown(df['ai_portfolio_value']).min()
    max_dd_base = compute_drawdown(df['baseline_value']).min()
    
    win_rate = (df['rolling_excess'] > 0).mean() * 100.0

    stats_df = pd.DataFrame({
        "Metric": ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"],
        "AI Agent / Harness": [f"{ai_return:+.2f}%", f"{sharpe_ai:.2f}", f"{max_dd_ai:.2f}%"],
        "Buy & Hold": [f"{baseline_return:+.2f}%", f"{sharpe_base:.2f}", f"{max_dd_base:.2f}%"]
    }).set_index("Metric")

    print("\n" + "=" * 55)
    print(f"   STATISTICAL EVALUATION SUMMARY ({ticker.upper()})")
    print("=" * 55)
    print(stats_df.to_string())
    print("-" * 55)
    print(f"Short-Term ({rolling_window}-Day) Win Rate vs Benchmark: {win_rate:.1f}% of trading days")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    targets = [
        ("react_results_INTC.json", "ReAct_performance_v2_INTC.png"),
    ]
    
    for json_file, png_file in targets:
        plot_scientific_performance(json_file, png_file)