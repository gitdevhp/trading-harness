import os
import json
import pandas as pd
import matplotlib
# Non-interactive backend so it runs smoothly on HPC / Slurm nodes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_vanilla_performance(input_json="NoHarness_NVDA.json", output_png="vanilla_performance.png"):
    if not os.path.exists(input_json):
        print(f"❌ Error: Could not find '{input_json}'. Make sure you ran evaluate.py with USE_HARNESS = False.")
        return

    # Load JSON output from evaluate.py
    with open(input_json, 'r') as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    df = pd.DataFrame(data['daily_logs'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Set up styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(14, 8), 
        sharex=True, 
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # ----------------------------------------------------
    # PANEL 1: EQUITY CURVE COMPARISON
    # ----------------------------------------------------
    ai_return = metrics.get('ai_return_pct', 0)
    baseline_return = metrics.get('baseline_return_pct', 0)

    ax1.plot(
        df.index, 
        df['baseline_value'], 
        label=f'Buy & Hold Benchmark [{baseline_return:+.2f}%]', 
        color='#7f8c8d', 
        linestyle='--', 
        linewidth=2.0, 
        alpha=0.85
    )

    ax1.plot(
        df.index, 
        df['ai_portfolio_value'], 
        label=f'AI Agent (Vanilla) [{ai_return:+.2f}%]', 
        color='#e74c3c', 
        linewidth=2.5
    )

    ax1.set_title("Vanilla AI Agent Performance vs. Buy & Hold Benchmark", fontsize=15, fontweight='bold', pad=12)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11, fontweight='bold')
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # PANEL 2: STOCK PRICE & TRADE EXECUTION MARKERS
    # ----------------------------------------------------
    ax2.plot(
        df.index, 
        df['price'], 
        label='Stock Price', 
        color='#3498db', 
        linewidth=1.5
    )

    # Extract BUY and SELL trades from log
    buys = df[df['trade_executed'].str.startswith('BOUGHT', na=False)]
    sells = df[df['trade_executed'].str.startswith('SOLD', na=False)]

    if not buys.empty:
        ax2.scatter(
            buys.index, 
            buys['price'], 
            marker='^', 
            color='#27ae60', 
            s=120, 
            label='BUY Action', 
            zorder=5
        )

    if not sells.empty:
        ax2.scatter(
            sells.index, 
            sells['price'], 
            marker='v', 
            color='#c0392b', 
            s=120, 
            label='SELL Action', 
            zorder=5
        )

    ax2.set_ylabel("Stock Price ($)", fontsize=11, fontweight='bold')
    ax2.yaxis.set_major_formatter('${x:,.2f}')
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Date formatting on X-axis
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"📊 Vanilla performance plot saved to: {output_png}")

if __name__ == "__main__":
    plot_vanilla_performance()