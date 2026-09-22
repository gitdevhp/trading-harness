"""
Visualize the adaptive routing backtest results.

Reads the JSON produced by run_monthly_adaptive.py and generates a
three-panel figure:

  Panel 1 — Cumulative portfolio value over time.  One line for the
             adaptive system; optionally overlaid with the four sub-systems
             and equal-weight for context.
  Panel 2 — System selection timeline: a colored horizontal bar per period
             showing which system was active.
  Panel 3 — Routing features over time: momentum breadth (fraction of
             assets above their 200d-SMA) and N-stocks annotation.

Usage:
    python -m ace_harness.plot_adaptive \
        --adaptive_json ./results/monthly_adaptive_convictionriskharness_adaptive_results.json \
        --baseline_jsons ./results/monthly_intra_convictionriskharness_results.json \
                         ./results/monthly_memory_only_convictionriskharness_adaptive_results.json \
                         ./results/monthly_dual_permanent_convictionriskharness_adaptive_results.json \
        --output_png ./results/adaptive_routing.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SYSTEM_COLORS = {
    "harness": "#607D8B",  # grey-blue
    "debater": "#2196F3",  # blue
    "memory":  "#FF9800",  # amber
    "ace":     "#4CAF50",  # green
}
SYSTEM_LABELS = {
    "harness": "HARNESS",
    "debater": "DEBATER",
    "memory":  "MEMORY",
    "ace":     "ACE",
}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _extract_equity_curve(data):
    """Extract (date, portfolio_value) from engine_monthly result JSON.
    The engine saves a list of records; accept both a bare list and a
    dict with a 'history' key for forward-compatibility."""
    history = data if isinstance(data, list) else data.get("history", [])
    dates = [h["date"] for h in history]
    values = [h["portfolio_value"] for h in history]
    return dates, values


def _extract_selection_log(data):
    """Return list of {date, system, reason, features} from adaptive meta."""
    history = data if isinstance(data, list) else data.get("history", [])
    rows = []
    for h in history:
        meta = h.get("meta") or {}
        adaptive = meta.get("adaptive") or {}
        if adaptive:
            rows.append({
                "date":     h["date"],
                "system":   adaptive.get("system", "harness"),
                "reason":   adaptive.get("reason", ""),
                "features": adaptive.get("features", {}),
            })
    return rows


def plot_adaptive(adaptive_json, baseline_jsons=None, output_png=None, initial_capital=1_000_000.0):
    data = _load_json(adaptive_json)
    dates, values = _extract_equity_curve(data)
    selection_log = _extract_selection_log(data)

    if not dates:
        print("No history found in adaptive JSON — is the backtest complete?")
        return

    # Normalize to cumulative return
    base = values[0]
    cum_rets = [v / base - 1.0 for v in values]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [3, 1, 1.5]})
    fig.suptitle("Adaptive System Routing — Portfolio Behaviour", fontsize=14, fontweight="bold")

    # ── Panel 1: Equity curves ───────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, [r * 100 for r in cum_rets], color="#1A237E", linewidth=2,
             label="Adaptive (routing)", zorder=5)

    # Overlay baselines if provided
    overlay_colors = ["#E53935", "#6A1B9A", "#F57F17", "#2E7D32"]
    overlay_labels = ["DEBATER", "MEMORY", "ACE", "HARNESS"]
    for path, color, lbl in zip(baseline_jsons or [], overlay_colors, overlay_labels):
        if not os.path.exists(path):
            continue
        bdata = _load_json(path)
        bdates, bvals = _extract_equity_curve(bdata)
        if not bvals:
            continue
        bbase = bvals[0]
        bcum = [v / bbase - 1.0 for v in bvals]
        ax1.plot(bdates, [r * 100 for r in bcum], linewidth=1.2, alpha=0.6,
                 linestyle="--", color=color, label=lbl)

    ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Cumulative Return", fontsize=10)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2: System selection timeline ───────────────────────────────────
    ax2 = axes[1]
    if selection_log:
        for i, row in enumerate(selection_log):
            left = row["date"]
            right = selection_log[i + 1]["date"] if i + 1 < len(selection_log) else dates[-1]
            color = SYSTEM_COLORS.get(row["system"], "#999999")
            # Convert to numeric x-axis position using index
            li = dates.index(left) if left in dates else 0
            ri = dates.index(right) if right in dates else len(dates) - 1
            ax2.barh(0, ri - li, left=li, height=0.7, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
            # Label the system name inside wider bars
            if ri - li > 3:
                ax2.text(li + (ri - li) / 2, 0, SYSTEM_LABELS.get(row["system"], row["system"]),
                         ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        patches = [mpatches.Patch(color=c, label=SYSTEM_LABELS.get(s, s))
                   for s, c in SYSTEM_COLORS.items()]
        ax2.legend(handles=patches, loc="upper right", fontsize=7, ncol=4)

    ax2.set_yticks([])
    ax2.set_xlim(0, len(dates))
    ax2.set_xticks([])
    ax2.set_title("System Selected per Rebalance Period", fontsize=10)
    ax2.set_ylabel("Active\nSystem")

    # ── Panel 3: Routing features ─────────────────────────────────────────────
    ax3 = axes[2]
    if selection_log:
        feat_dates = [r["date"] for r in selection_log]
        breadths   = [r["features"].get("price_breadth", 0) * 100 for r in selection_log]
        dispersions = [r["features"].get("return_dispersion", 0) for r in selection_log]

        # Map feature dates to numeric positions
        feat_xs = [dates.index(d) if d in dates else 0 for d in feat_dates]

        ax3.step(feat_xs, breadths, where="post", color="#1565C0", linewidth=1.8,
                 label="Price Breadth (% above 200d-SMA)")
        ax3.axhline(40, color="#E53935", linewidth=1, linestyle="--", alpha=0.7, label="Bear threshold (40%)")
        ax3.axhline(55, color="#2E7D32", linewidth=1, linestyle="--", alpha=0.7, label="Bull threshold (55%)")

        ax3_r = ax3.twinx()
        ax3_r.step(feat_xs, dispersions, where="post", color="#FF6F00", linewidth=1.2,
                   alpha=0.6, linestyle=":", label="Return dispersion (std 1M-Mom)")
        ax3_r.set_ylabel("Dispersion (%)", fontsize=8, color="#FF6F00")
        ax3_r.tick_params(axis="y", labelcolor="#FF6F00")

    ax3.set_ylabel("Price Breadth (%)")
    ax3.set_ylim(0, 110)
    ax3.legend(loc="upper left", fontsize=7)
    ax3.set_xlim(0, len(dates))
    # Set x-tick labels to dates at sensible intervals
    tick_step = max(1, len(dates) // 12)
    tick_positions = list(range(0, len(dates), tick_step))
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([dates[i] for i in tick_positions], rotation=30, fontsize=7, ha="right")
    ax3.set_title("Routing Features", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Summary stats in text box
    if selection_log:
        counts = {}
        for r in selection_log:
            counts[r["system"]] = counts.get(r["system"], 0) + 1
        total = sum(counts.values())
        summary_lines = [f"{SYSTEM_LABELS.get(s, s)}: {c}/{total} periods"
                         for s, c in sorted(counts.items(), key=lambda x: -x[1])]
        summary = "\n".join(summary_lines)
        fig.text(0.82, 0.02, summary, fontsize=8, family="monospace",
                 verticalalignment="bottom",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if output_png:
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Saved -> {output_png}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize adaptive routing backtest")
    parser.add_argument("--adaptive_json", required=True)
    parser.add_argument("--baseline_jsons", nargs="*", default=[])
    parser.add_argument("--output_png", default=None)
    parser.add_argument("--initial_capital", type=float, default=1_000_000.0)
    args = parser.parse_args()
    plot_adaptive(args.adaptive_json, args.baseline_jsons, args.output_png, args.initial_capital)


if __name__ == "__main__":
    main()
