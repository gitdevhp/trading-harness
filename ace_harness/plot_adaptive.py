"""
Visualize adaptive self-improvement behaviour matching the paper's metrics.

Three-panel figure following the structure of Tables 8/9 in the ReMo paper:

  Panel 1 — Cumulative portfolio returns (adaptive vs. baselines).
  Panel 2 — Consolidation decisions per rebalance period:
               admitted / consolidated / skipped — matching the paper's
               "consolidator calls per task" and "admitted" fraction metrics.
  Panel 3 — Memory size over time (bullets in playbook) — matching the
               paper's final memory size metric.

For the adaptive routing run produced by run_monthly_adaptive.py, also
overlays the per-period system selection (harness/debater/memory/ace) as
a color band across the top of Panel 2.

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

    # ── Panel 2: Consolidation decisions (admitted / consolidated / skipped) ─
    ax2 = axes[1]
    rebalance_entries = [h for h in (data if isinstance(data, list) else data.get("history", []))
                         if h.get("meta")]
    if rebalance_entries:
        xs = [dates.index(h["date"]) if h["date"] in dates else 0 for h in rebalance_entries]
        admitted_xs = [x for h, x in zip(rebalance_entries, xs) if h["meta"].get("admitted")]
        consolidated_xs = [x for h, x in zip(rebalance_entries, xs) if h["meta"].get("consolidated")]
        skipped_xs = [x for h, x in zip(rebalance_entries, xs) if not h["meta"].get("admitted")]

        ax2.scatter(admitted_xs, [1.5] * len(admitted_xs), marker="o", color="#2E7D32",
                    s=60, zorder=4, label=f"Admitted ({len(admitted_xs)})")
        ax2.scatter(consolidated_xs, [1.0] * len(consolidated_xs), marker="^", color="#1565C0",
                    s=60, zorder=4, label=f"Consolidated ({len(consolidated_xs)})")
        ax2.scatter(skipped_xs, [0.5] * len(skipped_xs), marker="x", color="#B71C1C",
                    s=50, zorder=4, label=f"Skipped / not admitted ({len(skipped_xs)})")

        # Overlay system selection band if adaptive routing was used
        for h, x in zip(rebalance_entries, xs):
            system = (h["meta"].get("adaptive") or {}).get("system")
            if system:
                ax2.axvspan(x - 0.4, x + 0.4, alpha=0.15,
                            color=SYSTEM_COLORS.get(system, "#999999"), zorder=1)

    ax2.set_yticks([0.5, 1.0, 1.5])
    ax2.set_yticklabels(["Skipped", "Consolidated", "Admitted"], fontsize=7)
    ax2.set_xlim(0, len(dates))
    ax2.set_xticks([])
    ax2.legend(loc="upper right", fontsize=7, ncol=3)
    ax2.set_title("Consolidation Decisions per Rebalance (AdaReMo-style gate)", fontsize=10)
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: Memory size over time ────────────────────────────────────────
    ax3 = axes[2]
    if rebalance_entries:
        mem_xs = [dates.index(h["date"]) if h["date"] in dates else 0 for h in rebalance_entries]
        mem_sizes = [h["meta"].get("memory_size", 0) for h in rebalance_entries]
        ax3.step(mem_xs, mem_sizes, where="post", color="#4A148C", linewidth=2,
                 label="Playbook bullets (memory size)")
        ax3.fill_between(mem_xs, mem_sizes, alpha=0.12, color="#4A148C", step="post")

    ax3.set_ylabel("Bullets in Playbook")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.set_xlim(0, len(dates))
    tick_step = max(1, len(dates) // 12)
    tick_positions = list(range(0, len(dates), tick_step))
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([dates[i] for i in tick_positions], rotation=30, fontsize=7, ha="right")
    ax3.set_title("Memory Growth (saturates when bullets plateau)", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Summary stats matching Tables 8/9 in the paper
    if rebalance_entries:
        total_rebal = len(rebalance_entries)
        n_admitted = sum(1 for h in rebalance_entries if h["meta"].get("admitted"))
        n_consol = sum(1 for h in rebalance_entries if h["meta"].get("consolidated"))
        final_mem = max((h["meta"].get("memory_size", 0) for h in rebalance_entries), default=0)
        pct_admitted = 100 * n_admitted / max(total_rebal, 1)
        pct_consol = 100 * n_consol / max(total_rebal, 1)
        summary = (
            f"Rebalances:      {total_rebal}\n"
            f"Admitted:        {n_admitted} ({pct_admitted:.0f}%)\n"
            f"Consolidated:    {n_consol} ({pct_consol:.0f}%)\n"
            f"Final memory:    {final_mem} bullets"
        )
        # Add system selection breakdown if adaptive routing
        sel_counts = {}
        for h in rebalance_entries:
            s = (h["meta"].get("adaptive") or {}).get("system")
            if s:
                sel_counts[s] = sel_counts.get(s, 0) + 1
        if sel_counts:
            summary += "\n--- routing ---"
            for s, c in sorted(sel_counts.items(), key=lambda x: -x[1]):
                summary += f"\n{SYSTEM_LABELS.get(s, s):10s}: {c}/{total_rebal}"
        fig.text(0.82, 0.02, summary, fontsize=7.5, family="monospace",
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
