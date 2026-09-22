"""
Visualize AutoGovern self-improvement behaviour matching the paper's metrics.

Four-panel figure following Tables 8/9 in the ReMo paper:

  Panel 1 — Cumulative portfolio returns (AutoGovern vs. baselines).
  Panel 2 — Refinement rounds used per rebalance (1/2/3) — AutoGovern's
               g_ref signal terminates early when no actionable fix remains,
               so short-round months directly show where token budget was saved.
  Panel 3 — Consolidation decisions per rebalance:
               admitted / consolidated / skipped — matching the paper's
               "consolidator calls per task" and "admitted" fraction metrics.
  Panel 4 — Memory size over time (bullets in playbook) — matching the
               paper's final memory size metric.

Backwards-compatible: if meta.adaptive.system exists (old routing run),
overlays the per-period system selection as a color band on Panel 3.

Usage:
    python -m ace_harness.plot_adaptive \
        --adaptive_json ./results/monthly_adaptive_autogover_results.json \
        --baseline_jsons ./results/monthly_intra_convictionriskharness_results.json \
                         ./results/monthly_memory_only_convictionriskharness_adaptive_results.json \
                         ./results/monthly_dual_permanent_convictionriskharness_adaptive_results.json \
        --output_png ./results/autogover_behaviour.png
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

    if not dates:
        print("No history found in adaptive JSON — is the backtest complete?")
        return

    base = values[0]
    cum_rets = [v / base - 1.0 for v in values]

    rebalance_entries = [h for h in (data if isinstance(data, list) else data.get("history", []))
                         if h.get("meta")]

    # Detect whether this is an AutoGovern run (has rounds list) or old routing run
    is_autogover = any("rounds" in (h.get("meta") or {}) for h in rebalance_entries)
    n_panels = 4 if is_autogover else 3
    height_ratios = [3, 1, 1, 1.5] if is_autogover else [3, 1, 1.5]

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 13 if is_autogover else 11),
                             gridspec_kw={"height_ratios": height_ratios})
    title = "AutoGovern (AdaReMo Alg 2) — Budget-Governed Self-Improvement" if is_autogover \
            else "Adaptive System Routing — Portfolio Behaviour"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Panel 1: Equity curves ───────────────────────────────────────────────
    ax1 = axes[0]
    curve_label = "AutoGovern" if is_autogover else "Adaptive (routing)"
    ax1.plot(dates, [r * 100 for r in cum_rets], color="#1A237E", linewidth=2,
             label=curve_label, zorder=5)

    overlay_colors = ["#E53935", "#6A1B9A", "#F57F17", "#2E7D32"]
    overlay_labels = ["DEBATER", "MEMORY", "ACE / dual_permanent", "HARNESS"]
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

    panel_idx = 1

    # ── Panel 2 (AutoGovern only): Refinement rounds used per rebalance ──────
    if is_autogover:
        ax_rounds = axes[panel_idx]
        panel_idx += 1
        if rebalance_entries:
            xs = [dates.index(h["date"]) if h["date"] in dates else 0 for h in rebalance_entries]
            round_counts = [len((h["meta"].get("rounds") or [])) for h in rebalance_entries]
            # bar color by how many rounds: 1=green (budget saved), 2=amber, 3=red (full budget)
            bar_colors = {1: "#2E7D32", 2: "#F57F17", 3: "#B71C1C"}
            for x, rc in zip(xs, round_counts):
                ax_rounds.bar(x, rc, color=bar_colors.get(rc, "#607D8B"), width=0.6,
                              alpha=0.8, zorder=3)
            # Legend patches
            patches = [mpatches.Patch(color=bar_colors[k], label=f"{k} round{'s' if k > 1 else ''}")
                       for k in sorted(bar_colors)]
            avg_rounds = sum(round_counts) / max(len(round_counts), 1)
            ax_rounds.axhline(avg_rounds, color="black", linewidth=1, linestyle="--",
                              label=f"avg {avg_rounds:.2f} rounds")
            patches.append(mpatches.Patch(color="black", label=f"avg {avg_rounds:.2f} rounds"))
            ax_rounds.set_yticks([1, 2, 3])
            ax_rounds.set_yticklabels(["1", "2", "3 (max)"], fontsize=7)
            ax_rounds.set_ylim(0, 3.5)
            ax_rounds.set_xlim(0, len(dates))
            ax_rounds.set_xticks([])
            ax_rounds.legend(handles=patches, loc="upper right", fontsize=7, ncol=4)
            ax_rounds.set_title(
                "Refinement Rounds per Rebalance (g_ref early-exit — fewer rounds = token budget saved)",
                fontsize=10)
            ax_rounds.grid(True, alpha=0.2, axis="y")

    # ── Panel 3: Consolidation decisions ─────────────────────────────────────
    ax2 = axes[panel_idx]
    panel_idx += 1
    if rebalance_entries:
        xs = [dates.index(h["date"]) if h["date"] in dates else 0 for h in rebalance_entries]
        admitted_xs   = [x for h, x in zip(rebalance_entries, xs) if h["meta"].get("admitted")]
        consol_xs     = [x for h, x in zip(rebalance_entries, xs) if h["meta"].get("consolidated")]
        skipped_xs    = [x for h, x in zip(rebalance_entries, xs) if not h["meta"].get("admitted")]

        ax2.scatter(admitted_xs, [1.5] * len(admitted_xs), marker="o", color="#2E7D32",
                    s=60, zorder=4, label=f"Admitted ({len(admitted_xs)})")
        ax2.scatter(consol_xs, [1.0] * len(consol_xs), marker="^", color="#1565C0",
                    s=60, zorder=4, label=f"Consolidated ({len(consol_xs)})")
        ax2.scatter(skipped_xs, [0.5] * len(skipped_xs), marker="x", color="#B71C1C",
                    s=50, zorder=4, label=f"Not admitted ({len(skipped_xs)})")

        # Old routing run: overlay system selection band
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
    ax2.set_title("Consolidation Gate per Rebalance (admitted ∧ g_sto ∧ ¬frozen)", fontsize=10)
    ax2.grid(True, alpha=0.2)

    # ── Panel 4: Memory size over time ────────────────────────────────────────
    ax3 = axes[panel_idx]
    if rebalance_entries:
        mem_xs = [dates.index(h["date"]) if h["date"] in dates else 0 for h in rebalance_entries]
        mem_sizes = [h["meta"].get("memory_size", 0) for h in rebalance_entries]
        ax3.step(mem_xs, mem_sizes, where="post", color="#4A148C", linewidth=2,
                 label="Playbook bullets")
        ax3.fill_between(mem_xs, mem_sizes, alpha=0.12, color="#4A148C", step="post")
        # Mark where frozen kicks in (memory flatlines)
        if len(mem_sizes) >= 2:
            freeze_x = next((mem_xs[i] for i in range(1, len(mem_sizes))
                             if mem_sizes[i] == mem_sizes[i - 1] and
                             all(mem_sizes[j] == mem_sizes[i] for j in range(i, len(mem_sizes)))),
                            None)
            if freeze_x is not None:
                ax3.axvline(freeze_x, color="#B71C1C", linewidth=1.2, linestyle="--",
                            label="frozen (SATURATED)")

    ax3.set_ylabel("Bullets in Playbook")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.set_xlim(0, len(dates))
    tick_step = max(1, len(dates) // 12)
    tick_positions = list(range(0, len(dates), tick_step))
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([dates[i] for i in tick_positions], rotation=30, fontsize=7, ha="right")
    ax3.set_title("Memory Growth (red dashed = SATURATED freeze point)", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Summary box
    if rebalance_entries:
        total_rebal = len(rebalance_entries)
        n_admitted = sum(1 for h in rebalance_entries if h["meta"].get("admitted"))
        n_consol   = sum(1 for h in rebalance_entries if h["meta"].get("consolidated"))
        final_mem  = max((h["meta"].get("memory_size", 0) for h in rebalance_entries), default=0)
        pct_admitted = 100 * n_admitted / max(total_rebal, 1)
        pct_consol   = 100 * n_consol   / max(total_rebal, 1)
        summary = (
            f"Rebalances:   {total_rebal}\n"
            f"Admitted:     {n_admitted} ({pct_admitted:.0f}%)\n"
            f"Consolidated: {n_consol} ({pct_consol:.0f}%)\n"
            f"Final memory: {final_mem} bullets"
        )
        if is_autogover:
            round_counts = [len((h["meta"].get("rounds") or [])) for h in rebalance_entries]
            from collections import Counter
            rc = Counter(round_counts)
            avg_r = sum(round_counts) / max(len(round_counts), 1)
            summary += f"\n--- rounds ---\navg: {avg_r:.2f}"
            for k in sorted(rc):
                summary += f"\n{k} round{'s' if k>1 else ''}: {rc[k]}/{total_rebal}"
        else:
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
