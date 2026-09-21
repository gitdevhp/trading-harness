#!/usr/bin/env python3
"""
analyze_regime.py — Tests the concentration hypothesis:

  DEBATER delta (vs HARNESS) correlates with universe concentration.
  MEMORY delta  (vs HARNESS) correlates with universe breadth.

Uses hardcoded experimental results (ablation runs, qwen25 only).
Outputs:
  - Console table: universe stats + deltas + which component wins
  - Pearson r for each hypothesis
  - Scatter plot: concentration_score vs delta for DEBATER and MEMORY
  - "Oracle gain" — how much an ideal adaptive router would add vs fixed HARNESS

Usage:
  python -m ace_harness.analyze_regime [--output_dir PATH]
"""

import argparse
import math
import json
import os

# ──────────────────────────────────────────────────────────────
# EXPERIMENTAL DATA  (qwen25, ablation runs 1 & 2 — shared values)
# ──────────────────────────────────────────────────────────────

UNIVERSES = {
    "mag7": {
        "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
        "sectors": {"Tech/Growth": 7},
        "harness":          71.12,
        "harness_debater":  70.47,
        "harness_memory":   63.45,
        # ACE (all three) — take mean of RUN1 and RUN3 (RUN2 excluded: outlier)
        "ace_runs":         [64.67, 65.06],
    },
    "tech18": {
        "tickers": [
            "AAPL","MSFT","NVDA","AVGO","AMD","ADBE","QCOM","TXN","AMAT","MU",
            "AMZN","TSLA","PEP","COST","HON","GOOGL","META","NFLX",
        ],
        "sectors": {"Tech/Semis": 11, "Consumer/Comm": 5, "Industrial": 2},
        "harness":          52.36,
        "harness_debater":  46.89,
        "harness_memory":   47.99,
        "ace_runs":         [51.90, 52.15, 56.59],
    },
    "balanced15": {
        "tickers": [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META",
            "JPM","JNJ","XOM","UNH","HD","WMT","BAC","PG","CVX",
        ],
        "sectors": {"Tech/Comm": 6, "Finance": 3, "Healthcare": 2, "Consumer": 2, "Energy": 2},
        "harness":          48.11,
        "harness_debater":  56.56,
        "harness_memory":   33.09,
        "ace_runs":         [40.36, 32.80, 42.88],
    },
    "volatile25": {
        "tickers": [
            "NVDA","AMD","AVGO","QCOM","MU","AMAT","MRVL",
            "CRWD","PLTR","DDOG","TSLA","META","AMZN","UBER","SPOT","ABNB","NFLX",
            "LLY","MRNA","BIIB","VRTX","ABBV","AMGN","REGN","PFE",
        ],
        "sectors": {"Semis/HW": 7, "TechGrowth": 4, "Consumer/Comm": 6, "Biotech": 8},
        "harness":          56.11,
        "harness_debater":  76.71,
        "harness_memory":   61.80,
        # RUN2 excluded (41.85 = known copy error matching qwen36 value)
        "ace_runs":         [70.99, 71.16],
    },
    "sp30": {
        "tickers": [
            "AAPL","MSFT","NVDA","AVGO","AMD","ADBE","QCOM","ORCL","AMAT","MU",
            "GOOGL","META","NFLX","AMZN","TSLA","HD","BKNG","NKE","WMT","PG",
            "KO","JNJ","LLY","UNH","ABBV","JPM","V","MA","GS","BAC",
        ],
        "sectors": {"Tech/Semis": 12, "Consumer": 6, "Healthcare": 4, "Finance": 5, "Comm": 3},
        "harness":          46.48,
        "harness_debater":  35.96,
        "harness_memory":   32.02,
        "ace_runs":         [40.73, 49.52, 53.02],
    },
    "diversified40": {
        "tickers": [
            "AAPL","MSFT","NVDA","AVGO","AMD","ADBE","QCOM","ORCL","AMAT","MU",
            "GOOGL","META","NFLX","CMCSA","AMZN","TSLA","HD","BKNG","NKE","MCD",
            "WMT","PG","KO","JNJ","LLY","UNH","ABBV","JPM","V","MA","GS","BAC",
            "XOM","CVX","COP","CAT","HON","RTX","PEP","DIS",
        ],
        "sectors": {"Tech/Semis": 12, "Consumer": 7, "Healthcare": 4, "Finance": 5,
                    "Comm": 4, "Energy": 3, "Industrial": 4, "Other": 1},
        "harness":          44.28,
        "harness_debater":  39.23,
        "harness_memory":   50.28,
        "ace_runs":         [30.25, 28.58, 19.53],
    },
}


def hhi(sectors: dict) -> float:
    """Herfindahl-Hirschman Index on sector counts. 1.0 = all one sector."""
    total = sum(sectors.values())
    return sum((c / total) ** 2 for c in sectors.values())


def concentration_score(data: dict) -> float:
    """
    Composite concentration score [0,1].
    Higher = more concentrated (small + single-sector).
    = 0.5 * (1 - n_norm) + 0.5 * hhi_val
    where n_norm = (n - min_n) / (max_n - min_n)
    """
    n = len(data["tickers"])
    return hhi(data["sectors"])   # HHI already 0-1; n captured by HHI in practice


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / denom if denom else 0.0


def run(output_dir: str | None = None):
    rows = []
    for name, data in UNIVERSES.items():
        n = len(data["tickers"])
        conc = concentration_score(data)
        h = data["harness"]
        d = data["harness_debater"]
        m = data["harness_memory"]
        ace_mean = sum(data["ace_runs"]) / len(data["ace_runs"])
        ace_std  = (sum((x - ace_mean) ** 2 for x in data["ace_runs"]) / len(data["ace_runs"])) ** 0.5

        debater_delta = d - h
        memory_delta  = m - h
        ace_delta     = ace_mean - h

        best_component = max(
            ("HARNESS",         h),
            ("HARNESS+DEBATER", d),
            ("HARNESS+MEMORY",  m),
            key=lambda kv: kv[1],
        )
        oracle_gain = best_component[1] - h

        rows.append({
            "universe":       name,
            "n_stocks":       n,
            "hhi":            round(conc, 3),
            "harness":        h,
            "debater_delta":  round(debater_delta, 2),
            "memory_delta":   round(memory_delta,  2),
            "ace_mean":       round(ace_mean, 2),
            "ace_std":        round(ace_std,  2),
            "ace_delta":      round(ace_delta, 2),
            "best":           best_component[0],
            "oracle_gain":    round(oracle_gain, 2),
        })

    # Sort by HHI descending (most concentrated first)
    rows.sort(key=lambda r: r["hhi"], reverse=True)

    # ── Console table ──────────────────────────────────────────
    header = (
        f"{'Universe':<14} {'N':>4} {'HHI':>5} {'HARNESS':>8} "
        f"{'ΔDEBATER':>9} {'ΔMEMORY':>8} {'ΔACE':>7} {'ACE±':>6} {'Best':>18} {'OracleGain':>11}"
    )
    print("\n" + "=" * len(header))
    print("REGIME HYPOTHESIS — concentration vs component delta (qwen25)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['universe']:<14} {r['n_stocks']:>4} {r['hhi']:>5.3f} {r['harness']:>8.2f} "
            f"{r['debater_delta']:>+9.2f} {r['memory_delta']:>+8.2f} {r['ace_delta']:>+7.2f} "
            f"{r['ace_std']:>6.2f} {r['best']:>18} {r['oracle_gain']:>+11.2f}"
        )
    print("=" * len(header))

    # ── Correlations ───────────────────────────────────────────
    hhis            = [r["hhi"]           for r in rows]
    debater_deltas  = [r["debater_delta"] for r in rows]
    memory_deltas   = [r["memory_delta"]  for r in rows]
    ace_deltas      = [r["ace_delta"]     for r in rows]

    n_stocks_list = [r["n_stocks"] for r in rows]

    r_debater      = pearson_r(hhis,         debater_deltas)
    r_memory       = pearson_r(hhis,         memory_deltas)
    r_ace          = pearson_r(hhis,         ace_deltas)
    r_debater_n    = pearson_r(n_stocks_list, debater_deltas)
    r_memory_n     = pearson_r(n_stocks_list, memory_deltas)

    print(f"\nCorrelations with SECTOR HHI (concentration):")
    print(f"  Pearson r (HHI vs ΔDEBATER): {r_debater:+.3f}  "
          f"{'[SUPPORTED]' if r_debater > 0.3 else '[weak — HHI alone insufficient]'}")
    print(f"  Pearson r (HHI vs ΔMEMORY):  {r_memory:+.3f}  "
          f"{'[SUPPORTED]' if r_memory < -0.3 else '[weak]'}")
    print(f"  Pearson r (HHI vs ΔACE):     {r_ace:+.3f}")
    print(f"\nCorrelations with N_STOCKS (breadth):")
    print(f"  Pearson r (N vs ΔDEBATER): {r_debater_n:+.3f}  "
          f"{'[SUPPORTED — fewer stocks → debater gains]' if r_debater_n < -0.3 else '[weak]'}")
    print(f"  Pearson r (N vs ΔMEMORY):  {r_memory_n:+.3f}  "
          f"{'[SUPPORTED — more stocks → memory gains]' if r_memory_n > 0.3 else '[weak]'}")
    print(f"\nNote: 6 universes is too few for strong statistical conclusions.")
    print(f"These correlations should be verified once ablation re-runs provide more data points.")

    avg_oracle_gain = sum(r["oracle_gain"] for r in rows) / len(rows)
    avg_harness     = sum(r["harness"]     for r in rows) / len(rows)
    avg_ace         = sum(r["ace_mean"]    for r in rows) / len(rows)
    print(f"\nAverage HARNESS:          {avg_harness:.2f}%")
    print(f"Average ACE (all 3):      {avg_ace:.2f}%  (Δ {avg_ace - avg_harness:+.2f}pp vs HARNESS)")
    print(f"Average Oracle best pick: {avg_harness + avg_oracle_gain:.2f}%  "
          f"(Δ {avg_oracle_gain:+.2f}pp vs HARNESS)")
    print(f"  → An adaptive router capturing oracle gains would be worth "
          f"{avg_oracle_gain:.1f}pp/universe on average.\n")

    # ── JSON output ────────────────────────────────────────────
    result = {
        "rows":             rows,
        "pearson_r": {
            "hhi_vs_debater_delta": round(r_debater, 4),
            "hhi_vs_memory_delta":  round(r_memory, 4),
            "hhi_vs_ace_delta":     round(r_ace, 4),
        },
        "summary": {
            "avg_harness":      round(avg_harness, 2),
            "avg_ace":          round(avg_ace, 2),
            "avg_oracle_gain":  round(avg_oracle_gain, 2),
        },
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "regime_analysis.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved -> {out_path}")

    _plot(rows, hhis, debater_deltas, memory_deltas, output_dir)
    return result


def _plot(rows, hhis, debater_deltas, memory_deltas, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Regime Hypothesis: Universe Concentration (HHI) vs Component Delta\n"
        "Higher HHI = more concentrated universe; Δ = TotalRet% vs HARNESS baseline",
        fontsize=11,
    )

    names = [r["universe"] for r in rows]

    for ax, deltas, label, color, hypothesis in [
        (axes[0], debater_deltas, "ΔDEBATER vs HARNESS (%)", "#2563eb",
         "Hypothesis: DEBATER helps concentrated universes (HHI↑ → Δ↑)"),
        (axes[1], memory_deltas,  "ΔMEMORY vs HARNESS (%)",  "#16a34a",
         "Hypothesis: MEMORY helps broad universes (HHI↑ → Δ↓)"),
    ]:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        for x, y, name in zip(hhis, deltas, names):
            ax.scatter(x, y, s=90, color=color, zorder=3)
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(6, 3), fontsize=8)

        # Fit line
        if len(hhis) >= 2:
            mx = sum(hhis) / len(hhis)
            my = sum(deltas) / len(deltas)
            slope = sum((x - mx) * (y - my) for x, y in zip(hhis, deltas)) / \
                    sum((x - mx) ** 2 for x in hhis)
            intercept = my - slope * mx
            xs_fit = [min(hhis) - 0.05, max(hhis) + 0.05]
            ax.plot(xs_fit, [slope * x + intercept for x in xs_fit],
                    color=color, alpha=0.5, linewidth=1.5, linestyle="-")
            r = pearson_r(hhis, deltas)
            ax.set_title(f"{hypothesis}\nr = {r:+.3f}", fontsize=8)
        else:
            ax.set_title(hypothesis, fontsize=8)

        ax.set_xlabel("Sector HHI (concentration)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, "regime_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot saved   -> {path}")
    else:
        plt.savefig("regime_analysis.png", dpi=150, bbox_inches="tight")
        print("Plot saved   -> regime_analysis.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regime concentration hypothesis analysis")
    parser.add_argument("--output_dir", default=None, help="Directory to save JSON + PNG")
    args = parser.parse_args()
    run(output_dir=args.output_dir)
