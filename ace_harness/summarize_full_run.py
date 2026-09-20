"""
Aggregate metrics from a full comparative run across models and universes.

Reads:
  REACT_DIR/results/
    qwen_raw_{model_tag}_{universe_tag}.json            -> QWEN
    react_no_harness_{model_tag}_{universe_tag}.json    -> QWEN+REACT
    react_gpt_harness_{model_tag}_{universe_tag}.json   -> HARNESS
  ACE_BASE/qwen{model_tag}_{universe_tag}/
    monthly_intra_*_results.json                        -> HARNESS+DEBATER
    monthly_memory_only_*_results.json                  -> HARNESS+MEMORY
    monthly_dual_permanent_*_results.json               -> HARNESS+MEMORY+DEBATER

Baselines (Equal-Weight, Risk-Parity, 60/40, Min-Variance, Cov-Risk-Parity,
Black-Litterman) are computed once per universe from the embedded prices in
the ACE result files — they do not depend on the model.

Outputs:
    OUTPUT_DIR/full_comparison.json   all rows as JSON
    OUTPUT_DIR/full_comparison.txt    ASCII table grouped by universe
"""
import argparse
import glob
import json
import os

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

TRADING_DAYS = 252
INITIAL_CAPITAL = 1_000_000.0
FEE_RATE = 0.0015

# File stem → display label for ReAct result files
REACT_STEMS = {
    "qwen_raw":           "QWEN",
    "react_no_harness":   "QWEN+REACT",
    "react_gpt_harness":  "HARNESS",
}

# Glob pattern → display label for ACE result files
# More-specific patterns must come before less-specific ones so a file that
# matches both (e.g. dual_permanent_adamo vs dual_permanent) is captured by
# the right label first.
ACE_GLOBS = {
    "monthly_baseline_adamo_*_results.json":       "HARNESS+ADAMO",
    "monthly_baseline_remo_*_results.json":        "HARNESS+REMO",
    "monthly_intra_*_results.json":                "HARNESS+DEBATER",
    "monthly_memory_only_*_results.json":          "HARNESS+MEMORY",
    "monthly_dual_permanent_adamo_*_results.json": "HARNESS+MEM+DEB+ADAMO",
    "monthly_dual_permanent_*_results.json":       "HARNESS+MEMORY+DEBATER",
}

AI_ORDER = [
    "QWEN", "QWEN+REACT", "HARNESS",
    "HARNESS+REMO", "HARNESS+ADAMO",
    "HARNESS+DEBATER", "HARNESS+MEMORY", "HARNESS+MEMORY+DEBATER",
    "HARNESS+MEM+DEB+ADAMO",
]
BASELINE_ORDER = [
    "Equal-Weight", "Risk-Parity", "60/40",
    "Min-Variance", "Cov-Risk-Parity", "Black-Litterman",
]
DISPLAY_ORDER = AI_ORDER + BASELINE_ORDER


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _dr(v):
    return np.diff(v) / v[:-1]

def _cagr(v):
    n = len(v) - 1
    if n <= 0 or v[0] <= 0:
        return 0.0
    return float((v[-1] / v[0]) ** (TRADING_DAYS / n) - 1) * 100

def _sharpe(r):
    s = np.std(r)
    return float(np.mean(r) * TRADING_DAYS / (s * np.sqrt(TRADING_DAYS) + 1e-9))

def _sortino(r):
    neg = r[r < 0]
    d = np.sqrt(np.mean(neg ** 2)) if len(neg) else 1e-9
    return float(np.mean(r) * TRADING_DAYS / (d * np.sqrt(TRADING_DAYS) + 1e-9))

def _mdd(v):
    peak = np.maximum.accumulate(v)
    return float(((v - peak) / peak).min() * 100)

def _calmar(v):
    return float(_cagr(v) / (abs(_mdd(v)) + 1e-9))

def _alpha_beta(pr, br):
    if len(pr) < 2 or len(br) < 2:
        return 0.0, 1.0
    cov = np.cov(pr, br)
    beta = float(cov[0, 1] / (cov[1, 1] + 1e-12))
    alpha = float((np.mean(pr) - beta * np.mean(br)) * TRADING_DAYS * 100)
    return alpha, beta

def compute_metrics(values, dates, bench_nav=None):
    values = np.array(values, dtype=float)
    if len(values) < 2:
        return None
    r = _dr(values)
    alpha, beta = 0.0, 1.0
    if bench_nav is not None and len(bench_nav) == len(values):
        br = _dr(np.array(bench_nav, dtype=float))
        alpha, beta = _alpha_beta(r, br)
    return {
        "start": dates[0],
        "end": dates[-1],
        "final_value": round(float(values[-1]), 2),
        "total_return_pct": round(float((values[-1] / values[0] - 1) * 100), 2),
        "cagr_pct": round(_cagr(values), 2),
        "ann_vol_pct": round(float(np.std(r) * np.sqrt(TRADING_DAYS) * 100), 2),
        "sharpe": round(_sharpe(r), 3),
        "sortino": round(_sortino(r), 3),
        "calmar": round(_calmar(values), 3),
        "max_drawdown_pct": round(_mdd(values), 2),
        "alpha_ann_pct": round(alpha, 2),
        "beta": round(beta, 3),
    }


# ---------------------------------------------------------------------------
# Price DataFrame from embedded result-file prices
# ---------------------------------------------------------------------------

def build_price_df(ace_dir):
    if not _HAS_PANDAS:
        return None
    for path in sorted(glob.glob(os.path.join(ace_dir, "*_results.json"))):
        try:
            with open(path) as f:
                records = json.load(f)
            price_recs = {}
            for r in records:
                p = r.get("prices")
                if isinstance(p, dict) and p:
                    price_recs[r["date"]] = p
            if len(price_recs) < 5:
                continue
            dates = sorted(price_recs.keys())
            counts = {}
            for day in price_recs.values():
                for t in day:
                    counts[t] = counts.get(t, 0) + 1
            threshold = 0.80 * len(dates)
            tickers = sorted(t for t, c in counts.items() if c >= threshold)
            if not tickers:
                continue
            data = {t: [price_recs[d].get(t, float("nan")) for d in dates] for t in tickers}
            df = pd.DataFrame(data, index=dates).dropna()
            if len(df) > 5:
                return df
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Baseline NAV curves  (one monthly rebalance per calendar month)
# ---------------------------------------------------------------------------

def _rebal_set(df):
    ts = pd.DatetimeIndex(list(df.index))
    dates, prev = [], None
    for orig, t in zip(df.index, ts):
        m = (t.year, t.month)
        if m != prev:
            dates.append(orig)
            prev = m
    return set(dates)

def _run(df, weight_fn):
    n = df.shape[1]
    rebal = _rebal_set(df)
    shares, cash = np.zeros(n), INITIAL_CAPITAL  # deploy capital on first rebalance day
    nav = [INITIAL_CAPITAL]
    for i, d in enumerate(df.index):
        prices = df.iloc[i].values
        if np.any(prices <= 0):
            nav.append(nav[-1])
            continue
        value = float(np.dot(shares, prices) + cash)
        if d in rebal:
            w = weight_fn(df.iloc[:i + 1], i)
            w = np.clip(w, 0, None)
            s = w.sum()
            w = w / s if s > 0 else np.ones(n) / n
            turnover = float(np.sum(np.abs(w * value - shares * prices)) / (value + 1e-9))
            value -= turnover * FEE_RATE * value
            shares = (w * value) / prices
            cash = 0.0
        nav.append(float(np.dot(shares, prices) + cash))
    return np.array(nav[1:])

def _ew(df):
    return _run(df, lambda sub, i: np.ones(sub.shape[1]) / sub.shape[1])

def _rp(df, lb=63):
    def w(sub, i):
        n = sub.shape[1]
        win = sub.iloc[max(0, i - lb):i + 1]
        if len(win) < 5:
            return np.ones(n) / n
        vol = np.std(win.pct_change().dropna().values, axis=0)
        vol = np.where(vol < 1e-8, 1e-8, vol)
        iv = 1 / vol
        return iv / iv.sum()
    return _run(df, w)

def _6040(df, bond_ann=0.045):
    if not _HAS_PANDAS:
        return None
    bond_d = (1 + bond_ann) ** (1 / TRADING_DAYS) - 1
    n = df.shape[1]
    rebal = _rebal_set(df)
    val0 = INITIAL_CAPITAL * (1 - FEE_RATE * 0.60)
    shares = (0.60 / n * val0) / df.iloc[0].values
    bond_val = 0.40 * val0
    nav = []
    for i, d in enumerate(df.index):
        prices = df.iloc[i].values
        bond_val *= (1 + bond_d)
        value = float(np.dot(shares, prices) + bond_val)
        if d in rebal and i > 0:
            turnover = float(np.sum(np.abs(0.60 * value / n - shares * prices)) / (value + 1e-9))
            value -= turnover * FEE_RATE * value
            shares = (0.60 * value / n) / prices
            bond_val = 0.40 * value
        nav.append(float(np.dot(shares, prices) + bond_val))
    return np.array(nav)

def _mv(df, lb=126):
    if not _HAS_SCIPY:
        return None
    def w(sub, i):
        n = sub.shape[1]
        win = sub.iloc[max(0, i - lb):i + 1]
        if len(win) < 10:
            return np.ones(n) / n
        ret = win.pct_change().dropna().values
        cov = np.cov(ret.T) + np.eye(n) * 1e-8
        w0 = np.ones(n) / n
        res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                       bounds=[(0, 1)] * n,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                       options={"ftol": 1e-9, "maxiter": 200})
        return res.x if res.success else w0
    return _run(df, w)

def _crp(df, lb=126):
    if not _HAS_SCIPY:
        return None
    def w(sub, i):
        n = sub.shape[1]
        win = sub.iloc[max(0, i - lb):i + 1]
        if len(win) < 10:
            return np.ones(n) / n
        ret = win.pct_change().dropna().values
        cov = np.cov(ret.T) + np.eye(n) * 1e-8
        target = 1.0 / n
        def obj(w):
            pv = w @ cov @ w
            rc = w * (cov @ w) / (pv + 1e-12)
            return float(np.sum((rc - target) ** 2))
        w0 = np.ones(n) / n
        res = minimize(obj, w0, method="SLSQP", bounds=[(1e-4, 1)] * n,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                       options={"ftol": 1e-12, "maxiter": 500})
        return res.x if res.success else w0
    return _run(df, w)

def _bl(df, lb=126, tau=0.025, ra=2.5):
    if not _HAS_SCIPY:
        return None
    def w(sub, i):
        n = sub.shape[1]
        win = sub.iloc[max(0, i - lb):i + 1]
        if len(win) < 21:
            return np.ones(n) / n
        ret = win.pct_change().dropna().values
        cov = np.cov(ret.T) + np.eye(n) * 1e-8
        pi = ra * cov @ (np.ones(n) / n)
        recent = win.iloc[-21:]
        Q = ((recent.iloc[-1] / recent.iloc[0]) - 1).values
        P, Omega = np.eye(n), tau * np.diag(np.diag(cov))
        tc = tau * cov
        M = np.linalg.inv(np.linalg.inv(tc) + P.T @ np.linalg.inv(Omega) @ P)
        mu_bl = M @ (np.linalg.inv(tc) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
        def neg_sh(w):
            return -(w @ mu_bl) / (np.sqrt(w @ cov @ w + 1e-12) + 1e-9)
        w0 = np.ones(n) / n
        res = minimize(neg_sh, w0, method="SLSQP", bounds=[(0, 0.30)] * n,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                       options={"ftol": 1e-9, "maxiter": 300})
        return res.x if res.success else w0
    return _run(df, w)

def compute_baselines(price_df):
    curves = {}
    for name, fn in [
        ("Equal-Weight",    _ew),
        ("Risk-Parity",     _rp),
        ("60/40",           _6040),
        ("Min-Variance",    _mv),
        ("Cov-Risk-Parity", _crp),
        ("Black-Litterman", _bl),
    ]:
        try:
            nav = fn(price_df)
            if nav is not None:
                curves[name] = nav
        except Exception as e:
            print(f"    [{name}] failed: {e}")
    return curves


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_records(path):
    with open(path) as f:
        d = json.load(f)
    return d if isinstance(d, list) else d.get("daily_nav", d.get("results", []))

def find_ace_file(ace_dir, pattern):
    matches = sorted(glob.glob(os.path.join(ace_dir, pattern)))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Table writer
# ---------------------------------------------------------------------------

COLS = [
    ("system",           "System",     "<", 28),
    ("total_return_pct", "TotalRet%",  ">", 11),
    ("cagr_pct",         "CAGR%",      ">",  8),
    ("ann_vol_pct",      "AnnVol%",    ">",  9),
    ("max_drawdown_pct", "MaxDD%",     ">",  8),
    ("sharpe",           "Sharpe",     ">",  8),
    ("sortino",          "Sortino",    ">",  9),
    ("calmar",           "Calmar",     ">",  8),
    ("alpha_ann_pct",    "Alpha%",     ">",  8),
    ("beta",             "Beta",       ">",  7),
]

def _fmt(row):
    line = ""
    for key, _, align, w in COLS:
        val = row.get(key, "")
        line += f"{val:{align}{w}.2f}" if isinstance(val, float) else f"{val:{align}{w}}"
    return line

def write_table(all_rows, path, universe_tags, model_tags):
    header = "".join(f"{h:{a}{w}}" for _, h, a, w in COLS)
    sep = "-" * len(header)

    idx = {}
    for r in all_rows:
        idx[(r["model"], r["universe"], r["system"])] = r

    lines = []
    for ut in universe_tags:
        lines += [f"\n{'=' * len(header)}", f"Universe: {ut}", "=" * len(header)]

        # AI systems (one section per model)
        for mt in model_tags:
            lines += [f"\n  -- qwen{mt} --", "  " + sep, "  " + header, "  " + sep]
            for sys in AI_ORDER:
                row = idx.get((f"qwen{mt}", ut, sys))
                lines.append("  " + (_fmt(row) if row else f"{sys:<28}  (missing)"))
            lines.append("  " + sep)

        # Baselines (model-independent — shown once per universe)
        lines += ["\n  -- Baselines (model-independent) --", "  " + sep, "  " + header, "  " + sep]
        for sys in BASELINE_ORDER:
            row = idx.get(("baseline", ut, sys))
            lines.append("  " + (_fmt(row) if row else f"{sys:<28}  (missing)"))
        lines.append("  " + sep)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Summarize full comparative run")
    parser.add_argument("--react_dir",     required=True, help="Path to ReAct/results/")
    parser.add_argument("--ace_base",      required=True, help="Path to ace_results/ base directory")
    parser.add_argument("--output_dir",    default=".",   help="Where to write outputs")
    parser.add_argument("--model_tags",    nargs="+", default=["25", "36"])
    parser.add_argument("--universe_tags", nargs="+",
                        default=["tech18", "mag7", "balanced15",
                                 "sp30", "diversified40", "volatile25"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []

    # Baselines are computed once per universe from embedded ACE prices
    baseline_navs = {}      # universe_tag -> {"Equal-Weight": np.array, ...}
    price_dfs     = {}      # universe_tag -> pd.DataFrame (cached)
    nav_curves_by_key = {}  # (model_tag, universe_tag, system_label) -> np.array
    dates_by_key      = {}  # (model_tag, universe_tag, system_label) -> [str]

    for ut in args.universe_tags:
        print(f"\n{'=' * 60}")
        print(f"Universe: {ut}")
        print(f"{'=' * 60}")

        # Build price_df from the first available ACE result directory
        if ut not in price_dfs:
            for mt in args.model_tags:
                ace_dir = os.path.join(args.ace_base, f"qwen{mt}_{ut}")
                df = build_price_df(ace_dir)
                if df is not None:
                    price_dfs[ut] = df
                    print(f"  Price data: {len(df.columns)} tickers × {len(df)} days "
                          f"(from qwen{mt})")
                    break
            else:
                print(f"  WARNING: no price data found for {ut} — baselines skipped")

        # Compute baselines once
        if ut not in baseline_navs:
            df = price_dfs.get(ut)
            if df is not None:
                print("  Computing baselines...")
                baseline_navs[ut] = compute_baselines(df)
            else:
                baseline_navs[ut] = {}

        ew_nav = baseline_navs[ut].get("Equal-Weight")

        # Record baseline metrics (keyed as model="baseline")
        df = price_dfs.get(ut)
        for name, nav in baseline_navs[ut].items():
            dates = list(df.index) if df is not None else []
            bench = list(ew_nav) if (ew_nav is not None and len(ew_nav) == len(nav)) else None
            m = compute_metrics(nav, dates, bench)
            if m:
                m.update(model="baseline", universe=ut, system=name)
                all_rows.append(m)
                print(f"    {name:<26} TotalRet={m['total_return_pct']:>7.2f}%")

        # AI systems — one block per model
        for mt in args.model_tags:
            ace_dir = os.path.join(args.ace_base, f"qwen{mt}_{ut}")
            print(f"\n  Model: qwen{mt}")

            # ReAct systems
            for stem, label in REACT_STEMS.items():
                path = os.path.join(args.react_dir, f"{stem}_{mt}_{ut}.json")
                if not os.path.exists(path):
                    print(f"    {label:<26} MISSING: {path}")
                    continue
                records = load_records(path)
                values = [r["portfolio_value"] for r in records]
                dates  = [r["date"] for r in records]
                bench  = list(ew_nav) if (ew_nav is not None and len(ew_nav) == len(values)) else None
                m = compute_metrics(values, dates, bench)
                if m:
                    m.update(model=f"qwen{mt}", universe=ut, system=label)
                    all_rows.append(m)
                    nav_curves_by_key[(f"qwen{mt}", ut, label)] = np.array(values)
                    dates_by_key[(f"qwen{mt}", ut, label)] = dates
                    print(f"    {label:<26} TotalRet={m['total_return_pct']:>7.2f}%  Sharpe={m['sharpe']:>6.3f}")

            # ACE systems
            for pattern, label in ACE_GLOBS.items():
                path = find_ace_file(ace_dir, pattern)
                if not path:
                    print(f"    {label:<26} MISSING: ace/{pattern}")
                    continue
                records = load_records(path)
                values = [r["portfolio_value"] for r in records]
                dates  = [r["date"] for r in records]
                bench  = list(ew_nav) if (ew_nav is not None and len(ew_nav) == len(values)) else None
                m = compute_metrics(values, dates, bench)
                if m:
                    m.update(model=f"qwen{mt}", universe=ut, system=label)
                    all_rows.append(m)
                    nav_curves_by_key[(f"qwen{mt}", ut, label)] = np.array(values)
                    dates_by_key[(f"qwen{mt}", ut, label)] = dates
                    print(f"    {label:<26} TotalRet={m['total_return_pct']:>7.2f}%  Sharpe={m['sharpe']:>6.3f}")

    # Save JSON
    out_json = os.path.join(args.output_dir, "full_comparison.json")
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nJSON saved  -> {out_json}")

    # Save ASCII table
    out_txt = os.path.join(args.output_dir, "full_comparison.txt")
    write_table(all_rows, out_txt, args.universe_tags, args.model_tags)
    print(f"Table saved -> {out_txt}")

    # Print table to stdout
    with open(out_txt) as f:
        print(f.read())

    # -----------------------------------------------------------------------
    # Full comparison charts — one per (model, universe)
    # Shows all AI systems as solid lines + all baselines as dashed lines.
    # Requires nav_curves collected during the main loop above.
    # -----------------------------------------------------------------------
    if _HAS_MPL and nav_curves_by_key:
        print("\nGenerating full comparison charts...")
        _AI_COLORS = {
            "QWEN":                  "#9e9e9e",
            "QWEN+REACT":            "#607d8b",
            "HARNESS":               "#2196f3",
            "HARNESS+DEBATER":       "#ff9800",
            "HARNESS+MEMORY":        "#4caf50",
            "HARNESS+MEMORY+DEBATER":"#9c27b0",
            "HARNESS+MEM+DEB+ADAMO": "#e91e63",
        }
        _BL_COLORS = {
            "Equal-Weight":    "#000000",
            "Risk-Parity":     "#795548",
            "60/40":           "#ff5722",
            "Min-Variance":    "#009688",
            "Cov-Risk-Parity": "#3f51b5",
            "Black-Litterman": "#ffc107",
        }
        for mt in args.model_tags:
            for ut in args.universe_tags:
                fig, ax = plt.subplots(figsize=(13, 6))
                plotted = False

                # AI systems
                for sys_label in AI_ORDER:
                    key = (f"qwen{mt}", ut, sys_label)
                    nav = nav_curves_by_key.get(key)
                    if nav is None:
                        continue
                    dates_key = dates_by_key.get(key, [])
                    color = _AI_COLORS.get(sys_label, "#333333")
                    x = list(range(len(nav)))
                    ax.plot(x, nav / nav[0] * 100, color=color, linewidth=1.8,
                            label=sys_label, zorder=3)
                    plotted = True

                # Baselines
                bn = baseline_navs.get(ut, {})
                df_b = price_dfs.get(ut)
                for bl_label in BASELINE_ORDER:
                    nav = bn.get(bl_label)
                    if nav is None:
                        continue
                    color = _BL_COLORS.get(bl_label, "#888888")
                    x = list(range(len(nav)))
                    ax.plot(x, nav / nav[0] * 100, color=color, linewidth=1.0,
                            linestyle="--", alpha=0.7, label=bl_label, zorder=2)
                    plotted = True

                if not plotted:
                    plt.close(fig)
                    continue

                ax.set_title(f"Full Comparison — qwen{mt} / {ut}", fontsize=13)
                ax.set_xlabel("Trading Day")
                ax.set_ylabel("Normalized NAV (start=100)")
                ax.axhline(100, color="#cccccc", linewidth=0.8, linestyle=":")
                ax.legend(fontsize=7, ncol=2, loc="upper left")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out_png = os.path.join(args.output_dir,
                                       f"full_comparison_qwen{mt}_{ut}.png")
                fig.savefig(out_png, dpi=130)
                plt.close(fig)
                print(f"  Chart -> {out_png}")
        print("Charts done.")


if __name__ == "__main__":
    main()
