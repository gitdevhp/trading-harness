"""
Compare ACE harness backtest outputs with full risk-adjusted metrics.

Usage:
    python -m ace_harness.compare_results results/*.json
    python -m ace_harness.compare_results          # scans ./results/ automatically

Metrics reported:
    Total Return (%), CAGR (%), Ann. Volatility (%), Max Drawdown (%),
    Sharpe Ratio, Sortino Ratio, Calmar Ratio, Alpha (% p.a.), Beta
    (alpha and beta are vs. the equal-weight portfolio of the same universe,
    computed from the price data embedded in each result file)
"""
import json
import os
import sys

import numpy as np

TRADING_DAYS = 252


def load(path):
    with open(path) as f:
        return json.load(f)


def _daily_returns(values):
    return np.diff(values) / values[:-1]


def _cagr(values):
    n = len(values) - 1
    if n <= 0 or values[0] <= 0:
        return 0.0
    return float((values[-1] / values[0]) ** (TRADING_DAYS / n) - 1) * 100


def _sharpe(daily_ret):
    std = np.std(daily_ret)
    return float(np.mean(daily_ret) * TRADING_DAYS / (std * np.sqrt(TRADING_DAYS) + 1e-9))


def _sortino(daily_ret):
    neg = daily_ret[daily_ret < 0]
    downside = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 1e-9
    return float(np.mean(daily_ret) * TRADING_DAYS / (downside * np.sqrt(TRADING_DAYS) + 1e-9))


def _max_drawdown(values):
    peak = np.maximum.accumulate(values)
    return float(((values - peak) / peak).min() * 100)


def _calmar(values):
    cagr = _cagr(values)
    mdd = abs(_max_drawdown(values))
    return float(cagr / (mdd + 1e-9))


def _alpha_beta(port_ret, bench_ret):
    if len(port_ret) < 2 or len(bench_ret) < 2:
        return 0.0, 1.0
    cov = np.cov(port_ret, bench_ret)
    beta = float(cov[0, 1] / (cov[1, 1] + 1e-12))
    alpha = float((np.mean(port_ret) - beta * np.mean(bench_ret)) * TRADING_DAYS * 100)
    return alpha, beta


def _equal_weight_benchmark(records):
    """Reconstruct a daily equal-weight NAV from the price data embedded in
    each result record. Returns a numpy array aligned with the portfolio values,
    or None if prices are absent."""
    price_rows = []
    for r in records:
        prices = r.get("prices")
        if not isinstance(prices, dict) or not prices:
            return None
        vals = [v for v in prices.values() if isinstance(v, (int, float)) and v > 0]
        if vals:
            price_rows.append(vals)

    if not price_rows:
        return None

    # Build equal-weight NAV: on each day compute the return of a 1/n portfolio
    n_assets = len(price_rows[0])
    nav = [1.0]
    for i in range(1, len(price_rows)):
        prev = np.array(price_rows[i - 1], dtype=float)
        curr = np.array(price_rows[i], dtype=float)
        if len(prev) != len(curr) or np.any(prev == 0):
            nav.append(nav[-1])
            continue
        ret = float(np.mean(curr / prev - 1.0))
        nav.append(nav[-1] * (1 + ret))

    return np.array(nav)


def metrics(records, bench_nav=None):
    values = np.array([r["portfolio_value"] for r in records], dtype=float)
    dates = [r["date"] for r in records]

    daily_ret = _daily_returns(values)
    total_return = float((values[-1] / values[0] - 1) * 100)
    cagr = _cagr(values)
    ann_vol = float(np.std(daily_ret) * np.sqrt(TRADING_DAYS) * 100)
    sharpe = _sharpe(daily_ret)
    sortino = _sortino(daily_ret)
    mdd = _max_drawdown(values)
    calmar = _calmar(values)

    alpha, beta = 0.0, 1.0
    if bench_nav is not None and len(bench_nav) == len(values):
        bench_ret = _daily_returns(bench_nav)
        alpha, beta = _alpha_beta(daily_ret, bench_ret)

    return {
        "start": dates[0], "end": dates[-1],
        "final_value": round(float(values[-1]), 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(mdd, 2),
        "alpha_ann_pct": round(alpha, 2),
        "beta": round(beta, 3),
    }


def main():
    import glob

    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob("results/*_results.json"))
    if not paths:
        print("No result files found. Pass JSON paths as arguments or run from the directory containing results/.")
        return

    rows = []
    for path in paths:
        try:
            data = load(path)
            records = data if isinstance(data, list) else data.get("daily_nav", data.get("results", []))
            if not records:
                print(f"{path} -> no records found, skipping")
                continue
            bench = _equal_weight_benchmark(records)
            m = metrics(records, bench)
            label = os.path.splitext(os.path.basename(path))[0]
            label = label.replace("monthly_", "").replace("_results", "")
            m["system"] = label
            rows.append(m)
        except FileNotFoundError:
            print(f"{path} -> not found")
        except Exception as e:
            print(f"{path} -> error: {e}")

    if not rows:
        return

    cols = [
        ("system",           "System",         "<", 30),
        ("total_return_pct", "TotalRet%",      ">", 11),
        ("cagr_pct",         "CAGR%",          ">", 8),
        ("ann_vol_pct",      "AnnVol%",        ">", 9),
        ("max_drawdown_pct", "MaxDD%",         ">", 8),
        ("sharpe",           "Sharpe",         ">", 8),
        ("sortino",          "Sortino",        ">", 9),
        ("calmar",           "Calmar",         ">", 8),
        ("alpha_ann_pct",    "Alpha%",         ">", 8),
        ("beta",             "Beta",           ">", 7),
    ]

    header = "".join(f"{h:{align}{w}}" for _, h, align, w in cols)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for m in rows:
        line = ""
        for key, _, align, w in cols:
            val = m.get(key, "")
            if isinstance(val, float):
                line += f"{val:{align}{w}.2f}"
            else:
                line += f"{val:{align}{w}}"
        print(line)
    print(sep + "\n")

    # Also save a JSON summary alongside the result files
    out_dir = os.path.dirname(os.path.abspath(paths[0]))
    out_path = os.path.join(out_dir, "comparison_metrics.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Metrics saved -> {out_path}")


if __name__ == "__main__":
    main()
