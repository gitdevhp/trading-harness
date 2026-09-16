"""
Compare the three harnesses' backtest output.

Usage: python -m ace_harness.compare_results results/intra_results.json results/inter_results.json results/dual_results.json
(no args -> looks for that exact set of files under ./results/)
"""
import json
import sys

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def metrics(results):
    values = np.array([r["portfolio_value"] for r in results], dtype=float)
    dates = [r["date"] for r in results]

    total_return_pct = (values[-1] / values[0] - 1) * 100
    daily_ret = np.diff(values) / values[:-1]
    ann_vol_pct = float(np.std(daily_ret) * np.sqrt(252) * 100)
    sharpe = float((np.mean(daily_ret) * 252) / (np.std(daily_ret) * np.sqrt(252) + 1e-9))

    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    max_dd_pct = float(drawdown.min() * 100)

    return {
        "start": dates[0], "end": dates[-1], "final_value": round(float(values[-1]), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "annualized_vol_pct": round(ann_vol_pct, 2),
        "sharpe": round(sharpe, 2), "max_drawdown_pct": round(max_dd_pct, 2),
    }


def main():
    paths = sys.argv[1:] or [
        "results/intra_results.json", "results/inter_results.json", "results/dual_results.json",
    ]
    rows = []
    for path in paths:
        try:
            m = metrics(load(path))
            m["file"] = path
            rows.append(m)
        except FileNotFoundError:
            print(f"{path} -> not found (run that harness first)")

    if not rows:
        return

    header = f"{'file':<28}{'total_return%':>15}{'ann_vol%':>12}{'sharpe':>9}{'max_dd%':>10}"
    print(header)
    print("-" * len(header))
    for m in rows:
        print(f"{m['file']:<28}{m['total_return_pct']:>15}{m['annualized_vol_pct']:>12}"
              f"{m['sharpe']:>9}{m['max_drawdown_pct']:>10}")


if __name__ == "__main__":
    main()
