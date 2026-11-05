"""Rule-based baselines for a run directory, in make_table's 14 columns.

    python -m ace_harness.baselines campaign_2024/run1 [--out campaign_2024/baselines.tsv]

The six quantitative baselines of the earlier tables (summarize_full_run.py):
Equal-Weight, Risk-Parity, 60/40, Min-Variance, Cov-Risk-Parity and Black-Litterman,
rebalanced on the first trading day of every month with the same 15 bps fee, on the
exact prices the LLM systems traded (the prices embedded in an ace cell's results, i.e.
the campaign's snapshot). They make no LLM call and do not depend on the asset order,
so any run directory of a campaign gives the same numbers. Metrics use make_table's
definitions (compare_results, rounded once), so the rows sit next to the LLM rows; the
second block is the mean over universes, as in summarize_runs. Min-Variance,
Cov-Risk-Parity and Black-Litterman need scipy and are skipped (and named) without it.
"""
import argparse
import os
import statistics
import sys

from ace_harness import make_table
from ace_harness import summarize_full_run as sfr

ORDER = ["Equal-Weight", "Risk-Parity", "60/40", "Min-Variance", "Cov-Risk-Parity", "Black-Litterman"]
_METRIC_KEYS = ("total_return_pct", "cagr_pct", "ann_vol_pct", "sharpe", "sortino", "calmar", "max_drawdown_pct")


def price_frame(root, universe):
    """Daily closes (date x ticker) embedded in the first ace cell of this universe."""
    for subdir, _ in make_table.SYSTEMS:
        cell = os.path.join(root, universe, subdir)
        if os.path.isdir(cell):
            df = sfr.build_price_df(cell)
            if df is not None:
                return df
    return None


def build(root, model="baseline"):
    rows, missing, skipped = [], [], set()
    for universe in make_table.UNIVERSES:
        df = price_frame(root, universe)
        if df is None:
            missing.append(universe)
            continue
        curves = sfr.compute_baselines(df)
        skipped |= set(ORDER) - set(curves)
        for name in ORDER:
            if name in curves:
                m = make_table.raw_metrics([{"portfolio_value": v} for v in curves[name]])
                rows.append([model, name, universe] + [make_table._num(m[k]) for k in _METRIC_KEYS]
                            + ["", "", "", ""])
    return rows, missing, sorted(skipped, key=ORDER.index)


def cross_universe(rows):
    """Mean over universes of each baseline's (rounded) table values, as summarize_runs does."""
    out = [["System", "n_universes"] + make_table.HEADER[3:10]]
    for name in ORDER:
        mine = [r for r in rows if r[1] == name]
        if mine:
            out.append([name, str(len(mine))] + [str(round(statistics.fmean(float(r[i]) for r in mine), 2))
                                                 for i in range(3, 10)])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="a run directory of the campaign, e.g. campaign_2024/run1")
    parser.add_argument("--out", help="also write the per-universe table here")
    args = parser.parse_args()
    rows, missing, skipped = build(args.root.rstrip("/"))
    if not rows:
        sys.exit(f"no ace results with embedded prices under {args.root}")
    table = "\n".join(["BASELINES", "\t".join(make_table.HEADER)] + ["\t".join(r) for r in rows]) + "\n"
    sys.stdout.write(table)
    sys.stdout.write("\nBASELINES, mean over universes\n" + "\n".join("\t".join(r) for r in cross_universe(rows)) + "\n")
    if missing:
        print(f"\n# no prices found for: {missing}", file=sys.stderr)
    if skipped:
        print(f"# skipped (needs scipy): {skipped}", file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table)


if __name__ == "__main__":
    main()
