"""Aggregate replicate tables (RUN 1, RUN 2, RUN 3 — the TSVs make_table
writes, or blocks exported from the sheet) into the paper's tables:

  per-universe:   mean and sd across runs for every metric, per system/universe
  cross-universe: per run, the mean over universes; then mean and sd across runs

    python -m ace_harness.summarize_runs OUT/run1/table.tsv OUT/run2/table.tsv OUT/run3/table.tsv \
        [--out_dir OUT/summary]

sd is the sample standard deviation across runs (n-1); it is blank with one run.
Every input must contain the same (system, universe) cells.
"""
import argparse
import os
import statistics
import sys

from ace_harness.make_table import HEADER, SYSTEMS, UNIVERSES

METRICS = HEADER[3:]


def read_table(path):
    """{(system, universe): {metric: float or None}} from the first block of a table TSV."""
    rows, headers = {}, 0
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            cells = [c.strip() for c in line.rstrip("\r\n").split("\t")]
            if cells and cells[0] == "Model":
                headers += 1
                if headers > 1:
                    break
                continue
            if headers and len(cells) >= 10 and cells[1]:
                cells += [""] * (len(HEADER) - len(cells))
                rows[(cells[1], cells[2])] = {m: float(v) if v else None for m, v in zip(METRICS, cells[3:])}
    return rows


def _mean_sd(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def _fmt(v, nd=2):
    return "" if v is None else str(round(v, nd))


def summarize(tables):
    keys = set(tables[0])
    for t in tables[1:]:
        if set(t) != keys:
            raise ValueError(f"runs disagree on cells: {sorted(keys ^ set(t))}")
    order = [label for _, label in SYSTEMS]
    systems = [s for s in order if any(k[0] == s for k in keys)]

    per_universe = [["System", "Universe", "n_runs"] + [f"{m} mean" for m in METRICS] + [f"{m} sd" for m in METRICS]]
    for system in systems:
        for universe in UNIVERSES:
            if (system, universe) not in keys:
                continue
            stats = [_mean_sd([t[(system, universe)][m] for t in tables]) for m in METRICS]
            per_universe.append([system, universe, str(len(tables))] + [_fmt(mu) for mu, _ in stats]
                                + [_fmt(sd) for _, sd in stats])

    cross = [["System", "n_universes", "n_runs"] + [f"{m} mean" for m in METRICS] + [f"{m} sd" for m in METRICS]]
    for system in systems:
        unis = [u for u in UNIVERSES if (system, u) in keys]
        per_run = {m: [_mean_sd([t[(system, u)][m] for u in unis])[0] for t in tables] for m in METRICS}
        stats = [_mean_sd(per_run[m]) for m in METRICS]
        cross.append([system, str(len(unis)), str(len(tables))] + [_fmt(mu) for mu, _ in stats]
                     + [_fmt(sd) for _, sd in stats])
    return per_universe, cross


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tables", nargs="+", help="one table TSV per run")
    parser.add_argument("--out_dir", help="also write per_universe.tsv and cross_universe.tsv here")
    args = parser.parse_args()
    per_universe, cross = summarize([read_table(p) for p in args.tables])
    for title, rows, name in (("CROSS-UNIVERSE (mean over universes per run; mean, sd across runs)", cross,
                               "cross_universe.tsv"),
                              ("PER-UNIVERSE (mean, sd across runs)", per_universe, "per_universe.tsv")):
        text = "\n".join("\t".join(r) for r in rows) + "\n"
        sys.stdout.write(f"{title}\n{text}\n")
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, name), "w") as f:
                f.write(text)


if __name__ == "__main__":
    main()
