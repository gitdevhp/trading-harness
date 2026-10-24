"""Build the paper's per-run results table (tab-separated, paste into Sheets).

Reads a run directory laid out as {root}/{universe}/{subdir}/ (the layout
written by run_run1_refresh.sh and run_dual_mem_deb_runs23.sh) and prints one
row per (system, universe) with the same columns and metric definitions as
the existing RUN 1 table (compare_results.metrics). Token columns come from
the <stem>_tokens.json sidecar; they are blank for the ReAct scripts, which
do not record tokens.

    python -m ace_harness.make_table --root OUT/run1 --title "RUN 1" \
        [--base old_run1.tsv] [--out OUT/run1/table.tsv]

--base merges with an existing table (same columns): every cell found under
--root replaces the base row; all other base rows are kept as they are. Only
the first table block of --base is read (a sheet export holding RUN 1, RUN 2,
... would otherwise mix runs). --expect "sp30:qwen sp30:react ..." makes a
missing re-run cell an error instead of silently keeping the invalid base row.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from ace_harness import accounting
from ace_harness import compare_results as cr

UNIVERSES = ["tech18", "mag7", "balanced15", "sp30", "diversified40", "volatile25"]
# (subdir, paper label), in table order
SYSTEMS = [
    ("qwen", "QWEN"), ("react", "REACT"), ("harness", "HARNESS"),
    ("memory", "HARNESS+MEMORY"), ("debater", "HARNESS+DEBATER"), ("dual", "DUAL"), ("adap", "ADAP"),
    ("remo", "REMO"), ("autogov", "AUTOGOV"), ("debater_stop", "DEBATER+STOP"),
]
HEADER = ["Model", "System", "Universe", "TotalRet%", "CAGR%", "AnnVol%", "Sharpe", "Sortino", "Calmar",
          "MaxDD%", "LLM_Calls", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"]
_SIDECAR_SUFFIXES = ("_tokens.json", "_manifest.json", "_evidence.json", "_riskparams.json")


def find_result(cell_dir):
    """The single results file in a cell directory, or None."""
    files = glob.glob(os.path.join(cell_dir, "*_results.json"))
    if not files:  # ReAct scripts name their output freely
        files = [f for f in glob.glob(os.path.join(cell_dir, "*.json")) if not f.endswith(_SIDECAR_SUFFIXES)
                 and not f.endswith("comparison_metrics.json")]
    if len(files) > 1:
        raise RuntimeError(f"{cell_dir}: expected one results file, found {sorted(files)}")
    return files[0] if files else None


def _num(v):
    return "" if v is None else str(round(float(v), 2))


def raw_metrics(records):
    """compare_results' definitions, unrounded, so the table rounds once
    (compare_results.metrics rounds Sharpe/Sortino/Calmar to 3 dp first)."""
    v = np.array([r["portfolio_value"] for r in records], dtype=float)
    r = cr._daily_returns(v)
    return {"total_return_pct": (v[-1] / v[0] - 1) * 100, "cagr_pct": cr._cagr(v),
            "ann_vol_pct": float(np.std(r) * np.sqrt(cr.TRADING_DAYS) * 100), "sharpe": cr._sharpe(r),
            "sortino": cr._sortino(r), "calmar": cr._calmar(v), "max_drawdown_pct": cr._max_drawdown(v)}


def load_records(path):
    with open(path) as f:
        records = json.load(f)
    return records if isinstance(records, list) else records.get("daily_nav", records.get("results", []))


def cell_row(model, label, universe, path):
    m = raw_metrics(load_records(path))
    manifest_path = accounting.sidecar_paths(path)["manifest"]
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            if json.load(f).get("status") != "complete":
                raise RuntimeError(f"{path}: manifest status is not 'complete' (crashed run?)")
    tok = (accounting.load_tokens(path) or {}).get("totals", {})
    return [model, label, universe, _num(m["total_return_pct"]), _num(m["cagr_pct"]), _num(m["ann_vol_pct"]),
            _num(m["sharpe"]), _num(m["sortino"]), _num(m["calmar"]), _num(m["max_drawdown_pct"]),
            str(tok.get("calls", "")), str(tok.get("prompt", "")), str(tok.get("completion", "")),
            str(tok.get("total", ""))]


def read_base(path, model):
    rows, headers = {}, 0
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            cells = [c.strip() for c in line.rstrip("\r\n").split("\t")]
            if cells and cells[0] == "Model":
                headers += 1
                if headers > 1:  # a second block (RUN 2, ...) starts: stop
                    break
                continue
            if len(cells) >= 10 and cells[0] == model and cells[1]:
                rows[(cells[1], cells[2])] = (cells + [""] * len(HEADER))[:len(HEADER)]
    return rows


def build(root, model="qwen25", base=None, expect=None):
    new = {}
    for universe in UNIVERSES:
        for subdir, label in SYSTEMS:
            path = find_result(os.path.join(root, universe, subdir))
            if path:
                new[(label, universe)] = cell_row(model, label, universe, path)
    labels = dict(SYSTEMS)
    missing = [cell for cell in (expect or []) if (labels[cell.split(":")[1]], cell.split(":")[0]) not in new]
    if missing:
        raise RuntimeError(f"expected re-run cells have no finished results: {missing}")
    base_rows = read_base(base, model) if base else {}
    rows, sources = [], []
    for universe in UNIVERSES:
        for _, label in SYSTEMS:
            key = (label, universe)
            if key in new:
                rows.append(new[key])
                sources.append(("NEW", key))
            elif key in base_rows:
                rows.append(base_rows[key])
                sources.append(("kept", key))
    return rows, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="run directory, e.g. OUT_ROOT/run1")
    parser.add_argument("--title", default="RUN 1")
    parser.add_argument("--model", default="qwen25")
    parser.add_argument("--base", help="existing table (TSV) to merge new cells into")
    parser.add_argument("--out", help="also write the table to this file")
    parser.add_argument("--expect", default="", help='space-separated "universe:subdir" cells that must exist')
    args = parser.parse_args()

    rows, sources = build(args.root, args.model, args.base, args.expect.split())
    lines = [args.title, "\t".join(HEADER)] + ["\t".join(r) for r in rows]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    n_new = sum(1 for s, _ in sources if s == "NEW")
    print(f"\n# {n_new} rows from {args.root}; {len(sources) - n_new} kept from --base", file=sys.stderr)
    for status, (label, universe) in sources:
        if status == "NEW":
            print(f"#   NEW  {universe:<14} {label}", file=sys.stderr)


if __name__ == "__main__":
    main()
