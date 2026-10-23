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
--root replaces the base row; all other base rows are kept as they are.
"""
import argparse
import glob
import json
import os
import sys

from ace_harness import accounting
from ace_harness.compare_results import metrics

UNIVERSES = ["tech18", "mag7", "balanced15", "sp30", "diversified40", "volatile25"]
# (subdir, paper label), in table order
SYSTEMS = [
    ("qwen", "QWEN"), ("react", "REACT"), ("harness", "HARNESS"),
    ("memory", "HARNESS+MEMORY"), ("debater", "HARNESS+DEBATER"), ("dual", "DUAL"), ("adap", "ADAP"),
    ("remo", "REMO"), ("debater_stop", "DEBATER+STOP"),
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


def cell_row(model, label, universe, path):
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        records = records.get("daily_nav", records.get("results", []))
    m = metrics(records)
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


def read_base(path):
    rows = {}
    with open(path) as f:
        for line in f:
            cells = line.rstrip("\n").split("\t")
            if len(cells) >= 10 and cells[0] != "Model" and cells[1]:
                rows[(cells[1].strip(), cells[2].strip())] = (cells + [""] * len(HEADER))[:len(HEADER)]
    return rows


def build(root, model="qwen25", base=None):
    new = {}
    for universe in UNIVERSES:
        for subdir, label in SYSTEMS:
            path = find_result(os.path.join(root, universe, subdir))
            if path:
                new[(label, universe)] = cell_row(model, label, universe, path)
    base_rows = read_base(base) if base else {}
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
    args = parser.parse_args()

    rows, sources = build(args.root, args.model, args.base)
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
