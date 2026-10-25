"""Audit finished run directories from their sidecars: read-only, no LLM, no network.

    python -m ace_harness.audit_run refresh_2024/run1 [refresh_2024/run2 ...] [--json audit.json]

Stdlib only, so it also runs as a plain file, e.g. fetched without touching a
checkout a job is using:  python ~/audit_run.py refresh_2024/run1

Per ace cell ({run}/{universe}/{system}) it reports what the SLURM log cannot:
  days      backtest days covered, checked against the requested window
  refl      Reflector calls and how many stopped at their output cap. Such a reply is cut
            off, its JSON does not parse and its lessons and bullet checks are dropped
            without any message. Runs since the Reflector budget fix count cut-off replies
            directly (finish_reason "length") and unparsed ones; for older runs a reply
            whose completion equals a known cap (550/800) is counted, since vLLM stops at
            exactly max_tokens.
  cur       Curator calls; the Curator is skipped when the Reflector returned no lessons,
            so "cur < refl" counts slow loops that added nothing. cur@cap: Curator replies
            that stopped at their cap (800; older runs: single-call tasks only).
            Cut-off replies of any other role (Solver, Debater) are flagged too.
  critic    reviews, literal "accept" verdicts, replies that did not parse (the legacy
            loops read a reply without a verdict as "accept"), and any verdict string
            other than accept/revise
  eqw       Solver outputs that are exactly equal weight with 0% cash over the whole
            universe: the equal_weight fallback when no allocation parsed
  gov       ADAP / REMO / AUTOGOV / DEBATER+STOP: stop reasons, freeze modes, demand and
            how many decisions showed the Solver any memory
  rmax      longest single Reflector reply (tokens), to compare with its budget
  mem       bullets in the final playbook
plus provenance (status, git sha, git_dirty) and data_hash agreement within and
across runs. Across the runs given it also flags cells that ran on different
commits, runs that requested different windows, and runs whose price snapshot
files (the driver's price_cache.sha256) differ. Exit status 1 if anything is flagged.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from datetime import date

# Output caps for runs made before llm_client counted cut-off replies (finish_reason
# "length"): the Reflector's caps up to RUN 1 and the Curator's (agents.Consolidator).
REFLECTOR_CAPS = (550, 800)
CURATOR_CAP = 800
# make_table's order; tests/test_core.py checks they still match.
UNIVERSES = ("tech18", "mag7", "balanced15", "sp30", "diversified40", "volatile25")
SYSTEMS = ("qwen", "react", "harness", "memory", "debater", "dual", "adap", "remo", "autogov", "debater_stop")
_NOT_RESULTS = ("_tokens.json", "_manifest.json", "_evidence.json", "_riskparams.json", "comparison_metrics.json")
_BULLET_RE = re.compile(r"^\[[a-z]{1,3}-\d{5}\] helpful=")  # memory.ExperienceMemory.save
_WINDOW_SLACK_DAYS = 6  # first/last trading day vs requested dates (holidays, weekends)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _result_files(cell_dir):
    files = glob.glob(os.path.join(cell_dir, "*_results.json"))
    if not files:  # the ReAct scripts name their output freely
        files = [f for f in glob.glob(os.path.join(cell_dir, "*.json")) if not f.endswith(_NOT_RESULTS)]
    return sorted(files)


def _stem(results_path):
    suffix = "_results.json"
    return results_path[:-len(suffix)] if results_path.endswith(suffix) else os.path.splitext(results_path)[0]


def _records(path):
    data = _load_json(path)
    return data if isinstance(data, list) else data.get("daily_nav", data.get("results", []))


def _is_equal_weight(alloc, n_assets):
    if not isinstance(alloc, dict) or not n_assets:
        return False
    assets = [k for k in alloc if k != "CASH"]
    if len(assets) != n_assets:
        return False
    target = 100.0 / n_assets
    try:
        return abs(float(alloc.get("CASH", 0.0))) < 1e-6 and all(abs(float(alloc[a]) - target) < 1e-3
                                                                 for a in assets)
    except (TypeError, ValueError):
        return False


def audit_cell(cell_dir):
    universe, system = os.path.normpath(cell_dir).split(os.sep)[-2:]
    out = {"cell": f"{universe}/{system}", "universe": universe, "system": system, "flags": []}
    files = _result_files(cell_dir)
    if len(files) != 1:
        out["flags"].append(f"{len(files)} results files" if files else "no results file")
        return out
    path = files[0]
    recs = _records(path)
    dates = [r.get("date") for r in recs if isinstance(r, dict) and r.get("date")]
    out.update(days=len(dates), first=dates[0] if dates else None, last=dates[-1] if dates else None)

    stem = _stem(path)
    if os.path.exists(os.path.join(cell_dir, "PROVENANCE")):
        with open(os.path.join(cell_dir, "PROVENANCE")) as f:
            prov = dict(kv.split("=", 1) for kv in f.read().split() if "=" in kv)
        out.update(kind="react", status="complete", git_sha=prov.get("commit"))
        return out
    if not os.path.exists(stem + "_manifest.json"):
        out.update(kind="unknown")
        out["flags"].append("no manifest")
        return out

    man = _load_json(stem + "_manifest.json")
    args = man.get("args") or {}
    out.update(kind="ace", status=man.get("status"), git_sha=man.get("git_sha"), git_dirty=man.get("git_dirty"),
               data_hash=man.get("data_hash"), risk_tuner=man.get("risk_tuner"),
               start=args.get("start"), end=args.get("end"))
    n_assets = len(man.get("tickers") or [])

    tok = _load_json(stem + "_tokens.json") if os.path.exists(stem + "_tokens.json") else {}
    refl = {"calls": 0, "tasks": 0, "at_cap": 0, "multi_call_tasks": 0, "unparsed": 0, "max_completion": 0}
    cur = {"calls": 0, "single_call_tasks": 0, "at_cap": 0}
    truncated, unparsed = collections.Counter(), collections.Counter()  # per role, counted by llm_client
    counted = "truncated" in (tok.get("totals") or {})
    for task in tok.get("per_task") or []:
        roles = task.get("by_role") or {}
        for role, counts in roles.items():
            truncated[role] += counts.get("truncated") or 0
            unparsed[role] += counts.get("unparsed") or 0
        r = roles.get("reflector")
        if r and r.get("calls"):
            refl["calls"] += r["calls"]
            refl["tasks"] += 1
            refl["unparsed"] += r.get("unparsed") or 0
            if r["calls"] == 1:
                refl["max_completion"] = max(refl["max_completion"], r.get("completion") or 0)
            if counted:
                refl["at_cap"] += r.get("truncated") or 0
            elif r["calls"] == 1:
                refl["at_cap"] += r.get("completion") in REFLECTOR_CAPS
            else:
                refl["multi_call_tasks"] += 1
        c = roles.get("consolidator")
        if c and c.get("calls"):
            cur["calls"] += c["calls"]
            if counted:
                cur["at_cap"] += c.get("truncated") or 0
            elif c["calls"] == 1:
                cur["single_call_tasks"] += 1
                cur["at_cap"] += (c.get("completion") or 0) >= CURATOR_CAP
    out.update(reflector=refl, curator=cur, totals=tok.get("totals"), truncation_counted=counted,
               truncated={k: v for k, v in truncated.items() if v}, unparsed={k: v for k, v in unparsed.items() if v},
               budgets=man.get("output_budgets") or {})

    decisions = []
    if os.path.exists(stem + "_decisions.jsonl"):
        with open(stem + "_decisions.jsonl") as f:
            decisions = [json.loads(line) for line in f if line.strip()]
    critic = {"reviews": 0, "accept": 0, "parse_fail": 0, "parse_fail_read_as_accept": 0,
              "should_store_true": 0, "should_refine_false": 0, "verdicts": collections.Counter()}
    solver = {"outputs": 0, "equal_weight": 0}
    gov = {"tasks": 0, "stops": collections.Counter(), "modes": collections.Counter(),
           "demand": collections.Counter(), "transitions": 0, "shown_known": 0, "shown_any": 0,
           "new_bullets": 0, "confirmed": 0}
    for rec in decisions:
        rounds = rec.get("rounds") or []
        allocs = [rd.get("allocations") for rd in rounds] or [rec.get("pre_harness") or rec.get("executed")]
        solver["outputs"] += len(allocs)
        solver["equal_weight"] += sum(_is_equal_weight(a, n_assets) for a in allocs)
        for rd in rounds:
            if not rd.get("reviewed"):
                continue
            critic["reviews"] += 1
            verdict = rd.get("verdict")
            critic["verdicts"][str(verdict)] += 1
            if rd.get("parse_ok") is False:
                critic["parse_fail"] += 1
                critic["parse_fail_read_as_accept"] += verdict == "accept"
            elif verdict == "accept":
                critic["accept"] += 1
            critic["should_store_true"] += bool(rd.get("should_store"))
            critic["should_refine_false"] += rd.get("should_refine") is False
        g = rec.get("governance")
        if isinstance(g, dict):
            gov["tasks"] += 1
            gov["stops"][str(g.get("stop_reason"))] += 1
            gov["modes"][str(g.get("mode"))] += 1
            gov["demand"][str(g.get("demand"))] += 1
            gov["transitions"] += g.get("transition") is not None
            if "n_visible" in g:
                gov["shown_known"] += 1
                gov["shown_any"] += (g.get("n_visible") or 0) > 0
            gov["new_bullets"] += g.get("n_new_bullets") or 0
            gov["confirmed"] += g.get("n_confirmed") or 0
    memory_size = None
    if os.path.exists(stem + "_playbook.txt"):  # final playbook: one "[id] helpful=.." line per bullet
        with open(stem + "_playbook.txt") as f:
            memory_size = sum(bool(_BULLET_RE.match(line)) for line in f)
    out.update(tasks=len(decisions), critic=critic, solver=solver, gov=gov if gov["tasks"] else None,
               memory_size=memory_size)

    if out["status"] != "complete":
        out["flags"].append(f"manifest status {out['status']!r}")
    if refl["at_cap"]:
        out["flags"].append(f"{refl['at_cap']}/{refl['calls']} Reflector replies stopped at the output cap"
                            + (" (cut off)" if counted else " (cut off: lessons and bullet checks dropped)"))
    if cur["at_cap"]:
        of = cur["calls"] if counted else cur["single_call_tasks"]
        out["flags"].append(f"{cur['at_cap']}/{of} Curator replies stopped at the output cap (cut off)")
    for role in sorted(set(out["truncated"]) - {"reflector", "consolidator"}):
        out["flags"].append(f"{out['truncated'][role]} {role} replies stopped at max_tokens (cut off)")
    for role in sorted(out["unparsed"]):
        out["flags"].append(f"{out['unparsed'][role]} {role} replies did not parse (their content was dropped)")
    if critic["parse_fail"]:
        out["flags"].append(f"{critic['parse_fail']}/{critic['reviews']} critic replies did not parse "
                            f"({critic['parse_fail_read_as_accept']} read as accept)")
    odd = sorted(v for v in critic["verdicts"] if v not in ("accept", "revise"))
    if odd:
        out["flags"].append(f"verdict strings other than accept/revise: {odd}")
    if solver["equal_weight"]:
        out["flags"].append(f"{solver['equal_weight']}/{solver['outputs']} Solver outputs are exact equal "
                            "weight (fallback when no allocation parsed)")
    return out


def _check_windows(cells):
    """Flag cells that do not cover the requested window or have a different
    day count from the rest of the run (one ticker's data ending early shortens
    the whole universe's backtest without an error)."""
    ace = [c for c in cells if c.get("kind") == "ace" and c.get("start")]
    windows = collections.Counter((c["start"], c["end"]) for c in ace)
    if len(windows) > 1:
        for c in ace:
            c["flags"].append(f"requested window {c['start']}..{c['end']} differs within the run: {dict(windows)}")
    if not windows:
        return None
    start, end = windows.most_common(1)[0][0]
    days = collections.Counter(c["days"] for c in cells if c.get("days"))
    mode_days = days.most_common(1)[0][0] if days else None
    for c in cells:
        if not c.get("first"):
            continue
        late = (date.fromisoformat(c["first"]) - date.fromisoformat(start)).days
        early = (date.fromisoformat(end) - date.fromisoformat(c["last"])).days
        if late > _WINDOW_SLACK_DAYS or early > _WINDOW_SLACK_DAYS:
            c["flags"].append(f"covers {c['first']}..{c['last']}, requested {start}..{end}")
        elif c["days"] != mode_days:
            c["flags"].append(f"{c['days']} days, the rest of the run has {mode_days}")
    return start, end


def _snapshot(root):
    """{file: sha256} of the price snapshot the run read (the driver's price_cache.sha256)."""
    path = os.path.join(root, "price_cache.sha256")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {parts[-1].lstrip("*"): parts[0] for parts in (line.split() for line in f) if len(parts) >= 2}


def audit_run(root):
    cells = []
    present = sorted(d for d in glob.glob(os.path.join(root, "*", "*")) if os.path.isdir(d))
    order = {(u, s): i for i, (u, s) in enumerate((u, s) for u in UNIVERSES for s in SYSTEMS)}
    present.sort(key=lambda d: order.get(tuple(os.path.normpath(d).split(os.sep)[-2:]), len(order)))
    for cell_dir in present:
        cells.append(audit_cell(cell_dir))
    window = _check_windows(cells)
    return {"root": root, "window": window, "cells": cells, "snapshot": _snapshot(root)}


def _campaign_flags(runs):
    """Replicates must share one commit, one requested window and one price snapshot."""
    flags = []
    shas = collections.defaultdict(list)
    for run in runs:
        for c in run["cells"]:
            if c.get("git_sha"):
                shas[c["git_sha"][:7]].append(f"{run['root']}/{c['cell']}")
    if len(shas) > 1:
        flags.append("cells ran on different commits: " +
                     "; ".join(f"{sha} ({len(cells)} cells, e.g. {cells[0]})" for sha, cells in sorted(shas.items())))
    windows = {run["root"]: run["window"] for run in runs if run["window"]}
    if len(set(windows.values())) > 1:
        flags.append(f"runs requested different windows: {windows}")
    snaps = [(run["root"], run["snapshot"]) for run in runs if run.get("snapshot")]
    for root, snap in snaps[1:]:
        base_root, base = snaps[0]
        changed = sorted(f for f in set(base) & set(snap) if base[f] != snap[f])
        if changed:
            flags.append(f"{root} read different price files than {base_root}: {changed[:5]}"
                         + (f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""))
    return flags


def _hash_report(runs):
    """data_hash per universe: within each run (cells download prices when they
    start, so a dividend between two cells changes adjusted prices) and across runs."""
    lines = []
    for universe in UNIVERSES:
        per_run = []
        for run in runs:
            hashes = {c["system"]: c["data_hash"] for c in run["cells"]
                      if c["universe"] == universe and c.get("data_hash")}
            per_run.append(hashes)
            distinct = set(hashes.values())
            if len(distinct) > 1:
                groups = collections.defaultdict(list)
                for s, h in hashes.items():
                    groups[h[:10]].append(s)
                lines.append(f"  {run['root']} {universe}: DIFFERS within the run {dict(groups)}")
        firsts = [sorted(set(h.values())) for h in per_run if h]
        if len(firsts) > 1 and any(f != firsts[0] for f in firsts[1:]):
            lines.append(f"  {universe}: differs across runs " +
                         " | ".join(f"{run['root']}: {[h[:10] for h in f]}" for run, f in zip(runs, firsts)))
    return lines


def _fmt_counter(counter):
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(counter.items(), key=lambda kv: -kv[1])) + "}"


def render(runs):
    lines = []
    for run in runs:
        cells = run["cells"]
        window = "..".join(run["window"]) if run["window"] else "?"
        lines.append(f"AUDIT {run['root']}  ({len(cells)} cells, requested {window})")
        lines.append(f"{'cell':<24} {'status':<9} {'sha':<8} {'dirty':<6} {'days':>4} {'tasks':>5} "
                     f"{'refl':>4} {'@cap':>4} {'rmax':>5} {'cur':>4} {'@cap':>4} {'reviews':>7} {'accept':>6} "
                     f"{'pfail':>5} {'eqw':>4} {'mem':>4}")
        for c in cells:
            if c.get("kind") != "ace":
                lines.append(f"{c['cell']:<24} {c.get('status') or '-':<9} {(c.get('git_sha') or '-')[:7]:<8} "
                             f"{'-':<6} {c.get('days', '-'):>4}   ({c.get('kind', '?')} cell)")
                continue
            r, cu, cr, so = c["reflector"], c["curator"], c["critic"], c["solver"]
            lines.append(f"{c['cell']:<24} {str(c['status']):<9} {(c['git_sha'] or '-')[:7]:<8} "
                         f"{str(c['git_dirty']):<6} {c['days']:>4} {c['tasks']:>5} {r['calls']:>4} "
                         f"{r['at_cap']:>4} {r['max_completion'] or '-':>5} {cu['calls']:>4} {cu['at_cap']:>4} "
                         f"{cr['reviews']:>7} "
                         f"{cr['accept']:>6} {cr['parse_fail']:>5} {so['equal_weight']:>4} "
                         f"{c['memory_size'] if c['memory_size'] is not None else '-':>4}")
        gov_cells = [c for c in cells if c.get("gov")]
        if gov_cells:
            lines.append("governance (autogov_v2 cells):")
            for c in gov_cells:
                g = c["gov"]
                # exposure "all" (REMO/AUTOGOV) logs no n_visible: the Solver sees the whole memory
                shown = f"{g['shown_any']}/{g['shown_known']}" if g["shown_known"] else "n/a"
                lines.append(f"  {c['cell']:<22} stops={_fmt_counter(g['stops'])} modes={_fmt_counter(g['modes'])} "
                             f"demand={_fmt_counter(g['demand'])} transitions={g['transitions']} "
                             f"memory shown {shown} new={g['new_bullets']} "
                             f"confirmed={g['confirmed']}")
        verdicts = collections.Counter()
        for c in cells:
            if c.get("critic"):
                verdicts.update(c["critic"]["verdicts"])
        if verdicts:
            lines.append(f"critic verdict strings, all cells: {_fmt_counter(verdicts)}")
        lines.append("")
    hash_lines = _hash_report(runs)
    lines.append("data_hash: " + ("identical within every universe" if not hash_lines else "see below"))
    lines.extend(hash_lines)
    budgets = sorted({json.dumps(c["budgets"], sort_keys=True) for run in runs for c in run["cells"] if c.get("budgets")})
    if budgets:
        lines.append("output budgets (tokens): " + " | ".join(budgets))
    snaps = [run for run in runs if run.get("snapshot")]
    if snaps:
        lines.append("price snapshot: " + ", ".join(f"{run['root']} {len(run['snapshot'])} files" for run in snaps))
    flags = [f"{run['root']}/{c['cell']}: {flag}" for run in runs for c in run["cells"] for flag in c["flags"]]
    flags += _campaign_flags(runs)
    dirty = sorted({run["root"] for run in runs for c in run["cells"] if c.get("git_dirty")})
    lines.append("")
    lines.append("FLAGS" + ("" if flags or dirty else ": none"))
    lines.extend(f"  - {f}" for f in flags)
    lines.extend(f"  - {root}: manifests record git_dirty=true (uncommitted changes when the cells ran)"
                 for root in dirty)
    return "\n".join(lines) + "\n", bool(flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", help="run directories, e.g. refresh_2024/run1")
    parser.add_argument("--json", help="also write the full audit as JSON here")
    args = parser.parse_args()
    runs = [audit_run(root.rstrip("/")) for root in args.runs]
    text, flagged = render(runs)
    sys.stdout.write(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(runs, f, indent=1, default=dict)
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
