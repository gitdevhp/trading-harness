"""Run artifacts that make every result auditable, written next to each
`<stem>_results.json` (which stays a JSON list so existing readers keep working):

  <stem>_tokens.json      totals, per-role and per-task LLM usage
  <stem>_manifest.json    provenance: git SHA, args, models, versions, data hash
  <stem>_decisions.jsonl  one record per rebalance: inputs, every round's
                          proposal, executed weights, governance state, tokens
  <stem>_evidence.json    ADAP-v2 realized-evidence counts per memory bullet

plus prepare_memory(), which refuses to silently reuse a playbook left behind
by an earlier (e.g. crashed) run of the same backtest window.
"""
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

from ace_harness import llm_client

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUFFIXES = {
    "tokens": "_tokens.json",
    "manifest": "_manifest.json",
    "decisions": "_decisions.jsonl",
    "evidence": "_evidence.json",
}
_SECRET_ENV = ("ACE_LLM_API_KEY",)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sidecar_paths(output_file: str) -> dict:
    suffix = "_results.json"
    stem = output_file[:-len(suffix)] if output_file.endswith(suffix) else os.path.splitext(output_file)[0]
    return {key: stem + ext for key, ext in _SUFFIXES.items()}


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)


def load_tokens(output_file: str):
    """Token sidecar for a results file, or None if it was never written
    (all runs before this module existed)."""
    path = sidecar_paths(output_file)["tokens"]
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_tokens_sidecar(output_file: str, per_task=None) -> dict:
    data = {
        "totals": llm_client.get_token_counts(),
        "by_role": llm_client.get_token_counts_by_role(),
        "per_task": per_task or [],
    }
    write_json(sidecar_paths(output_file)["tokens"], data)
    return data


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=_REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def data_hash(universe) -> str:
    """Hash of the exact price frames a run consumed, so runs on different
    download dates (adjusted prices drift as new dividends arrive) are
    distinguishable."""
    h = hashlib.sha256()
    for anon in sorted(universe.data_cache):
        h.update(anon.encode())
        h.update(universe.data_cache[anon].to_csv(float_format="%.6f").encode())
    return h.hexdigest()


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "yfinance", "openai"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def build_manifest(output_file, system, args: dict, universe, extra=None) -> dict:
    status = _git("status", "--porcelain")
    manifest = {
        "status": "running",
        "system": system,
        "output_file": os.path.abspath(output_file),
        "started_utc": _utc_now(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "argv": list(sys.argv),
        "args": args,
        "models": {"solver": llm_client.SOLVER_MODEL, "debater": llm_client.DEBATER_MODEL,
                   "consolidator": llm_client.CONSOLIDATOR_MODEL},
        "llm_base_url": llm_client.API_BASE_URL,
        "temperature": 0.0,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("ACE_", "SLURM_JOB_ID", "SLURM_ARRAY")) and k not in _SECRET_ENV},
        "versions": _versions(),
        "host": platform.node(),
        "tickers": list(universe.raw_universe),
        "anon_map": dict(universe.anon_map),
        "data_hash": data_hash(universe),
    }
    manifest.update(extra or {})
    return manifest


def finalize_manifest(path, **updates):
    with open(path) as f:
        manifest = json.load(f)
    manifest.update(updates)
    manifest["status"] = "complete"
    manifest["finished_utc"] = _utc_now()
    write_json(path, manifest)
    return manifest


def _move_aside(path):
    stale = f"{path}.stale.{_utc_now().replace(':', '')}"
    shutil.move(path, stale)
    print(f"WARNING: moved pre-existing {path} -> {stale} (starting from EMPTY memory; "
          f"pass --resume_memory only to warm-start a NEW period from a finished run)")
    return stale


def prepare_memory(memory_path, resume=False, evidence_path=None):
    """(ExperienceMemory, info). A playbook found at memory_path was written by
    an earlier run of this same output directory — typically a crashed run of
    the same window, whose lessons come from LATER months than the ones about
    to be replayed. Reusing it would be look-ahead, so by default it is moved
    aside and memory starts empty."""
    from ace_harness.memory import ExperienceMemory
    info = {"resumed_memory": False, "stale_files_moved": []}
    if resume and os.path.exists(memory_path):
        memory = ExperienceMemory.load(memory_path)
        info["resumed_memory"] = True
    else:
        for p in (memory_path, evidence_path):
            if p and os.path.exists(p):
                info["stale_files_moved"].append(_move_aside(p))
        memory = ExperienceMemory()
    info["initial_memory_bullets"] = len(memory.bullets)
    return memory, info


def _usage_delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in ("calls", "prompt", "completion", "total")}


def wrap_with_decision_log(decision_fn, log_path, universe, risk_harness=None):
    """Outermost wrapper: appends one JSON line per rebalance and keeps a
    per-task token list on wrapped.per_task. Never changes the decision.
    Per-round candidates are also run through the risk harness when it is
    stateless, so the realized value of each round can be computed offline
    without any extra LLM call."""
    stateless = bool(getattr(risk_harness, "STATELESS", False))
    open(log_path, "w").close()
    per_task = []

    def wrapped(current_date, portfolio_state, decision_log, rebalance_days):
        tok0, roles0 = llm_client.get_token_counts(), llm_client.get_token_counts_by_role()
        executed, meta = decision_fn(current_date, portfolio_state, decision_log, rebalance_days)
        tok1, roles1 = llm_client.get_token_counts(), llm_client.get_token_counts_by_role()
        tokens = _usage_delta(tok0, tok1)
        tokens["by_role"] = {}
        for role, counts in roles1.items():
            delta = _usage_delta(roles0.get(role, {}), counts)
            if delta["calls"]:
                tokens["by_role"][role] = delta
        meta = meta or {}

        rounds = []
        for rd in meta.get("rounds") or []:
            alloc = rd.get("allocations")
            review = rd.get("review") or {}
            entry = {"round": rd.get("round"), "allocations": alloc, "reviewed": bool(review)}
            if review:
                for key in ("verdict", "direction", "should_refine", "should_store", "parse_ok"):
                    entry[key] = review.get(key)
                entry["n_lessons"] = len(review.get("lessons") or [])
                entry["feedback"] = (review.get("feedback") or "")[:300]
            if stateless and alloc:
                entry["harnessed"] = risk_harness.apply(alloc, current_date, portfolio_state["portfolio_value"])
            rounds.append(entry)

        record = {
            "task_idx": len(per_task),
            "date": current_date,
            "portfolio_value": portfolio_state["portfolio_value"],
            "pre_weights": portfolio_state["allocations_pct"],
            "close_prices": universe.close_prices(current_date),
            "executed": executed,
            "pre_harness": meta.get("pre_harness_allocations"),
            "risk_params": meta.get("risk_params"),
            "rounds": rounds,
            "admitted": meta.get("admitted"),
            "consolidated": meta.get("consolidated"),
            "memory_size": meta.get("memory_size"),
            "governance": meta.get("governance"),
            "tokens": tokens,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")
        per_task.append({"date": current_date, **tokens})
        return executed, meta

    wrapped.per_task = per_task
    return wrapped


def read_decisions(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
