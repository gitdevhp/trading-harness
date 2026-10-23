"""Deterministic, prompt-sensitive LLM stub + fixed synthetic backtests.

HashChat derives every reply from a hash of the full request, so ANY change
to any prompt (wording, memory block, feedback, screener) changes the reply,
which changes downstream allocations and the call sequence. Recording these
fingerprints at a known commit and asserting them later proves a code path
is behaviourally unchanged.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROLE_PREFIXES = (
    ("You are an autonomous ReAct Portfolio Manager", "solver"),
    ("You are a portfolio Debater", "debater"),
    ("You are a portfolio Reflector", "reflector"),
    ("You are the Curator", "consolidator"),
    ("You are a Risk Tuner", "risk_tuner"),
)
GOLDEN_SYSTEMS = ("baseline", "intra", "memory_only", "dual", "memory_only+rt", "dual+rt")
_ID_RE = re.compile(r"\[([a-z]{1,3}-\d{5})\]")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def role_of(system_prompt: str) -> str:
    for prefix, role in ROLE_PREFIXES:
        if system_prompt.startswith(prefix):
            return role
    return "other"


def make_universe(tickers=("A", "B", "C", "D", "E"), start="2023-01-02", n_days=520, seed=7):
    from ace_harness.market import MarketUniverse
    rng = np.random.default_rng(seed)
    universe = MarketUniverse(list(tickers))
    dates = pd.bdate_range(start, periods=n_days)
    for i, anon in enumerate(universe.anon_universe):
        drift = 0.0002 + 0.0003 * (i % 3)
        prices = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n_days))
        universe.data_cache[anon] = pd.DataFrame({"Open": prices * (1 + rng.normal(0, 0.002, n_days)),
                                                  "Close": prices}, index=dates)
    return universe


class HashChat:
    """Stub for agents.chat. Records (role, sha(system), sha(all messages))."""

    def __init__(self, assets, debater_override=None, solver_override=None):
        self.assets = list(assets)
        self.calls = []
        self.debater_override = debater_override  # callable(bits, system_prompt) -> dict
        self.solver_override = solver_override    # callable(bits, system_prompt) -> dict of weights

    def __call__(self, model, messages, temperature=0.0, max_tokens=700, stop=None):
        system = messages[0]["content"]
        role = role_of(system)
        full = "\x1f".join(m["role"] + ":" + m["content"] for m in messages)
        self.calls.append((role, sha(system), sha(full)))
        h = sha(full)
        bits = int(h, 16)

        if role == "solver":
            if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("Observation:"):
                if self.solver_override is not None:
                    weights = self.solver_override(bits, system)
                else:
                    weights = {a: int(h[2 * i:2 * i + 2], 16) % 40 for i, a in enumerate(self.assets)}
                    weights["CASH"] = int(h[-2:], 16) % 30
                return "Thought: allocate.\nAction: Target_Allocations[" + json.dumps(weights) + "]"
            return "Thought: look at the data.\nAction: get_market_screener[]"

        if role == "debater":
            if self.debater_override is not None:
                out = self.debater_override(bits, system)
                return out if isinstance(out, str) else json.dumps(out)
            return json.dumps({
                "verdict": "accept" if bits % 3 == 0 else "revise",
                "direction": ("increase_conviction", "decrease_risk", "well_calibrated")[(bits >> 2) % 3],
                "feedback": f"Resize per screener (fb-{h[:8]}).",
                "lessons": [{"lesson": f"When 6M-Mom > {10 + (bits >> 4) % 20}% size up to "
                                       f"{5 + (bits >> 9) % 10}% (d{h[:6]})",
                             "confidence": ("high", "medium", "low")[(bits >> 12) % 3]}],
                "should_store": bool((bits >> 14) & 1),
                "should_refine": bool((bits >> 15) & 1),
            })

        if role == "reflector":
            ids = list(dict.fromkeys(_ID_RE.findall(system)))
            checks = [{"id": bid, "verdict": "confirmed" if int(sha(h + bid), 16) % 2 else "contradicted",
                       "reason": "realized move"} for bid in ids[:4]]
            out = {"lessons": [{"lesson": f"When 1M-Vol > {20 + bits % 20}% cut to {3 + (bits >> 5) % 5}% "
                                          f"(r{h[:6]})",
                                "confidence": ("high", "medium", "low")[(bits >> 7) % 3]}],
                   "bullet_checks": checks}
            cands = re.findall(r"^(C\d+):", system, re.MULTILINE)
            if cands:
                out["candidate_checks"] = [
                    {"id": c, "verdict": ("confirmed", "contradicted", "unverifiable")[int(sha(h + c), 16) % 3],
                     "reason": "realized move"} for c in cands]
            return json.dumps(out)

        if role == "consolidator":
            user = messages[1]["content"]
            ids = list(dict.fromkeys(_ID_RE.findall(user)))
            ops = [{"op": "add", "section": ("RISK LESSONS", "CONVICTION SIGNALS")[bits % 2],
                    "content": f"Rule {h[:10]}: cap at {4 + bits % 9}% when 1M-Vol > 30%"}]
            if ids:
                ops.append({"op": "update", "id": ids[bits % len(ids)], "helpful": 1})
            return json.dumps(ops)

        if role == "risk_tuner":
            return json.dumps({"target_vol": ((bits % 5) - 2) * 0.005})
        return "{}"


def build_decision_fn(system, universe, tmpdir, max_rounds=3):
    """Mirror run_monthly.run_system's wiring for the golden systems.
    A '+rt' suffix enables the RiskTuner (run_monthly's default for memory systems)."""
    import os
    from ace_harness import harnesses
    from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
    from ace_harness.memory import ExperienceMemory
    from ace_harness.risk_harness import ConvictionHarness

    base, _, flag = system.partition("+")
    solver = Solver(universe, include_screener_tool=True, fallback_mode="equal_weight",
                    max_tokens=800, truncate_context=False, prompt_style="monthly")
    debater = Debater()
    rh = ConvictionHarness(universe)
    tuner = RiskTuner() if flag == "rt" else None
    mem_path = os.path.join(tmpdir, f"{base}_playbook.txt")
    rp_path = os.path.join(tmpdir, f"{base}_riskparams.json")
    if base == "baseline":
        return harnesses.wrap_with_risk_harness(harnesses.make_baseline(universe, solver), rh)
    if base == "intra":
        return harnesses.wrap_with_risk_harness(
            harnesses.make_intra_task(universe, solver, debater, max_rounds=max_rounds), rh)
    if base == "memory_only":
        return harnesses.make_memory_only(universe, solver, debater, Consolidator(), ExperienceMemory(),
                                          mem_path, risk_harness=rh, risk_tuner=tuner, risk_params_path=rp_path)
    if base == "dual":
        return harnesses.make_dual_timescale(universe, solver, debater, Consolidator(), ExperienceMemory(),
                                             mem_path, max_rounds=max_rounds, risk_harness=rh,
                                             risk_tuner=tuner, risk_params_path=rp_path)
    raise ValueError(system)


def run_backtest_fingerprint(decision_fn, universe, chat, tmpdir):
    import os
    from ace_harness import engine_monthly
    out = os.path.join(tmpdir, "fp_results.json")
    results = engine_monthly.run_backtest(universe, decision_fn, "2024-01-02", "2024-12-31",
                                          1_000_000.0, out, 20)
    alloc_text = json.dumps([{k: round(v, 4) for k, v in sorted(r["allocations"].items())} for r in results])
    roles = {}
    for role, _, _ in chat.calls:
        roles[role] = roles.get(role, 0) + 1
    return {
        "n_calls": len(chat.calls),
        "role_counts": roles,
        "calls_hash": sha(json.dumps(chat.calls)),
        "alloc_hash": sha(alloc_text),
        "final_nav": round(results[-1]["portfolio_value"], 2),
    }


def run_with_stub(make_fn, universe=None, **chat_kwargs):
    """Run a 2024 synthetic backtest for make_fn(universe, tmpdir) under HashChat."""
    import tempfile
    import ace_harness.agents as agents_mod
    universe = universe or make_universe()
    chat = HashChat(universe.anon_universe, **chat_kwargs)
    original = agents_mod.chat
    agents_mod.chat = chat
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return run_backtest_fingerprint(make_fn(universe, tmp), universe, chat, tmp), chat
    finally:
        agents_mod.chat = original


def run_golden_system(system):
    fp, _ = run_with_stub(lambda u, tmp: build_decision_fn(system, u, tmp))
    return fp


def prompt_hashes():
    """SHA-256 of each role's prompt text under fixed inputs."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Solver, Debater, Consolidator
    from ace_harness.memory import ExperienceMemory

    captured = []

    def spy(model, messages, **kw):
        captured.append("\x1f".join(m["content"] for m in messages))
        if role_of(messages[0]["content"]) == "solver":
            return 'Action: Target_Allocations[{"ASSET_A": 50, "CASH": 50}]'
        if role_of(messages[0]["content"]) == "consolidator":
            return "[]"
        return "{}"

    universe = make_universe()
    state = {"portfolio_value": 1_000_000.0, "cash_pct": 0.0,
             "allocations_pct": {a: 20.0 for a in universe.anon_universe} | {"CASH": 0.0}}
    mem = ExperienceMemory()
    mem.seed_with_principles()
    playbook = mem.format_for_prompt()
    screener = universe.get_market_screener("2024-06-03")
    alloc = {a: 20.0 for a in universe.anon_universe} | {"CASH": 0.0}
    prices0 = universe.close_prices("2024-05-01")
    prices1 = universe.close_prices("2024-06-03")

    original = agents_mod.chat
    agents_mod.chat = spy
    out = {}
    try:
        solver = Solver(universe, include_screener_tool=True, fallback_mode="equal_weight",
                        max_tokens=800, prompt_style="monthly")
        debater = Debater()
        cases = [
            ("solver_plain", lambda: solver.decide("2024-06-03", state, 20)),
            ("solver_memory_feedback", lambda: solver.decide("2024-06-03", state, 20, playbook_text=playbook,
                                                             feedback_text="Size up.",
                                                             direction="increase_conviction")),
            ("debater_no_playbook", lambda: debater.intra_task_review("2024-06-03", screener, "status", alloc,
                                                                      round_num=1)),
            ("debater_playbook", lambda: debater.intra_task_review("2024-06-03", screener, "status", alloc,
                                                                   playbook_text=playbook, round_num=2)),
            ("reflector_default", lambda: debater.post_task_reflect("2024-05-01", "2024-06-03", alloc, prices0,
                                                                    prices1, playbook_text=playbook,
                                                                    decision_screener_text=screener)),
            ("consolidator", lambda: Consolidator().consolidate(
                mem, [{"lesson": "When 6M-Mom > 15% size up", "confidence": "high"}], mem.sections)),
        ]
        for name, call in cases:
            captured.clear()
            call()
            out[name] = sha("\x1e".join(captured))
    finally:
        agents_mod.chat = original
    return out
