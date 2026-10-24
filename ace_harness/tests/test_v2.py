"""Offline tests for ADAP-v2 (governance.py + autogov_v2.py).
Run: python -m ace_harness.tests.test_v2
"""
import json
import os
import re
import tempfile

from ace_harness.tests import golden_util as gu

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden.json")
_ID_RE = re.compile(r"\[([a-z]{1,3}-\d{5})\]")


def _v2_factory(preset, rt=False, overrides=None, **cfg_fixed):
    from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
    from ace_harness.autogov_v2 import make_autogover_v2
    from ace_harness.governance import make_config
    from ace_harness.memory import ExperienceMemory
    from ace_harness.risk_harness import ConvictionHarness

    def make(u, tmp):
        solver = Solver(u, include_screener_tool=True, fallback_mode="equal_weight", max_tokens=800,
                        prompt_style="monthly")
        cfg = make_config(preset, overrides, use_risk_tuner=rt, **cfg_fixed)
        make.memory = ExperienceMemory()
        make.fn = make_autogover_v2(u, solver, Debater(), Consolidator(), make.memory,
                                    os.path.join(tmp, "pb.txt"), cfg, risk_harness=ConvictionHarness(u),
                                    risk_tuner=RiskTuner() if rt else None,
                                    risk_params_path=os.path.join(tmp, "rp.json"), verbose=False)
        return make.fn
    return make


def test_presets_reproduce_dual_and_debater_exactly():
    with open(_GOLDEN) as f:
        golden = json.load(f)["systems"]
    for preset, rt, key in [("dual_equiv", False, "dual"), ("dual_equiv", True, "dual+rt"),
                            ("debater_equiv", False, "intra")]:
        fp, _ = gu.run_with_stub(_v2_factory(preset, rt))
        assert fp == golden[key], f"{preset} (rt={rt}) diverged from {key}: {fp} vs {golden[key]}"
    print("[v2] dual_equiv == DUAL (with/without RiskTuner) and debater_equiv == DEBATER: identical call sequences")


class ScriptedChat:
    """Role-scripted stub; records every call as (role, system, user)."""

    def __init__(self, assets, solver_fn=None, debater_fn=None, reflector_fn=None, consolidator_fn=None):
        self.assets, self.calls = list(assets), []
        self.solver_fn = solver_fn or (lambda n, system: {a: 10 + 5 * (n % 3) for a in self.assets} | {"CASH": 10})
        self.debater_fn = debater_fn or (lambda n, system: {"verdict": "accept", "lessons": []})
        self.reflector_fn = reflector_fn or (lambda system: {"lessons": [], "bullet_checks": []})
        self.consolidator_fn = consolidator_fn or (lambda user: [])
        self.n_decide = 0
        self.n_debate = 0

    def __call__(self, model, messages, temperature=0.0, max_tokens=700, stop=None):
        system, user = messages[0]["content"], messages[1]["content"]
        role = gu.role_of(system)
        self.calls.append((role, system, user))
        if role == "solver":
            if not messages[-1]["content"].startswith("Observation:"):
                return "Action: get_market_screener[]"
            self.n_decide += 1
            return "Action: Target_Allocations[" + json.dumps(self.solver_fn(self.n_decide, system)) + "]"
        if role == "debater":
            self.n_debate += 1
            out = self.debater_fn(self.n_debate, system)
            return out if isinstance(out, str) else json.dumps(out)
        if role == "reflector":
            return json.dumps(self.reflector_fn(system))
        if role == "consolidator":
            return json.dumps(self.consolidator_fn(user))
        return "{}"

    def count(self, role):
        return sum(1 for r, _, _ in self.calls if r == role)


def _one_task(chat, preset="v2", **cfg_fixed):
    """Run a single decision (no history) and return (executed, meta)."""
    import ace_harness.agents as agents_mod
    u = gu.make_universe()
    chat.assets = u.anon_universe
    original = agents_mod.chat
    agents_mod.chat = chat
    try:
        with tempfile.TemporaryDirectory() as tmp:
            fn = _v2_factory(preset, **cfg_fixed)(u, tmp)
            state = {"portfolio_value": 1e6, "cash_pct": 0.0,
                     "allocations_pct": {a: 20.0 for a in u.anon_universe} | {"CASH": 0.0}}
            return fn("2024-06-03", state, {}, 20)
    finally:
        agents_mod.chat = original


def _revise(**kw):
    return lambda n, s: {"verdict": "revise", "direction": "increase_conviction", "feedback": f"fb{n}",
                         "lessons": [], "should_store": False, "should_refine": True, **kw}


def test_submits_last_round_and_skips_terminal_review():
    chat = ScriptedChat([], debater_fn=_revise())
    executed, meta = _one_task(chat)
    gov = meta["governance"]
    assert gov["stop_reason"] == "terminal" and gov["n_reviews"] == 2 and gov["submitted_round"] == 3
    assert meta["pre_harness_allocations"] == meta["rounds"][-1]["allocations"], "must execute the last proposal"
    assert chat.count("debater") == 2, "the round-K proposal must not be sent to the critic"
    print("[v2] executes the round-K proposal (like DEBATER/DUAL) and skips the useless terminal review")


def test_gref_stop():
    chat = ScriptedChat([], debater_fn=_revise(should_refine=False))
    _, meta = _one_task(chat)
    gov = meta["governance"]
    assert gov["stop_reason"] == "g_ref" and gov["n_reviews"] == 1 and gov["submitted_round"] == 1
    assert chat.n_decide == 1
    print("[v2] critic g_ref=false stops refinement after one review")


def test_convergence_stop():
    chat = ScriptedChat([], solver_fn=lambda n, s: {"ASSET_A": 50, "CASH": 50}, debater_fn=_revise())
    _, meta = _one_task(chat)
    gov = meta["governance"]
    assert gov["stop_reason"] == "converged" and gov["n_reviews"] == 1 and gov["n_rounds"] == 2
    print("[v2] an unchanged revision stops the loop without another critic call")


def test_parse_fail_is_not_an_admission():
    chat = ScriptedChat([], debater_fn=lambda n, s: "I cannot produce JSON for this.")
    _, meta = _one_task(chat)
    gov = meta["governance"]
    assert gov["stop_reason"] == "parse_fail" and meta["admitted"] is False and gov["n_staged_next"] == 0
    assert gov["parse_fail"] and gov["submitted_round"] == 1
    print("[v2] an unparseable critic reply stops like DEBATER does but is not counted as an admission")


def test_debater_bool_and_verdict_parsing():
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Debater
    replies = iter([json.dumps({"verdict": " Accept ", "should_store": "false", "should_refine": "TRUE"}),
                    json.dumps({"verdict": "reject"}), "no json here"])
    original = agents_mod.chat
    agents_mod.chat = lambda *a, **k: next(replies)
    try:
        a = Debater().intra_task_review("2024-06-03", "", "", {"CASH": 100})
        b = Debater().intra_task_review("2024-06-03", "", "", {"CASH": 100})
        c = Debater().intra_task_review("2024-06-03", "", "", {"CASH": 100})
    finally:
        agents_mod.chat = original
    # verdict_norm mirrors the legacy read exactly: only the literal "accept" (or no verdict) stops
    assert a["verdict_norm"] == "revise" and a["parse_ok"] and a["verdict"] == " Accept "
    assert a["store_signal"] is False and a["refine_signal"] is True, 'v2 reads "false" as False'
    assert a["should_store"] is True, "legacy field keeps its legacy bool() parsing (ADAP-v1 unchanged)"
    assert b["verdict_norm"] == "revise" and not b["parse_ok"], "unknown verdicts keep refining, as before"
    assert c["verdict_norm"] == "accept" and not c["parse_ok"] and c["verdict"] == "accept"
    print('[agents] v2 signals parse "false" correctly; legacy fields and stop semantics unchanged')


def test_presets_match_legacy_on_odd_verdict_casing():
    """A capitalised "Accept" keeps DUAL/DEBATER refining; the presets must do the same."""
    from ace_harness.tests.golden_util import build_decision_fn

    def odd(bits, system):
        return {"verdict": ("Accept", "revise", " accept ", "accept")[bits % 4], "feedback": "f", "lessons": [],
                "should_store": bool(bits & 2), "should_refine": bool(bits & 4)}
    for preset, legacy in (("dual_equiv", "dual"), ("debater_equiv", "intra")):
        got, _ = gu.run_with_stub(_v2_factory(preset), debater_override=odd)
        want, _ = gu.run_with_stub(lambda u, tmp, s=legacy: build_decision_fn(s, u, tmp), debater_override=odd)
        assert got == want, (preset, got, want)
    print("[v2] presets stay call-for-call identical to DUAL/DEBATER even for 'Accept' / ' accept ' replies")


def _run_months(chat, preset="v2", months=4, **cfg_fixed):
    """Engine run over the first `months` rebalances of 2024 under a ScriptedChat."""
    import ace_harness.agents as agents_mod
    from ace_harness import engine_monthly
    u = gu.make_universe()
    chat.assets = u.anon_universe
    ends = {1: "2024-02-02", 2: "2024-03-04", 3: "2024-04-02", 4: "2024-05-02", 5: "2024-06-04",
            6: "2024-07-02", 7: "2024-08-02", 8: "2024-09-03", 9: "2024-10-02", 10: "2024-11-04",
            11: "2024-12-31"}
    metas = []
    original = agents_mod.chat
    agents_mod.chat = chat
    try:
        with tempfile.TemporaryDirectory() as tmp:
            factory = _v2_factory(preset, **cfg_fixed)
            inner = factory(u, tmp)

            def fn(*args):
                n_before = len(chat.calls)
                out = inner(*args)
                metas.append((out[1], chat.calls[n_before:]))
                return out
            engine_monthly.run_backtest(u, fn, "2024-01-02", ends[months], 1e6, os.path.join(tmp, "r.json"), 20)
            return metas, factory.memory
    finally:
        agents_mod.chat = original


def test_deferred_outcome_verified_storage():
    lesson_a = "When 6M-Mom > 20% and price above 200d-SMA, size up to 12%"
    lesson_b = "When 1M-Vol > 40%, cap the position at 4%"
    debate = iter([{"verdict": "revise", "feedback": "x", "should_store": True, "should_refine": False,
                    "lessons": [{"lesson": lesson_a, "confidence": "medium"},
                                {"lesson": lesson_b, "confidence": "high"},
                                {"lesson": "low-confidence idea", "confidence": "low"}]}]
                  + [{"verdict": "accept", "lessons": []}] * 10)

    def reflector(system):
        checks = []
        if "CANDIDATE lessons" in system:
            assert "C0: " + lesson_a in system and "C1: " + lesson_b in system
            assert "low-confidence idea" not in system, "low-confidence lessons are never nominated"
            checks = [{"id": "C0", "verdict": "confirmed"}, {"id": "C1", "verdict": "contradicted"}]
        return {"lessons": [], "bullet_checks": [], "candidate_checks": checks}

    chat = ScriptedChat([], debater_fn=lambda n, s: next(debate), reflector_fn=reflector,
                        consolidator_fn=lambda user: [{"op": "add", "section": "CONVICTION SIGNALS",
                                                       "content": json.loads(user.split("lessons:\n")[-1])[0]["lesson"]}]
                        if json.loads(user.split("lessons:\n")[-1]) else [])
    metas, memory = _run_months(chat, months=2)
    task1_calls, task2_calls = metas[0][1], metas[1][1]
    assert not any(r == "consolidator" for r, _, _ in task1_calls), "no fast-loop Curator call"
    curator_users = [u for r, _, u in task2_calls if r == "consolidator"]
    assert len(curator_users) == 1 and lesson_a in curator_users[0] and lesson_b not in curator_users[0]
    assert [b["content"] for b in memory.bullets.values()] == [lesson_a]
    assert metas[1][0]["governance"]["n_confirmed"] == 1
    print("[v2] critic-nominated lessons are stored only after next month's realized returns confirm them")


def test_evidence_gated_exposure():
    rule = "When 6M-Mom > 15% size up to 10%"
    seen = {}

    def reflector(system):
        ids = [i for i in _ID_RE.findall(system) if i != "xyz-00001"]  # skip the prompt's own example id
        if "PROBATIONARY" in system:
            seen["probation_ids"] = ids
        return {"lessons": [{"lesson": rule, "confidence": "high"}] if not ids else [],
                "bullet_checks": [{"id": i, "verdict": "confirmed"} for i in ids]}

    chat = ScriptedChat([], reflector_fn=reflector,
                        consolidator_fn=lambda user: ([{"op": "add", "section": "CONVICTION SIGNALS", "content": rule}]
                                                      if rule in user.split("lessons:")[-1] else []))
    metas, memory = _run_months(chat, months=3)
    solver_systems = [[s for r, s, _ in calls if r == "solver"] for _, calls in metas]
    assert all("EXPERIENCE MEMORY" not in s for s in solver_systems[0] + solver_systems[1]), \
        "an unvalidated bullet must not reach the Solver"
    bid = next(iter(memory.bullets))
    assert seen.get("probation_ids") == [bid], "the new bullet is graded while on probation"
    assert any(rule in s and "EXPERIENCE MEMORY" in s for s in solver_systems[2]), \
        "after a realized confirmation the bullet is shown"
    assert metas[2][0]["governance"]["n_visible"] == 1
    print("[v2] memory reaches the Solver only after realized data confirms it (probation -> visible)")


def test_freeze_controller_unit():
    from ace_harness.governance import FreezeController
    fc = FreezeController(window=3, tau=1 / 3, probe_every=3)
    modes = []
    for t in range(12):
        mode = fc.begin_task()
        modes.append(mode)
        fc.end_task(mode, demand=1 if t == 9 else 0, slow_loop_ran=t > 0)
    # tasks 0-3 open (freeze needs 3 slow loops), 4-5 frozen, 6 probe, 7-8 frozen, 9 probe (demand) -> reopen
    assert modes == ["open"] * 4 + ["frozen", "frozen", "probe", "frozen", "frozen", "probe", "open", "open"], modes
    assert fc.n_freezes == 1 and fc.n_probes == 2 and fc.n_reopens == 1 and not fc.frozen
    print("[governance] sliding-window freeze, periodic probes and reopen behave as specified")


def test_freeze_skips_memory_calls():
    chat = ScriptedChat([], debater_fn=lambda n, s: {"verdict": "accept", "lessons": [], "should_store": False},
                        reflector_fn=lambda s: {"lessons": [{"lesson": "x 1M-Vol > 30%", "confidence": "high"}],
                                                "bullet_checks": []})
    metas, _ = _run_months(chat, months=11)
    modes = [m["governance"]["mode"] for m, _ in metas]
    assert modes[:4] == ["open"] * 4 and modes[4:6] == ["frozen", "frozen"] and modes[6] == "probe", modes
    for (meta, calls), mode in list(zip(metas, modes))[1:]:  # task 1 has no history to reflect on
        n_mem = sum(1 for r, _, _ in calls if r in ("reflector", "consolidator"))
        assert (n_mem == 0) == (mode == "frozen"), (mode, n_mem)
    print(f"[v2] frozen tasks make no Reflector/Curator calls; probes still run (modes={modes})")


def test_no_lookahead():
    """Prices after T must not influence any prompt or decision up to T."""
    import ace_harness.agents as agents_mod
    from ace_harness import engine_monthly
    cut = "2024-07-01"
    runs = []
    for perturb in (False, True):
        u = gu.make_universe()
        if perturb:
            for df in u.data_cache.values():
                df.loc[df.index > cut, ["Open", "Close"]] *= 1.5
        chat = gu.HashChat(u.anon_universe)
        original = agents_mod.chat
        agents_mod.chat = chat
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn = _v2_factory("v2")(u, tmp)
                engine_monthly.run_backtest(u, fn, "2024-01-02", cut if not perturb else "2024-12-31", 1e6,
                                            os.path.join(tmp, "r.json"), 20)
        finally:
            agents_mod.chat = original
        runs.append(chat.calls)
    assert runs[1][:len(runs[0])] == runs[0] and len(runs[1]) > len(runs[0])
    print(f"[v2] no look-ahead: first {len(runs[0])} calls identical when all post-{cut} prices change")


def test_rewritten_bullet_returns_to_probation():
    """A Curator rewrite of a validated bullet must not reach the Solver until
    realized data confirms the new text."""
    rule, rewrite = "When 6M-Mom > 15% size up to 10%", "UNVERIFIED: put 60% in the top 1M-Mom asset"
    step = {"n": 0}

    def reflector(system):
        step["n"] += 1
        ids = [i for i in _ID_RE.findall(system) if i != "xyz-00001"]
        if step["n"] == 1:
            return {"lessons": [{"lesson": rule, "confidence": "high"}], "bullet_checks": []}
        return {"lessons": [{"lesson": rewrite, "confidence": "high"}] if step["n"] == 3 else [],
                "bullet_checks": [{"id": i, "verdict": "confirmed"} for i in ids]}

    def curator(user):
        lessons = json.loads(user.split("lessons:\n")[-1])
        ids = _ID_RE.findall(user)
        if lessons and lessons[0]["lesson"] == rule:
            return [{"op": "add", "section": "CONVICTION SIGNALS", "content": rule}]
        if lessons and lessons[0]["lesson"] == rewrite and ids:
            return [{"op": "replace", "id": ids[0], "content": rewrite}]
        return []

    chat = ScriptedChat([], reflector_fn=reflector, consolidator_fn=curator)
    metas, _ = _run_months(chat, months=4)
    solver_sys = [[s for r, s, _ in calls if r == "solver"] for _, calls in metas]
    assert any(rule in s for s in solver_sys[2]), "confirmed bullet visible in task 3"
    assert not any(rewrite in s for s in solver_sys[3]), "rewritten text must not be shown before confirmation"
    assert metas[3][0]["governance"]["rewritten_ids"], "the rewrite is logged"
    print("[v2] a rewritten bullet goes back to probation; ungraded text never reaches the Solver")


def test_r1_policy_reviews_round_k():
    chat = ScriptedChat([], debater_fn=lambda n, s: {"verdict": "accept" if n == 3 else "revise", "feedback": "f",
                                                     "lessons": [], "should_refine": True})
    _, meta = _one_task(chat, preset="v2", submit_policy="r1_unless_admitted")
    gov = meta["governance"]
    assert gov["n_reviews"] == 3 and gov["stop_reason"] == "accept" and gov["submitted_round"] == 3
    print("[v2] under r1_unless_admitted the round-K proposal is still reviewed and can be admitted")


def test_single_round_never_freezes():
    chat = ScriptedChat([], reflector_fn=lambda s: {"lessons": [], "bullet_checks": []})
    metas, _ = _run_months(chat, months=8, max_rounds=1)
    assert all(m["governance"]["demand"] is None and m["governance"]["mode"] == "open" for m, _ in metas)
    print("[v2] with K=1 the critic is never consulted, demand is unknown, and memory is not frozen")


def test_malformed_reflector_fields():
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Debater
    from ace_harness.governance import EvidenceStore
    from ace_harness.memory import ExperienceMemory
    replies = iter([json.dumps({"lessons": {"x": 1}, "bullet_checks": {"y": 2}, "candidate_checks": {"C0": "ok"}}),
                    json.dumps({"lessons": [], "candidate_checks": [{"id": "C01", "verdict": "confirmed"}]})])
    original = agents_mod.chat
    agents_mod.chat = lambda *a, **k: next(replies)
    try:
        cands = [{"lesson": "a", "confidence": "high"}, {"lesson": "b", "confidence": "high"}]
        r1 = Debater().post_task_reflect("d0", "d1", {}, {}, {}, candidate_lessons=cands)
        r2 = Debater().post_task_reflect("d0", "d1", {}, {}, {}, candidate_lessons=cands)
    finally:
        agents_mod.chat = original
    assert r1["lessons"] == [] and r1["bullet_checks"] == [] and r1["candidate_checks"] == []
    assert r2["candidate_checks"][0]["id"] == "C1"
    mem, ev = ExperienceMemory(), EvidenceStore()
    ev.apply_checks([{"id": "ris-00001", "verdict": "confirmed"}], mem)
    bid = mem.add_bullet("RISK LESSONS", "new bullet")
    ev.sync(mem)
    assert bid == "ris-00001" and ev.net(bid) == 0, "a check citing a not-yet-existing id must not pre-validate it"
    print("[agents+governance] malformed Reflector fields are ignored; phantom ids never pre-validate a bullet")


def test_ablation_tags_and_reserved_overrides():
    from ace_harness import run_monthly
    saved = run_monthly.build_universe
    run_monthly.build_universe = lambda *a: gu.make_universe()
    try:
        for kwargs in ({"gov_overrides": {"use_risk_tuner": True}}, {"gov_overrides": {"max_rounds": 2}}):
            try:
                run_monthly.run_system("adap_v2", ["A"], "2024-01-02", "2024-02-02", "/nonexistent", **kwargs)
                raise AssertionError(kwargs)
            except ValueError:
                pass
    finally:
        run_monthly.build_universe = saved
    print("[run_monthly] --gov cannot override flags that also drive the tag and manifest")


def test_run_monthly_end_to_end():
    """run_monthly wiring for adap_v2 and intra: results stay a list, sidecars
    and manifest are written, governance is logged, tokens are per role."""
    from ace_harness import accounting, run_monthly
    from ace_harness.tests.test_core import with_real_chat
    u = gu.make_universe()
    saved = run_monthly.build_universe
    run_monthly.build_universe = lambda tickers, start, end: gu.make_universe()
    try:
        with with_real_chat(gu.HashChat(u.anon_universe)), tempfile.TemporaryDirectory() as tmp:
            for system in ("adap_v2", "intra"):
                out = run_monthly.run_system(system, ["A", "B", "C", "D", "E"], "2024-01-02", "2024-12-31", tmp,
                                             adaptive_risk=False)
                assert isinstance(json.load(open(out)), list)
                paths = accounting.sidecar_paths(out)
                manifest = json.load(open(paths["manifest"]))
                assert manifest["status"] == "complete" and manifest["temperature"] == 0.0
                assert manifest["risk_tuner"] is False and manifest["tokens"]["calls"] > 0
                recs = accounting.read_decisions(paths["decisions"])
                assert len(recs) == 11
                tokens = accounting.load_tokens(out)
                assert "solver" in tokens["by_role"] and "debater" in tokens["by_role"]
                if system == "adap_v2":
                    assert manifest["governance_hash"] and manifest["governance"]["exposure_mode"] == "evidence_gated"
                    assert all(r["governance"]["stop_reason"] for r in recs)
                    assert os.path.exists(paths["evidence"])
            out = run_monthly.run_system("intra", ["A"], "2024-01-02", "2024-03-04", tmp, max_rounds=2,
                                         debater_sizing_context=True)
            assert os.path.basename(out) == "monthly_intra_sizing_k2_convictionriskharness_results.json"
            try:
                run_monthly.run_system("intra", ["A"], "2024-01-02", "2024-12-31", tmp,
                                       gov_overrides={"freeze_mode": "off"})
                raise AssertionError("--gov on a non-v2 system must fail")
            except ValueError:
                pass
    finally:
        run_monthly.build_universe = saved
    print("[run_monthly] adap_v2 + intra end-to-end: sidecars, manifest, governance log, per-role tokens")


def test_gov_override_parsing():
    from ace_harness.governance import make_config, parse_override
    assert parse_override("freeze_mode=off") == ("freeze_mode", "off")
    assert parse_override("use_critic_gref=false") == ("use_critic_gref", False)
    assert parse_override("convergence_eps_pp=0") == ("convergence_eps_pp", 0.0)
    for bad in ("nonexistent=1", "use_critic_gref=maybe"):
        try:
            parse_override(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        make_config("v2", {"exposure_mode": "everything"})
        raise AssertionError("invalid choice accepted")
    except ValueError:
        pass
    print("[governance] --gov overrides are typed and validated")


def run_all():
    test_presets_reproduce_dual_and_debater_exactly()
    test_submits_last_round_and_skips_terminal_review()
    test_gref_stop()
    test_convergence_stop()
    test_parse_fail_is_not_an_admission()
    test_debater_bool_and_verdict_parsing()
    test_presets_match_legacy_on_odd_verdict_casing()
    test_rewritten_bullet_returns_to_probation()
    test_r1_policy_reviews_round_k()
    test_single_round_never_freezes()
    test_malformed_reflector_fields()
    test_ablation_tags_and_reserved_overrides()
    test_deferred_outcome_verified_storage()
    test_evidence_gated_exposure()
    test_freeze_controller_unit()
    test_freeze_skips_memory_calls()
    test_no_lookahead()
    test_gov_override_parsing()
    test_run_monthly_end_to_end()
    print("V2 TESTS PASSED")


if __name__ == "__main__":
    run_all()
