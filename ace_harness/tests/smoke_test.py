"""Run: python -m ace_harness.tests.smoke_test
Verifies engine <-> solver <-> debater <-> consolidator <-> memory wiring
with synthetic prices and a stubbed LLM — no network or model server needed.
"""
import json
import os
import random
import tempfile

import numpy as np
import pandas as pd

import ace_harness.llm_client as llm_client

_RNG = random.Random(0)


def _fake_chat(model, messages, temperature=0.0, max_tokens=700, stop=None):
    system = messages[0]["content"]
    if "Risk Tuner" in system:
        # covers param names across all three risk harnesses; update_params
        # silently ignores whichever keys don't apply to the active harness
        return json.dumps({
            "target_vol": -0.01, "stop_vol_multiplier": 0.1,
            "stop_drawdown": -0.01, "sma_cut": 0.05, "momentum_cut": 0.05,
        })
    if "Curator" in system:
        user = messages[1]["content"]
        lessons_part = user.split("New candidate lessons:")[-1].strip()
        if lessons_part and lessons_part != "[]":
            return json.dumps([{"op": "add", "section": "RISK LESSONS", "content": "Watch turnover-driven fee drag."}])
        return "[]"
    if "Reflector" in system:
        return json.dumps({"lessons": [
            {"lesson": "Diversify across at least three assets before scaling any single position.", "confidence": "medium"},
        ]})
    if "Debater" in system:
        verdict = "accept" if _RNG.random() > 0.3 else "revise"
        return json.dumps({
            "verdict": verdict,
            "feedback": "Reduce concentration in the largest position.",
            "lessons": [{"lesson": "Avoid single-asset weights above ~25% given no hard risk cap.", "confidence": "low"}],
        })
    return ('Thought: Allocating based on screener.\n'
            'Action: Target_Allocations[{"ASSET_A": 15, "ASSET_B": 15, "ASSET_C": 10, "CASH": 60}]')


llm_client.chat = _fake_chat

from ace_harness.market import MarketUniverse
from ace_harness.engine import run_backtest
from ace_harness import engine_monthly
from ace_harness.memory import ExperienceMemory
from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
from ace_harness.risk_harness import (
    GPTInstitutionalRiskHarness, SimpleTrailingRiskHarness, SimpleMomentumHarness, ConvictionHarness,
)
from ace_harness import harnesses


def make_synthetic_universe(tickers=("A", "B", "C"), n_days=140, seed=0):
    rng = np.random.default_rng(seed)
    universe = MarketUniverse(list(tickers))
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    for anon in universe.anon_universe:
        prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days))
        df = pd.DataFrame({
            "Open": prices * (1 + rng.normal(0, 0.001, n_days)),
            "Close": prices,
        }, index=dates)
        universe.data_cache[anon] = df
    return universe


def run_one(system_name, use_risk_harness=False, adaptive_risk=False,
            risk_harness_cls=GPTInstitutionalRiskHarness, include_screener_tool=True):
    universe = make_synthetic_universe()
    solver = Solver(universe, include_screener_tool=include_screener_tool)
    debater = Debater()

    with tempfile.TemporaryDirectory() as tmp:
        out_file = os.path.join(tmp, f"{system_name}.json")
        memory = None
        risk_harness = risk_harness_cls(universe) if use_risk_harness else None
        risk_tuner = RiskTuner() if adaptive_risk else None
        risk_params_path = os.path.join(tmp, "riskparams.json") if use_risk_harness else None
        memory_path = os.path.join(tmp, "playbook.txt")

        if system_name == "intra":
            decision_fn = harnesses.make_intra_task(universe, solver, debater, max_rounds=2)
            if risk_harness is not None:
                decision_fn = harnesses.wrap_with_risk_harness(decision_fn, risk_harness)
        elif system_name == "inter":
            memory = ExperienceMemory()
            decision_fn = harnesses.make_inter_task(
                universe, solver, debater, Consolidator(), memory, memory_path,
                risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
            )
        elif system_name == "dual":
            memory = ExperienceMemory()
            decision_fn = harnesses.make_dual_timescale(
                universe, solver, debater, Consolidator(), memory, memory_path, max_rounds=2,
                risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
            )
        elif system_name == "dual_session":
            decision_fn = harnesses.make_dual_session(
                universe, solver, debater, Consolidator(), max_rounds=3, risk_harness=risk_harness,
            )
        elif system_name == "dual_permanent":
            memory = ExperienceMemory()
            decision_fn = harnesses.make_dual_permanent(
                universe, solver, debater, Consolidator(), memory, memory_path,
                risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
            )
        else:
            raise ValueError(system_name)

        tag = system_name
        if use_risk_harness:
            tag += "+adaptive_risk" if adaptive_risk else "+riskharness"

        results = run_backtest(
            universe, decision_fn, start_date="2024-03-01", end_date="2024-07-01",
            initial_capital=100000.0, output_file=out_file, rebalance_days=10, warmup_days=20,
        )
        assert len(results) > 0
        assert all("portfolio_value" in r for r in results)
        for r in results:
            assert abs(sum(r["allocations"].values()) - 100.0) < 0.5, r["allocations"]
        print(f"[{tag}] {len(results)} days simulated, final value = {results[-1]['portfolio_value']:.2f}")
        if memory is not None:
            print(f"[{tag}] playbook bullets after run: {len(memory.bullets)}")
            assert os.path.exists(memory_path), "persistent-memory system never wrote its .txt playbook"
        if system_name == "dual_session":
            assert not os.path.exists(memory_path), "dual_session must never persist memory to disk"
        if adaptive_risk:
            final_params = risk_harness.get_params()
            assert final_params != risk_harness_cls.DEFAULT_PARAMS, "risk tuner never changed any parameter"
            print(f"[{tag}] risk params drifted from defaults: {final_params}")


def run_monthly_one():
    """Exercises engine_monthly.py: same-close execution, calendar-month
    rebalance trigger, equal-weight start, with dual_permanent (the
    diagram's closed loop) + ConvictionHarness (the current default) +
    adaptive tuning + the exact 'monthly' prompt style + both tools
    present (matching every version of the pasted script), reusing
    harnesses.make_dual_permanent completely unchanged."""
    universe = make_synthetic_universe(n_days=260)  # ~1 year of business days -> several month boundaries
    solver = Solver(universe, include_screener_tool=True, fallback_mode="equal_weight",
                     prompt_style="monthly", max_tokens=300, truncate_context=False)
    debater = Debater()
    memory = ExperienceMemory()
    risk_harness = ConvictionHarness(universe)
    risk_tuner = RiskTuner()

    with tempfile.TemporaryDirectory() as tmp:
        memory_path = os.path.join(tmp, "playbook.txt")
        risk_params_path = os.path.join(tmp, "riskparams.json")
        out_file = os.path.join(tmp, "monthly.json")

        decision_fn = harnesses.make_dual_permanent(
            universe, solver, debater, Consolidator(), memory, memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

        results = engine_monthly.run_backtest(
            universe, decision_fn, start_date="2024-01-02", end_date="2024-12-31",
            initial_capital=1_000_000.0, output_file=out_file,
        )
        assert len(results) > 0
        for r in results:
            assert abs(sum(r["allocations"].values()) - 100.0) < 0.5, r["allocations"]
        assert os.path.exists(memory_path)
        final_params = risk_harness.get_params()
        assert final_params != ConvictionHarness.DEFAULT_PARAMS, "risk tuner never changed any parameter"
        print(f"[monthly/dual_permanent] {len(results)} days simulated, "
              f"final value = {results[-1]['portfolio_value']:.2f}, "
              f"playbook bullets = {len(memory.bullets)}, risk params = {final_params}")


def test_dual_permanent_revision_cycle():
    """dual_permanent must call the Debater exactly once per task; on a
    "revise" verdict the Solver must get a SECOND call that actually
    contains the Debater's feedback text — this is the bug fix: previously
    the loop ended after round 1 regardless of verdict, so "revise"
    feedback was collected but never used."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Solver as _Solver, Debater as _Debater, Consolidator as _Consolidator

    original_chat = agents_mod.chat
    try:
        calls = {"solver": 0, "debater": 0}
        saw_feedback = []

        def spy(model, messages, **kw):
            system = messages[0]["content"]
            if "Debater" in system:
                calls["debater"] += 1
                return json.dumps({"verdict": "revise", "feedback": "Too concentrated.", "lessons": []})
            if "Reflector" in system:
                return json.dumps({"lessons": []})
            if "Curator" in system:
                return "[]"
            calls["solver"] += 1
            saw_feedback.append("Too concentrated." in system)
            return 'Action: Target_Allocations[{"ASSET_A": 100, "CASH": 0}]'

        agents_mod.chat = spy
        u = make_synthetic_universe(n_days=140)
        solver = _Solver(u)
        debater = _Debater()
        memory = ExperienceMemory()
        with tempfile.TemporaryDirectory() as tmp:
            decision_fn = harnesses.make_dual_permanent(
                u, solver, debater, _Consolidator(), memory, os.path.join(tmp, "mem.txt")
            )
            portfolio_state = {
                "portfolio_value": 100000, "cash_pct": 100,
                "allocations_pct": {a: 0 for a in u.anon_universe} | {"CASH": 100},
            }
            raw_alloc, meta = decision_fn("2024-06-03", portfolio_state, {}, 20)

        assert calls["debater"] == 1, f"expected exactly 1 Debater call, got {calls['debater']}"
        assert calls["solver"] == 2, f"expected exactly 2 Solver calls on a 'revise' verdict, got {calls['solver']}"
        assert saw_feedback == [False, True], f"feedback must appear only on the 2nd Solver call: {saw_feedback}"
        assert len(meta["rounds"]) == 2
    finally:
        agents_mod.chat = original_chat
    print("[harnesses] dual_permanent propose->debate->revise cycle confirmed (1 debate, feedback actually used)")


def test_monthly_prompt_style_and_conviction_harness():
    """prompt_style='monthly' must produce the exact wording from the
    latest pasted script, and ConvictionHarness must run cleanly."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Solver as _Solver

    original_chat = agents_mod.chat
    try:
        captured = {}

        def spy(model, messages, **kw):
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return 'Action: Target_Allocations[{"ASSET_A": 100, "CASH": 0}]'

        agents_mod.chat = spy
        u = make_synthetic_universe(n_days=140)
        s = _Solver(u, prompt_style="monthly", max_tokens=300)
        s.decide("2024-06-03", {"portfolio_value": 100000, "cash_pct": 100,
                                  "allocations_pct": {a: 0 for a in u.anon_universe} | {"CASH": 100}}, 20)

        assert "Decision horizon: 20 trading days." in captured["system"]
        assert "Available tools:" in captured["system"]
        assert "no access to ticker names" in captured["system"]
        assert "ReAct format:" in captured["system"]
        assert "Objective:" not in captured["system"], "monthly style must not carry the engine style's Objective line"
        assert captured["user"] == "Set target allocations for the next 20 trading days."
    finally:
        agents_mod.chat = original_chat

    risk_harness = ConvictionHarness(u)
    result = risk_harness.apply({a: 100.0 / len(u.anon_universe) for a in u.anon_universe} | {"CASH": 0.0},
                                  "2024-06-03", 100000.0)
    assert abs(sum(result.values()) - 100.0) < 0.5
    print("[solver] prompt_style='monthly' wording confirmed; [risk_harness] ConvictionHarness runs cleanly")


def test_memory_txt_roundtrip_and_hand_edit():
    """Save a playbook, hand-edit the .txt file the way a person would
    (change a count, delete a line, add a plain new line), reload, and
    confirm every edit was respected."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "playbook.txt")
        mem = ExperienceMemory()
        id_a = mem.add_bullet("STRATEGIES & INSIGHTS", "Rule A")
        id_b = mem.add_bullet("STRATEGIES & INSIGHTS", "Rule B")
        mem.update_counts(id_a, helpful=3)
        mem.update_counts(id_b, helpful=1, harmful=1)
        mem.save(path)

        with open(path) as f:
            content = f.read()
        assert f"[{id_a}]" in content and f"[{id_b}]" in content

        # hand-edit: bump id_a's helpful count, delete id_b's line entirely,
        # and add a brand-new plain-text line with no [id]/counts at all
        lines = content.splitlines()
        lines = [ln.replace(f"[{id_a}] helpful=3", f"[{id_a}] helpful=9") for ln in lines]
        lines = [ln for ln in lines if f"[{id_b}]" not in ln]
        insert_at = next(i for i, ln in enumerate(lines) if ln.startswith("## STRATEGIES")) + 1
        lines.insert(insert_at, "Hand-typed rule with no id yet")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        reloaded = ExperienceMemory.load(path)
        assert reloaded.bullets[id_a]["helpful"] == 9, "hand-edited count was not respected on reload"
        contents = [b["content"] for b in reloaded.bullets.values()]
        assert "Rule B" not in contents, "hand-deleted bullet's content reappeared on reload"
        assert "Hand-typed rule with no id yet" in contents, "hand-typed plain line was not adopted as a new bullet"
        print("[memory] .txt round-trip + hand-edit test passed")


def test_context_truncation():
    """truncate_context=True must keep the request window bounded
    regardless of turn count; the default (False) must keep growing it,
    so existing run_all.py systems are never silently affected."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Solver as _Solver

    original_chat = agents_mod.chat
    try:
        for truncate, expected in [(True, [2, 4, 4]), (False, [2, 4, 6])]:
            lengths = []
            turn = {"n": 0}

            def spy(model, messages, **kw):
                lengths.append(len(messages))
                turn["n"] += 1
                if turn["n"] < 3:
                    return 'Thought: checking.\nAction: get_portfolio_status[]'
                return 'Thought: done.\nAction: Target_Allocations[{"ASSET_A": 100, "CASH": 0}]'

            agents_mod.chat = spy
            u = make_synthetic_universe(tickers=("A",), n_days=250)
            s = _Solver(u, truncate_context=truncate)
            s.decide("2024-06-01",
                      {"portfolio_value": 100000, "cash_pct": 100, "allocations_pct": {"ASSET_A": 0, "CASH": 100}},
                      20)
            assert lengths == expected, f"truncate_context={truncate}: expected {expected}, got {lengths}"
    finally:
        agents_mod.chat = original_chat
    print("[solver] context truncation on/off both behave correctly")


def test_self_correcting_nudges():
    """A Solver seeing a no-Action reply, then an invalid-allocation
    reply, then a valid one, must nudge appropriately both times and
    recover the valid decision."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Solver as _Solver

    original_chat = agents_mod.chat
    try:
        captured = []
        turn = {"n": 0}

        def spy(model, messages, **kw):
            captured.append(messages[-1]["content"] if messages[-1]["role"] == "user" else None)
            turn["n"] += 1
            if turn["n"] == 1:
                return 'Thought: no action here.'
            if turn["n"] == 2:
                return 'Action: Target_Allocations[{"NOT_A_REAL_ASSET": 100}]'
            return 'Action: Target_Allocations[{"ASSET_A": 100, "CASH": 0}]'

        agents_mod.chat = spy
        u = make_synthetic_universe(tickers=("A",), n_days=250)
        s = _Solver(u)
        result, _ = s.decide("2024-06-01",
                              {"portfolio_value": 100000, "cash_pct": 100, "allocations_pct": {"ASSET_A": 0, "CASH": 100}},
                              20)
        assert "must provide an Action" in captured[1]
        assert "invalid" in captured[2].lower()
        assert result.get("ASSET_A", 0) == 100.0, "should have recovered the valid decision on turn 3"
    finally:
        agents_mod.chat = original_chat
    print("[solver] self-correcting retry nudges confirmed")


def test_common_trading_days_intersection():
    """A date present in only one asset's calendar must never appear in
    common_trading_days() — this is what prevents a KeyError when a
    later step looks that date up on every asset."""
    u = MarketUniverse(["A", "B"])
    dates_a = pd.bdate_range("2024-01-02", periods=10)
    dates_b = dates_a.delete(5)
    u.data_cache["ASSET_A"] = pd.DataFrame({"Open": range(10), "Close": range(10)}, index=dates_a)
    u.data_cache["ASSET_B"] = pd.DataFrame({"Open": range(9), "Close": range(9)}, index=dates_b)

    common = u.common_trading_days()
    missing_day = dates_a[5].strftime("%Y-%m-%d")
    assert missing_day not in common
    assert len(common) == 9
    print("[market] common_trading_days() correctly excludes asset-specific gaps")


def test_all_zero_falls_back_to_full_cash():
    """An all-zero allocation (Solver output, or ConvictionHarness output
    after every asset gets stopped/scaled to zero) must normalize to 100%
    CASH, not an all-zero dict that sums to nothing — matches
    normalize_allocations() in the pasted script exactly."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Solver as _Solver

    original_chat = agents_mod.chat
    try:
        def spy(model, messages, **kw):
            return 'Action: Target_Allocations[{"ASSET_A": 0, "ASSET_B": 0, "CASH": 0}]'
        agents_mod.chat = spy
        u = make_synthetic_universe(tickers=("A", "B"), n_days=140)
        s = _Solver(u)
        result, _ = s.decide("2024-06-01",
                              {"portfolio_value": 100000, "cash_pct": 100,
                               "allocations_pct": {"ASSET_A": 0, "ASSET_B": 0, "CASH": 100}}, 20)
        assert result == {"ASSET_A": 0.0, "ASSET_B": 0.0, "CASH": 100.0}, result
    finally:
        agents_mod.chat = original_chat

    conv = ConvictionHarness(u)
    result = conv.apply({"ASSET_A": 0.0, "ASSET_B": 0.0, "CASH": 0.0}, "2024-06-01", 100000.0)
    assert result == {"ASSET_A": 0.0, "ASSET_B": 0.0, "CASH": 100.0}, result
    print("[solver+risk_harness] all-zero input correctly falls back to 100% CASH, not an empty allocation")


def test_reflection_receives_decision_time_screener():
    """post_task_reflect must actually see the screener snapshot from the
    decision date when the engine captured one — this is what lets a
    lesson cite a specific indicator value instead of a platitude."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Debater as _Debater

    original_chat = agents_mod.chat
    try:
        captured = {}

        def spy(model, messages, **kw):
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return json.dumps({"lessons": []})

        agents_mod.chat = spy
        d = _Debater()
        marker = "6M-Mom=-42.00% <<UNIQUE_SCREENER_MARKER>>"
        d.post_task_reflect(
            "2024-06-01", "2024-06-03", {"ASSET_A": 100.0}, {"ASSET_A": 10.0}, {"ASSET_A": 9.0},
            decision_screener_text=marker,
        )
        assert marker in captured["system"], "decision-date screener text never reached the Reflector's prompt"

        # and confirm the quality-bar / anti-platitude instruction is actually present
        assert "specific" in captured["system"].lower()
        assert "diversify" in captured["system"].lower()  # named as a banned example
    finally:
        agents_mod.chat = original_chat
    print("[harnesses+agents] post_task_reflect receives decision-time screener; quality-bar instruction present")


def test_consolidator_prompt_requires_specificity():
    """Consolidator's prompt must instruct it to reject numberless
    platitudes for new 'add' ops and to actively merge duplicates."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Consolidator as _Consolidator
    from ace_harness.memory import ExperienceMemory as _ExperienceMemory

    original_chat = agents_mod.chat
    try:
        captured = {}

        def spy(model, messages, **kw):
            captured["system"] = messages[0]["content"]
            return "[]"

        agents_mod.chat = spy
        c = _Consolidator()
        mem = _ExperienceMemory()
        c.consolidate(mem, [{"lesson": "Diversify broadly.", "confidence": "high"}], mem.sections)
        assert "platitude" in captured["system"].lower() or "no number" in captured["system"].lower() \
            or "not qualify" in captured["system"].lower()
        assert "merge" in captured["system"].lower()
    finally:
        agents_mod.chat = original_chat
    print("[agents] Consolidator prompt requires specificity and active merging")


def test_debater_and_consolidator_are_symmetric():
    """The Debater's intra_task_review must be able to argue FOR more
    conviction, not just against it, and must return a 'direction' field.
    post_task_reflect must explicitly ask about underweighted winners,
    not just losing positions. The Consolidator must know CONVICTION
    SIGNALS is not a lesser section. This is the actual fix for the
    one-way-ratchet-toward-caution failure mode observed after a real
    backtest: memory converging on "diversify"/"cap positions" and
    systematically under-sizing the year's biggest winners."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Debater as _Debater, Consolidator as _Consolidator
    from ace_harness.memory import ExperienceMemory as _ExperienceMemory, DEFAULT_SECTIONS

    assert "CONVICTION SIGNALS" in DEFAULT_SECTIONS

    original_chat = agents_mod.chat
    try:
        captured = {}

        def spy_intra(model, messages, **kw):
            captured["intra_system"] = messages[0]["content"]
            return json.dumps({"verdict": "revise", "direction": "increase_conviction",
                                "feedback": "Size up given strong momentum.", "lessons": []})

        agents_mod.chat = spy_intra
        d = _Debater()
        review = d.intra_task_review("2024-06-03", "ASSET_A: 6M-Mom=25.0%", "status", {"ASSET_A": 5.0})
        assert review["direction"] == "increase_conviction"
        assert "increase_conviction" in captured["intra_system"] or "MORE conviction" in captured["intra_system"]
        assert "decrease_risk" in captured["intra_system"] or "LESS conviction" in captured["intra_system"]

        def spy_reflect(model, messages, **kw):
            captured["reflect_system"] = messages[0]["content"]
            return json.dumps({"lessons": []})

        agents_mod.chat = spy_reflect
        d.post_task_reflect("2024-06-01", "2024-06-03", {"ASSET_A": 5.0}, {"ASSET_A": 10.0}, {"ASSET_A": 12.0})
        assert "near-zero" in captured["reflect_system"] or "underweight" in captured["reflect_system"].lower() \
            or "small weight" in captured["reflect_system"]

        def spy_consolidate(model, messages, **kw):
            captured["consolidate_system"] = messages[0]["content"]
            return "[]"

        agents_mod.chat = spy_consolidate
        c = _Consolidator()
        mem = _ExperienceMemory()
        c.consolidate(mem, [{"lesson": "test", "confidence": "high"}], mem.sections)
        assert "CONVICTION SIGNALS" in captured["consolidate_system"]
        assert "not a lesser section" in captured["consolidate_system"] or "do not systematically favor" \
            in captured["consolidate_system"].lower()
    finally:
        agents_mod.chat = original_chat
    print("[agents] Debater/Consolidator symmetric framing confirmed (can argue for conviction, not just caution)")


def test_bullet_checks_ground_truth_scoring():
    """bullet_checks give a DIRECT, evidence-grounded path to a bullet's
    helpful/harmful count — separate from the Consolidator's text-based
    updates. This is the fix for '23 Debater calls and zero harmful
    counts': a bullet can now be marked contradicted purely because
    realized data disagreed with it, with no new lesson required."""
    from ace_harness.agents import _normalize_bullet_checks
    from ace_harness.harnesses import _apply_bullet_checks
    from ace_harness.memory import ExperienceMemory as _ExperienceMemory

    # normalization: only the first _MAX_BULLET_CHECKS_PER_CALL raw items are ever
    # considered (cap applies before filtering), and invalid ones within that window are dropped
    raw = [
        {"id": "str-00001", "verdict": "confirmed", "reason": "worked"},
        {"id": "str-00002", "verdict": "contradicted", "reason": "did not hold"},
        {"id": "", "verdict": "confirmed", "reason": "no id, must be dropped"},
        {"id": "str-00003", "verdict": "maybe", "reason": "invalid verdict, must be dropped"},
        "not a dict",
    ] + [{"id": f"extra-{i}", "verdict": "confirmed"} for i in range(10)]  # never reached: cap is 5
    normalized = _normalize_bullet_checks(raw)
    assert len(normalized) == 2
    assert {c["id"] for c in normalized} == {"str-00001", "str-00002"}

    # application: confirmed -> helpful+1, contradicted -> harmful+1, unknown id -> silently ignored
    mem = _ExperienceMemory()
    bid = mem.add_bullet("STRATEGIES & INSIGHTS", "Some rule")
    _apply_bullet_checks(mem, [
        {"id": bid, "verdict": "confirmed", "reason": "x"},
        {"id": "nonexistent-id", "verdict": "contradicted", "reason": "x"},
    ])
    assert mem.bullets[bid]["helpful"] == 1 and mem.bullets[bid]["harmful"] == 0

    _apply_bullet_checks(mem, [{"id": bid, "verdict": "contradicted", "reason": "x"}])
    assert mem.bullets[bid]["helpful"] == 1 and mem.bullets[bid]["harmful"] == 1
    print("[agents+harnesses] bullet_checks correctly ground bullet scoring in realized outcomes, not just text")


def test_baseline_and_memory_only_isolation():
    """baseline must call the Solver exactly once with no Debater
    involvement at all. memory_only must update memory via reflection
    (including bullet_checks) but never call intra_task_review — the
    debate/revision step specifically must be absent."""
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Solver as _Solver, Debater as _Debater, Consolidator as _Consolidator

    original_chat = agents_mod.chat
    try:
        calls = {"solver": 0, "debater_intra": 0, "debater_reflect": 0}

        def spy(model, messages, **kw):
            system = messages[0]["content"]
            if "pressure-testing a proposal BEFORE execution" in system:
                calls["debater_intra"] += 1
                return json.dumps({"verdict": "accept", "direction": "well_calibrated", "feedback": "", "lessons": []})
            if "Reflector" in system:
                calls["debater_reflect"] += 1
                return json.dumps({"lessons": [], "bullet_checks": []})
            calls["solver"] += 1
            return 'Action: Target_Allocations[{"ASSET_A": 100, "CASH": 0}]'

        agents_mod.chat = spy
        u = make_synthetic_universe(n_days=140)
        solver = _Solver(u)
        portfolio_state = {"portfolio_value": 100000, "cash_pct": 100,
                            "allocations_pct": {a: 0 for a in u.anon_universe} | {"CASH": 100}}

        # baseline: exactly 1 Solver call, 0 Debater calls of any kind
        decision_fn = harnesses.make_baseline(u, solver)
        decision_fn("2024-06-03", portfolio_state, {}, 20)
        assert calls == {"solver": 1, "debater_intra": 0, "debater_reflect": 0}, calls

        # memory_only: reflection runs on the SECOND task (needs decision_log history),
        # but intra_task_review must never be called regardless of history
        calls = {"solver": 0, "debater_intra": 0, "debater_reflect": 0}
        with tempfile.TemporaryDirectory() as tmp:
            memory = ExperienceMemory()
            decision_fn = harnesses.make_memory_only(
                u, solver, _Debater(), _Consolidator(), memory, os.path.join(tmp, "mem.txt")
            )
            decision_log = {}
            raw1, meta1 = decision_fn("2024-06-03", portfolio_state, decision_log, 20)
            decision_log["2024-06-03"] = {"targets": raw1, "close_prices": u.close_prices("2024-06-03"), "meta": meta1,
                                            "screener": u.get_market_screener("2024-06-03")}
            decision_fn("2024-07-01", portfolio_state, decision_log, 20)
        assert calls["debater_intra"] == 0, "memory_only must never call the pre-execution debate"
        assert calls["debater_reflect"] == 1, "memory_only must still reflect on the prior task"
        assert calls["solver"] == 2
    finally:
        agents_mod.chat = original_chat
    print("[harnesses] baseline (zero Debater calls) and memory_only (reflection yes, debate no) both confirmed")


if __name__ == "__main__":
    test_memory_txt_roundtrip_and_hand_edit()
    test_dual_permanent_revision_cycle()
    test_monthly_prompt_style_and_conviction_harness()
    test_context_truncation()
    test_self_correcting_nudges()
    test_common_trading_days_intersection()
    test_all_zero_falls_back_to_full_cash()
    test_reflection_receives_decision_time_screener()
    test_consolidator_prompt_requires_specificity()
    test_debater_and_consolidator_are_symmetric()
    test_bullet_checks_ground_truth_scoring()
    test_baseline_and_memory_only_isolation()
    for name in ["intra", "inter", "dual", "dual_session", "dual_permanent"]:
        run_one(name)
    run_one("dual_permanent", use_risk_harness=True)
    run_one("dual_permanent", use_risk_harness=True, adaptive_risk=True)
    # yesharness.py-style config: no screener tool + SimpleTrailingRiskHarness
    run_one("dual_permanent", use_risk_harness=True, risk_harness_cls=SimpleTrailingRiskHarness,
            include_screener_tool=False)
    run_monthly_one()
    print("SMOKE TEST PASSED")
