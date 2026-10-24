"""Offline tests for shared fixes and run accounting.
Run: python -m ace_harness.tests.test_core
"""
import json
import os
import re
import tempfile
import types

from ace_harness.tests import golden_util as gu

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden.json")


def test_golden_paths_unchanged():
    """HARNESS/DEBATER/MEMORY/DUAL (with and without RiskTuner) must reproduce
    the fingerprints recorded at the unmodified commit, so their existing
    result rows stay valid wherever the asset-label fix does not apply."""
    with open(_GOLDEN) as f:
        golden = json.load(f)
    assert gu.prompt_hashes() == golden["prompts"], "a shared prompt text changed"
    for system, expected in golden["systems"].items():
        got = gu.run_golden_system(system)
        assert got == expected, f"{system} changed behaviour:\n{got}\nvs\n{expected}"
    print(f"[golden] {len(golden['systems'])} baseline paths + prompts identical to "
          f"{golden['recorded_at_commit'][:8]}")


def test_labels_alphanumeric_and_40_asset_parse():
    import ace_harness.agents as agents_mod
    from ace_harness.agents import Solver
    from ace_harness.market import MarketUniverse

    u = gu.make_universe(tickers=[f"T{i:02d}" for i in range(40)], n_days=300)
    labels = u.anon_universe
    assert labels[:26] == [f"ASSET_{chr(65 + i)}" for i in range(26)], "first 26 labels must not change"
    assert labels[26:28] == ["ASSET_AA", "ASSET_AB"] and labels[-1] == "ASSET_AN"
    assert all(re.match(r"^ASSET_[A-Z]+$", a) for a in labels) and len(set(labels)) == 40

    reply = "Action: Target_Allocations[" + json.dumps({a: 2 for a in labels} | {"CASH": 20}) + "]"
    original = agents_mod.chat
    agents_mod.chat = lambda *a, **k: reply
    try:
        state = {"portfolio_value": 1e6, "cash_pct": 0.0, "allocations_pct": {a: 2.5 for a in labels} | {"CASH": 0}}
        alloc, _ = Solver(u, include_screener_tool=False).decide("2023-12-01", state, 20)
    finally:
        agents_mod.chat = original
    assert len(alloc) == 41 and abs(alloc["CASH"] - 20.0) < 1e-6, "all 40 assets + CASH must parse"
    MarketUniverse(["A", "B"])  # constructor still accepts small universes
    print("[market] labels past 26 assets are alphabetic and a 40-asset allocation parses fully")


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens, self.completion_tokens, self.total_tokens = p, c, p + c


class FakeOpenAI:
    """Stands in for the OpenAI client so llm_client.chat (and its token
    counters) run for real while replies come from a stub."""

    def __init__(self, reply_fn):
        def create(model, messages, temperature, max_tokens, stop):
            text = reply_fn(model, messages, temperature=temperature, max_tokens=max_tokens, stop=stop)
            p = sum(len(m["content"]) for m in messages) // 4
            msg = types.SimpleNamespace(content=text)
            return types.SimpleNamespace(usage=_Usage(p, len(text) // 4),
                                         choices=[types.SimpleNamespace(message=msg)])
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))


def with_real_chat(stub):
    """Context manager: route agents.chat through the real llm_client.chat
    backed by FakeOpenAI(stub), so token accounting is exercised."""
    import contextlib
    import ace_harness.agents as agents_mod
    import ace_harness.llm_client as llm

    @contextlib.contextmanager
    def cm():
        saved = (agents_mod.chat, llm._client)
        llm._client = FakeOpenAI(stub)
        agents_mod.chat = getattr(llm, "_unpatched_chat", llm.chat)
        llm.reset_token_counts()
        try:
            yield
        finally:
            agents_mod.chat, llm._client = saved
    return cm()


def test_role_classification():
    from ace_harness.llm_client import classify_role
    cases = {"You are an autonomous ReAct Portfolio Manager on x": "solver",
             "You are a portfolio Debater pressure-testing": "debater",
             "You are a portfolio Reflector analyzing": "reflector",
             "You are the Curator maintaining": "consolidator",
             "You are a Risk Tuner adjusting": "risk_tuner",
             "Something else": "other"}
    for system, role in cases.items():
        assert classify_role([{"role": "system", "content": system}]) == role, system
    print("[llm_client] per-role classification correct")


def test_decision_log_and_token_sidecar():
    from ace_harness import accounting, engine_monthly
    from ace_harness.risk_harness import ConvictionHarness

    u = gu.make_universe()
    stub = gu.HashChat(u.anon_universe)
    with with_real_chat(stub), tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "monthly_intra_convictionriskharness_results.json")
        paths = accounting.sidecar_paths(out)
        fn = gu.build_decision_fn("intra", u, tmp)
        fn = accounting.wrap_with_decision_log(fn, paths["decisions"], u, ConvictionHarness(u))
        results = engine_monthly.run_backtest(u, fn, "2024-01-02", "2024-12-31", 1e6, out, 20)
        tok = accounting.write_tokens_sidecar(out, fn.per_task)
        assert isinstance(json.load(open(out)), list), "results JSON must stay a list"
        loaded = accounting.load_tokens(out)
        assert loaded["totals"]["calls"] == len(stub.calls) > 0
        assert set(loaded["by_role"]) == {"solver", "debater"}
        assert sum(t["calls"] for t in loaded["per_task"]) == tok["totals"]["calls"]
        recs = accounting.read_decisions(paths["decisions"])
        assert len(recs) == 11 and all(r["rounds"] and "harnessed" in r["rounds"][0] for r in recs)
        # the logged pre-trade weights are the engine's exact inputs: post-trade
        # NAV on each rebalance day = pre-trade value minus the turnover fee
        nav = {r["date"]: r["portfolio_value"] for r in results}
        for rec in recs:
            turnover = sum(abs(rec["executed"].get(a, 0.0) - rec["pre_weights"][a]) / 100.0 for a in u.anon_universe)
            expected = rec["portfolio_value"] * (1 - turnover * engine_monthly.FEE_RATE)
            assert abs(nav[rec["date"]] - expected) < 0.02, (rec["date"], nav[rec["date"]], expected)
    print("[accounting] decision log (11 rebalances) + per-role token sidecar persisted; results stay a list")


def test_fresh_memory_guard():
    from ace_harness import accounting
    from ace_harness.memory import ExperienceMemory
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "monthly_dual_playbook.txt")
        m = ExperienceMemory()
        m.add_bullet("RISK LESSONS", "Lesson learned in December")
        m.save(path)
        mem, info = accounting.prepare_memory(path)
        assert len(mem.bullets) == 0 and not os.path.exists(path) and info["stale_files_moved"]
        source = os.path.join(tmp, "finished_2024_playbook.txt")
        m.save(source)
        mem, info = accounting.prepare_memory(path, init_from=source)
        assert len(mem.bullets) == 1 and info["init_memory_from"] and os.path.exists(source)
        m.save(path)
        try:
            accounting.prepare_memory(path, init_from=path)
            raise AssertionError("warm-starting from this run's own output path must be refused")
        except ValueError:
            pass
    print("[accounting] stale playbook always moved aside; --init_memory reads another run, never overwrites it")


def test_make_table_merges_new_cells_into_base():
    """New cells replace base rows; other base rows are kept verbatim; ReAct
    dict outputs and ace list outputs are both read; tokens come from sidecars."""
    from ace_harness import accounting, engine_monthly, make_table
    from ace_harness.compare_results import metrics

    base_text = (
        "RUN 1\n" + "\t".join(make_table.HEADER) + "\n"
        "qwen25\tQWEN\ttech18\t35.69\t35.83\t21.41\t1.54\t1.64\t2.13\t-16.85\t\t\t\t\n"
        "qwen25\tHARNESS+DEBATER\tsp30\t53.58\t53.84\t24.61\t1.88\t1.92\t3.23\t-16.65\t131\t299597\t23551\t323148\n"
        "qwen25\tADAP\ttech18\t38.01\t38.18\t26.87\t1.34\t1.31\t2\t-19.1\t137\t286000\t24479\t310479\n"
    )
    u = gu.make_universe()
    with with_real_chat(gu.HashChat(u.anon_universe)), tempfile.TemporaryDirectory() as tmp:
        cell = os.path.join(tmp, "run1", "sp30", "debater")
        os.makedirs(cell)
        out = os.path.join(cell, "monthly_intra_convictionriskharness_results.json")
        fn = accounting.wrap_with_decision_log(gu.build_decision_fn("intra", u, tmp),
                                               accounting.sidecar_paths(out)["decisions"], u)
        results = engine_monthly.run_backtest(u, fn, "2024-01-02", "2024-12-31", 1e6, out, 20)
        accounting.write_tokens_sidecar(out, fn.per_task)
        react_cell = os.path.join(tmp, "run1", "sp30", "qwen")
        os.makedirs(react_cell)
        with open(os.path.join(react_cell, "qwen_results_raw.json"), "w") as f:
            json.dump({"summary": {}, "rebalances": [], "daily_nav": results}, f)
        base = os.path.join(tmp, "base.tsv")
        with open(base, "w") as f:
            f.write(base_text)
        rows, sources = make_table.build(os.path.join(tmp, "run1"), base=base)

    by_key = {(r[1], r[2]): r for r in rows}
    assert by_key[("QWEN", "tech18")][3] == "35.69", "untouched base rows are kept verbatim"
    assert by_key[("ADAP", "tech18")][3] == "38.01"
    new = by_key[("HARNESS+DEBATER", "sp30")]
    m = metrics(results)
    assert new[3] == str(round(m["total_return_pct"], 2)) and new[3] != "53.58", "rerun cell replaces base row"
    assert int(new[10]) > 0 and new[13], "token columns filled from the sidecar"
    assert by_key[("QWEN", "sp30")][3] == new[3] and by_key[("QWEN", "sp30")][10] == "", "ReAct dict read; no tokens"
    assert [r[2] for r in rows] == ["tech18", "tech18", "sp30", "sp30"], "canonical universe order"
    assert sum(1 for s, _ in sources if s == "NEW") == 2

    # single rounding from raw values (compare_results.metrics pre-rounds to 3 dp)
    raw = make_table.raw_metrics(results)
    assert new[6] == str(round(raw["sharpe"], 2)) and new[8] == str(round(raw["calmar"], 2))
    print("[make_table] RUN 1 table: new cells replace base rows, others kept; ReAct + ace outputs read")


def test_make_table_base_blocks_and_expect():
    from ace_harness import make_table
    header = "\t".join(make_table.HEADER)
    text = ("RUN 1\r\n" + header + "\r\n" + "qwen25\tQWEN\ttech18\t1\t1\t1\t1\t1\t1\t-1\t\t\t\t\r\n"
            + "qwen36\tQWEN\ttech18\t9\t9\t9\t9\t9\t9\t-9\t\t\t\t\r\n"
            + "\r\nRUN 2\r\n" + header + "\r\n" + "qwen25\tQWEN\ttech18\t2\t2\t2\t2\t2\t2\t-2\t\t\t\t\r\n")
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "sheet.tsv")
        with open(base, "w", newline="") as f:
            f.write(text)
        rows = make_table.read_base(base, "qwen25")
        assert rows[("QWEN", "tech18")][3] == "1", "only RUN 1 block and only this model are read"
        os.makedirs(os.path.join(tmp, "run1"))
        try:
            make_table.build(os.path.join(tmp, "run1"), base=base, expect=["sp30:qwen"])
            raise AssertionError("a missing expected cell must raise")
        except RuntimeError:
            pass
    print("[make_table] reads only the first block of a multi-run sheet; missing expected cells raise")





def run_all():
    test_golden_paths_unchanged()
    test_labels_alphanumeric_and_40_asset_parse()
    test_role_classification()
    test_decision_log_and_token_sidecar()
    test_fresh_memory_guard()
    test_make_table_merges_new_cells_into_base()
    test_make_table_base_blocks_and_expect()
    print("CORE TESTS PASSED")


if __name__ == "__main__":
    run_all()
