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


class CappedFakeOpenAI:
    """FakeOpenAI that stops a reply at max_tokens, as vLLM does: the text is
    cut, completion_tokens equals max_tokens and finish_reason is "length"."""

    def __init__(self, reply_fn):
        def create(model, messages, temperature, max_tokens, stop):
            text = reply_fn(model, messages, temperature=temperature, max_tokens=max_tokens, stop=stop)
            finish = "stop"
            if len(text) // 4 > max_tokens:
                text, finish = text[:max_tokens * 4], "length"
            p = sum(len(m["content"]) for m in messages) // 4
            choice = types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason=finish)
            return types.SimpleNamespace(usage=_Usage(p, len(text) // 4), choices=[choice])
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))


def with_real_chat(stub, client=FakeOpenAI):
    """Context manager: route agents.chat through the real llm_client.chat
    backed by client(stub), so token accounting is exercised."""
    import contextlib
    import ace_harness.agents as agents_mod
    import ace_harness.llm_client as llm

    @contextlib.contextmanager
    def cm():
        saved = (agents_mod.chat, llm._client)
        llm._client = client(stub)
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


def test_audit_run():
    """audit_run finds what the SLURM log cannot show: Reflector replies cut
    off at their output cap (and the Curator calls that therefore never
    happen), unparseable or oddly cased critic verdicts, a cell whose
    backtest is shorter than requested, and data_hash drift across runs."""
    import inspect
    import shutil
    import ace_harness.agents as agents_mod
    from ace_harness import audit_run, make_table, run_monthly

    src = inspect.getsource(agents_mod)
    assert agents_mod.REFLECTOR_MAX_TOKENS == 2048 and "max_tokens=REFLECTOR_MAX_TOKENS)" in src
    assert "temperature=0.0, max_tokens=800)\n        return _extract_json_array" in src and audit_run.CURATOR_CAP == 800
    assert list(audit_run.UNIVERSES) == make_table.UNIVERSES
    assert list(audit_run.SYSTEMS) == [s for s, _ in make_table.SYSTEMS]
    assert "ace_harness" not in "".join(l for l in inspect.getsource(audit_run).splitlines(True)
                                        if l.startswith(("import", "from"))), "audit_run must stay stdlib-only"
    assert audit_run._is_equal_weight({"A": 50.0, "B": 50.0, "CASH": 0.0}, 2)
    assert not audit_run._is_equal_weight({"A": 50.0, "B": 50.0, "CASH": 0.0}, 3)
    assert not audit_run._is_equal_weight({"A": 45.0, "B": 45.0, "CASH": 10.0}, 2)
    from ace_harness.memory import ExperienceMemory
    m = ExperienceMemory()
    m.add_bullet("RISK LESSONS", "Cap at 5% when 1M-Vol > 30%")
    m.add_bullet("CONVICTION SIGNALS", "Size up when 6M-Mom > 20%")
    with tempfile.TemporaryDirectory() as tmp:
        m.save(os.path.join(tmp, "pb.txt"))
        with open(os.path.join(tmp, "pb.txt")) as f:
            assert sum(bool(audit_run._BULLET_RE.match(line)) for line in f) == 2, "playbook bullet count"

    u = gu.make_universe()
    base_stub = gu.HashChat(u.anon_universe)

    def long_reflector(model, messages, **kw):  # a Reflector reply longer than its output budget
        text = base_stub(model, messages, **kw)
        if messages[0]["content"].startswith("You are a portfolio Reflector"):
            out = json.loads(text)
            out["lessons"] = [{"lesson": "When 1M-Vol > 30% cut to 4% " + "x" * 9000, "confidence": "high"}]
            return json.dumps(out)
        return text

    def odd_critic(bits, system):  # half the replies have no JSON, half say "Accept"
        return "no json here" if bits % 2 else {"verdict": "Accept", "feedback": "fine", "lessons": []}

    saved = run_monthly.build_universe
    run_monthly.build_universe = lambda tickers, start, end: gu.make_universe()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run1 = os.path.join(tmp, "run1")
            def cell(system):
                path = os.path.join(run1, "tech18", system)
                os.makedirs(path, exist_ok=True)
                return path
            with with_real_chat(long_reflector, client=CappedFakeOpenAI):
                run_monthly.run_system("memory_only", list("ABCDE"), "2024-01-02", "2024-12-31", cell("memory"),
                                       adaptive_risk=False)
                run_monthly.run_system("adap_v2", list("ABCDE"), "2024-01-02", "2024-12-31", cell("adap"),
                                       adaptive_risk=False)
            with with_real_chat(gu.HashChat(u.anon_universe, debater_override=odd_critic)):
                run_monthly.run_system("intra", list("ABCDE"), "2024-01-02", "2024-12-31", cell("debater"))
            with open(os.path.join(cell("qwen"), "qwen_results_raw.json"), "w") as f:
                json.dump({"daily_nav": [{"date": "2024-01-02", "portfolio_value": 1.0},
                                         {"date": "2024-06-28", "portfolio_value": 1.1}]}, f)
            with open(os.path.join(cell("qwen"), "PROVENANCE"), "w") as f:
                f.write("commit=abc123 script=qwen_port.py model=m date=2026-10-25T00:00:00Z\n")

            run = audit_run.audit_run(run1)
            by = {c["system"]: c for c in run["cells"]}
            assert [c["system"] for c in run["cells"]] == ["qwen", "memory", "debater", "adap"], "table order"
            assert run["window"] == ("2024-01-02", "2024-12-31")
            mem = by["memory"]
            assert mem["status"] == "complete" and mem["tasks"] == 11
            assert mem["reflector"]["calls"] == mem["reflector"]["at_cap"] == mem["reflector"]["unparsed"] == 10, \
                "every cut-off (and therefore unparsed) reply is counted"
            assert mem["truncation_counted"] and mem["truncated"] == {"reflector": 10}
            assert mem["curator"]["calls"] == 0, "no lessons survive, so the Curator never runs"
            assert mem["memory_size"] == 0 and any("Reflector replies stopped" in f for f in mem["flags"])
            deb = by["debater"]
            assert deb["critic"]["parse_fail"] == deb["critic"]["parse_fail_read_as_accept"] > 0
            assert deb["critic"]["verdicts"]["Accept"] > 0 and deb["critic"]["accept"] == 0
            assert any("Accept" in f for f in deb["flags"]) and any("did not parse" in f for f in deb["flags"])
            adap = by["adap"]
            assert adap["gov"]["tasks"] == 11 and sum(adap["gov"]["stops"].values()) == 11
            assert any("covers 2024-01-02..2024-06-28" in f for f in by["qwen"]["flags"])
            assert by["qwen"]["git_sha"] == "abc123" and by["qwen"]["kind"] == "react"

            run2 = os.path.join(tmp, "run2")
            shutil.copytree(run1, run2)
            # a run from before the cut-off counters: fall back to completion == a known cap
            tok_path = glob_one(os.path.join(run2, "tech18", "memory", "*_tokens.json"))
            tok = json.load(open(tok_path))
            tok["totals"].pop("truncated")
            for task in tok["per_task"]:
                for counts in task["by_role"].values():
                    counts.pop("truncated"), counts.pop("unparsed")
                if "reflector" in task["by_role"]:
                    task["by_role"]["reflector"]["completion"] = 550
            json.dump(tok, open(tok_path, "w"))
            legacy = {c["system"]: c for c in audit_run.audit_run(run2)["cells"]}["memory"]
            assert not legacy["truncation_counted"] and legacy["reflector"]["at_cap"] == 10
            man_path = glob_one(os.path.join(run2, "tech18", "debater", "*_manifest.json"))
            man = json.load(open(man_path))
            man["data_hash"] = "f" * 64
            json.dump(man, open(man_path, "w"))
            text, flagged = audit_run.render([run, audit_run.audit_run(run2)])
            assert flagged and "FLAGS\n" in text and "Reflector replies stopped" in text
            assert "DIFFERS within the run" in text and "differs across runs" in text
    finally:
        run_monthly.build_universe = saved
    print("[audit_run] cut-off Reflector replies, skipped Curator calls, odd/unparsed verdicts, short "
          "windows and data_hash drift are all flagged")


def test_preflight():
    import pandas as pd
    from ace_harness import preflight
    days = pd.bdate_range("2024-01-02", "2025-12-31")

    def fake_download(ticker, start, end):
        assert (start, end) == ("2024-01-02", "2026-01-01"), "365-day lookback, end inclusive"
        if ticker == "BAD":
            raise ValueError("no data")
        idx = days[days < "2025-07-16"] if ticker == "SHORT" else days
        close = pd.Series(100.0, index=idx)
        if ticker == "JUMP":
            close[idx >= "2025-03-03"] = 50.0
        return pd.DataFrame({"Close": close})

    lines, problems = preflight.check("2025-01-01", "2025-12-31", download=fake_download,
                                      universes={"u1": ["OK", "JUMP"], "u2": ["OK", "SHORT"], "u3": ["BAD"]})
    text = "\n".join(lines)
    assert any(p.startswith("SHORT: ends 2025-07-15") for p in problems)
    assert any(p.startswith("BAD: download failed") for p in problems) and len(problems) == 2
    assert "('2025-03-03', -50.0)" in text and "u2             common trading days=140 2025-01-01..2025-07-15" in text
    assert "u3             missing ['BAD']" in text
    print("[preflight] failed downloads, short tickers and big one-day moves are reported")


def test_price_cache():
    """With ACE_PRICE_CACHE set, the ACE engine and all three ReAct baselines
    download each ticker once per campaign and every later cell reads the
    identical frame; without it they download as before."""
    import importlib.util
    import pandas as pd
    from ace_harness import accounting, market, price_cache

    days = pd.bdate_range("2023-01-03", "2024-12-31")
    calls = []

    def fake_yf_download(ticker, start=None, end=None, **kw):
        calls.append((ticker, start, end))
        noise = len(calls) * 1e-5  # Yahoo's adjusted prices differ slightly on every request
        close = pd.Series([100.0 + noise + i for i in range(len(days))], index=days)
        return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0})

    saved_env, saved_yf = os.environ.get(price_cache.ENV), market.yf.download
    market.yf.download = fake_yf_download
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ.pop(price_cache.ENV, None)
            u1, u2 = market.MarketUniverse(["AAA", "BBB"]), market.MarketUniverse(["AAA", "BBB"])
            u1.prefetch("2024-01-02", "2024-12-31")
            u2.prefetch("2024-01-02", "2024-12-31")
            assert len(calls) == 4 and accounting.data_hash(u1) != accounting.data_hash(u2), "no cache: as before"

            os.environ[price_cache.ENV] = os.path.join(tmp, "price_cache")
            calls.clear()
            u1, u2 = market.MarketUniverse(["AAA", "BBB"]), market.MarketUniverse(["AAA", "BBB"])
            u1.prefetch("2024-01-02", "2024-12-31")
            u2.prefetch("2024-01-02", "2024-12-31")
            assert len(calls) == 2 and accounting.data_hash(u1) == accounting.data_hash(u2), "one download, same data"
            for anon in u1.anon_universe:
                pd.testing.assert_frame_equal(u1.data_cache[anon], u2.data_cache[anon])

            react_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ReAct")
            for script, start in (("qwen_port", "2024-01-02"), ("noharness", "2024-01-02"),
                                  ("yesharnessgpt", "2023-01-02")):
                frames = []
                for _ in range(2):
                    spec = importlib.util.spec_from_file_location(f"_react_{script}",
                                                                  os.path.join(react_dir, f"{script}.py"))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    mod.yf.download = fake_yf_download  # same yfinance module object as market.yf
                    mod.setup_universe(["AAA", "BBB"])
                    calls.clear()
                    mod.prefetch_data("2024-01-02", "2024-12-31")
                    frames.append((len(calls), dict(mod.GLOBAL_DATA_CACHE)))
                (n_first, first), (n_second, second) = frames
                assert n_second == 0 and set(first) == set(second), (script, n_first, n_second)
                for anon in first:
                    pd.testing.assert_frame_equal(first[anon], second[anon])
                assert any(f.startswith(f"{script}__AAA__{start}__") for f in os.listdir(os.environ[price_cache.ENV]))
    finally:
        market.yf.download = saved_yf
        if saved_env is None:
            os.environ.pop(price_cache.ENV, None)
        else:
            os.environ[price_cache.ENV] = saved_env
    print("[price_cache] ACE + QWEN/REACT/HARNESS download once per campaign; later cells read identical frames")


def glob_one(pattern):
    import glob
    matches = glob.glob(pattern)
    assert len(matches) == 1, (pattern, matches)
    return matches[0]





def test_summarize_runs():
    from ace_harness import make_table, summarize_runs
    header = "\t".join(make_table.HEADER)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for k, (a, b) in enumerate([(10, 20), (12, 22), (14, 30)]):
            path = os.path.join(tmp, f"run{k + 1}.tsv")
            with open(path, "w") as f:
                f.write(f"RUN {k + 1}\n{header}\n"
                        f"qwen25\tADAP\ttech18\t{a}\t1\t1\t1\t1\t1\t-1\t100\t\t\t1000\n"
                        f"qwen25\tADAP\tmag7\t{b}\t1\t1\t1\t1\t1\t-1\t200\t\t\t3000\n")
            paths.append(path)
        per_universe, cross = summarize_runs.summarize([summarize_runs.read_table(p) for p in paths])
    col = cross[0].index("TotalRet% mean")
    assert cross[1][0] == "ADAP" and cross[1][col] == "18.0"            # mean of 15, 17, 22
    assert cross[1][cross[0].index("TotalRet% sd")] == str(round(3.605551275, 2))
    assert per_universe[1][per_universe[0].index("TotalRet% mean")] == "12.0"
    assert cross[1][cross[0].index("LLM_Calls mean")] == "150.0"
    print("[summarize_runs] per-universe and cross-universe mean/sd across runs computed correctly")


def run_all():
    test_golden_paths_unchanged()
    test_labels_alphanumeric_and_40_asset_parse()
    test_role_classification()
    test_decision_log_and_token_sidecar()
    test_fresh_memory_guard()
    test_make_table_merges_new_cells_into_base()
    test_make_table_base_blocks_and_expect()
    test_summarize_runs()
    test_audit_run()
    test_preflight()
    test_price_cache()
    print("CORE TESTS PASSED")


if __name__ == "__main__":
    run_all()
