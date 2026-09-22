"""
decision_fn(current_date, portfolio_state, decision_log, rebalance_days)
-> (raw_allocations, meta) — every structure here produces one of these,
which plugs straight into engine.run_backtest.

Six named systems, forming a deliberate ablation ladder — each adds
exactly one thing on top of the last, using the identical engine, Solver,
and risk harness code, so any difference between them is guaranteed to
come from that one addition and nothing else:

  baseline        — Solver only. No Debater, no memory, no revision —
                     just the raw ReAct proposal (optionally passed
                     through a risk harness). The zero-intervention
                     control: run this alongside any other system to see
                     what the Debater/Consolidator/Memory actually change,
                     using the exact same engine/solver/harness code
                     rather than a separate script that might differ in
                     other ways too.
  intra           — + Solver <-> Debater loop within a task. Still no
                     memory at all — isolates the Debater's effect alone.
  inter           — Experience Memory feeds Solver directly (no in-task
                     loop); Debater reflects post-hoc on realized outcomes,
                     Consolidator updates memory for the NEXT task.
  dual            — general/configurable dual-timescale: fast intra-task
                     debate loop (max_rounds, tunable) + slow inter-task
                     memory update, both wired to the same persistent memory.
  dual_session    — many debate rounds within ONE task, feeding a
                     day-scoped Experience Memory that grows across those
                     rounds — then is thrown away. Nothing survives past
                     one task; there is no "next day" memory at all.
  memory_only     — ABLATION CONTROL for dual_permanent: same persistent
                     memory + Consolidator + adaptive risk-tuning, but NO
                     pre-execution Debater review/revision at all
                     (alias for inter — see make_memory_only's docstring).
  dual_permanent  — exactly ONE Debater review per task, but that review
                     can trigger exactly ONE Solver revision before the
                     result is final (LLM -> Debater -> LLM(revised) ->
                     Result — not a loop-until-accept). Experience Memory
                     persists for the entire backtest, never cleared.
"""
from ace_harness.memory import ExperienceMemory


def _apply_bullet_checks(memory, bullet_checks):
    """Direct, evidence-grounded scoring: bullet_checks come from the
    Reflector comparing an EXISTING bullet's claim to what actually
    happened this period, not from a new lesson's text merely agreeing
    or disagreeing with it. update_counts() silently no-ops on an
    unrecognized id, so a hallucinated bullet id is harmless."""
    for check in (bullet_checks or []):
        if check["verdict"] == "confirmed":
            memory.update_counts(check["id"], helpful=1)
        elif check["verdict"] == "contradicted":
            memory.update_counts(check["id"], harmful=1)


def make_baseline(universe, solver):
    """The ablation control: the Solver's raw proposal, unmodified by any
    Debater review, memory context, or revision. Wrap with
    wrap_with_risk_harness if you want a risk harness applied (matching
    how every other system optionally gets one) — the point of this
    system is isolating the Debater/Consolidator/Memory's effect, not the
    risk harness's, so keep the risk harness setting the SAME across
    whatever you're comparing this against.
    """
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        raw_alloc, trace = solver.decide(current_date, portfolio_state, rebalance_days)
        return raw_alloc, {"trace": trace}
    return decision_fn


def make_intra_task(universe, solver, debater, max_rounds: int = 3):
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        screener = universe.get_market_screener(current_date)
        status_text = (f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                        f"Cash: {portfolio_state['cash_pct']:.1f}%")
        feedback = None
        raw_alloc, trace, rounds_log = None, None, []

        direction = None
        for r in range(1, max_rounds + 1):
            raw_alloc, trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                              feedback_text=feedback, direction=direction)
            review = debater.intra_task_review(current_date, screener, status_text, raw_alloc,
                                               round_num=r)
            rounds_log.append({"round": r, "allocations": raw_alloc, "review": review})
            if review["verdict"] == "accept":
                break
            feedback = review["feedback"]
            direction = review.get("direction")

        return raw_alloc, {"rounds": rounds_log}
    return decision_fn


def make_inter_task(universe, solver, debater, consolidator, memory, memory_path,
                     risk_harness=None, risk_tuner=None, risk_params_path=None):
    """risk_harness/risk_tuner are optional. If risk_harness is given, its
    output (not the Solver's raw allocation) is what gets returned/executed.
    If risk_tuner is ALSO given, the same post-task reflection that updates
    the text playbook also proposes bounded parameter deltas for the risk
    harness — i.e. memory tunes the risk layer, not just the Solver prompt.
    """
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        past_dates = sorted(decision_log.keys())
        if past_dates:
            prev_date = past_dates[-1]
            prev = decision_log[prev_date]
            # Use the playbook snapshot the Solver actually saw at decision time,
            # so bullet grading in post_task_reflect references the correct bullets.
            decision_playbook = (prev.get("meta") or {}).get("playbook_snapshot") or memory.format_for_prompt()
            realized_prices = universe.close_prices(current_date)
            reflection = debater.post_task_reflect(
                prev_date, current_date, prev["targets"], prev["close_prices"], realized_prices,
                playbook_text=decision_playbook, decision_screener_text=prev.get("screener"),
            )
            _apply_bullet_checks(memory, reflection.get("bullet_checks"))
            ops = consolidator.consolidate(memory, reflection["lessons"], memory.sections)
            memory.apply_delta_ops(ops)
            memory.save(memory_path)

            if risk_harness is not None and risk_tuner is not None:
                deltas = risk_tuner.propose_adjustments(
                    risk_harness.get_params(), risk_harness.get_param_bounds(),
                    reflection["lessons"], reflection["realized_return_pct"],
                )
                risk_harness.update_params(deltas)
                if risk_params_path:
                    risk_harness.save_params(risk_params_path)

        playbook_text = memory.format_for_prompt()
        raw_alloc, trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                          playbook_text=playbook_text)
        meta = {"trace": trace, "playbook_snapshot": playbook_text}

        if risk_harness is not None:
            final_alloc = risk_harness.apply(raw_alloc, current_date, portfolio_state["portfolio_value"])
            meta["pre_harness_allocations"] = raw_alloc
            meta["risk_params"] = risk_harness.get_params()
            return final_alloc, meta

        return raw_alloc, meta
    return decision_fn


def make_dual_timescale(universe, solver, debater, consolidator, memory, memory_path, max_rounds: int = 3,
                         risk_harness=None, risk_tuner=None, risk_params_path=None):
    """Same risk_harness/risk_tuner behavior as make_inter_task, layered on
    top of the dual-timescale (fast intra-task + slow inter-task) structure.
    Risk-parameter tuning is deliberately driven only by the slow,
    realized-outcome reflection (not the same-day intra-task debate) —
    a quantitative knob should move on evidence, not on a single
    unverified round of argument.
    """
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        past_dates = sorted(decision_log.keys())
        if past_dates:
            prev_date = past_dates[-1]
            prev = decision_log[prev_date]
            decision_playbook = (prev.get("meta") or {}).get("playbook_snapshot") or memory.format_for_prompt()
            realized_prices = universe.close_prices(current_date)
            reflection = debater.post_task_reflect(
                prev_date, current_date, prev["targets"], prev["close_prices"], realized_prices,
                playbook_text=decision_playbook, decision_screener_text=prev.get("screener"),
            )
            _apply_bullet_checks(memory, reflection.get("bullet_checks"))
            ops = consolidator.consolidate(memory, reflection["lessons"], memory.sections)
            memory.apply_delta_ops(ops)

            if risk_harness is not None and risk_tuner is not None:
                deltas = risk_tuner.propose_adjustments(
                    risk_harness.get_params(), risk_harness.get_param_bounds(),
                    reflection["lessons"], reflection["realized_return_pct"],
                )
                risk_harness.update_params(deltas)
                if risk_params_path:
                    risk_harness.save_params(risk_params_path)

        screener = universe.get_market_screener(current_date)
        status_text = (f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                        f"Cash: {portfolio_state['cash_pct']:.1f}%")
        playbook_text = memory.format_for_prompt()

        feedback = None
        direction = None
        raw_alloc, trace, rounds_log, all_lessons = None, None, [], []

        for r in range(1, max_rounds + 1):
            raw_alloc, trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                              playbook_text=playbook_text, feedback_text=feedback,
                                              direction=direction)
            review = debater.intra_task_review(current_date, screener, status_text, raw_alloc,
                                                playbook_text=playbook_text, round_num=r)
            rounds_log.append({"round": r, "allocations": raw_alloc, "review": review})
            all_lessons.extend(review.get("lessons", []))
            if review["verdict"] == "accept":
                break
            feedback = review["feedback"]
            direction = review.get("direction")

        ops = consolidator.consolidate(memory, all_lessons, memory.sections)
        memory.apply_delta_ops(ops)
        memory.save(memory_path)

        meta = {"rounds": rounds_log, "playbook_snapshot": playbook_text}
        if risk_harness is not None:
            final_alloc = risk_harness.apply(raw_alloc, current_date, portfolio_state["portfolio_value"])
            meta["pre_harness_allocations"] = raw_alloc
            meta["risk_params"] = risk_harness.get_params()
            return final_alloc, meta

        return raw_alloc, meta
    return decision_fn


def make_dual_session(universe, solver, debater, consolidator, max_rounds: int = 5, risk_harness=None):
    """SYSTEM A — session-scoped dual-timescale.

    Many debate rounds happen within a single task: each round's lessons
    are consolidated into a day-scoped Experience Memory that later rounds
    *within that same task* can read back (so round 3 can build on what
    rounds 1-2 already established) — but that memory is a fresh,
    throwaway object created at the top of every decision_fn call, so
    nothing survives to the next rebalance date. Two timescales exist
    (round-by-round revision, and consolidated memory informing later
    rounds) but both are confined to one task; there is no persistent
    memory file, and no post-task reflection, by design.

    risk_harness is optional and applied statically (no adaptive tuning —
    there's no persistent memory here to drive it with).
    """
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        session_memory = ExperienceMemory()  # fresh every task; discarded when this call returns

        screener = universe.get_market_screener(current_date)
        status_text = (f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                        f"Cash: {portfolio_state['cash_pct']:.1f}%")
        feedback = None
        direction = None
        raw_alloc, trace, rounds_log = None, None, []

        for r in range(1, max_rounds + 1):
            playbook_text = session_memory.format_for_prompt()
            raw_alloc, trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                              playbook_text=playbook_text, feedback_text=feedback,
                                              direction=direction)
            review = debater.intra_task_review(current_date, screener, status_text, raw_alloc,
                                                playbook_text=playbook_text, round_num=r)
            rounds_log.append({"round": r, "allocations": raw_alloc, "review": review})

            ops = consolidator.consolidate(session_memory, review.get("lessons", []), session_memory.sections)
            session_memory.apply_delta_ops(ops)

            if review["verdict"] == "accept":
                break
            feedback = review["feedback"]
            direction = review.get("direction")

        meta = {"rounds": rounds_log, "session_bullets_at_end": len(session_memory.bullets)}
        if risk_harness is not None:
            final_alloc = risk_harness.apply(raw_alloc, current_date, portfolio_state["portfolio_value"])
            meta["pre_harness_allocations"] = raw_alloc
            return final_alloc, meta

        return raw_alloc, meta
    return decision_fn


def make_memory_only(universe, solver, debater, consolidator, memory, memory_path,
                      risk_harness=None, risk_tuner=None, risk_params_path=None):
    """ABLATION CONTROL for dual_permanent: the exact same persistent
    memory, Consolidator, and (if given) adaptive risk-tuning — but with
    NO pre-execution Debater pressure-test/revision at all. The Solver
    proposes once, informed by memory, and that's final.

    This is functionally identical to make_inter_task — "inter" already
    IS "dual_permanent minus the debate step" — this is just an explicit,
    clearly-named alias so the ablation relationship to dual_permanent is
    obvious without having to know that already. Reflection
    (post_task_reflect) still runs and still updates memory from realized
    outcomes: that's "the memory system," not "the debate," and isolating
    the debate's effect specifically means keeping memory's own update
    path intact while removing only the pre-execution argument.

    Run this side by side with `baseline` (no memory, no debate) and
    `dual_permanent` (memory + debate) for a 3-point ladder: does memory
    alone help, does adding the debate on top help further, hurt, or wash
    out relative to memory alone?
    """
    return make_inter_task(
        universe, solver, debater, consolidator, memory, memory_path,
        risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
    )


def make_dual_permanent(universe, solver, debater, consolidator, memory, memory_path,
                         risk_harness=None, risk_tuner=None, risk_params_path=None):
    """SYSTEM B — permanent-memory dual-timescale.

    Exactly the loop from the diagram: Solver proposes -> Debater reviews
    ONCE -> if the verdict is "revise", the Solver gets exactly ONE more
    turn informed by that feedback, and THAT becomes the final result (the
    revision is never re-reviewed — this is a single propose/debate/revise
    cycle, not a loop-until-accept). `memory` is the SAME ExperienceMemory
    object across the entire backtest — both the debate's own lessons and
    the post-task reflection on the PREVIOUS task's realized outcome feed
    the Consolidator, and nothing is ever cleared.
    """
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        # slow loop: reflect on the previous task's realized outcome first
        past_dates = sorted(decision_log.keys())
        if past_dates:
            prev_date = past_dates[-1]
            prev = decision_log[prev_date]
            decision_playbook = (prev.get("meta") or {}).get("playbook_snapshot") or memory.format_for_prompt()
            realized_prices = universe.close_prices(current_date)
            reflection = debater.post_task_reflect(
                prev_date, current_date, prev["targets"], prev["close_prices"], realized_prices,
                playbook_text=decision_playbook, decision_screener_text=prev.get("screener"),
            )
            _apply_bullet_checks(memory, reflection.get("bullet_checks"))
            ops = consolidator.consolidate(memory, reflection["lessons"], memory.sections)
            memory.apply_delta_ops(ops)
            # Save after slow-loop ops so reflection updates survive a fast-loop crash.
            memory.save(memory_path)

            if risk_harness is not None and risk_tuner is not None:
                deltas = risk_tuner.propose_adjustments(
                    risk_harness.get_params(), risk_harness.get_param_bounds(),
                    reflection["lessons"], reflection["realized_return_pct"],
                )
                risk_harness.update_params(deltas)
                if risk_params_path:
                    risk_harness.save_params(risk_params_path)

        # fast loop: exactly one propose -> debate -> revise cycle
        playbook_text = memory.format_for_prompt()
        screener = universe.get_market_screener(current_date)
        status_text = (f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                        f"Cash: {portfolio_state['cash_pct']:.1f}%")

        first_alloc, first_trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                                   playbook_text=playbook_text)
        review = debater.intra_task_review(current_date, screener, status_text, first_alloc,
                                            playbook_text=playbook_text, round_num=1)
        rounds_log = [{"round": 1, "allocations": first_alloc, "review": review}]

        all_lessons = list(review.get("lessons", []))
        # AdaReMo Algorithm 2: admitted gate (g_ref already controlled the loop;
        # now gate consolidation on admitted=True, matching policy.memory_decision()).
        admitted = review["verdict"] == "accept"
        # g_sto: critic's per-episode store decision (Algorithm 2 / AdaReMo).
        # Defaults True so ReMo-mode behavior (Algorithm 1: always consolidate if
        # admitted) is preserved when the model doesn't return this field.
        should_store = review.get("should_store", True)

        if admitted:
            final_alloc = first_alloc
        else:
            final_alloc, second_trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                                        playbook_text=playbook_text, feedback_text=review["feedback"],
                                                        direction=review.get("direction"))
            rounds_log.append({"round": 2, "allocations": final_alloc})

        # Consolidate fast-loop lessons only if: admitted AND should_store AND
        # memory is not saturated (mirrors policy.memory_decision("stored") path).
        consolidated = False
        if admitted and should_store and not memory.is_saturated():
            ops = consolidator.consolidate(memory, all_lessons, memory.sections)
            memory.apply_delta_ops(ops)
            consolidated = True
        memory.save(memory_path)

        meta = {
            "rounds": rounds_log,
            "playbook_snapshot": playbook_text,
            "admitted": admitted,
            "consolidated": consolidated,
            "memory_size": len(memory.bullets),
        }
        if risk_harness is not None:
            harnessed = risk_harness.apply(final_alloc, current_date, portfolio_state["portfolio_value"])
            meta["pre_harness_allocations"] = final_alloc
            meta["risk_params"] = risk_harness.get_params()
            return harnessed, meta

        return final_alloc, meta
    return decision_fn


def make_baseline_adamo(universe, solver, reward_model, risk_harness=None):
    """Harness + AdaReMo only — no memory, no debater.

    After each period, AdaReMo is updated with the realized per-asset returns
    and the executed allocation. From period min_samples onward, the Solver's
    prompt receives a realized-return signal: which assets actually delivered
    in this backtest, with average and most-recent monthly return shown, plus
    a Ridge-regression sizing weight. The signal is directive ("increase /
    reduce allocation") so the at-temperature-0 Solver uses it rather than
    treating it as decorative text.

    No lookahead: AdaReMo only sees data from periods already completed.
    """
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        past_dates = sorted(decision_log.keys())
        if past_dates:
            prev_date = past_dates[-1]
            prev = decision_log[prev_date]
            realized_prices = universe.close_prices(current_date)

            # Compute per-asset realized returns for the previous period
            per_asset_returns = {}
            for asset, entry_price in (prev.get("close_prices") or {}).items():
                exit_price = realized_prices.get(asset)
                if exit_price and entry_price:
                    per_asset_returns[asset] = round(
                        (exit_price - entry_price) / entry_price * 100.0, 2
                    )

            if per_asset_returns:
                reward_model.record(prev["targets"], per_asset_returns)
                reward_model.fit()

        reward_signal = reward_model.signal_text()

        raw_alloc, trace = solver.decide(
            current_date, portfolio_state, rebalance_days,
            playbook_text=reward_signal if reward_signal else None,
        )

        meta = {"trace": trace}
        if risk_harness is not None:
            final_alloc = risk_harness.apply(raw_alloc, current_date, portfolio_state["portfolio_value"])
            meta["pre_harness_allocations"] = raw_alloc
            meta["risk_params"] = risk_harness.get_params()
            return final_alloc, meta

        return raw_alloc, meta
    return decision_fn


def make_baseline_remo(universe, solver, reward_model, risk_harness=None):
    """Harness + ReMo (plain Reward Model) — no memory, no debater, no risk-adjustment.

    Identical flow to make_baseline_adamo but uses SimpleRewardModel, which
    ranks assets by raw realized returns without Sharpe/Sortino/drawdown scoring.
    Useful as a cleaner ablation between pure return-chasing and risk-adjusted guidance.
    """
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        past_dates = sorted(decision_log.keys())
        if past_dates:
            prev_date = past_dates[-1]
            prev = decision_log[prev_date]
            realized_prices = universe.close_prices(current_date)

            per_asset_returns = {}
            for asset, entry_price in (prev.get("close_prices") or {}).items():
                exit_price = realized_prices.get(asset)
                if exit_price and entry_price:
                    per_asset_returns[asset] = round(
                        (exit_price - entry_price) / entry_price * 100.0, 2
                    )

            if per_asset_returns:
                reward_model.record(prev["targets"], per_asset_returns)
                reward_model.fit()

        reward_signal = reward_model.signal_text()

        raw_alloc, trace = solver.decide(
            current_date, portfolio_state, rebalance_days,
            playbook_text=reward_signal if reward_signal else None,
        )

        meta = {"trace": trace}
        if risk_harness is not None:
            final_alloc = risk_harness.apply(raw_alloc, current_date, portfolio_state["portfolio_value"])
            meta["pre_harness_allocations"] = raw_alloc
            meta["risk_params"] = risk_harness.get_params()
            return final_alloc, meta

        return raw_alloc, meta
    return decision_fn


def make_dual_permanent_adamo(universe, solver, debater, consolidator, memory, memory_path,
                               reward_model, risk_harness=None, risk_tuner=None, risk_params_path=None):
    """dual_permanent + AdaReMo: identical flow to make_dual_permanent, with one
    addition — after each period the AdaptiveRewardModel is updated with the
    previous allocation and its realized weighted return, then its signal (a
    Ridge-regression prediction of which asset weights correlated with better
    returns historically) is appended to the Solver's context as a weak prior.
    No lookahead: the model only sees data from periods already completed.
    """
    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        past_dates = sorted(decision_log.keys())
        if past_dates:
            prev_date = past_dates[-1]
            prev = decision_log[prev_date]
            decision_playbook = (prev.get("meta") or {}).get("playbook_snapshot") or memory.format_for_prompt()
            realized_prices = universe.close_prices(current_date)
            reflection = debater.post_task_reflect(
                prev_date, current_date, prev["targets"], prev["close_prices"], realized_prices,
                playbook_text=decision_playbook, decision_screener_text=prev.get("screener"),
            )
            _apply_bullet_checks(memory, reflection.get("bullet_checks"))
            ops = consolidator.consolidate(memory, reflection["lessons"], memory.sections)
            memory.apply_delta_ops(ops)
            memory.save(memory_path)

            # Update AdaReMo with per-asset realized returns (computed from prices,
            # not the scalar portfolio return in reflection["realized_return_pct"])
            per_asset_returns = {}
            for asset, entry_price in (prev.get("close_prices") or {}).items():
                exit_price = realized_prices.get(asset)
                if exit_price and entry_price:
                    per_asset_returns[asset] = round(
                        (exit_price - entry_price) / entry_price * 100.0, 2
                    )
            if per_asset_returns:
                reward_model.record(prev["targets"], per_asset_returns)
                reward_model.fit()

            if risk_harness is not None and risk_tuner is not None:
                deltas = risk_tuner.propose_adjustments(
                    risk_harness.get_params(), risk_harness.get_param_bounds(),
                    reflection["lessons"], reflection["realized_return_pct"],
                )
                risk_harness.update_params(deltas)
                if risk_params_path:
                    risk_harness.save_params(risk_params_path)

        playbook_text = memory.format_for_prompt()
        reward_signal = reward_model.signal_text()
        combined_context = (
            f"{playbook_text}\n\n{reward_signal}" if playbook_text and reward_signal
            else playbook_text or reward_signal or None
        )

        screener = universe.get_market_screener(current_date)
        status_text = (f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                        f"Cash: {portfolio_state['cash_pct']:.1f}%")

        first_alloc, first_trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                                   playbook_text=combined_context)
        review = debater.intra_task_review(current_date, screener, status_text, first_alloc,
                                            playbook_text=playbook_text, round_num=1)
        rounds_log = [{"round": 1, "allocations": first_alloc, "review": review}]
        all_lessons = list(review.get("lessons", []))

        if review["verdict"] == "accept":
            final_alloc = first_alloc
        else:
            final_alloc, second_trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                                        playbook_text=combined_context, feedback_text=review["feedback"],
                                                        direction=review.get("direction"))
            rounds_log.append({"round": 2, "allocations": final_alloc})

        ops = consolidator.consolidate(memory, all_lessons, memory.sections)
        memory.apply_delta_ops(ops)
        memory.save(memory_path)

        meta = {"rounds": rounds_log, "playbook_snapshot": playbook_text}
        if risk_harness is not None:
            harnessed = risk_harness.apply(final_alloc, current_date, portfolio_state["portfolio_value"])
            meta["pre_harness_allocations"] = final_alloc
            meta["risk_params"] = risk_harness.get_params()
            return harnessed, meta

        return final_alloc, meta
    return decision_fn


def make_autogover(universe, solver, debater, consolidator, memory, memory_path,
                   max_rounds: int = 3,
                   risk_harness=None, risk_tuner=None, risk_params_path=None):
    """AutoGovern / AdaReMo Algorithm 2 — budget-governed dual-timescale system.

    This is NOT a mode router. Memory is ALWAYS on. The critic governs how
    many refinement rounds happen and whether to consolidate.

    Structure (mirrors peer's remo/agent.py ReMoAgent.run_task exactly):

      1. SLOW LOOP (inter-task): post_task_reflect on the previous month's
         realized prices → bullet_checks → Consolidator updates memory.
         Identical to make_dual_permanent's slow loop.

      2. FAST LOOP (intra-task, K=max_rounds): every round the Solver is
         conditioned on memory (memory is never switched off):
           - Solver (memory) → Debater review (also sees memory)
           - admitted  = (verdict == "accept")      → break, consolidate if g_sto
           - g_ref==False (no actionable fix)        → break, don't consolidate
           - else: pass feedback to next Solver round

      3. CONSOLIDATION gate (Algorithm 2 line 11):
           admitted AND g_sto AND NOT frozen
         where frozen is a STICKY flag (Algorithm 2 lines 1 + 15):
           frozen ← frozen ∨ SATURATED(M)
         Once True it never resets — matching the algorithm exactly.

    All existing systems (baseline, intra, inter, dual_permanent, …) are
    unchanged — this is an addition, not a replacement.
    """
    # Algorithm 2 line 1: frozen ← false (initialized once, persists across all tasks)
    _frozen = [False]  # mutable closure cell so decision_fn can update it

    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        # --- SLOW LOOP (inter-task) -----------------------------------------
        past_dates = sorted(decision_log.keys())
        if past_dates:
            prev_date = past_dates[-1]
            prev = decision_log[prev_date]
            decision_playbook = (prev.get("meta") or {}).get("playbook_snapshot") or memory.format_for_prompt()
            realized_prices = universe.close_prices(current_date)
            reflection = debater.post_task_reflect(
                prev_date, current_date, prev["targets"], prev["close_prices"], realized_prices,
                playbook_text=decision_playbook, decision_screener_text=prev.get("screener"),
            )
            _apply_bullet_checks(memory, reflection.get("bullet_checks"))
            # Gate slow-loop consolidation on frozen (Algorithm 2: "leave M unchanged"
            # when frozen). track_saturation=False keeps slow-loop calls out of
            # _consolidation_sizes so SATURATED(M) only reflects fast-loop learning.
            if not _frozen[0]:
                ops = consolidator.consolidate(memory, reflection["lessons"], memory.sections)
                memory.apply_delta_ops(ops, track_saturation=False)
            memory.save(memory_path)

            if risk_harness is not None and risk_tuner is not None:
                deltas = risk_tuner.propose_adjustments(
                    risk_harness.get_params(), risk_harness.get_param_bounds(),
                    reflection["lessons"], reflection.get("realized_return_pct", {}),
                )
                risk_harness.update_params(deltas)
                if risk_params_path:
                    risk_harness.save_params(risk_params_path)

        # --- FAST LOOP (intra-task, K rounds, memory always on) ---------------
        screener = universe.get_market_screener(current_date)
        status_text = (f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                       f"Cash: {portfolio_state['cash_pct']:.1f}%")
        playbook_text = memory.format_for_prompt()  # snapshot once; same for all rounds
        n_stocks = len(universe.anon_universe)

        feedback = None
        direction = None
        raw_alloc, rounds_log, all_lessons = None, [], []
        admitted = False
        g_sto = False

        print(f"[autogov] {current_date} fast-loop start  frozen={_frozen[0]}  mem_bullets={len(memory.bullets)}")
        for r in range(1, max_rounds + 1):
            # Solver ALWAYS receives the memory snapshot (never switched off)
            raw_alloc, _trace = solver.decide(current_date, portfolio_state, rebalance_days,
                                              playbook_text=playbook_text,
                                              feedback_text=feedback, direction=direction)
            review = debater.intra_task_review(current_date, screener, status_text, raw_alloc,
                                               playbook_text=playbook_text, round_num=r,
                                               n_universe_stocks=n_stocks)
            rounds_log.append({"round": r, "allocations": raw_alloc, "review": review})
            all_lessons.extend(review.get("lessons", []))

            admitted = review["verdict"] == "accept"
            g_ref = review.get("should_refine", True)   # continue loop?
            g_sto = review.get("should_store", False)   # worth consolidating?

            print(f"[autogov]   r{r}: verdict={review['verdict']}  dir={review.get('direction','')}  "
                  f"g_ref={g_ref}  g_sto={g_sto}  "
                  f"feedback={review.get('feedback','')[:80]!r}")

            if admitted:
                break  # accepted: done

            if not g_ref:
                break  # critic: no actionable fix left — stop without admission

            feedback = review["feedback"]
            direction = review.get("direction")

        print(f"[autogov]   result: admitted={admitted}  g_sto={g_sto}  frozen={_frozen[0]}")

        # --- CONSOLIDATION gate (Algorithm 2 line 11) -------------------------
        consolidated = False
        if admitted and g_sto and not _frozen[0]:
            ops = consolidator.consolidate(memory, all_lessons, memory.sections)
            memory.apply_delta_ops(ops)
            consolidated = True
        memory.save(memory_path)
        # Algorithm 2 line 15: frozen ← frozen ∨ SATURATED(M) — monotonic, never resets
        _frozen[0] = _frozen[0] or memory.is_saturated()

        meta = {
            "rounds": rounds_log,
            "playbook_snapshot": playbook_text,
            "admitted": admitted,
            "consolidated": consolidated,
            "memory_size": len(memory.bullets),
        }

        if risk_harness is not None:
            harnessed = risk_harness.apply(raw_alloc, current_date, portfolio_state["portfolio_value"])
            meta["pre_harness_allocations"] = raw_alloc
            meta["risk_params"] = risk_harness.get_params()
            return harnessed, meta

        return raw_alloc, meta
    return decision_fn


def make_adaptive_router(universe, system_fns: dict, router_fn=None):
    """Adaptive system selector: at each rebalance, computes observable
    universe-structure features and routes to the best-fit system.

    system_fns: dict mapping system name → decision_fn, e.g.
        {"debater": intra_fn, "ace": ace_fn, "memory": mem_fn, "harness": base_fn}
    router_fn: callable(universe, current_date) -> (system_name, reason, features)
        If None, uses the default rule-based router from adaptive_router.py.

    The system selection and features are logged in the returned meta dict so
    the per-period selection history can be read from the results JSON and
    visualized by plot_adaptive.py.
    """
    from ace_harness import adaptive_router as _ar

    def _default_router(universe, current_date):
        features = _ar.compute_features(universe, current_date)
        system, reason = _ar.route(features)
        return system, reason, features

    _router = router_fn if router_fn is not None else _default_router

    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        system_name, reason, features = _router(universe, current_date)
        fn = system_fns.get(system_name) or system_fns.get("harness")
        raw_alloc, meta = fn(current_date, portfolio_state, decision_log, rebalance_days)
        meta = dict(meta or {})
        meta["adaptive"] = {
            "system": system_name,
            "reason": reason,
            "features": features,
        }
        return raw_alloc, meta

    return decision_fn


def wrap_with_risk_harness(decision_fn, risk_harness):
    """Applies an optional risk-management layer (see risk_harness.py) on
    top of ANY decision_fn, so the self-improvement structure and the risk
    layer (on/off) are independent, separately-testable axes rather than
    baked together.
    """
    def wrapped(current_date, portfolio_state, decision_log, rebalance_days):
        raw_alloc, meta = decision_fn(current_date, portfolio_state, decision_log, rebalance_days)
        harnessed = risk_harness.apply(raw_alloc, current_date, portfolio_state["portfolio_value"])
        meta = dict(meta or {})
        meta["pre_harness_allocations"] = raw_alloc
        return harnessed, meta
    return wrapped
