"""ADAP-v2: AutoGovern with outcome-grounded governance (see governance.py).

Per monthly task:
  A. mode = open | frozen | probe              (FreezeController)
  B. slow loop, unless frozen: the Reflector grades last month's decision,
     the visible and probationary bullets, and the lessons the critic
     nominated last month, against realized returns; reflection lessons plus
     CONFIRMED nominations go to one Curator call.
  C. visible memory = bullets with realized evidence (or all, or none).
  D. fast loop, up to K rounds: Solver -> critic; stops on accept, critic
     g_ref, convergence, or the final round (no terminal review).
  E. execute the last proposal.
  F. storage, unless frozen: nominate lessons for next month's verification
     (or consolidate now, in the immediate_* ablations).
  G. demand -> freeze / probe / reopen.

Compared with legacy make_autogover (ADAP-v1, kept unchanged for
reproduction), this removes three non-governance differences from DUAL: the
round-1 submission whenever the critic did not accept, the ADAP-only critic
prompt, and the RiskTuner being always on.
"""
from ace_harness.governance import EvidenceStore, FreezeController, l1_pp, stage_lessons
from ace_harness.harnesses import _apply_bullet_checks


def make_autogover_v2(universe, solver, debater, consolidator, memory, memory_path, cfg,
                      evidence=None, evidence_path=None,
                      risk_harness=None, risk_tuner=None, risk_params_path=None, verbose=True):
    memory_on = cfg.memory == "on"
    gated = cfg.exposure_mode == "evidence_gated"
    deferred = cfg.storage_mode == "deferred_verified"
    freeze = (FreezeController(cfg.freeze_window, cfg.freeze_tau, cfg.probe_every)
              if memory_on and cfg.freeze_mode == "window_probe" else None)
    evidence = evidence if evidence is not None else EvidenceStore()
    tuner = risk_tuner if (cfg.use_risk_tuner and risk_harness is not None) else None
    n_stocks = len(universe.anon_universe) if cfg.debater_sizing_context else None
    pending = {"staged": []}  # lessons nominated last task, awaiting realized verification

    def consolidate(lessons, gov):
        # Like DUAL: the Curator makes no LLM call for an empty lesson list, but
        # apply_delta_ops still runs (it prunes bullets the checks just discredited).
        before = {bid: b["content"] for bid, b in memory.bullets.items()}
        ops = consolidator.consolidate(memory, lessons, memory.sections)
        added = memory.apply_delta_ops(ops)["added_ids"]
        evidence.sync(memory)
        rewritten = [bid for bid, b in memory.bullets.items() if bid in before and b["content"] != before[bid]]
        if gated:  # evidence belongs to the graded text: a rewrite goes back to probation
            evidence.reset_confirmations(rewritten)
        gov["n_new_bullets"] += len(added)
        gov["added_ids"] += added
        gov["rewritten_ids"] += rewritten
        return bool(lessons)

    def decision_fn(current_date, portfolio_state, decision_log, rebalance_days):
        mode = freeze.begin_task() if freeze else "open"
        frozen_before = freeze.frozen if freeze else False
        staged, pending["staged"] = pending["staged"], []
        gov = {"mode": mode, "n_staged_graded": 0, "n_confirmed": 0, "n_new_bullets": 0,
               "added_ids": [], "rewritten_ids": [], "slow_loop": False,
               "n_staged_dropped": len(staged) if mode == "frozen" else 0}  # frozen: memory stays unchanged
        consolidated = False

        # --- B. slow loop: realized-outcome reflection --------------------------
        past_dates = sorted(decision_log.keys())
        if memory_on and past_dates and mode != "frozen":
            prev_date = past_dates[-1]
            prev = decision_log[prev_date]
            snapshot = (prev.get("meta") or {}).get("playbook_snapshot")
            decision_playbook = snapshot if gated else (snapshot or memory.format_for_prompt())
            probation = evidence.probation_ids(memory, cfg.exposure_min_net, cfg.max_probation_shown) if gated else []
            reflection = debater.post_task_reflect(
                prev_date, current_date, prev["targets"], prev["close_prices"], universe.close_prices(current_date),
                playbook_text=decision_playbook, decision_screener_text=prev.get("screener"),
                candidate_lessons=staged if (deferred and staged) else None,
                probation_text=memory.format_bullets(probation, counts=evidence.counts_for(probation)),
                bullet_check_cap=cfg.reflect_bullet_check_cap if gated else None,
            )
            _apply_bullet_checks(memory, reflection.get("bullet_checks"))
            evidence.apply_checks(reflection.get("bullet_checks"), memory)
            checks = reflection.get("candidate_checks", []) if deferred else []
            confirmed_ids = {c["id"] for c in checks if c["verdict"] == "confirmed"}
            confirmed = [lesson for i, lesson in enumerate(staged) if f"C{i}" in confirmed_ids]
            gov.update(n_staged_graded=len(checks), n_confirmed=len(confirmed))
            consolidate(list(reflection["lessons"]) + confirmed, gov)
            consolidated = bool(confirmed)  # last task's critic lessons entered memory now
            gov["slow_loop"] = True

            if tuner is not None:
                deltas = tuner.propose_adjustments(risk_harness.get_params(), risk_harness.get_param_bounds(),
                                                   reflection["lessons"], reflection["realized_return_pct"])
                risk_harness.update_params(deltas)
                if risk_params_path:
                    risk_harness.save_params(risk_params_path)

        # --- C. what the Solver and critic see ---------------------------------
        if not memory_on:
            visible = None
        elif gated:
            ids = evidence.visible_ids(memory, cfg.exposure_min_net)
            visible = memory.format_bullets(ids, counts=evidence.counts_for(ids))
            gov["n_visible"] = len(ids)
        else:
            visible = memory.format_for_prompt()

        # --- D. fast loop: refinement with governed stops -----------------------
        screener = universe.get_market_screener(current_date)
        status_text = (f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                       f"Cash: {portfolio_state['cash_pct']:.1f}%")
        feedback = direction = None
        allocs, rounds_log, reviewed = [], [], []
        stop_reason, admitted = "exhausted", False
        for r in range(1, cfg.max_rounds + 1):
            alloc, _ = solver.decide(current_date, portfolio_state, rebalance_days, playbook_text=visible,
                                     feedback_text=feedback, direction=direction)
            allocs.append(alloc)
            if r >= 2 and cfg.convergence_eps_pp > 0 and l1_pp(allocs[-1], allocs[-2]) < cfg.convergence_eps_pp:
                rounds_log.append({"round": r, "allocations": alloc})
                stop_reason = "converged"
                break
            if r == cfg.max_rounds and cfg.skip_terminal_review and cfg.submit_policy == "last":
                rounds_log.append({"round": r, "allocations": alloc})
                stop_reason = "terminal"
                break
            review = debater.intra_task_review(current_date, screener, status_text, alloc,
                                               playbook_text=visible, round_num=r, n_universe_stocks=n_stocks)
            rounds_log.append({"round": r, "allocations": alloc, "review": review})
            reviewed.append(review)
            if review["verdict_norm"] == "accept":
                admitted = review["parse_ok"]
                stop_reason = "accept" if admitted else "parse_fail"
                break
            if cfg.use_critic_gref and review["parse_ok"] and not review["refine_signal"]:
                stop_reason = "g_ref"
                break
            feedback, direction = review["feedback"], review.get("direction")

        # --- E. submission -------------------------------------------------------
        submitted_round = len(allocs)
        if cfg.submit_policy == "r1_unless_admitted" and not admitted:
            submitted_round = 1
        final_alloc = allocs[submitted_round - 1]

        # --- F. storage ------------------------------------------------------------
        if memory_on and mode != "frozen":
            if cfg.storage_mode == "deferred_verified":
                pending["staged"] = stage_lessons(reviewed, cfg, admitted)
            else:
                all_lessons = [lesson for rv in reviewed for lesson in rv.get("lessons", [])]
                if cfg.storage_mode == "immediate_all":        # DUAL
                    lessons = all_lessons
                elif cfg.storage_mode == "immediate_admitted":  # ReMo Algorithm 1
                    lessons = all_lessons if admitted else None
                else:                                           # immediate_gsto
                    lessons = stage_lessons(reviewed, cfg, admitted)
                if lessons is not None:
                    consolidated = consolidate(lessons, gov) or consolidated
        if memory_on:
            memory.save(memory_path)
            if evidence_path:
                evidence.save(evidence_path)

        # --- G. demand -> freeze / probe / reopen ----------------------------------
        if cfg.demand_signal == "gsto":
            parsed_reviews = [rv for rv in reviewed if rv["parse_ok"]]
            # no parseable critic review this task (e.g. max_rounds=1): no information
            demand = int(any(rv["store_signal"] for rv in parsed_reviews)) if parsed_reviews else None
        else:
            demand = int(gov["n_new_bullets"] > 0)
        transition = freeze.end_task(mode, demand, gov["slow_loop"]) if freeze else None

        gov.update({
            "stop_reason": stop_reason, "submitted_round": submitted_round, "n_rounds": len(allocs),
            "n_reviews": len(reviewed), "demand": demand, "frozen_before": frozen_before,
            "transition": transition, "n_staged_next": len(pending["staged"]),
            "parse_fail": any(not rv["parse_ok"] for rv in reviewed),
            **({"freeze": freeze.state()} if freeze else {}),
        })
        if verbose:
            print(f"[autogov2] {current_date} mode={mode} stop={stop_reason} rounds={len(allocs)} "
                  f"reviews={len(reviewed)} visible={gov.get('n_visible', '-')} confirmed={gov['n_confirmed']} "
                  f"new={gov['n_new_bullets']} demand={demand} transition={transition}")

        meta = {"rounds": rounds_log, "playbook_snapshot": visible, "admitted": admitted,
                "consolidated": consolidated, "memory_size": len(memory.bullets), "governance": gov}
        if risk_harness is not None:
            meta["pre_harness_allocations"] = final_alloc
            meta["risk_params"] = risk_harness.get_params()
            return risk_harness.apply(final_alloc, current_date, portfolio_state["portfolio_value"]), meta
        return final_alloc, meta

    return decision_fn
