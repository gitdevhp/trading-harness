"""
The roles from the ACE paper (Generator / Reflector / Curator), kept
together in one file since each is short.

Solver       = your ReAct agent, unchanged tools/parsing/prompt structure,
               with two OPTIONAL prompt blocks: an Experience Memory block
               and a Debater-feedback block.
Debater      = the Reflector, framed as a pressure-tester the Solver has
               to satisfy — NOT a purely adversarial critic. It must
               argue in whichever direction the evidence supports:
               for MORE conviction/concentration when signals are
               strong, or for LESS when they're weak — a one-sided
               "always argue for caution" Debater systematically biases
               the whole system toward under-sizing winners. Produces
               LESSONS as {"lesson": str, "confidence": "high"|"medium"|
               "low"} rather than flat strings, so the Consolidator can
               tell a well-evidenced claim from a speculative one.
Consolidator = the Curator. Sees the FULL playbook (including bullets
               already marked harmful, via memory.format_all_for_review())
               so it can recognize duplicates of discredited ideas, and
               can add / reinforce / contradict / REPLACE (rewrite,
               merge) / remove — not just append.
RiskTuner    = optional. Turns the same lessons into bounded numeric
               deltas for the risk harness's parameters.
"""
import json
import re

from ace_harness.llm_client import chat, SOLVER_MODEL, DEBATER_MODEL, CONSOLIDATOR_MODEL

_VALID_CONFIDENCE = {"high", "medium", "low"}
_MAX_LESSONS_PER_CALL = 5

_LESSON_QUALITY_BAR = """A lesson must reference a SPECIFIC, checkable condition from the screener \
(a named metric and a threshold — e.g. "6M-Mom < -10%", "price below 200d-SMA", "1M-Vol > 30%") paired \
with a specific consequence for allocation. Reject anything that would be true in almost any month \
regardless of the data — that includes generic advice like "diversify", "avoid volatile assets", \
"reassess allocations regularly", or "limit any single position" UNLESS it names the specific \
threshold that makes it actionable (e.g. "cap any single asset above 25% of the portfolio" is fine; \
"limit the maximum allocation to any single asset" alone is not — it has no number in it). \
If you cannot state the condition as a number or named indicator, mark it "low" confidence or drop it. \
Lessons must be judged SYMMETRICALLY: a specific, evidenced case for sizing UP or concentrating into a \
strong signal (e.g. "composite momentum > 15% with price above both 50d and 200d SMA justified a larger \
position") is exactly as valid as a case for sizing down or diversifying — do not produce only \
caution-flavored lessons. A playbook that only ever learns to be more conservative is failing at this job \
as surely as one that only ever learns to be more aggressive."""


def _normalize_lessons(raw):
    """Coerce whatever the model returned for "lessons" into a clean list
    of {"lesson": str, "confidence": "high"|"medium"|"low"}, capped, so a
    malformed or overlong response can't flood the Consolidator."""
    out = []
    for item in (raw or [])[:_MAX_LESSONS_PER_CALL]:
        if isinstance(item, dict):
            text = str(item.get("lesson", "")).strip()
            confidence = str(item.get("confidence", "low")).lower()
            if confidence not in _VALID_CONFIDENCE:
                confidence = "low"
        elif isinstance(item, str):
            text, confidence = item.strip(), "low"
        else:
            continue
        if text:
            out.append({"lesson": text, "confidence": confidence})
    return out


_MAX_BULLET_CHECKS_PER_CALL = 5
_VALID_VERDICTS = {"confirmed", "contradicted"}


def _normalize_bullet_checks(raw):
    """Coerce "bullet_checks" into a clean list of {"id": str, "verdict":
    "confirmed"|"contradicted", "reason": str}, capped. This is the direct,
    evidence-grounded path to a bullet's helpful/harmful count — separate
    from (and a check on) the Consolidator's text-similarity-based updates."""
    out = []
    for item in (raw or [])[:_MAX_BULLET_CHECKS_PER_CALL]:
        if not isinstance(item, dict):
            continue
        bid = str(item.get("id", "")).strip()
        verdict = str(item.get("verdict", "")).strip().lower()
        if not bid or verdict not in _VALID_VERDICTS:
            continue
        out.append({"id": bid, "verdict": verdict, "reason": str(item.get("reason", "")).strip()})
    return out


class Solver:
    def __init__(self, universe, model: str = SOLVER_MODEL, max_steps: int = 4,
                 include_screener_tool: bool = True, fallback_mode: str = "cash",
                 max_tokens: int = 600, truncate_context: bool = False, prompt_style: str = "engine"):
        self.universe = universe
        self.model = model
        self.max_steps = max_steps
        self.include_screener_tool = include_screener_tool  # False matches yesharness.py's leaner solver
        self.max_tokens = max_tokens  # per-turn output budget
        # If True, each turn only resends [system, user, *last 2 messages] instead
        # of the whole growing transcript — keeps long debates/large screeners
        # from blowing the context window, at the cost of the model losing
        # visibility into earlier tool calls by the time it finalizes a decision.
        # Off by default so it never silently changes existing systems' behavior.
        self.truncate_context = truncate_context
        if fallback_mode not in ("cash", "equal_weight"):
            raise ValueError("fallback_mode must be 'cash' or 'equal_weight'")
        self.fallback_mode = fallback_mode  # what to do if the model's output never parses
        if prompt_style not in ("engine", "monthly"):
            raise ValueError("prompt_style must be 'engine' or 'monthly'")
        # "engine": run_all.py family's exact wording (Rebalance every N days, Objective line).
        # "monthly": your fair-portbench/institutional-harness script's exact wording
        # (Decision horizon, "Available tools", "no access to ticker names..." line,
        # "ReAct format" header) — used by run_monthly.py so that pathway's Solver
        # prompt is byte-for-byte the same as your pasted DECISION_FUNCTION.
        self.prompt_style = prompt_style

    def _fallback_decision(self):
        if self.fallback_mode == "equal_weight":
            n = len(self.universe.anon_universe)
            return {a: 100.0 / n for a in self.universe.anon_universe} | {"CASH": 0.0}
        return {"CASH": 100.0}

    def _tools(self, current_date, portfolio_state):
        def get_portfolio_status(arg: str = "") -> str:
            alloc_str = ", ".join(f"{k}: {v:.1f}%" for k, v in portfolio_state["allocations_pct"].items())
            return (f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | "
                    f"Cash: {portfolio_state['cash_pct']:.1f}%\nAllocations: {alloc_str}")

        tools = {"get_portfolio_status": get_portfolio_status}

        if self.include_screener_tool:
            def get_market_screener(arg: str = "") -> str:
                return self.universe.get_market_screener(current_date)
            tools["get_market_screener"] = get_market_screener

        return tools

    def _build_prompt(self, current_date, rebalance_days, tools_list, memory_block, feedback_block):
        if self.prompt_style == "monthly":
            system_prompt = f"""You are an autonomous ReAct Portfolio Manager on {current_date}.
Assets: {self.universe.anon_universe} + CASH
Decision horizon: {rebalance_days} trading days.

Available tools:
{tools_list}

The tools expose the complete information available to you.
You have no access to ticker names, news, fundamentals, future data,
future prices, or any information not returned by a tool.
{memory_block}{feedback_block}
Rules:
- Long-only.
- No leverage.
- CASH allowed.
- Do not invent assets.
- Final action MUST be Target_Allocations.

ReAct format:
Thought: <brief reasoning>
Action: get_market_screener[]
Observation: <tool result>
Thought: <brief reasoning>
Action: Target_Allocations[{{"ASSET_A": 10, ..., "CASH": 0}}]

The Target_Allocations percentages must be non-negative and
will be normalized to 100%."""
            user_content = f"Set target allocations for the next {rebalance_days} trading days."
            invalid_nudge = ("The Target_Allocations action was invalid. Return valid JSON-style asset "
                              "percentage allocations using only the provided assets and CASH.")
            return system_prompt, user_content, invalid_nudge

        system_prompt = f"""You are an autonomous ReAct Portfolio Manager evaluating targets on {current_date}.
Trading Horizon: Rebalance every {rebalance_days} trading days.
Allocations are normally held fixed between scheduled rebalances.
Assets: {self.universe.anon_universe} + CASH
{memory_block}{feedback_block}
Tools:
{tools_list}

Objective: Construct a medium-term portfolio optimized for risk-adjusted returns over the coming {rebalance_days} trading days.

Rules:
- Long-only, no leverage. CASH allowed.
- Only use the provided assets — never invent an asset or assume information not provided to you.
- Use only information available as of {current_date}. Never assume or reference real-world
  ticker names — reason only from the anonymized asset labels and the data provided to you.

Format:
Thought: <Reasoning step>
Action: <tool_name>[]
Observation: <tool response>
...
Thought: <Final allocation decision>
Action: Target_Allocations[{{"ASSET_A": 15, "ASSET_B": 15, ..., "CASH": 10}}]"""
        user_content = (
            f"Date: {current_date}. Analyze market conditions and set target allocations "
            f"for the next {rebalance_days} trading days."
        )
        invalid_nudge = ("The Target_Allocations action was invalid. Return valid asset percentage "
                          "allocations using only the provided assets and CASH.")
        return system_prompt, user_content, invalid_nudge

    def decide(self, current_date, portfolio_state, rebalance_days, playbook_text=None, feedback_text=None, direction=None):
        u = self.universe
        tools = self._tools(current_date, portfolio_state)
        tools_list = "\n".join(f"- {name}[]" for name in tools)

        memory_block = (
            f"\n\nEXPERIENCE MEMORY (lessons from prior tasks — weigh higher helpful/harmful counts more):\n{playbook_text}\n"
            if playbook_text else ""
        )
        if feedback_text:
            if direction == "increase_conviction":
                label = "DEBATER ARGUES YOU ARE UNDERSIZING A STRONG SIGNAL — address before finalizing:"
            elif direction == "decrease_risk":
                label = "DEBATER CHALLENGES THIS POSITION SIZE — address before finalizing:"
            else:
                label = "DEBATER'S REVIEW OF YOUR PRIOR PROPOSAL — address before finalizing:"
            feedback_block = f"\n\n{label}\n{feedback_text}\n"
        else:
            feedback_block = ""

        system_prompt, user_content, invalid_nudge = self._build_prompt(
            current_date, rebalance_days, tools_list, memory_block, feedback_block
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_decision = None
        transcript = []

        for _ in range(self.max_steps):
            if self.truncate_context:
                # Resend only system + user + the last exchange, not the whole
                # growing transcript — a market screener plus a playbook plus
                # several rounds of feedback can otherwise push a long-running
                # debate past the model's context window.
                request_messages = [messages[0], messages[1]]
                if len(messages) > 2:
                    request_messages.extend(messages[-2:])
            else:
                request_messages = messages

            reply = chat(self.model, request_messages, temperature=0.0, max_tokens=self.max_tokens,
                         stop=["Observation:"])
            messages.append({"role": "assistant", "content": reply})
            transcript.append(reply)

            action_match = re.search(r"Action:\s*([A-Za-z_]+)\[(.*?)\]", reply, re.DOTALL)
            if not action_match:
                messages.append({"role": "user",
                                  "content": "Continue using the required ReAct format. You must provide an Action."})
                continue

            action_name, action_arg = action_match.group(1), action_match.group(2).strip()

            if action_name == "Target_Allocations":
                valid_keys = set(u.anon_universe) | {"CASH"}
                parsed = re.findall(r'["\']?([A-Za-z0-9_]+)["\']?\s*:\s*(-?\d+(?:\.\d+)?)', action_arg)
                clean = {k: max(0.0, float(v)) for k, v in parsed if k in valid_keys}
                if clean:
                    raw_decision = clean
                    break
                messages.append({"role": "user", "content": invalid_nudge})
                continue

            if action_name in tools:
                obs_text = f"Observation: {tools[action_name](action_arg)}"
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."
            messages.append({"role": "user", "content": obs_text})
            transcript.append(obs_text)

        if raw_decision is None:
            raw_decision = self._fallback_decision()

        total = sum(raw_decision.values())
        if total <= 0:
            # matches normalize_allocations() in the monthly script family: an
            # all-zero parsed decision falls back to 100% cash, not all-zero.
            normalized = {a: 0.0 for a in u.anon_universe} | {"CASH": 100.0}
        else:
            normalized = {k: round((v / total * 100.0), 6) for k, v in raw_decision.items()}
        return normalized, "\n".join(transcript)


def _extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _extract_json_array(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


class Debater:
    """Argues against the Solver's proposal. intra_task_review() is one
    round of that argument, run before execution — the Solver either wins
    the point (verdict "accept") or has to revise and come back. Run this
    method multiple times in a loop for a real back-and-forth debate, or
    once for a single round of scrutiny (see harnesses.make_dual_permanent).

    Lessons come back as {"lesson": str, "confidence": "high"|"medium"|
    "low"} — the confidence is what lets the Consolidator tell a claim
    backed by clear evidence from one it's merely floating.
    """

    def __init__(self, model: str = DEBATER_MODEL):
        self.model = model

    def intra_task_review(self, current_date, screener_text, portfolio_status_text,
                           proposed_allocations, playbook_text=None, round_num=1,
                           n_universe_stocks=None):
        memory_block = f"\n\nExperience memory context:\n{playbook_text}\n" if playbook_text else ""

        # Scale what "large" means to the universe — equal weight defines the floor.
        if n_universe_stocks and n_universe_stocks > 0:
            eq_wt = round(100.0 / n_universe_stocks, 1)
            large_threshold = round(2.5 * eq_wt, 1)
            sizing_context = (
                f"\nUniverse size: {n_universe_stocks} stocks. "
                f"Equal-weight baseline = {eq_wt:.1f}% per stock. "
                f"Only challenge a position when it deviates meaningfully from this baseline — "
                f"a 'large' position for this universe means >{large_threshold:.1f}% (2.5× equal weight). "
                f"Positions near equal weight are NORMAL here and must not be challenged "
                f"without a specific negative screener signal."
            )
        else:
            sizing_context = ""

        system_prompt = f"""You are a portfolio Debater pressure-testing a proposal BEFORE execution (round {round_num}). Your job is to check whether the SIZING matches the evidence — not to reflexively push toward caution.
You must be willing to argue in EITHER direction, based only on what the screener actually shows:
- Argue for MORE conviction / a LARGER position when an asset shows strong, aligned signals (e.g. positive momentum across multiple windows, price above both 50d and 200d SMA, healthy breadth) but the proposal sizes it small or excludes it — being needlessly cautious in the face of a strong signal is a real error, not a safe default.
- Argue for LESS conviction / a SMALLER position or more diversification when a position is large relative to weak, mixed, or contradictory signals, or when turnover looks excessive given trading fees.
- Concede ("accept") when the sizing is actually proportionate to the strength of the evidence, in either direction.
You do NOT know future prices — never argue from hindsight, only from what's in the screener right now.
IMPORTANT: A risk harness (volatility targeting, position caps, drawdown limits) is applied AFTER this proposal is finalized. It already handles generic risk guardrails. Your role is SIGNAL MISREAD detection only: wrong direction on an asset, a missed strong signal that was priced out, or a large position with directly contradictory screener metrics. If the sizing looks reasonable given the signals, concede.
If you argue "decrease_risk", you MUST cite the SPECIFIC screener metric and value that directly contradicts the proposed position (e.g. "6M-Mom is -8% yet allocated 20%"). Without a named, checkable metric that contradicts the sizing, verdict must be "accept".
CRITICAL — verdict vs. should_refine: these are independent decisions.
- verdict="accept": the signal-to-sizing match is reasonable. Use this whenever your only remaining concern is something the risk harness already handles (position caps, volatility limits, drawdown guards) — the harness will correct it, so accept the signal read and let the harness do its job.
- verdict="revise": there is a genuine, specific signal misread that the Solver can actually fix in the next round (wrong direction, a clearly contradicted large position). Only use "revise" when you have concrete, actionable feedback that the Solver can act on.
- should_refine=true: your "revise" feedback is specific enough that another round would plausibly improve the proposal.
- should_refine=false: only pair with verdict="revise" when the disagreement is genuine but unresolvable (e.g. the screener is ambiguous and no rewrite will fix it). Do NOT set should_refine=false just because the risk harness will handle it — in that case, set verdict="accept" instead.
{sizing_context}
{memory_block}
Also propose at most {_MAX_LESSONS_PER_CALL} candidate lessons for the shared playbook — lessons that argue for sizing up on strong signals are just as valuable as lessons that argue for caution.
{_LESSON_QUALITY_BAR}

Also output "should_store": true if at least one lesson is genuinely novel relative to the existing playbook, false if the lessons are already well-covered (saves a Consolidator call). Default to false — only true when you can articulate why this is meaningfully new.

Also output "should_refine": true if your "revise" feedback is specific and actionable enough that another Solver round would plausibly improve the proposal; false if the disagreement is genuine but unresolvable. If verdict is "accept", always set should_refine to false.

Respond ONLY with JSON, no other text:
{{"verdict": "accept" or "revise", "direction": "increase_conviction" or "decrease_risk" or "well_calibrated", "feedback": "1-3 sentences making your strongest argument in that direction, or why you concede it", "lessons": [{{"lesson": "short reusable rule with a specific condition", "confidence": "high|medium|low"}}], "should_store": true or false, "should_refine": true or false}}"""
        user_prompt = f"""Date: {current_date}
Market screener:
{screener_text}

Portfolio status:
{portfolio_status_text}

Proposed target allocations: {proposed_allocations}"""

        reply = chat(self.model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                      temperature=0.0, max_tokens=600)
        parsed = _extract_json(reply) or {}
        direction = parsed.get("direction", "well_calibrated")
        if direction not in ("increase_conviction", "decrease_risk", "well_calibrated"):
            direction = "well_calibrated"
        # g_sto (AdaReMo Algorithm 2): should the lessons from this round be
        # stored in persistent memory? Default False (conservative): only store
        # when the Debater explicitly confirms the lessons are novel.
        should_store = bool(parsed.get("should_store", False))
        # g_ref (AdaReMo Algorithm 2): should the refinement loop continue?
        # False when the critic has no actionable fix to offer — stop early
        # even if verdict is "revise" (0 additional rounds needed).
        should_refine = bool(parsed.get("should_refine", True))
        return {
            "verdict": parsed.get("verdict", "accept"),
            "direction": direction,
            "feedback": parsed.get("feedback", ""),
            "lessons": _normalize_lessons(parsed.get("lessons")),
            "should_store": should_store,
            "should_refine": should_refine,
        }

    def post_task_reflect(self, decision_date, next_date, executed_allocations,
                           decision_prices, realized_prices, playbook_text=None, decision_screener_text=None):
        realized_return_pct = {}
        for asset, entry_price in decision_prices.items():
            exit_price = realized_prices.get(asset)
            if exit_price and entry_price:
                realized_return_pct[asset] = round((exit_price - entry_price) / entry_price * 100.0, 2)

        memory_block = f"\n\nExisting experience memory (these exact bullets were shown to the Solver for this decision):\n{playbook_text}\n" if playbook_text else ""
        screener_block = (
            f"\n\nMarket screener AS OF the decision date ({decision_date}), i.e. the exact numbers "
            f"the Solver had available when it made this call:\n{decision_screener_text}\n"
            if decision_screener_text else ""
        )
        system_prompt = f"""You are a portfolio Reflector analyzing what happened after a decision was executed.
Compare the allocation made on {decision_date} to realized per-asset returns through {next_date}. Extract at most {_MAX_LESSONS_PER_CALL} concrete, REUSABLE, GENERAL lessons (about regime signals, sizing, timing, diversification, or turnover/fee cost) — not just a restatement of "asset X went up".
{memory_block}{screener_block}
{_LESSON_QUALITY_BAR}
A lesson here should connect a SPECIFIC value from the decision-date screener to the realized outcome — e.g. "when 6M-Mom was below -10% at decision time, the asset kept underperforming over the next month" is usable; "the market was volatile so returns were mixed" is not.

Check BOTH directions explicitly:
- Positions that lost value or underperformed despite a large allocation — what in the decision-date screener should have signaled more caution?
- Assets that gained significantly (e.g. double-digit realized return) but were allocated near-zero or a small weight — a large miss like this is just as important to learn from as a loss. Look at the decision-date screener for that asset: was there a signal (strong momentum, price above its SMAs) that should have justified more conviction, and the sizing was too conservative? If the screener genuinely gave no leading signal for the move, say so and don't fabricate a lesson from it — not every missed gain was foreseeable, and a false lesson is worse than none.

Rate each lesson's confidence: "high" only if this outcome clearly demonstrates a general rule, "medium" if it's suggestive but could be one data point, "low" if it's speculative.

Separately from new lessons, GRADE the EXISTING bullets you were shown (the ones in brackets like [xyz-00001] above) against what actually happened this period. For any bullet whose claim you can directly check against this month's realized data — not vague ones you can't verify — say whether it was confirmed or contradicted. This is the ONLY way a bullet's harmful count can ever grow from real evidence rather than another lesson merely disagreeing with it in text, so take it seriously: if a bullet said something like "cap concentrated positions" and a concentrated position this period massively outperformed, that is a contradiction, not something to politely skip. Only cite bullet ids that literally appear in the memory block above — never invent one.

Respond ONLY with JSON, no other text:
{{"lessons": [{{"lesson": "short reusable rule with a specific condition", "confidence": "high|medium|low"}}], "bullet_checks": [{{"id": "<exact existing bullet id>", "verdict": "confirmed" or "contradicted", "reason": "1 short sentence citing the realized number"}}]}}"""
        user_prompt = f"""Executed allocations on {decision_date}: {executed_allocations}
Realized per-asset returns to {next_date} (%): {realized_return_pct}"""

        reply = chat(self.model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                      temperature=0.0, max_tokens=550)
        parsed = _extract_json(reply) or {}
        return {
            "lessons": _normalize_lessons(parsed.get("lessons")),
            "bullet_checks": _normalize_bullet_checks(parsed.get("bullet_checks")),
            "realized_return_pct": realized_return_pct,
        }


class Consolidator:
    """Curates the playbook. Sees the FULL memory (via
    memory.format_all_for_review(), including soft-hidden/harmful
    bullets) so it has enough context to recognize a duplicate of
    something already discredited, rather than only ever seeing the
    top-ranked subset the Solver sees."""

    def __init__(self, model: str = CONSOLIDATOR_MODEL):
        self.model = model

    def consolidate(self, memory, lessons, sections):
        if not lessons:
            return []
        existing = memory.format_all_for_review()
        system_prompt = f"""You are the Curator maintaining an evolving trading playbook. Your job is to keep it SHORT, GENERAL, SPECIFIC, and NON-REDUNDANT — not to log everything that happens. Prefer a few strong, checkable bullets over many vague ones.
Sections available: {sections}
"CONVICTION SIGNALS" is for lessons about when the evidence justified sizing UP or concentrating — it is NOT a lesser section than "RISK LESSONS" or "MISTAKES TO AVOID". Do not systematically favor caution-flavored lessons over conviction-flavored ones; weigh each purely by the specificity and strength of its evidence. A playbook that only ever grows more conservative is a curation failure, not a safe default.
You see the FULL existing playbook below, including bullets already marked harmful, and NEW candidate lessons, each with a confidence rating.

Rules (apply in this exact order):
1. MERGE FIRST — before reading the new lessons, scan ALL pairs of existing bullets for near-duplicates (same idea, different wording). For every such pair: {{"op": "replace", "id": "<the higher net-score one>", "content": "<merged, sharper wording that keeps both specifics>"}} and {{"op": "remove", "id": "<the lower net-score one>"}}. A compact playbook with fewer, stronger bullets always beats a long one with redundant ones.
2. A "low" confidence lesson may only REINFORCE or CONTRADICT an existing bullet (an "update" op) — it must never spawn a brand new bullet by itself.
3. A "high" or "medium" confidence lesson may spawn a new bullet, but ONLY if nothing existing already captures it, AND ONLY if it names a specific threshold, metric, or condition (a screener field, a percentage, a named regime) — not a platitude that's true in almost any month. "Avoid volatile assets", "diversify", "reassess regularly", or "limit position size" with no number attached do NOT qualify for "add"; route them to "update" on an existing bullet instead, or drop them.
4. Lesson duplicates/reinforces an existing bullet -> {{"op": "update", "id": "<id>", "helpful": 1}}
5. Lesson contradicts / is disproven by an existing bullet -> {{"op": "update", "id": "<id>", "harmful": 1}}
6. Lesson clearly generalizes or sharpens an existing bullet's wording (same idea, better phrasing, or adds the missing number) -> {{"op": "replace", "id": "<id>", "content": "<improved, more specific wording>"}}
7. Genuinely new, non-duplicate, high/medium-confidence, condition-specific lesson -> {{"op": "add", "section": "<one of the sections above>", "content": "<concise rule that names a specific threshold or metric>"}}

Respond ONLY with a JSON array of at most 8 ops, no other text."""
        user_prompt = f"Full existing playbook:\n{existing}\n\nNew candidate lessons:\n{json.dumps(lessons)}"

        reply = chat(self.model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                      temperature=0.0, max_tokens=800)
        return _extract_json_array(reply)[:8]


class RiskTuner:
    """Optional role: proposes small, bounded deltas to the risk harness's
    NUMERIC parameters (not the playbook text), using the same
    confidence-rated lessons the Consolidator uses. This is what lets
    Experience Memory actually strengthen the risk-management layer
    itself, not just the Solver's prompt — deltas are always clamped by
    the caller (risk_harness.update_params) to a fixed safe range, so
    this role can shift risk appetite but never remove a guardrail."""

    def __init__(self, model: str = CONSOLIDATOR_MODEL):
        self.model = model

    def propose_adjustments(self, current_params, param_bounds, lessons, realized_return_pct):
        if not lessons:
            return {}
        system_prompt = f"""You are a Risk Tuner adjusting a risk-management filter's numeric parameters based on realized trading outcomes.
Current parameters: {json.dumps(current_params)}
Allowed ranges (hard limits, enforced regardless of what you propose): {json.dumps(param_bounds)}
Each lesson below has a confidence rating — only act on "high" or "medium" confidence lessons; ignore "low" confidence ones entirely.
Propose SMALL deltas (no more than ~10% of a parameter's range in one step) to at most 2 parameters, only if the evidence clearly supports a change. If no change is warranted, return {{}}.
Respond ONLY with JSON mapping parameter name to delta, e.g. {{"target_vol": -0.005}}. No other text."""
        user_prompt = f"Lessons from reflection:\n{json.dumps(lessons)}\n\nRealized per-asset returns (%):\n{json.dumps(realized_return_pct)}"

        reply = chat(self.model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                      temperature=0.0, max_tokens=200)
        parsed = _extract_json(reply)
        return parsed if isinstance(parsed, dict) else {}
