# ACE-style self-improvement harnesses around your Solver

Wraps your no-harness ReAct portfolio solver in ACE-style self-improvement
(arXiv 2510.04618, github.com/ace-agent/ace): **Solver** (Generator),
**Debater** (Reflector), **Consolidator** (Curator), and an evolving
**Experience Memory** playbook — a human-editable `.txt` file of sectioned
bullets with helpful/harmful counters, updated via small delta ops rather
than full rewrites.

11-file flat package, one concept per file:

| File | What it is |
|---|---|
| `market.py` | Data + screener. `prefetch()` retries failed downloads with backoff and a `Ticker.history()` fallback; `get_market_screener()` includes 20d/50d/200d SMA + momentum + vol. |
| `engine.py` | T+1-open execution, fixed-day-count rebalance, warmup + rebase — the original protocol. Injectable `decision_fn`. |
| `engine_monthly.py` | Same-close execution, calendar-month rebalance, no warmup, equal-weight start — matches your "fair portbench" script's protocol exactly. Same `decision_fn` interface as `engine.py`, so every harness works on either engine unchanged. |
| `agents.py` | `Solver`, `Debater`, `Consolidator`, `RiskTuner`. |
| `memory.py` | `ExperienceMemory` — the playbook, saved as `.txt`. |
| `harnesses.py` | Five named systems (below) + `wrap_with_risk_harness`. |
| `risk_harness.py` | `GPTInstitutionalRiskHarness`, `SimpleTrailingRiskHarness`, `SimpleMomentumHarness` — three interchangeable optional post-processing filters. |
| `run_all.py` | CLI for `engine.py`. |
| `run_monthly.py` | CLI for `engine_monthly.py` — defaults to the exact setup in your diagram (see below). |
| `compare_results.py` | Side-by-side metrics (return, vol, Sharpe, drawdown). |
| `llm_client.py` | Shared chat wrapper, configurable per-role via env vars. |

## Your diagram, mapped to what's here

**Data → Harness → LLM → Result** is your pasted "fair portbench" script.
**Result → Debater → Consolidator → Memory → back into the Harness** is
the part you asked me to build — except it already existed: `Debater`,
`Consolidator`, `ExperienceMemory`, and `RiskTuner` (which is what makes
the Harness *dynamic* — it reads Memory's lessons and adjusts the
Harness's own parameters) were all built for `dual_permanent` already.
The only genuinely new work was giving your Data+Harness+LLM+Result
pipeline an engine to plug into — `engine_monthly.py` — because it
executes differently (same-close, not next-open) and rebalances
differently (calendar month, not a day-count interval) from everything
else in this package. Once that engine has the same `decision_fn`
interface as `engine.py`, `harnesses.make_dual_permanent` runs against
it completely unchanged — that's the whole "loop back into the Harness"
arrow in your drawing.

```bash
python -m ace_harness.run_monthly \
    --tickers AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX \
    --start 2024-01-01 --end 2024-12-31
```
Defaults: `dual_permanent`, `ConvictionHarness` (your latest script's
`apply_institutional_harness` — conviction overlay + vol-targeting +
breadth multiplier, deliberately no per-asset stop or drawdown guard),
`--adaptive_risk` on, **both** `get_market_screener` and
`get_portfolio_status` available, equal-weight fallback, no context
truncation, 300 max output tokens per turn — i.e. your script's exact
Data/Harness/LLM/Result, with the Debater→Consolidator→Memory→dynamic-
Harness loop closed around it.

**Correction to something I got wrong earlier:** I had previously
defaulted this to *excluding* the screener tool. That was a mistake —
every version of your script (the simple-harness one and this
institutional-harness one) always exposes **both** tools. Fixed; the
default is now `include_screener_tool=True`, with `--no_screener` if you
ever want to drop it.

**Prompt parity, made explicit rather than approximate.** `Solver` now
takes a `prompt_style` ("engine" or "monthly") so the two script
families each get their own byte-for-byte-matched wording instead of one
paraphrased template trying to serve both — `run_monthly.py` always
passes `prompt_style="monthly"`, which reproduces your latest
`DECISION_FUNCTION`'s system prompt exactly: same "Available tools" /
"no access to ticker names..." / "ReAct format" wording, same
JSON-style-allocation nudge text, no truncation, `max_tokens=300`.
Verified with a test that checks each of these phrases is present
verbatim. `run_all.py`'s Solver usage (`prompt_style="engine"`, the
default) is untouched.

**What I stripped out as unnecessary**, per your ask — these duplicate
things `engine_monthly.py`/`compare_results.py` already do consistently
across every system in this package, so keeping them would've meant two
different output schemas to reconcile:
- the `write_result()` / `"fair-portbench-v1"` protocol JSON (summary +
  rebalances + daily_nav, with slippage/commission bps metadata) —
  replaced by the same flat `{date, prices, portfolio_value, allocations}`
  schema every other system in this package writes;
- per-order buy/sell notional tracking in `execute_rebalance()`;
- the in-script Sharpe/Sortino/Calmar/CAGR calculation — `compare_results.py`
  already computes return/vol/Sharpe/drawdown for every system's output.

**What I kept and generalized** because they were genuinely better than
what was already here: the retrying, self-healing data downloader (now
`market.py`'s `prefetch()`, for every system) and the richer market
snapshot (now `get_market_screener()`, also for every system).

**One inconsistency worth knowing about your original script:** its
`--rebalance-days` argument is parsed but never actually used to decide
when to rebalance — the trigger is always "new calendar month," and
`rebalance_days` only reaches the Solver's prompt text ("decision
horizon"). I preserved that behavior exactly in `engine_monthly.py`
(the flag only affects prompt phrasing) rather than silently making it
functional, since I wasn't sure which one you actually want — happy to
wire it up for real if you'd rather the flag do something.

## Two more fixes from your latest update, ported into the shared code

**Trading-day intersection, not just the first asset's calendar.**
`MarketUniverse.common_trading_days()` now matches your script's
`get_common_trading_days()` — it only uses dates present in *every*
asset's index, not just the first ticker's. Different tickers can have
slightly different available dates (holidays, halts, late IPOs); reading
one asset's price on a day it doesn't have raises a `KeyError`, so this
was a latent bug in `engine.py` too (it worked in the smoke test only
because the synthetic data there shares one calendar by construction).
Fixed in both `engine.py` and `engine_monthly.py`, verified with a
direct test where one asset is missing a day the other has.

**ReAct loop hardening, opt-in via `Solver(..., truncate_context=True)`.**
Two things from your update, both in `agents.py`:
- if a reply has no `Action:` at all, the Solver now nudges with
  *"Continue using the required ReAct format. You must provide an
  Action."* instead of silently looping;
- if `Target_Allocations` parses to nothing usable, it now nudges with
  *"...Return valid asset percentage allocations..."* and gives the
  model another turn, instead of giving up immediately.

Both apply to every `Solver` instance (verified: a Solver fed a
no-Action reply, then an invalid-asset reply, then a valid one,
correctly recovers by turn 3). The **context-window truncation**
(resend only `[system, user, last_2_messages]` each turn, to stop a
large screener + playbook + several debate rounds from blowing the
context budget) is a real tradeoff, not a strict improvement — it means
the model can lose sight of an earlier tool call's result by the time it
finalizes a decision. So it's **off by default** (existing `run_all.py`
systems are unaffected — verified byte-for-byte identical output before
and after this change) and only turned on in `run_monthly.py`, where it
matches your script's own design intent exactly.

## The Debater

What was called "Critic" is now **Debater** — same two methods
(`intra_task_review`, `post_task_reflect`), pressure-testing the Solver's
proposal rather than handing down a verdict. `intra_task_review` is one
round of that pressure-test: the Debater checks whether the SIZING
matches the evidence, and can argue in **either direction** — for more
conviction/concentration when signals are strong, or for less when
they're weak — returning a `"direction"` field
(`increase_conviction`/`decrease_risk`/`well_calibrated`) alongside the
verdict. Call it once for a single round of scrutiny, or in a loop for a
real back-and-forth.

**Why it's two-directional and not just adversarial, based on a real
result.** The original framing ("argue against the proposal") is a
one-way ratchet: it can only ever push toward less concentration, more
diversification, less turnover — never toward more conviction on a
strong signal. Run for a year against 18 large-cap tickers, the memory
converged almost entirely on "diversify" / "cap any single position" —
and the resulting `dual_permanent` run finished the year at $1.31M vs.
$1.46M for the same harness with no Debater/memory involved at all,
because 2024 was a year where concentrating in a few big winners (TSLA,
NFLX) paid off enormously, and a system that structurally can't argue
for concentration will systematically under-size exactly those trades.
That's not a bug in the plumbing — it's what an adversarial-only Debater
and a caution-only memory were designed to produce. Fixed by:

- `intra_task_review`'s prompt explicitly instructs arguing for MORE
  conviction when signals are strong and the proposal under-sizes them,
  not just for less when they're weak.
- `post_task_reflect` now explicitly checks for assets that gained
  significantly but were allocated near-zero — a missed win is checked
  for a leading signal just as seriously as a loss is.
- A new `"CONVICTION SIGNALS"` memory section exists alongside `"RISK
  LESSONS"`/`"MISTAKES TO AVOID"`, and the Consolidator is explicitly
  told it isn't a lesser section — weigh conviction-flavored and
  caution-flavored lessons purely by evidence quality, not by which
  direction they push.

Verified with a test that the Debater's prompt contains both directions,
the Reflector's prompt asks about underweighted winners, and the
Consolidator's prompt names the new section and the anti-bias
instruction. What this **can't** guarantee: that the LLM actually uses
the room it's now given to argue for conviction — only that the
one-sided instruction that was actively preventing it is gone. Re-run
the same year and diff the playbook to see whether it actually changes.

## Seven named systems — a deliberate ablation ladder

Each of these adds exactly one thing on top of the last, using the
identical engine/Solver/risk-harness code — so any difference between
two of them is guaranteed to come from that one addition and nothing
else, rather than from a subtly different standalone script.

```
baseline        ABLATION CONTROL — Solver only. No Debater, no memory,
                no revision. Run this next to anything else to see what
                the rest is actually contributing.
intra           + Solver <-> Debater loop within a task. No memory —
                isolates the Debater's effect alone, with nothing to
                remember between tasks.
inter           Permanent memory feeds Solver directly, no in-task loop.
                Debater reflects post-hoc on realized outcomes.
dual            General/configurable: fast intra-task loop (--max_rounds)
                + slow permanent-memory update, together.
dual_session    many debate rounds within ONE task, feeding a day-scoped
                memory that grows across those rounds, then is thrown
                away. Nothing survives past one task.
memory_only     ABLATION CONTROL for dual_permanent — identical
                persistent memory + Consolidator + adaptive risk-tuning,
                but NO pre-execution Debater review/revision at all
                (this is functionally an alias for `inter`, given an
                explicit name so the relationship to dual_permanent is
                obvious without already knowing that).
dual_permanent  Solver proposes -> Debater reviews ONCE -> if "revise",
                Solver gets exactly one more turn using that feedback ->
                that's the Result. Memory persists for the ENTIRE
                backtest, never cleared. Primary/default system.
```

**Isolating the Debater specifically:** run `baseline`, `memory_only`,
and `dual_permanent` against the same universe/dates and diff the
results — `memory_only` vs `baseline` tells you what memory alone is
doing; `dual_permanent` vs `memory_only` tells you what the debate/
revision step adds *on top of* memory, isolated from memory's own
effect:
```bash
python -m ace_harness.run_monthly --systems baseline memory_only dual_permanent \
    --risk_harness --risk_harness_type conviction
python -m ace_harness.compare_results \
    results/monthly_baseline_convictionriskharness_results.json \
    results/monthly_memory_only_convictionriskharness_adaptive_results.json \
    results/monthly_dual_permanent_convictionriskharness_adaptive_results.json
```

**Bug fix, worth knowing about if you ran this before:** `dual_permanent`
used to be implemented as `dual` with `max_rounds` pinned to 1. That's
broken — with `max_rounds=1`, the loop always ends after round 1
regardless of the Debater's verdict, so a "revise" verdict's feedback was
collected but the Solver never actually got to act on it. It's now a
bespoke propose → debate → revise cycle in `harnesses.make_dual_permanent`:
exactly one Debater review, and if it says "revise," exactly one more
Solver call that actually receives the feedback text. Verified with a
call-counting test: 1 Debater call, 2 Solver calls on "revise," and the
feedback string is only present in the second Solver call's prompt.

`dual_permanent` still makes **two** Debater calls per task in total —
`intra_task_review` (the actual review of today's proposal) and
`post_task_reflect` (a separate, non-adversarial reflection on the
*previous* task's realized outcome, which is what feeds memory forward).
`memory_only` keeps the second call and drops only the first — that's
the precise sense in which "the Debater" is isolated: the pre-execution
argument is gone, the post-hoc reflection that actually populates memory
is not, because removing that too would leave memory permanently empty
and wouldn't test memory's effect at all.

## Setup

```bash
pip install -r requirements.txt
```

OpenAI-compatible endpoint (defaults to local vLLM, same as your script):
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct-AWQ --port 8000
```

```bash
export ACE_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export ACE_SOLVER_MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
export ACE_DEBATER_MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"       # can differ from solver
export ACE_CONSOLIDATOR_MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
```

## Running it

```bash
python -m ace_harness.run_all \
    --tickers GLD LLY KRE TSLA GOOGL TLT XLU XLE XOM NVDA \
    --start 2023-03-15 --end 2026-04-01 \
    --output_dir ./results \
    --rebalance-days 20 --warmup-days 60
```

Runs `dual_permanent` by default. Writes `results/dual_permanent_results.json`
(same schema as your original output) and `results/dual_permanent_playbook.txt`
— open that file directly to read, in plain English, what the system has
learned so far.

Run other systems: `python -m ace_harness.run_all --systems intra inter dual dual_session dual_permanent`
(`dual`/`intra`/`dual_session` take `--max_rounds`; `dual_permanent` always uses 1.)

## The smart-memory design

Three things make the playbook more than an append-only log:

**1. It's a real, editable `.txt` file — not JSON.**
```
## STRATEGIES & INSIGHTS
[str-00014] helpful=6 harmful=0 :: In low-breadth regimes, cut gross exposure before rotating sector weights.
[str-00019] helpful=1 harmful=3 :: Chase 1M momentum leaders after a >10% breakout.
```
Open it in any editor. Delete a bad line, hand-adjust a `helpful=`/`harmful=`
count, or just type a new plain-text line under a section heading — it's
picked up as a fresh, untested bullet the next time the file loads.
Nothing needs to round-trip through JSON.

**2. Two different views, for two different readers.**
- `format_for_prompt()` — what the Solver/Debater see: ranked by net
  score, and it **soft-hides** bullets with a clearly negative,
  evidenced track record (net ≤ −1 with ≥ 2 votes). They're still in the
  file and still visible to the Consolidator, just not fed to the model
  as if they were good advice, until `prune()` eventually deletes them
  outright (net ≤ −3).
- `format_all_for_review()` — what the Consolidator sees: *everything*,
  soft-hidden bullets included, so it has full context to recognize a
  duplicate of something already discredited instead of re-adding it.

**3. Confidence-rated lessons, and a Curator that can rewrite, not just append.**
The Debater no longer hands back flat lesson strings — every lesson is
`{"lesson": "...", "confidence": "high"|"medium"|"low"}`. The
Consolidator's rules (in `agents.py`) use that directly:
- a `"low"`-confidence lesson may only reinforce or contradict an
  *existing* bullet — it can never spawn a new one on its own;
- a new bullet only gets added on `"high"`/`"medium"` confidence, and
  only if nothing existing already covers it;
- when a new lesson just says an existing bullet's idea better, the
  Consolidator can `"replace"` the wording in place (keeping its
  accumulated helpful/harmful history — a phrasing fix isn't a new claim);
- when two bullets have drifted into saying the same thing, it can merge
  them (`"replace"` the stronger one, `"remove"` the weaker one).

Every Consolidator call is capped (max 6 ops requested, max 8 ever
applied — `memory.MAX_OPS_PER_CALL`), so one bad response can't blow up
the playbook.

**4. A quality bar against platitudes, added after a real failure mode showed up.**
After a full year of monthly `dual_permanent` runs, the surviving playbook
was almost entirely generic finance-101 advice ("diversify", "avoid
volatile assets", "reassess regularly") with near-zero harmful counts —
technically non-redundant, but not actually teaching the Solver anything
it didn't already know, and with no mechanism to ever be corrected since
nothing that vague can be contradicted. Two changes address this:

- `_LESSON_QUALITY_BAR` (in `agents.py`, shared by both Debater prompts
  and the Consolidator's prompt) requires every lesson to name a specific
  metric and threshold from the screener — "6M-Mom < -10%", "price below
  200d-SMA" — and explicitly rejects the always-true phrasing above unless
  a number is attached to it.
- `post_task_reflect()` now actually receives the screener snapshot from
  the **decision date**, not just prices — both engines capture it into
  `decision_log[date]["screener"]` when a decision is made, and
  `harnesses.py` threads it through. Without this the Reflector had no
  way to cite a specific indicator value even if instructed to; it could
  only compare an allocation to a return, which is exactly what produces
  vague lessons.
- The Consolidator is now also told to actively look for two bullets that
  say the same thing and merge them, rather than only merging when a new
  lesson happens to point it out.

Verified with tests that check the decision-date screener text actually
reaches the Reflector's prompt, and that the quality-bar/anti-platitude
and merge instructions are present in both prompts. This raises the bar
the Debater/Consolidator are held to — it can't guarantee the LLM never
produces a platitude, but it removes the excuse (no data to be specific
with) and makes generic advice fail the Consolidator's explicit criteria
for a new bullet.

**5. Ground-truth scoring, not just self-consistency scoring.** Even
after (4), a real gap remained: a bullet's `harmful` count could ONLY
ever grow when a brand-new lesson happened to textually contradict it
via the Consolidator's judgment — there was no mechanism directly
checking "did this specific bullet's claim actually hold up against what
happened this period?" That's judging text against text, not text
against outcomes, and it's why a run could go a full year with dozens of
Debater calls and never see a single `harmful` count anywhere.

Fixed with `bullet_checks`: `post_task_reflect()` is now also shown
exactly which bullets (by their real `[id]`) were active in the Solver's
prompt for the decision being reflected on, and is asked to directly
verdict each one it can check — `"confirmed"` or `"contradicted"` —
against the realized return data, separately from proposing new lessons.
`harnesses._apply_bullet_checks()` applies these straight to
`memory.update_counts()`, bypassing the Consolidator's dedup/merge logic
entirely (an ID-referenced verdict is already unambiguous — there's
nothing to deduplicate). An unrecognized or hallucinated id is silently
ignored, verified with a dedicated test, along with the whole
confirmed→helpful / contradicted→harmful path and the parsing cap.

This doesn't guarantee the LLM will actually flag a bullet as
contradicted when it should — that still depends on the model reading
its own instructions carefully. What it fixes is that there was
previously no code path for it to do so at all, no matter how carefully
it was asked.

## Optional risk-management layer

Two interchangeable post-processing filters, independent of the
self-improvement structure — either wraps whatever any `decision_fn`
produces, unchanged:

| `--risk_harness_type` | Class | Matches |
|---|---|---|
| `gpt` (default in `run_all.py`) | `GPTInstitutionalRiskHarness` | `yesharnessgpt.py` — conviction overlay, vol-targeting, breadth overlay, conditional stops, drawdown guard (4 tunable params) |
| `simple` | `SimpleTrailingRiskHarness` | `yesharness.py` — trailing-peak drawdown stop, SMA de-risking, negative-momentum de-risking (3 tunable params) |
| `momentum` | `SimpleMomentumHarness` | your "fair portbench" (simple-harness) script's `apply_simple_harness` — SMA de-risking, negative-momentum de-risking, no trailing stop (2 tunable params) |
| `conviction` (default in `run_monthly.py`) | `ConvictionHarness` | your latest "fair portbench" (institutional-harness) script's `apply_institutional_harness` — conviction overlay + vol-targeting + breadth multiplier, no trailing stop or drawdown guard (3 tunable params) |

```bash
python -m ace_harness.run_all --risk_harness --risk_harness_type simple
```

## Exact-parity fixes (this round)

You asked me to line up every small thing between `ConvictionHarness` /
the Solver and your pasted script, with the Debater/Consolidator as the
only actual addition. Diffing carefully turned up three real gaps
(beyond the screener/tools ones already covered above), now fixed and
covered by dedicated tests:

- **`get_market_screener()` was missing the empty-data guard** your
  `build_market_snapshot()` has (`if closes.empty: raise ValueError(...)`)
  — added, with the same reverse-mapped-ticker error message.
- **Decimal precision on the screener's momentum/vol fields** was `.1f`
  (matching the older `run_all.py` script family) vs your `.2f`. Since
  this only changes how many digits the LLM *sees* in the prompt text —
  never the underlying float values anything computes from — I
  consolidated on `.2f` rather than build a dual-precision system for a
  cosmetic-only difference. Flagging it in case you'd rather it stayed
  configurable.
- **The all-zero fallback was a real bug, not cosmetic.** Your
  `normalize_allocations()` forces `CASH: 100.0` when every value is
  zero (e.g. everything got stopped out, or the model proposed all
  zeros). Both `Solver.decide()`'s final normalization and
  `ConvictionHarness.apply()`'s final normalization were instead
  returning an all-zero dict summing to nothing in that case. Fixed in
  both, verified with a test that forces an all-zero input into each and
  checks the result is exactly `{...: 0.0, "CASH": 100.0}`.
- **Rounding precision**: `Solver.decide()` and `ConvictionHarness.apply()`
  now round to 6 decimals (matching `normalize_allocations()`'s `round(...,
  6)`), not 2. `GPTInstitutionalRiskHarness`/`SimpleTrailingRiskHarness`/
  `SimpleMomentumHarness` keep their existing 2-decimal rounding — those
  belong to the earlier `yesharness.py`/`yesharnessgpt.py` script family,
  which used 2 decimals in its own originals, so "matching the source"
  means something different for each family.

Everything else — `FEE_RATE` (0.0015 here vs `engine.py`'s 0.0010),
`INITIAL_CAPITAL` ($1M here vs $100k), the conviction/vol-target/breadth
math itself, `_portfolio_vol`'s exact formula, `execute_rebalance`'s
turnover/fee math — was already an exact line-for-line match; verified
by re-diffing function by function against this paste.

## Matching yesharness.py's leaner Solver

`yesharness.py`'s ReAct agent only exposes `get_portfolio_status` — no
`get_market_screener` tool at all. `--no_screener` reproduces that:
```bash
python -m ace_harness.run_all --no_screener --risk_harness --risk_harness_type simple
```
`Solver(universe, include_screener_tool=False)` drops the tool from both
the tool dispatch table and the `Tools:` section of the system prompt
(built dynamically from whichever tools are actually available, so
there's no hardcoded prompt text to keep in sync).

## Letting memory tune the risk layer itself

Add `--adaptive_risk` (needs `--risk_harness` too, and only does
anything for `inter`/`dual`/`dual_permanent`, which have persistent
memory): each risk harness's own `params` dict and `PARAM_BOUNDS` (as a
class attribute — `GPTInstitutionalRiskHarness.PARAM_BOUNDS` and
`SimpleTrailingRiskHarness.PARAM_BOUNDS` are independent) get read via
`get_param_bounds()`, and `RiskTuner` proposes small deltas from the
same lessons that already update the text playbook — hard-clamped, so
memory can shift risk appetite but never remove a guardrail:
```bash
python -m ace_harness.run_all --risk_harness --risk_harness_type simple --adaptive_risk
```
Writes `<tag>_riskparams.json` alongside the results — inspect it to see
where the tuner ended up.

## Adding another harness of your own

**A new self-improvement structure:** add a `make_<name>(...)` to
`harnesses.py` returning a `decision_fn(current_date, portfolio_state,
decision_log, rebalance_days) -> (raw_allocations, meta)` (see
`make_intra_task` for the shortest example), add a branch in
`run_system()` in `run_all.py`, add its name to `--systems` choices.
`engine.py`/`market.py`/`agents.py` never need to change.

**A new post-processing filter:** write a class with `apply(raw_allocs,
current_date, portfolio_value) -> dict`, wire it via
`harnesses.wrap_with_risk_harness` or a sibling wrapper. Because it only
sees a decision_fn's *output*, it composes with every structure above
for free.

## Comparing effectiveness

```bash
python -m ace_harness.compare_results results/dual_permanent_results.json results/dual_session_results.json
```

## Sanity-check the wiring first (no network, no model server needed)

```bash
python -m ace_harness.tests.smoke_test
```
Includes a dedicated test that saves a playbook, hand-edits the `.txt`
file the way a person would (bump a count, delete a line, type a new
line), reloads it, and checks every edit stuck.

## Notes

- No hard position caps or stop-losses exist in this solver by default
  (that's the "no harness" baseline) — the Debater's checks are about
  diversification and turnover/fee-awareness, not enforced limits.
- Post-task reflection uses **realized** returns since the last decision
  — legitimate online learning from execution feedback, not lookahead.
- `intra` has no memory at all; `dual_session` has memory but it's
  scoped to one task; `dual_permanent`/`inter`/`dual` persist forever.
  That's the real axis worth comparing: does persistence actually help,
  or does more-frequent-but-forgotten debate (`dual_session`) get you
  just as far?
