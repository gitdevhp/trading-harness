"""Governance primitives for ADAP-v2 (make_autogover_v2). No LLM calls here.

ADAP-v2 keeps AutoGovern's three governance points and grounds them in the
one signal finance offers that AppWorld/Formula do not: the realized outcome
of a decision, which becomes known one rebalance later.

  1. Refinement (g_ref)  — the critic may stop refinement early; a round that
     no longer moves the proposal also stops it; under submit_policy='last'
     the critic is not consulted on the final round, where its verdict cannot
     change what is executed.
  2. Storage (g_sto)     — the critic only NOMINATES lessons. They are graded
     against next month's realized returns and stored only if confirmed.
  3. Freeze with probes  — memory updates stop when recent consolidation
     demand falls below a threshold; periodic probes can reopen memory.
Plus evidence-gated exposure: the Solver/critic only see bullets whose
current text realized data has confirmed (a Curator rewrite of a bullet's
text sends it back to probation until it is confirmed again).

Notes for the paper:
  * The critic judges should_store novelty against what it is shown, i.e. the
    validated subset; nominations are still deduplicated by the Curator, which
    sees the full memory, before anything is stored.
  * The Reflector gets a larger output budget (800 vs 550 tokens) only when it
    must also grade candidates/probationary bullets — part of the mechanism.
  * All GovernanceConfig defaults below were fixed before any ADAP-v2 run (not
    tuned on any evaluation period). The critic prompt itself is the one shared
    with DEBATER/DUAL; it was last revised on 2026-09-22 (b2c9a69) after
    inspecting 2024-Q1 debug runs, which applies to every critic-based system.

Each mechanism is one GovernanceConfig field, so every mechanism can be
ablated, and the 'dual_equiv' / 'debater_equiv' presets reproduce DUAL and
DEBATER exactly (same LLM call sequence).
"""
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass

_CONF_RANK = {"low": 0, "medium": 1, "high": 2}
_CHOICES = {
    "submit_policy": ("last", "r1_unless_admitted"),
    "memory": ("on", "off"),
    "storage_mode": ("deferred_verified", "admitted_gsto", "immediate_gsto", "immediate_admitted", "immediate_all"),
    "stage_min_confidence": tuple(_CONF_RANK),
    "exposure_mode": ("evidence_gated", "all"),
    "freeze_mode": ("window_probe", "off"),
    "demand_signal": ("gsto", "adds"),
}


@dataclass(frozen=True)
class GovernanceConfig:
    max_rounds: int = 3
    submit_policy: str = "last"            # execute the last proposal, as DEBATER and DUAL do
    use_critic_gref: bool = True           # critic should_refine=false stops refinement
    skip_terminal_review: bool = True      # no critic call on round K (its verdict cannot change the trade)
    convergence_eps_pp: float = 2.0        # stop when a revision moves < this many pp (L1); 0 disables
    memory: str = "on"
    storage_mode: str = "deferred_verified"
    stage_requires_admission: bool = False
    stage_min_confidence: str = "medium"
    max_staged_lessons: int = 5
    exposure_mode: str = "evidence_gated"
    exposure_min_net: int = 1              # realized confirmations minus contradictions needed to be shown
    max_probation_shown: int = 10          # unvalidated bullets offered to the Reflector for grading
    reflect_bullet_check_cap: int = 10
    freeze_mode: str = "window_probe"
    freeze_window: int = 3
    freeze_tau: float = 1 / 3
    probe_every: int = 3
    demand_signal: str = "gsto"
    debater_sizing_context: bool = False   # shared with DEBATER/DUAL via --debater_sizing_context
    use_risk_tuner: bool = False           # follows run_monthly's rule for memory systems

    def __post_init__(self):
        for field, allowed in _CHOICES.items():
            if getattr(self, field) not in allowed:
                raise ValueError(f"{field} must be one of {allowed}, got {getattr(self, field)!r}")
        if self.max_rounds < 1 or self.freeze_window < 1 or self.probe_every < 1:
            raise ValueError("max_rounds, freeze_window and probe_every must be >= 1")


_OFF_CONTROL = dict(use_critic_gref=False, skip_terminal_review=False, convergence_eps_pp=0.0)
PRESETS = {
    "v2": {},
    # control-flow governance without memory: isolates stops from memory
    "debater_stop": dict(memory="off", freeze_mode="off"),
    # exact re-implementations, used by the equivalence tests
    "debater_equiv": dict(memory="off", freeze_mode="off", **_OFF_CONTROL),
    "dual_equiv": dict(storage_mode="immediate_all", exposure_mode="all", freeze_mode="off", **_OFF_CONTROL),
    # faithful ReMo Algorithm 1: consolidate a task's lessons only if admitted
    "remo": dict(storage_mode="immediate_admitted", exposure_mode="all", freeze_mode="off", **_OFF_CONTROL),
    # faithful AutoGovern (the peer's Algorithm 2): critic g_ref stop; consolidate
    # iff admitted AND g_sto AND not frozen; sliding-window freeze with probes;
    # every attempt reviewed; all memory shown. 'v2' = this + outcome grounding.
    "autogov": dict(storage_mode="admitted_gsto", exposure_mode="all", skip_terminal_review=False,
                    convergence_eps_pp=0.0),
}


def make_config(preset="v2", overrides=None, **fixed) -> GovernanceConfig:
    values = dict(PRESETS[preset])
    values.update(fixed)
    values.update(overrides or {})
    return GovernanceConfig(**values)


def parse_override(text: str):
    """'key=value' -> (key, typed value), typed by the GovernanceConfig field."""
    key, sep, raw = text.partition("=")
    fields = {f.name: f for f in dataclasses.fields(GovernanceConfig)}
    if not sep or key not in fields:
        raise ValueError(f"unknown governance override {text!r}; fields: {sorted(fields)}")
    kind = type(getattr(GovernanceConfig(), key))
    if kind is bool:
        if raw.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"{key} expects true/false, got {raw!r}")
        return key, raw.lower() in ("true", "1")
    return key, kind(raw)


def config_dict(cfg: GovernanceConfig) -> dict:
    return dataclasses.asdict(cfg)


def config_hash(cfg: GovernanceConfig) -> str:
    return hashlib.sha256(json.dumps(config_dict(cfg), sort_keys=True).encode()).hexdigest()[:12]


def l1_pp(a: dict, b: dict) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def stage_lessons(reviewed_rounds, cfg: GovernanceConfig, admitted: bool):
    """Lessons the critic nominated for storage this task (g_sto): from
    parseable reviews with store_signal, at least stage_min_confidence."""
    if cfg.stage_requires_admission and not admitted:
        return []
    floor = _CONF_RANK[cfg.stage_min_confidence]
    out, seen = [], set()
    for review in reviewed_rounds:
        if not (review.get("parse_ok") and review.get("store_signal")):
            continue
        for lesson in review.get("lessons") or []:
            key = lesson["lesson"].strip().lower()
            if _CONF_RANK.get(lesson.get("confidence"), 0) >= floor and key not in seen:
                seen.add(key)
                out.append(lesson)
    return out[:cfg.max_staged_lessons]


class FreezeController:
    """Sliding-window freeze with probes. Demand d_t in {0,1} is recorded for
    open tasks (None = no information this task, e.g. the critic was never
    consulted, and is not recorded); once at least `min_slow_loops` outcome
    reflections have run and the mean demand over the last `window` recorded
    tasks is below `tau`, memory freezes. While frozen, every `probe_every`-th
    task is a probe that runs the full memory update; a probe with demand
    reopens memory and resets the window."""

    def __init__(self, window=3, tau=1 / 3, probe_every=3, min_slow_loops=None):
        self.window, self.tau, self.probe_every = window, tau, probe_every
        self.min_slow_loops = window if min_slow_loops is None else min_slow_loops
        self.frozen = False
        self.demand = []
        self.n_slow_loops = 0
        self._frozen_tasks = 0
        self.n_freezes = self.n_probes = self.n_reopens = 0

    def begin_task(self) -> str:
        if not self.frozen:
            return "open"
        self._frozen_tasks += 1
        if self._frozen_tasks % self.probe_every == 0:
            self.n_probes += 1
            return "probe"
        return "frozen"

    def end_task(self, mode, demand, slow_loop_ran):
        if mode == "frozen":
            return None
        if mode == "probe":
            if demand:  # None (no information) keeps memory frozen
                self.frozen, self.demand, self._frozen_tasks = False, [], 0
                self.n_reopens += 1
                return "reopened"
            return "stayed_frozen"
        self.n_slow_loops += int(bool(slow_loop_ran))
        if demand is not None:
            self.demand = (self.demand + [int(bool(demand))])[-self.window:]
        if (self.n_slow_loops >= self.min_slow_loops and len(self.demand) == self.window
                and sum(self.demand) / self.window < self.tau):
            self.frozen, self._frozen_tasks = True, 0
            self.n_freezes += 1
            return "froze"
        return None

    def state(self) -> dict:
        return {"frozen": self.frozen, "demand_window": list(self.demand), "n_freezes": self.n_freezes,
                "n_probes": self.n_probes, "n_reopens": self.n_reopens}


class EvidenceStore:
    """Realized-outcome evidence per memory bullet, counted ONLY from the
    Reflector's bullet_checks (a bullet's claim graded against realized
    returns) — not from the Curator's text-agreement updates."""

    def __init__(self, counts=None):
        self.counts = {bid: dict(c) for bid, c in (counts or {}).items()}

    def apply_checks(self, checks, memory):
        """Only ids that exist now: a check citing a not-yet-created id must
        not pre-validate whichever bullet the Curator creates under it."""
        for check in checks or []:
            if check["id"] in memory.bullets:
                entry = self.counts.setdefault(check["id"], {"conf": 0, "contra": 0})
                entry["conf" if check["verdict"] == "confirmed" else "contra"] += 1

    def reset_confirmations(self, ids):
        """A bullet whose text was rewritten must earn confirmation again;
        contradictions are kept so a discredited bullet stays hidden."""
        for bid in ids:
            self.counts.setdefault(bid, {"conf": 0, "contra": 0})["conf"] = 0

    def sync(self, memory):
        """Drop ids no longer in memory; new bullets start at 0/0."""
        self.counts = {bid: self.counts.get(bid, {"conf": 0, "contra": 0}) for bid in memory.bullets}

    def net(self, bid) -> int:
        c = self.counts.get(bid, {"conf": 0, "contra": 0})
        return c["conf"] - c["contra"]

    def visible_ids(self, memory, min_net):
        return [bid for bid in memory.bullets if self.net(bid) >= min_net]

    def probation_ids(self, memory, min_net, max_n):
        """Not yet validated and not discredited, newest first."""
        ids = [bid for bid in memory.bullets if 0 <= self.net(bid) < min_net]
        ids.sort(key=_id_number, reverse=True)
        return ids[:max_n]

    def counts_for(self, ids):
        return {bid: (self.counts.get(bid, {}).get("conf", 0), self.counts.get(bid, {}).get("contra", 0))
                for bid in ids}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.counts, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path):
            return cls()
        with open(path) as f:
            return cls(json.load(f))


def _id_number(bid):
    try:
        return int(bid.rsplit("-", 1)[-1])
    except ValueError:
        return -1
