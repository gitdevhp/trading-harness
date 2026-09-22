"""Experience Memory = the ACE "playbook": sectioned bullets, each with
an id and helpful/harmful counters, updated via small delta ops (add /
update-counts / replace / remove) instead of a full rewrite each time.

Persisted as a plain, human-readable .txt file (not JSON) — open it,
delete a bad line, hand-edit a helpful/harmful count, or just type a new
line under a section heading, and it's picked up on the next load. Lines
that don't parse are treated as freshly hand-added bullets rather than
silently dropped.

Two different views exist on purpose:
  format_for_prompt() — what the Solver/Debater see: ranked, and
      "soft-hides" bullets with a clearly negative track record (still
      on disk, just not shown) before prune() eventually deletes them.
  format_all_for_review() — what the Consolidator sees: EVERYTHING,
      including soft-hidden bullets, so it has full context to avoid
      recreating something already discredited.
"""
import itertools
import os
import re

DEFAULT_SECTIONS = [
    "STRATEGIES & INSIGHTS",
    "CONVICTION SIGNALS",
    "RISK LESSONS",
    "MARKET REGIME NOTES",
    "MISTAKES TO AVOID",
]

# Hard safety cap: even if a single Consolidator call returns a huge/
# malformed ops list, only the first MAX_OPS_PER_CALL are ever applied.
MAX_OPS_PER_CALL = 8

_SECTION_RE = re.compile(r"^##\s+(.*)$")
_BULLET_RE = re.compile(r"^\[(?P<id>[^\]]+)\]\s+helpful=(?P<helpful>-?\d+)\s+harmful=(?P<harmful>-?\d+)\s*::\s*(?P<content>.*)$")


class ExperienceMemory:
    def __init__(self, sections=None):
        self.sections = sections or list(DEFAULT_SECTIONS)
        self.bullets = {}  # id -> {"section", "content", "helpful", "harmful"}
        self._counter = itertools.count(1)
        # AdaReMo SATURATED tracking: record bullet count after each consolidation
        # so is_saturated() can detect when the memory has stopped growing.
        self._consolidation_sizes = []  # bullet counts after each apply_delta_ops call

    def _new_id(self, section: str) -> str:
        n = next(self._counter)
        slug = "".join(c for c in section.split()[0].lower() if c.isalnum())[:3] or "gen"
        return f"{slug}-{n:05d}"

    def add_bullet(self, section: str, content: str) -> str:
        if section not in self.sections:
            self.sections.append(section)
        bid = self._new_id(section)
        self.bullets[bid] = {"section": section, "content": content.strip(), "helpful": 0, "harmful": 0}
        return bid

    def replace_bullet(self, bullet_id: str, content: str):
        """Rewrite a bullet's wording in place (e.g. to generalize or merge
        it with another) while keeping its accumulated helpful/harmful
        track record — a wording refinement isn't a new claim."""
        content = (content or "").strip()
        if bullet_id in self.bullets and content:
            self.bullets[bullet_id]["content"] = content

    def update_counts(self, bullet_id: str, helpful: int = 0, harmful: int = 0):
        if bullet_id in self.bullets:
            self.bullets[bullet_id]["helpful"] += helpful
            self.bullets[bullet_id]["harmful"] += harmful

    def prune(self, min_net_score: int = -2, max_bullets_per_section: int = 25):
        to_remove = [
            bid for bid, b in self.bullets.items()
            if (b["helpful"] - b["harmful"]) <= min_net_score and (b["helpful"] + b["harmful"]) >= 3
        ]
        for bid in to_remove:
            self.bullets.pop(bid, None)
        for section in self.sections:
            ids = [bid for bid, b in self.bullets.items() if b["section"] == section]
            ids.sort(key=lambda i: self.bullets[i]["helpful"] - self.bullets[i]["harmful"], reverse=True)
            for bid in ids[max_bullets_per_section:]:
                self.bullets.pop(bid, None)

    def seed_with_principles(self):
        """Pre-populate memory with foundational momentum-signal principles so
        memory-based systems start with actionable heuristics rather than an empty
        playbook. Only seeds if memory is currently empty, so a loaded playbook
        is never clobbered."""
        if self.bullets:
            return
        seeds = [
            ("CONVICTION SIGNALS",
             "When 6M-Mom > 15% AND price is above both 50d-SMA and 200d-SMA, strong trending regime justifies a position >= 12%"),
            ("CONVICTION SIGNALS",
             "When 1M-Mom and 6M-Mom are both positive and price is above 200d-SMA, momentum is confirmed — sizing up to 15% is warranted"),
            ("RISK LESSONS",
             "When price is below 200d-SMA AND 6M-Mom < -5%, regime is bearish — cap that asset at 5% or less"),
            ("RISK LESSONS",
             "When 1M-Vol > 35% on an asset already allocated >= 15%, reduce to under 10% to control tail risk"),
            ("STRATEGIES & INSIGHTS",
             "Monthly 15bps fee means turnover above 60% costs over 9bps net drag — avoid churning positions that changed < 5% in fundamentals"),
            ("MARKET REGIME NOTES",
             "When the majority of assets show price below 50d-SMA, broad market weakness — raise CASH above 20%"),
            ("MISTAKES TO AVOID",
             "Allocating > 20% to any asset whose 6M-Mom is negative and price is below 50d-SMA has historically produced losses"),
        ]
        for section, content in seeds:
            self.add_bullet(section, content)

    def is_saturated(self, window: int = 5, min_growth: int = 1) -> bool:
        """AdaReMo Algorithm 2: SATURATED(M).

        Returns True when the last `window` consolidation calls collectively
        added fewer than `min_growth` net new bullets — the memory has stopped
        learning new reusable structure and further consolidation would only
        add prompt cost without knowledge gain.

        Never fires until at least `window` consolidations have happened, so
        early tasks always write their lessons; this matches the paper's
        observation that SATURATED "never fired" within their task streams.

        Only consolidation calls made with track_saturation=True (the default,
        used by the fast/intra-task loop) count toward this window. Slow-loop
        inter-task consolidations pass track_saturation=False so realized-outcome
        learning doesn't pollute the fast-loop saturation signal.
        """
        # Need window+1 entries to measure growth across exactly window intervals.
        if len(self._consolidation_sizes) < window + 1:
            return False
        oldest = self._consolidation_sizes[-(window + 1)]
        current = len(self.bullets)
        return (current - oldest) < min_growth

    def apply_delta_ops(self, ops, track_saturation: bool = True):
        """ops: list of dicts, one of
            {"op": "add", "section": str, "content": str}
            {"op": "update", "id": str, "helpful": int, "harmful": int}
            {"op": "replace", "id": str, "content": str}
            {"op": "remove", "id": str}
        Capped at MAX_OPS_PER_CALL regardless of what's passed. Malformed
        ops are silently skipped so a flaky LLM response never crashes
        the backtest.

        track_saturation: pass False from slow/inter-task loops so their
        consolidation calls don't count toward SATURATED(M). The saturation
        signal should only reflect fast-loop (admitted AND g_sto) consolidations.
        """
        for op in (ops or [])[:MAX_OPS_PER_CALL]:
            kind = op.get("op") if isinstance(op, dict) else None
            if kind == "add":
                content = (op.get("content") or "").strip()
                if content:
                    self.add_bullet(op.get("section", "STRATEGIES & INSIGHTS"), content)
            elif kind == "update":
                self.update_counts(op.get("id", ""), int(op.get("helpful", 0) or 0), int(op.get("harmful", 0) or 0))
            elif kind == "replace":
                self.replace_bullet(op.get("id", ""), op.get("content", ""))
            elif kind == "remove":
                self.bullets.pop(op.get("id", ""), None)
        self.prune()
        if track_saturation:
            # Record bullet count after this consolidation for SATURATED tracking.
            self._consolidation_sizes.append(len(self.bullets))

    def _sorted_entries(self, section):
        entries = [(bid, b) for bid, b in self.bullets.items() if b["section"] == section]
        entries.sort(key=lambda kv: (kv[1]["helpful"] - kv[1]["harmful"], kv[1]["helpful"] + kv[1]["harmful"]), reverse=True)
        return entries

    def format_for_prompt(self, max_bullets_per_section: int = 10) -> str:
        """Ranked view for the Solver/Debater. Soft-hides bullets with a
        clearly negative, evidenced track record (net <= -1 with >= 2
        votes) — they stay on disk and in format_all_for_review() until
        prune() physically deletes them, but they're not shown as if they
        were good advice in the meantime."""
        if not self.bullets:
            return "(Experience memory is empty — no prior lessons yet.)"
        lines = []
        for section in self.sections:
            entries = self._sorted_entries(section)
            visible = [(bid, b) for bid, b in entries
                       if not ((b["helpful"] - b["harmful"]) <= -1 and (b["helpful"] + b["harmful"]) >= 2)]
            if not visible:
                continue
            lines.append(f"## {section}")
            for bid, b in visible[:max_bullets_per_section]:
                lines.append(f"[{bid}] helpful={b['helpful']} harmful={b['harmful']} :: {b['content']}")
        return "\n".join(lines) if lines else "(Experience memory is empty — no prior lessons yet.)"

    def format_all_for_review(self) -> str:
        """Unfiltered view, for the Consolidator only — it needs to see
        soft-hidden/harmful bullets too, so it can recognize a duplicate
        of something already discredited instead of re-adding it."""
        if not self.bullets:
            return "(empty)"
        lines = []
        for section in self.sections:
            entries = self._sorted_entries(section)
            if not entries:
                continue
            lines.append(f"## {section}")
            for bid, b in entries:
                lines.append(f"[{bid}] helpful={b['helpful']} harmful={b['harmful']} :: {b['content']}")
        return "\n".join(lines)

    def save(self, path: str):
        lines = [
            "# Experience Memory Playbook",
            "# Auto-maintained by the Consolidator. Hand edits are preserved on reload:",
            "# edit a helpful=/harmful= count, delete a line, or add a new plain-text",
            "# line under a section heading (it becomes a new, untested bullet).",
            "",
        ]
        for section in self.sections:
            entries = self._sorted_entries(section)
            if not entries:
                continue
            lines.append(f"## {section}")
            for bid, b in entries:
                lines.append(f"[{bid}] helpful={b['helpful']} harmful={b['harmful']} :: {b['content']}")
            lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines).rstrip() + "\n")

    @classmethod
    def load(cls, path: str) -> "ExperienceMemory":
        mem = cls()
        if not os.path.exists(path):
            return mem

        with open(path) as f:
            raw_lines = f.readlines()

        current_section = None
        max_n = 0
        pending = []  # hand-added plain-text lines, assigned real ids after the counter is known

        for raw in raw_lines:
            line = raw.rstrip("\n").strip()
            if not line:
                continue
            if line.startswith("#") and not line.startswith("##"):
                continue  # comment line

            section_match = _SECTION_RE.match(line)
            if section_match:
                current_section = section_match.group(1).strip()
                if current_section not in mem.sections:
                    mem.sections.append(current_section)
                continue

            if current_section is None:
                continue  # stray content before any section heading

            bullet_match = _BULLET_RE.match(line)
            if bullet_match:
                bid = bullet_match.group("id")
                mem.bullets[bid] = {
                    "section": current_section,
                    "content": bullet_match.group("content").strip(),
                    "helpful": int(bullet_match.group("helpful")),
                    "harmful": int(bullet_match.group("harmful")),
                }
                try:
                    max_n = max(max_n, int(bid.split("-")[-1]))
                except ValueError:
                    pass
            else:
                # a hand-typed line with no [id]/counts — adopt it as a new bullet
                pending.append((current_section, line))

        mem._counter = itertools.count(max_n + 1)
        for section, content in pending:
            mem.add_bullet(section, content)

        return mem
