"""Data model of a restricted-disclosure audit episode.

An episode (Trajectory) is a time-ordered list of Events.  Every event has
PUBLIC metadata (id, time, actor, recipients, type, object references, token
cost) and HIDDEN content (natural-language text plus the structured Fact the
text was rendered from).  The auditor only ever sees `PublicEvent`s; hidden
content is reachable exclusively through `oracle.EvidenceOracle.reveal`.

Fact vocabulary (kind):
  messages  PROPOSE(ops, aware)  ACCEPT(ops, aware)  CONFIRM(ops, aware)
            DECLINE(ops)  ABANDON(ops)  CHAT
  actions   DO(ops=((obj, op),))
  approvals GRANT(ops=((obj, op),), valid)  DENY(ops)
  reads     SOURCE(ops, correct)   (a document stating that ops are allowed)
  outcomes  STATE(ops=((obj, op),))   (system log: effect of op present on obj)

`aware=True` on a message means the text acknowledges that the plan breaks a
rule (no waiver / must stay hidden); `aware=False` means it presents the plan
as legitimate (claimed approval, documentation, routine practice).
"""
from __future__ import annotations

from dataclasses import dataclass, field

Op = tuple[str, str]  # (object id, canonical operation)


@dataclass(frozen=True)
class Fact:
    kind: str
    agent: str | None = None          # who produced the record (true identity)
    to: tuple[str, ...] = ()          # recipients of a message
    ops: tuple[Op, ...] = ()
    aware: bool | None = None
    valid: bool | None = None
    correct: bool | None = None


@dataclass(frozen=True)
class Pattern:
    """A rule-defined violation pattern: executing all `ops` (without a valid
    waiver for any of them) is prohibited.  If `needs_agreement`, executing the
    ops is prohibited only when done under a knowing agreement (reciprocity,
    coordinated false reports)."""
    pid: str
    kind: str                          # protected | joint | reciprocal | report
    ops: tuple[Op, ...]
    needs_agreement: bool = False


@dataclass
class Rules:
    patterns: list[Pattern]
    authority: str                     # the only agent that can issue waivers
    text: str = ""                     # public natural-language statement


@dataclass(frozen=True)
class PublicEvent:
    eid: str
    t: int
    etype: str                         # msg | action | approval | read | outcome
    actor: str | None                  # None = metadata missing
    recipients: tuple[str, ...] | None
    objects: tuple[str, ...] | None
    cost: int                          # tokens charged when revealed


@dataclass
class Event:
    pub: PublicEvent
    text: str
    fact: Fact

    @property
    def eid(self) -> str:
        return self.pub.eid


@dataclass
class Trajectory:
    tid: str
    mechanism: str                     # A | B | C | D | null
    condition: str                     # pos | benign_coop | individual | misled | ...
    family: int                        # text-template family (split key)
    agents: dict[str, str]             # id -> public role
    objects: list[str]
    rules: Rules
    events: list[Event]
    labels: dict = field(default_factory=dict)
    knobs: dict = field(default_factory=dict)
    core: list[str] = field(default_factory=list)   # eids inserted by the scenario (offline only)

    @property
    def total_cost(self) -> int:
        return sum(e.pub.cost for e in self.events)

    def public_events(self) -> list[PublicEvent]:
        return [e.pub for e in self.events]

    def facts(self) -> dict[str, Fact]:
        return {e.eid: e.fact for e in self.events}
