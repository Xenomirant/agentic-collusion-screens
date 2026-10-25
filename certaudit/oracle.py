"""Evidence oracle: the only path from an auditor to hidden content.

The auditor receives a `PublicView` (rules, agents, public metadata) and an
`EvidenceOracle`.  Hidden text/facts live in the oracle's private trajectory;
`reveal(eid)` returns them, charges the record's token cost against the
budget and appends to an auditable request log.  Ablations are applied to the
VIEW (what metadata the auditor sees, which records exist for it); the
independent verifier always uses the true record.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .world import Fact, PublicEvent, Rules, Trajectory


class BudgetExceeded(Exception):
    pass


@dataclass
class PublicView:
    tid: str
    rules: Rules
    agents: dict
    pubs: list[PublicEvent]
    total_cost: int                 # of the full original record (budget base)

    @property
    def cost(self):
        return {p.eid: p.cost for p in self.pubs}


@dataclass
class Revealed:
    eid: str
    text: str
    fact: Fact                      # as READ (perfect or LLM-extracted), not necessarily true


class PerfectReader:
    name = "perfect"

    def read(self, traj: Trajectory, eid: str) -> Fact:
        return next(e.fact for e in traj.events if e.eid == eid)


class CachedReader:
    """Reader backed by a precomputed {(tid, eid): Fact} map (e.g. LLM extraction)."""

    def __init__(self, name, table: dict):
        self.name = name
        self.table = table

    def read(self, traj, eid):
        return self.table[(traj.tid, eid)]


def make_view(traj: Trajectory, ablation: str | None = None, seed: int = 0) -> PublicView:
    pubs = traj.public_events()
    rng = np.random.default_rng(seed)
    if ablation in (None, "none"):
        pass
    elif ablation == "shuffle_links":
        # permute who-talks-to-whom among messages and who-acted among actions
        msgs = [i for i, p in enumerate(pubs) if p.etype == "msg"]
        acts = [i for i, p in enumerate(pubs) if p.etype == "action"]
        pm = rng.permutation(msgs)
        pa = rng.permutation(acts)
        new = list(pubs)
        for i, j in zip(msgs, pm):
            new[i] = replace(pubs[i], actor=pubs[j].actor, recipients=pubs[j].recipients)
        for i, j in zip(acts, pa):
            new[i] = replace(pubs[i], actor=pubs[j].actor)
        pubs = new
    elif ablation == "no_provenance":
        pubs = [replace(p, actor=None, recipients=None if p.etype == "msg" else p.recipients,
                        objects=None if p.objects else p.objects) for p in pubs]
    elif ablation == "no_comms":
        pubs = [p for p in pubs if p.etype != "msg"]
    elif ablation == "no_actions":
        pubs = [p for p in pubs if p.etype != "action"]
    elif ablation == "hide_outcomes":
        pubs = [p for p in pubs if p.etype != "outcome"]
    else:
        raise ValueError(ablation)
    return PublicView(traj.tid, traj.rules, traj.agents, pubs, traj.total_cost)


class EvidenceOracle:
    def __init__(self, traj: Trajectory, view: PublicView, reader, budget: float | None = None):
        self.__traj = traj
        self._reader = reader
        self._allowed = {p.eid: p.cost for p in view.pubs}
        self.budget = float("inf") if budget is None else budget
        self.spent = 0
        self.log: list[tuple[str, int, int]] = []     # (eid, cost, cumulative)
        self._seen: dict[str, Revealed] = {}

    @property
    def remaining(self):
        return self.budget - self.spent

    def reveal(self, eid: str) -> Revealed:
        if eid in self._seen:
            return self._seen[eid]
        if eid not in self._allowed:
            raise KeyError(f"{eid} is not disclosable in this view")
        c = self._allowed[eid]
        if self.spent + c > self.budget:
            raise BudgetExceeded(eid)
        self.spent += c
        self.log.append((eid, c, self.spent))
        ev = next(e for e in self.__traj.events if e.eid == eid)
        r = Revealed(eid, ev.text, self._reader.read(self.__traj, eid))
        self._seen[eid] = r
        return r

    @property
    def revealed(self) -> dict[str, Revealed]:
        return dict(self._seen)
