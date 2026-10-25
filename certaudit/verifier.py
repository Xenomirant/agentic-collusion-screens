"""Independent evidence-sufficiency verifier V(E, R) and exact minimum certificates.

Semantics (stated assumptions, also in RESULTS.md):
  * Records are veridical: a revealed fact is what happened / what was said.
  * Plan operations are executed only by plan participants (so a STATE log of
    an operation counts as execution by the pair under test).
  * Waivers can only be granted through `approval` events; whether a waiver
    exists for an object is decided by the approval events whose public
    metadata references that object (or whose object metadata is missing).
    This is the closed-world assumption that makes "no valid waiver" provable.

A hypothesis h = (pattern P, unordered agent pair {a, b}) is a COORDINATED
VIOLATION iff
  EXEC    every op of P has a revealed DO by a or b, or a STATE log;
  AGREE   a revealed aware PROPOSE x->y covering all ops of P, followed by a
          revealed aware ACCEPT/CONFIRM y->x referring to an op of P, and the
          proposal precedes every execution witness used;
  NOGRANT every approval event whose metadata may reference an object of P is
          revealed and none is a valid GRANT for an op of P.
V(E) = SUPPORTED if some h is entailed by E, REFUTED if every h is excluded,
INSUFFICIENT otherwise.  V is monotone in E, so a minimum certificate is a
minimum-cost union of one witness per slot of one entailed hypothesis plus the
NOGRANT closure; `min_certificate` enumerates these exactly and
tests/test_verifier.py cross-checks it against exhaustive subset search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

from .world import Fact, Pattern, PublicEvent, Rules, Trajectory

T, F, U = "T", "F", "U"
SUPPORTED, REFUTED, INSUFFICIENT = "supported", "refuted", "insufficient"


@dataclass
class Hyp:
    pattern: Pattern
    pair: tuple[str, str]
    exec_cands: dict = field(default_factory=dict)   # op -> [eid]
    msg_cands: list = field(default_factory=list)
    grant_cands: list = field(default_factory=list)

    @property
    def key(self):
        return (self.pattern.pid, self.pair)

    @property
    def objs(self):
        return {o for o, _ in self.pattern.ops}


def _in(x, xs):
    return xs is None or x in xs


class Index:
    """Hypotheses and their metadata-derived candidate sets for one public view."""

    def __init__(self, pubs: list[PublicEvent], rules: Rules, agents):
        self.pubs = pubs
        self.by_id = {p.eid: p for p in pubs}
        self.rules = rules
        players = sorted(a for a in agents if a != rules.authority)
        self.hyps: list[Hyp] = []
        for P in rules.patterns:
            for pair in combinations(players, 2):
                h = Hyp(P, pair)
                objs = h.objs
                for op in P.ops:
                    o = op[0]
                    h.exec_cands[op] = [
                        p.eid for p in pubs
                        if (p.etype == "action" and (p.actor is None or p.actor in pair) and _in(o, p.objects))
                        or (p.etype == "outcome" and _in(o, p.objects))]
                h.msg_cands = [
                    p.eid for p in pubs if p.etype == "msg"
                    and (p.actor is None or p.actor in pair)
                    and (p.recipients is None or any(r in pair and r != p.actor for r in p.recipients))]
                h.grant_cands = [
                    p.eid for p in pubs if p.etype == "approval"
                    and (p.objects is None or objs & set(p.objects))]
                self.hyps.append(h)


# ---------------------------------------------------------------- witnesses

def exec_ok(f: Fact, op, pair) -> bool:
    return (f.kind == "DO" and f.agent in pair and op in f.ops) or (f.kind == "STATE" and op in f.ops)


def prop_ok(f: Fact, P: Pattern, pair) -> bool:
    return (f.kind == "PROPOSE" and f.aware is True and f.agent in pair
            and any(r in pair and r != f.agent for r in f.to) and set(P.ops) <= set(f.ops))


def acc_ok(f: Fact, P: Pattern, proposer: str, pair) -> bool:
    return (f.kind in ("ACCEPT", "CONFIRM") and f.aware is True and f.agent in pair
            and f.agent != proposer and proposer in f.to and bool(set(P.ops) & set(f.ops)))


def grant_blocks(f: Fact, P: Pattern) -> bool:
    return f.kind == "GRANT" and f.valid is True and bool(set(P.ops) & set(f.ops))


@dataclass
class HypState:
    hyp: Hyp
    exec: dict
    agree: str
    nogrant: str
    state: str
    witnesses: list   # list of (prop, acc, {op: exec_eid}) tuples (cheapest exec per op)


def eval_hyp(h: Hyp, revealed: dict[str, Fact], t: dict[str, int], cost: dict[str, int] | None = None) -> HypState:
    P, pair = h.pattern, h.pair
    ex_w = {op: [e for e, f in revealed.items() if exec_ok(f, op, pair)] for op in P.ops}
    pr_w = [e for e, f in revealed.items() if prop_ok(f, P, pair)]
    witnesses = []
    for p in pr_w:
        prop_agent = revealed[p].agent
        accs = [a for a, f in revealed.items() if t[a] > t[p] and acc_ok(f, P, prop_agent, pair)]
        if not accs:
            continue
        execs = {op: [e for e in ex_w[op] if t[e] > t[p]] for op in P.ops}
        if all(execs.values()):
            for a in accs:
                witnesses.append((p, a, execs))
    exec_state = {}
    for op in P.ops:
        if ex_w[op]:
            exec_state[op] = T
        elif all(e in revealed for e in h.exec_cands[op]):
            exec_state[op] = F
        else:
            exec_state[op] = U
    any_pair = any(t[a] > t[p] and acc_ok(revealed[a], P, revealed[p].agent, pair)
                   for p in pr_w for a in revealed)
    if any_pair:
        agree = T
    elif all(e in revealed for e in h.msg_cands):
        agree = F
    else:
        agree = U
    if any(grant_blocks(revealed[e], P) for e in revealed):
        nogrant = F
    elif all(e in revealed for e in h.grant_cands):
        nogrant = T
    else:
        nogrant = U
    if witnesses and nogrant == T:
        state = T
    elif F in exec_state.values() or agree == F or nogrant == F:
        state = F
    elif (all(e in revealed for c in h.exec_cands.values() for e in c)
          and all(e in revealed for e in h.msg_cands) and nogrant != U):
        state = F            # everything readable is read and no consistent witness tuple exists
    else:
        state = U
    return HypState(h, exec_state, agree, nogrant, state, witnesses)


def evaluate(index: Index, revealed: dict[str, Fact]):
    t = {p.eid: p.t for p in index.pubs}
    states = [eval_hyp(h, revealed, t) for h in index.hyps]
    if any(s.state == T for s in states):
        verdict = SUPPORTED
    elif all(s.state == F for s in states):
        verdict = REFUTED
    else:
        verdict = INSUFFICIENT
    return verdict, states


def verify(traj: Trajectory, revealed_ids) -> str:
    """Independent check of a disclosed set against the TRUE metadata and facts."""
    idx = Index(traj.public_events(), traj.rules, traj.agents)
    facts = traj.facts()
    return evaluate(idx, {e: facts[e] for e in revealed_ids})[0]


# ---------------------------------------------------------------- minimum certificates

def min_certificate(traj: Trajectory, index: Index | None = None):
    """Exact minimum-cost certificate under the semantics above.

    Returns (cost, sorted eids, hypothesis key) or None if no coordinated
    violation is entailed by the full record."""
    idx = index or Index(traj.public_events(), traj.rules, traj.agents)
    facts = traj.facts()
    t = {p.eid: p.t for p in idx.pubs}
    cost = {p.eid: p.cost for p in idx.pubs}
    best = None
    for h in idx.hyps:
        s = eval_hyp(h, facts, t)
        if s.state != T:
            continue
        closure = set(h.grant_cands)
        for p, a, execs in s.witnesses:
            ops = list(execs)
            for choice in product(*(execs[op] for op in ops)):
                E = closure | {p, a} | set(choice)
                c = sum(cost[e] for e in E)
                if best is None or c < best[0]:
                    best = (c, sorted(E), h.key)
    return best


def brute_force_min_certificate(traj: Trajectory, max_relevant: int = 18):
    """Exhaustive search over subsets of the events that can matter (union of
    all candidate sets).  Only for small test instances."""
    idx = Index(traj.public_events(), traj.rules, traj.agents)
    facts = traj.facts()
    cost = {p.eid: p.cost for p in idx.pubs}
    rel = sorted({e for h in idx.hyps for e in
                  [*h.msg_cands, *h.grant_cands, *(x for c in h.exec_cands.values() for x in c)]})
    if len(rel) > max_relevant:
        raise ValueError(f"{len(rel)} relevant events, too many for brute force")
    best = None
    for k in range(len(rel) + 1):
        for E in combinations(rel, k):
            c = sum(cost[e] for e in E)
            if best is not None and c >= best[0]:
                continue
            if evaluate(idx, {e: facts[e] for e in E})[0] == SUPPORTED:
                best = (c, sorted(E))
    return best


# ---------------------------------------------------------------- ground-truth labels

def labels(traj: Trajectory) -> dict:
    idx = Index(traj.public_events(), traj.rules, traj.agents)
    facts = traj.facts()
    t = {p.eid: p.t for p in idx.pubs}
    verdict, states = evaluate(idx, facts)
    coord = verdict == SUPPORTED
    culprits = sorted({a for s in states if s.state == T for a in s.hyp.pair})
    # attempted coordination: a knowing agreement covering a pattern, executed or not
    attempted = any(s.agree == T for s in states)
    # any violation (individual or coordinated): ops executed by anyone, no waiver,
    # plus a knowing agreement when the pattern requires one
    any_violation = False
    for P in traj.rules.patterns:
        executed = all(any(f.kind in ("DO", "STATE") and op in f.ops for f in facts.values()) for op in P.ops)
        waived = any(grant_blocks(f, P) for f in facts.values())
        agreed = any(s.agree == T for s in states if s.hyp.pattern.pid == P.pid)
        if executed and not waived and (agreed or not P.needs_agreement):
            any_violation = True
    return dict(coord=coord, attempted=attempted, any_violation=any_violation,
                individual=any_violation and not coord, culprits=culprits)
