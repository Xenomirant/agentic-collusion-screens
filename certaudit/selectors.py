"""Evidence selectors and the audit loop.

All selectors see only the PublicView, the facts they have revealed so far and
(for belief-based ones) the shared BeliefState computed from those.  Each call
returns one affordable, unrevealed event id, or None to stop.

Factorial core (same belief model, same information):
  suspicion   G0R0  individual records, most likely to add positive evidence (q/c)
  voi         G0R1  individual records, myopic expected entropy reduction of P(viol)/c
  comp        G1R0  hypothesis chosen by AND-aggregated value P(h)/E[cost to decide h];
                    inside it, positive slots first, waiver closure last
  combined    G1R1  same hypothesis choice; inside it refutation-first ordering
                    (slot with the smallest E[cost]/P(fail) first): tests the
                    alternative explanations (waiver, awareness, execution) early
Content-agnostic baselines: random, random_groups, graph, caes (a reconstruction of
the earlier cross-agent selector from its description; see RESULTS.md).
"""
from __future__ import annotations

import numpy as np

from .belief import Belief, BeliefState, Structure
from .oracle import EvidenceOracle, PublicView, make_view
from .verifier import REFUTED, SUPPORTED, T, U, verify


def _h(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class Selector:
    name = "base"
    uses_belief = True

    def reset(self, view: PublicView, st: Structure, rng):
        self.view, self.st, self.rng = view, st, rng

    def next(self, bs: BeliefState, revealed: dict, remaining: float):
        raise NotImplementedError

    def affordable(self, e, revealed, remaining):
        return e not in revealed and self.st.cost[e] <= remaining


# ---------------------------------------------------------------- content-agnostic

class RandomSel(Selector):
    name = "random"

    def reset(self, view, st, rng):
        super().reset(view, st, rng)
        self.order = list(rng.permutation([p.eid for p in view.pubs]))

    def next(self, bs, revealed, remaining):
        return next((e for e in self.order if self.affordable(e, revealed, remaining)), None)


def _neighbors(st: Structure, e, window):
    p = st.pub[e]
    out = []
    for q in st.view.pubs:
        if q.eid == e or abs(q.t - p.t) > window:
            continue
        linked = False
        if p.etype == "msg" and q.etype == "msg" and p.actor and q.actor and p.recipients and q.recipients:
            linked = {p.actor, *p.recipients} == {q.actor, *q.recipients}
        people = {p.actor, *(p.recipients or ())} - {None, "system"}
        if q.actor in people and q.etype in ("action", "msg"):
            linked = True
        if p.objects and q.objects and set(p.objects) & set(q.objects):
            linked = True
        if linked:
            out.append(q.eid)
    return sorted(out, key=lambda x: abs(st.t[x] - p.t))


class RandomGroupsSel(Selector):
    name = "random_groups"

    def reset(self, view, st, rng):
        super().reset(view, st, rng)
        self.queue = []

    def next(self, bs, revealed, remaining):
        while True:
            self.queue = [e for e in self.queue if self.affordable(e, revealed, remaining)]
            if self.queue:
                return self.queue.pop(0)
            pool = [p.eid for p in self.view.pubs if self.affordable(p.eid, revealed, remaining)]
            if not pool:
                return None
            seed = pool[int(self.rng.integers(len(pool)))]
            self.queue = [seed] + _neighbors(self.st, seed, 0.15 * self.st.T)[:3]


class GraphSel(Selector):
    """Metadata-graph traversal from actions on rule-referenced objects."""
    name = "graph"

    def reset(self, view, st, rng):
        super().reset(view, st, rng)
        robjs = {o for P in view.rules.patterns for o, _ in P.ops}
        seeds = sorted((p for p in view.pubs if p.etype in ("action", "outcome")
                        and (p.objects is None or set(p.objects) & robjs)), key=lambda p: p.t)
        order, seen = [], set()
        for s in seeds:
            for e in [s.eid] + _neighbors(st, s.eid, 0.15 * st.T)[:4]:
                if e not in seen:
                    seen.add(e)
                    order.append(e)
        order += [p.eid for p in sorted(view.pubs, key=lambda p: p.t) if p.eid not in seen]
        self.order = order

    def next(self, bs, revealed, remaining):
        return next((e for e in self.order if self.affordable(e, revealed, remaining)), None)


class CAESSel(Selector):
    """Reconstruction of the earlier cross-agent evidence selector: rank cross-agent
    provenance links (message x->y followed by an action of x or y on a
    rule-referenced object; approval followed by an action on the same object)
    by temporal proximity and reveal both endpoints of the strongest links first."""
    name = "caes"

    def reset(self, view, st, rng):
        super().reset(view, st, rng)
        robjs = {o for P in view.rules.patterns for o, _ in P.ops}
        acts = [p for p in view.pubs if p.etype in ("action", "outcome")
                and (p.objects is None or set(p.objects) & robjs)]
        links = []
        W = 0.3 * st.T
        for m in view.pubs:
            if m.etype == "msg":
                people = {m.actor, *(m.recipients or ())} - {None}
                for a in acts:
                    if 0 < a.t - m.t <= W and (a.actor in people or a.actor is None or not people):
                        links.append((1 / (1 + a.t - m.t), m.eid, a.eid))
            elif m.etype == "approval":
                for a in acts:
                    if 0 < a.t - m.t <= W and m.objects and a.objects and set(m.objects) & set(a.objects):
                        links.append((1 / (1 + a.t - m.t), m.eid, a.eid))
        links.sort(key=lambda x: -x[0])
        order, seen = [], set()
        for _, x, y in links:
            for e in (x, y):
                if e not in seen:
                    seen.add(e)
                    order.append(e)
        order += [p.eid for p in sorted(view.pubs, key=lambda p: p.t) if p.eid not in seen]
        self.order = order

    def next(self, bs, revealed, remaining):
        return next((e for e in self.order if self.affordable(e, revealed, remaining)), None)


# ---------------------------------------------------------------- belief-based

def _cand_scores(bs: BeliefState, roles=("exec", "prop", "acc")):
    """max q per candidate eid over open hypotheses' unresolved positive slots."""
    best = {}
    for hi, slots in enumerate(bs.slots):
        if bs.p_h[hi] <= 0 or bs.hstates[hi].state != U:
            continue
        for s in slots:
            if s.closure or s.state != U:
                continue
            for e, q in zip(s.cands, s.q):
                if q > best.get(e, -1):
                    best[e] = q
    return best


class SuspicionSel(Selector):
    name = "suspicion"

    def next(self, bs, revealed, remaining):
        # a hypothesis whose positive evidence is complete: check its waivers to certify
        for hi in np.argsort(-bs.p_h):
            slots = bs.slots[hi]
            if bs.hstates[hi].state == U and bs.p_h[hi] > 0 and all(s.state == T for s in slots if not s.closure):
                g = next(s for s in slots if s.closure)
                e = next((e for e in g.cands if self.affordable(e, revealed, remaining)), None)
                if e is not None:
                    return e
        sc = _cand_scores(bs)
        cands = [(q / self.st.cost[e], e) for e, q in sc.items() if self.affordable(e, revealed, remaining)]
        return max(cands)[1] if cands else None


class VOISel(Selector):
    name = "voi"

    def next(self, bs, revealed, remaining):
        # per-eid effects on each open hypothesis' slot probabilities
        eff = {}
        for hi, slots in enumerate(bs.slots):
            if bs.hstates[hi].state != U or bs.p_h[hi] <= 0:
                continue
            for si, s in enumerate(slots):
                if s.state != U:
                    continue
                for e, q in zip(s.cands, s.q):
                    eff.setdefault(e, []).append((hi, si, q))
        if not eff:
            return None
        ph = np.clip(bs.p_h.copy(), 0, 1 - 1e-12)
        base_log = np.log1p(-ph)
        tot = base_log.sum()
        H0 = _h(1 - np.exp(tot))
        best = None
        for e, lst in eff.items():
            if not self.affordable(e, revealed, remaining):
                continue
            qbar = max(q for _, _, q in lst)
            # positive / negative outcome: recompute the affected hypotheses
            per_h_pos, per_h_neg = {}, {}
            for hi, si, q in lst:
                s = bs.slots[hi][si]
                ps = max(s.p, 1e-12)
                if s.closure:
                    pos, neg = 0.0, min(1.0, ps / max(1 - q, 1e-12))
                else:
                    pos, neg = 1.0, max(0.0, 1 - (1 - ps) / max(1 - q, 1e-12))
                fpos = per_h_pos.setdefault(hi, ph[hi])
                fneg = per_h_neg.setdefault(hi, ph[hi])
                per_h_pos[hi] = fpos / ps * pos
                per_h_neg[hi] = fneg / ps * neg
            tp, tn = tot, tot
            for hi in per_h_pos:
                tp += np.log1p(-min(per_h_pos[hi], 1 - 1e-12)) - base_log[hi]
                tn += np.log1p(-min(per_h_neg[hi], 1 - 1e-12)) - base_log[hi]
            eh = qbar * _h(1 - np.exp(tp)) + (1 - qbar) * _h(1 - np.exp(tn))
            v = (H0 - eh) / self.st.cost[e]
            if best is None or v > best[0]:
                best = (v, e)
        return best[1] if best else None


class GroupSel(Selector):
    """AND-OR sequential evaluation: pick the hypothesis with the highest
    P(h)/E[cost to decide h]; inside it reveal the next record of the first
    unresolved slot in the chosen AND order."""

    def __init__(self, refute_first: bool):
        self.refute_first = refute_first
        self.name = "combined" if refute_first else "comp"

    def next(self, bs, revealed, remaining):
        best = None
        for hi, slots in enumerate(bs.slots):
            if bs.hstates[hi].state != U or bs.p_h[hi] <= 0:
                continue
            info = []
            for s in slots:
                if s.state != U:
                    continue
                keep = [i for i, e in enumerate(s.cands) if self.affordable(e, revealed, remaining)]
                if not keep:
                    continue
                q, c = s.q[keep], s.c[keep]
                surv = np.concatenate([[1.0], np.cumprod(1 - q)[:-1]])
                ec = float((c * surv).sum())
                info.append((s, ec, s.p, s.cands[keep[0]]))
            if not info:
                continue
            if self.refute_first:
                info.sort(key=lambda x: x[1] / max(1 - x[2], 1e-9))
            else:
                info.sort(key=lambda x: (x[0].closure, -x[2]))
            ech, alive = 0.0, 1.0
            for s, ec, p, _ in info:
                ech += alive * ec
                alive *= p
            v = bs.p_h[hi] / max(ech, 1e-9)
            if best is None or v > best[0]:
                best = (v, info[0][3])
        return best[1] if best else None


SELECTORS = {
    "random": RandomSel, "random_groups": RandomGroupsSel, "graph": GraphSel, "caes": CAESSel,
    "suspicion": SuspicionSel, "voi": VOISel,
    "comp": lambda: GroupSel(False), "combined": lambda: GroupSel(True),
}


# ---------------------------------------------------------------- audit loop

def certificate_records(bs: BeliefState):
    """Records that currently witness a SUPPORTED verdict (one entailed hypothesis)."""
    for hs in bs.hstates:
        if hs.state == T and hs.witnesses:
            p, a, execs = hs.witnesses[0]
            return {p, a, *(c[0] for c in execs.values())} | set(hs.hyp.grant_cands)
    return set()


def run_audit(traj, selector: Selector, qm, reader, ablation=None, budget_frac=1.0, seed=0, stop=True,
              confirm_reader=None):
    """`confirm_reader`: before stopping on SUPPORTED, re-read the certificate's records with a
    second (stronger) reader; no new disclosure, only extra inference.  If the certificate no
    longer verifies, the re-read facts replace the first reading and the search continues."""
    view = make_view(traj, ablation, seed)
    st = Structure(view)
    belief = Belief(st, qm)
    oracle = EvidenceOracle(traj, view, reader, budget=budget_frac * view.total_cost)
    rng = np.random.default_rng(seed)
    selector.reset(view, st, rng)
    revealed = {}
    bs = belief.compute(revealed)
    trace = [(0, 0, bs.p_viol, bs.verdict)]
    confirmed = set()
    n_confirm = 0
    while True:
        if stop and bs.verdict == SUPPORTED and confirm_reader is not None:
            todo = certificate_records(bs) - confirmed
            if todo:
                for e in todo:
                    revealed[e] = confirm_reader.read(traj, e)
                    confirmed.add(e)
                n_confirm += len(todo)
                bs = belief.compute(revealed)
                trace.append((len(revealed), oracle.spent, bs.p_viol, bs.verdict))
                continue
        if stop and bs.verdict in (SUPPORTED, REFUTED):
            break
        e = selector.next(bs, revealed, oracle.remaining)
        if e is None:
            break
        revealed[e] = oracle.reveal(e).fact
        if e in confirmed:
            revealed[e] = confirm_reader.read(traj, e)
        bs = belief.compute(revealed)
        trace.append((len(revealed), oracle.spent, bs.p_viol, bs.verdict))
    ids = [x[0] for x in oracle.log]
    # first prefix that is an independently verified certificate (V is monotone -> bisect)
    disc = None
    if ids and verify(traj, ids) == SUPPORTED:
        lo, hi = 1, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if verify(traj, ids[:mid]) == SUPPORTED:
                hi = mid
            else:
                lo = mid + 1
        disc = lo
    return dict(tid=traj.tid, selector=selector.name, ablation=ablation or "none", log=oracle.log,
                trace=trace, disc_index=disc,
                disc_cost=(oracle.log[disc - 1][2] if disc else None),
                total_cost=view.total_cost)
