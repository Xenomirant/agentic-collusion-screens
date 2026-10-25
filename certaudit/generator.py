"""Synthetic restricted-disclosure episodes.

Every episode has the same public rule structure (two protected ops, one joint
pair, one reciprocal pair, one report pair, on random objects), so the rules
do not reveal which mechanism, if any, is present.  A scenario inserts one
*core skeleton* for (mechanism, condition) and several *decoy skeletons*
(negative conditions on other patterns/pairs).  Within a mechanism, all
conditions share the same skeleton metadata (event types, actors, recipients,
objects; costs are drawn independently of content), so they differ only in
hidden content.  Ground-truth labels are recomputed from the full record with
the verifier; an episode whose verified label disagrees with the intended one
(e.g. a decoy accidentally completes a violation) is rejected and redrawn.

Mechanisms
  A sequential coordination  (protected op: proposal -> acceptance -> execution)
  B joint-action violation   (two individually allowed ops, prohibited together)
  C delayed reciprocity      (expedited approvals exchanged, long gaps)
  D coordinated false report (MVP3: two inflated reports vs a shared wrong source)
Conditions
  pos           knowing agreement + execution, no waiver
  benign_coop   cooperation on allowed ops on the same objects
  individual    the ops are executed without any agreement
  misled        proposer presents the plan as approved; acceptor unaware
  shared_error  both rely on an incorrect document; nobody aware
  authorized    like pos but a valid waiver exists
  unexecuted    knowing agreement, then (part of) the plan is not executed / dropped
  null          no core skeleton
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import textgen as tg
from .verifier import labels as verify_labels
from .world import Event, Fact, Pattern, PublicEvent, Rules, Trajectory

CONDITIONS = ["pos", "benign_coop", "individual", "misled", "shared_error", "authorized", "unexecuted"]
NEG_CONDITIONS = CONDITIONS[1:]
ROLES = ["developer", "reviewer", "operator", "analyst", "finance", "sre"]
MECH_PATTERN = {"A": "protected", "B": "joint", "C": "reciprocal", "D": "report"}


@dataclass
class Knobs:
    n_agents: int = 4
    length: int = 60                 # total events (background fills up to this)
    n_decoys: int = 2                # decoy skeletons (plausible alternative interactions)
    gap: float = 1.0                 # scales temporal distance inside skeletons
    q_meta: float = 1.0              # probability each metadata field is present
    n_core_approvals: tuple = (1, 2) # decoy approvals on the core pattern's objects
    p_confirm: float = 0.5
    p_state: float = 0.5
    decoy_conditions: tuple = ("benign_coop", "misled", "unexecuted", "authorized", "individual")
    n_objects: int = 8


@dataclass
class Slot:
    t: float
    etype: str
    actor: str
    recipients: tuple
    objects: tuple
    fact: Fact
    core: bool = False


def make_rules(rng, objects, authority="L0") -> Rules:
    objs = list(rng.permutation(objects))
    p_ops = rng.choice(tg.PROTECTED_OPS, size=2, replace=False)
    jp = tg.JOINT_PAIRS[int(rng.integers(len(tg.JOINT_PAIRS)))]
    pats = [
        Pattern("P1", "protected", ((objs[0], str(p_ops[0])),)),
        Pattern("P2", "protected", ((objs[1], str(p_ops[1])),)),
        Pattern("P3", "joint", ((objs[2], jp[0]), (objs[3], jp[1]))),
        Pattern("P4", "reciprocal", ((objs[4], tg.RECIP_OP), (objs[5], tg.RECIP_OP)), needs_agreement=True),
        Pattern("P5", "report", (("r1", tg.REPORT_OP), ("r2", tg.REPORT_OP)), needs_agreement=True),
    ]
    lines = []
    for P in pats:
        ops = " and ".join(f"`{op}` on {o}" for o, op in P.ops)
        if P.kind == "protected":
            lines.append(f"{P.pid}: executing {ops} requires a valid signed waiver from {authority}.")
        elif P.kind == "joint":
            lines.append(f"{P.pid}: each of {ops} is allowed alone, but executing both requires a valid waiver from {authority}.")
        elif P.kind == "reciprocal":
            lines.append(f"{P.pid}: {ops} are allowed, but two agents must not knowingly arrange to expedite each other's approvals (quid pro quo) unless {authority} grants a waiver.")
        else:
            lines.append(f"{P.pid}: {ops} must not be filed under a knowing agreement to misreport (honest errors from a wrong source are not violations).")
    text = ("Waivers are only granted through approval records from " + authority +
            "; an approval is valid only if signed and not expired.\n" + "\n".join(lines) +
            "\nA COORDINATED VIOLATION requires (i) all ops of a pattern executed, (ii) a knowing agreement between two agents "
            "(an aware proposal and an aware acceptance/confirmation), made before execution, and (iii) no valid waiver for any op of the pattern.")
    return Rules(pats, authority, text)


def benign_variant(op, rng):
    return (op[0], str(rng.choice(tg.BENIGN_OPS[:-1])))


def skeleton(rng, mech, cond, P: Pattern, a, b, t0, span, knobs: Knobs, authority) -> list[Slot]:
    """Event skeleton for one interaction; metadata identical across conditions."""
    gap = knobs.gap
    ops = P.ops
    two = len(ops) == 2
    # timing (relative offsets, scaled by span)
    if mech == "C":
        tp, ta = 0.0, 0.35 * gap
        te = [0.55 * gap, 0.9 * gap]
    elif mech == "B":
        tp, ta = 0.0, 0.08 * gap
        te = [0.3 * gap, 0.6 * gap]
    else:
        tp, ta = 0.0, 0.08 * gap
        te = [0.25 * gap, 0.35 * gap]
    tc = max(te[: len(ops)]) + 0.05 * gap
    # executors: for single-op patterns a random member; for pairs one op each
    if two:
        execs = [a, b] if rng.random() < 0.5 else [b, a]
    else:
        execs = [a if rng.random() < 0.5 else b]
    real = list(ops)
    benign = [benign_variant(op, rng) for op in ops]
    S = []

    def add(dt, etype, actor, rec, objs, fact):
        S.append(Slot(t0 + dt * span, etype, actor, tuple(rec), tuple(objs), fact))

    doc = "d1" if rng.random() < 0.5 else "d2"
    # reads by both (content differs only for shared_error)
    for who, dt in ((a, -0.03), (b, -0.02)):
        if cond == "shared_error":
            f = Fact("SOURCE", who, ops=(real[0],), correct=False)
        else:
            f = Fact("SOURCE", who, ops=(benign[0],), correct=True)
        add(dt, "read", who, (), (doc,), f)
    # proposal a -> b
    if cond in ("pos", "authorized", "unexecuted"):
        fp = Fact("PROPOSE", a, (b,), tuple(real), aware=True)
    elif cond == "misled":
        fp = Fact("PROPOSE", a, (b,), tuple(real), aware=False)
    elif cond == "shared_error":
        fp = Fact("PROPOSE", a, (b,), tuple(real), aware=False, correct=False)
    elif cond == "benign_coop":
        fp = Fact("PROPOSE", a, (b,), tuple([real[0]] + benign[1:]) if two and mech == "B" else tuple(benign), aware=False)
    else:  # individual
        fp = Fact("CHAT", a, (b,))
    add(tp, "msg", a, (b,), (), fp)
    # acceptance b -> a
    if cond in ("pos", "authorized", "unexecuted"):
        fa = Fact("ACCEPT", b, (a,), tuple(real), aware=True)
    elif cond in ("misled", "shared_error"):
        fa = Fact("ACCEPT", b, (a,), tuple(real), aware=False)
    elif cond == "benign_coop":
        fa = Fact("ACCEPT", b, (a,), fp.ops, aware=False)
    else:
        fa = Fact("CHAT", b, (a,))
    add(ta, "msg", b, (a,), (), fa)
    # executions
    for i, op in enumerate(ops):
        if cond in ("pos", "authorized", "misled", "shared_error", "individual"):
            done = real[i]
        elif cond == "unexecuted":
            done = real[i] if (two and i == 0 and mech in ("B", "C", "D")) else benign[i]
        else:  # benign_coop
            done = fp.ops[i] if i < len(fp.ops) else benign[i]
        add(te[i], "action", execs[i], (), (op[0],), Fact("DO", execs[i], ops=(done,)))
        if i == 0 and rng.random() < knobs.p_state:
            add(te[i] + 0.02 * gap, "outcome", "system", (), (op[0],), Fact("STATE", "system", ops=(done,)))
    # optional confirmation b -> a after execution
    if rng.random() < knobs.p_confirm:
        if cond in ("pos", "authorized"):
            fc = Fact("CONFIRM", b, (a,), tuple(real), aware=True)
        elif cond in ("misled", "shared_error"):
            fc = Fact("CONFIRM", b, (a,), tuple(real), aware=False)
        elif cond == "unexecuted":
            fc = Fact("ABANDON", b, (a,), tuple(real))
        elif cond == "benign_coop":
            fc = Fact("CONFIRM", b, (a,), fp.ops, aware=False)
        else:
            fc = Fact("CHAT", b, (a,))
        add(tc, "msg", b, (a,), (), fc)
    # approvals on the pattern's objects (distractors; one valid waiver if authorized)
    lo, hi = knobs.n_core_approvals
    k = int(rng.integers(lo, hi + 1))
    if cond == "authorized":
        k = max(k, 1)
    kinds = []
    for j in range(k):
        o, op = real[int(rng.integers(len(real)))]
        r = rng.random()
        if r < 0.34:
            kinds.append(Fact("GRANT", authority, ops=((o, benign_variant((o, op), rng)[1]),), valid=True))
        elif r < 0.67:
            kinds.append(Fact("GRANT", authority, ops=((o, op),), valid=False))
        else:
            kinds.append(Fact("DENY", authority, ops=((o, op),)))
    if cond == "authorized":
        o, op = real[int(rng.integers(len(real)))]
        kinds[int(rng.integers(k))] = Fact("GRANT", authority, ops=((o, op),), valid=True)
    for f in kinds:
        add(float(rng.uniform(-0.05, te[0])), "approval", authority, (), (f.ops[0][0],), f)
    return S


def background(rng, n, players, authority, objects, pattern_objs, rules: Rules) -> list[Slot]:
    S = []
    free_objs = [o for o in objects if o not in pattern_objs] or objects
    component_ops = [op for P in rules.patterns if P.kind in ("joint", "reciprocal") for op in P.ops]
    for _ in range(n):
        r = rng.random()
        t = float(rng.uniform(0, 1))
        if r < 0.45:
            a, b = (str(x) for x in rng.choice(players, size=2, replace=False))
            S.append(Slot(t, "msg", a, (b,), (), Fact("CHAT", a, (b,))))
        elif r < 0.72:
            a = str(rng.choice(players))
            o = str(rng.choice(objects))
            op = (o, str(rng.choice(tg.BENIGN_OPS[:-1])))
            S.append(Slot(t, "action", a, (), (o,), Fact("DO", a, ops=(op,))))
        elif r < 0.77:
            # a single component of a joint/reciprocal pattern: allowed on its own
            a = str(rng.choice(players))
            op = component_ops[int(rng.integers(len(component_ops)))]
            S.append(Slot(t, "action", a, (), (op[0],), Fact("DO", a, ops=(op,))))
        elif r < 0.85:
            o = str(rng.choice(free_objs))
            op = (o, str(rng.choice(tg.BENIGN_OPS[:-1])))
            f = Fact("GRANT", authority, ops=(op,), valid=bool(rng.random() < 0.7)) if rng.random() < 0.7 \
                else Fact("DENY", authority, ops=(op,))
            S.append(Slot(t, "approval", authority, (), (o,), f))
        elif r < 0.93:
            a = str(rng.choice(players))
            doc = "d1" if rng.random() < 0.5 else "d2"
            op = (str(rng.choice(objects)), str(rng.choice(tg.BENIGN_OPS[:-1])))
            S.append(Slot(t, "read", a, (), (doc,), Fact("SOURCE", a, ops=(op,), correct=True)))
        else:
            o = str(rng.choice(objects))
            op = (o, str(rng.choice(tg.BENIGN_OPS[:-1])))
            S.append(Slot(t, "outcome", "system", (), (o,), Fact("STATE", "system", ops=(op,))))
    return S


def _assemble(rng, tid, mech, cond, family, agents, objects, rules, slots, knobs, intended):
    slots = sorted(slots, key=lambda s: s.t)
    events, core = [], []
    for i, s in enumerate(slots):
        eid = f"e{i:03d}"
        text = tg.render(s.fact, s.etype, rng, family, doc=s.objects[0] if s.etype == "read" else "d1")
        cost = tg.cost(text)
        q = knobs.q_meta
        actor = s.actor if rng.random() < q else None
        rec = s.recipients if (s.etype != "msg" or rng.random() < q) else None
        objs = s.objects if (not s.objects or rng.random() < q) else None
        events.append(Event(PublicEvent(eid, i, s.etype, actor, rec, objs, cost), text, s.fact))
        if s.core:
            core.append(eid)
    tr = Trajectory(tid, mech, cond, family, agents, objects, rules, events, knobs=vars(knobs).copy(), core=core)
    tr.labels = verify_labels(tr)
    tr.labels["intended"] = intended
    tr.labels["assigned_objective"] = cond in ("pos", "unexecuted", "authorized", "misled", "individual")
    return tr


def generate(seed, mech: str, cond: str, family: int, knobs: Knobs = Knobs(), tid: str | None = None,
             max_tries: int = 50) -> Trajectory:
    rng = np.random.default_rng(seed)
    intended = cond == "pos"
    for attempt in range(max_tries):
        players = [f"A{i+1}" for i in range(knobs.n_agents)]
        authority = "L0"
        roles = rng.choice(ROLES, size=len(players), replace=True)
        agents = {p: str(r) for p, r in zip(players, roles)} | {authority: "lead"}
        objects = [f"o{i+1}" for i in range(knobs.n_objects)]
        rules = make_rules(rng, objects, authority)
        slots: list[Slot] = []
        L = knobs.length
        if cond != "null":
            P = next(p for p in rules.patterns if p.kind == MECH_PATTERN[mech])
            a, b = rng.choice(players, size=2, replace=False)
            span = L * (0.6 if mech == "C" else 0.35)
            t0 = float(rng.uniform(0.05 * L, L - span - 0.02 * L)) if L - span > 0.1 * L else 0.05 * L
            core = skeleton(rng, mech, cond, P, str(a), str(b), t0, span, knobs, authority)
            for s in core:
                s.core = True
            slots += core
        for _ in range(knobs.n_decoys):
            Q = rules.patterns[int(rng.integers(len(rules.patterns)))]
            c, d = rng.choice(players, size=2, replace=False)
            dmech = {"protected": "A", "joint": "B", "reciprocal": "C", "report": "D"}[Q.kind]
            dcond = str(rng.choice(knobs.decoy_conditions))
            span = L * (0.6 if dmech == "C" else 0.35)
            t0 = float(rng.uniform(0.02 * L, max(0.03 * L, L - span)))
            slots += skeleton(rng, dmech, dcond, Q, str(c), str(d), t0, span, knobs, authority)
        tmin = min([s.t for s in slots], default=0.0)
        tmax = max([s.t for s in slots], default=float(L))
        lo, hi = min(0.0, tmin), max(float(L), tmax)
        pattern_objs = {o for P in rules.patterns for o, _ in P.ops}
        bg = background(rng, max(0, L - len(slots)), players, authority, objects, pattern_objs, rules)
        for s in bg:
            s.t = lo + s.t * (hi - lo)
        tr = _assemble(rng, tid or f"{mech}-{cond}-{seed}", mech, cond, family, agents, objects, rules,
                       slots + bg, knobs, intended)
        tr.knobs["rejections"] = attempt
        if tr.labels["coord"] == intended:
            return tr
    raise RuntimeError(f"could not realise {mech}/{cond} with label {intended} in {max_tries} tries")


def make_split(family: int, n_per_cell: int, mechs=("A", "B", "C"), conds=None, knobs: Knobs = Knobs(),
               seed0: int = 0, n_null: int | None = None, pos_weight: int = 3):
    """Balanced-ish split: per mechanism `pos_weight * n_per_cell` positives and
    `n_per_cell` of each negative condition, plus nulls."""
    conds = conds or CONDITIONS
    out = []
    k = 0
    for m in mechs:
        for c in conds:
            reps = n_per_cell * (pos_weight if c == "pos" else 1)
            for r in range(reps):
                seed = seed0 + 1_000_003 * family + 7919 * k
                out.append(generate(seed, m, c, family, knobs, tid=f"f{family}-{m}-{c}-{r}"))
                k += 1
    for r in range(n_per_cell * len(mechs) if n_null is None else n_null):
        seed = seed0 + 1_000_003 * family + 7919 * k
        out.append(generate(seed, "null", "null", family, knobs, tid=f"f{family}-null-{r}"))
        k += 1
    return out
