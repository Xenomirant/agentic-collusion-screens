"""Auditor belief: metadata-only record priors combined through the rule's AND-OR structure.

For every hypothesis h = (pattern, pair) the certificate has slots
  exec[op] (OR over action/outcome candidates), prop and acc (OR over message
  candidates), and nogrant (AND over approval candidates: all must be clean).
A learned model gives, for each unrevealed candidate record and slot role,
q = P(record satisfies the role | public metadata, facts read so far).  The
models are trained on the training split with random reveal masks (so they
see partial-evidence contexts), and never see hidden content of the episode
being audited except through records already revealed.

  P(slot) = 1 - prod(1 - q)        for OR slots (exec, prop, acc)
  P(slot) = prod(1 - q)            for the nogrant closure
  P(h)    = prod over slots        (independence approximation)
  P(viol) = 1 - prod_h (1 - P(h))
Slots already decided by the three-valued verifier are clamped to 0/1.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .oracle import PublicView, make_view
from .verifier import (F, INSUFFICIENT, REFUTED, SUPPORTED, T, U, Index, acc_ok, eval_hyp, evaluate,
                       exec_ok, grant_blocks, prop_ok)
from .world import Fact

ROLES = ("exec", "prop", "acc", "grant")
KINDS = ("protected", "joint", "reciprocal", "report")
N_CTX = 13


def acc_any(f: Fact, P, pair) -> bool:
    return (f.kind in ("ACCEPT", "CONFIRM") and f.aware is True and f.agent in pair
            and any(r in pair and r != f.agent for r in f.to) and bool(set(P.ops) & set(f.ops)))


class Structure:
    """Static (metadata-only) structure and features of one public view."""

    def __init__(self, view: PublicView):
        self.view = view
        self.idx = Index(view.pubs, view.rules, view.agents)
        self.pub = {p.eid: p for p in view.pubs}
        self.t = {p.eid: p.t for p in view.pubs}
        self.cost = {p.eid: p.cost for p in view.pubs}
        self.T = max([p.t for p in view.pubs], default=0) + 1
        self.total = view.total_cost
        H = self.idx.hyps
        self.rows = {r: [] for r in ROLES}           # (hyp index, eid, op index)
        feats = {r: [] for r in ROLES}
        reads = [p for p in view.pubs if p.etype == "read"]
        for hi, h in enumerate(H):
            pair = h.pair
            objs = h.objs
            kind = [float(h.pattern.kind == k) for k in KINDS]
            msgs = sorted((self.t[e], e) for e in h.msg_cands)
            mt = np.array([x for x, _ in msgs]) if msgs else np.zeros(0)
            acts = sorted(self.t[p.eid] for p in view.pubs if p.etype == "action"
                          and (p.actor is None or p.actor in pair)
                          and (p.objects is None or objs & set(p.objects)))
            at = np.array(acts) if acts else np.zeros(0)
            rt = np.array(sorted(p.t for p in reads if p.actor is None or p.actor in pair)) if reads else np.zeros(0)
            n_appr = len(h.grant_cands)
            for k, e in enumerate(e for _, e in msgs):
                p = self.pub[e]
                te = p.t
                nxt_a = at[at > te]
                prv_r = rt[rt < te]
                reply = any(self.pub[x].actor != p.actor and 0 < self.t[x] - te <= 0.15 * self.T for _, x in msgs)
                before = any(self.pub[x].actor != p.actor and 0 < te - self.t[x] <= 0.15 * self.T for _, x in msgs)
                f = [te / self.T, np.log(p.cost), np.log1p(len(msgs)), k / max(1, len(msgs)),
                     np.log1p(len(nxt_a)), ((nxt_a[0] - te) / self.T) if len(nxt_a) else 1.0,
                     ((te - prv_r[-1]) / self.T) if len(prv_r) else 1.0, float(reply), float(before),
                     float(p.actor is not None), float(p.recipients is not None),
                     np.log1p(len(at) - len(nxt_a)), np.log1p(n_appr), *kind]
                for role in ("prop", "acc"):
                    self.rows[role].append((hi, e, -1))
                    feats[role].append(f)
            for oi, op in enumerate(h.pattern.ops):
                cands = sorted(h.exec_cands[op], key=lambda e: self.t[e])
                for k, e in enumerate(cands):
                    p = self.pub[e]
                    te = p.t
                    prv = mt[mt < te]
                    f = [te / self.T, np.log(p.cost), float(p.etype == "outcome"), float(p.actor is not None),
                         float(p.objects is not None), np.log1p(len(prv)),
                         ((te - prv[-1]) / self.T) if len(prv) else 1.0, np.log1p(len(cands)),
                         k / max(1, len(cands)), float(oi), float(len(h.pattern.ops)),
                         np.log1p(len(mt)), *kind]
                    self.rows["exec"].append((hi, e, oi))
                    feats["exec"].append(f)
            g = sorted(h.grant_cands, key=lambda e: self.t[e])
            for k, e in enumerate(g):
                p = self.pub[e]
                te = p.t
                nxt_a = at[at > te]
                f = [te / self.T, np.log(p.cost), float(n_appr), k / max(1, n_appr),
                     ((nxt_a[0] - te) / self.T) if len(nxt_a) else 1.0, float(p.objects is not None),
                     np.log1p(len(mt)), *kind]
                self.rows["grant"].append((hi, e, -1))
                feats["grant"].append(f)
        self.static = {r: np.array(feats[r], dtype=float).reshape(len(feats[r]), -1) for r in ROLES}
        self.row_h = {r: np.array([x[0] for x in self.rows[r]], dtype=int) for r in ROLES}
        self.row_e = {r: [x[1] for x in self.rows[r]] for r in ROLES}
        self.row_t = {r: np.array([self.t[x[1]] for x in self.rows[r]], dtype=float) for r in ROLES}
        self.row_c = {r: np.array([self.cost[x[1]] for x in self.rows[r]], dtype=float) for r in ROLES}
        self.row_op = {r: np.array([x[2] for x in self.rows[r]], dtype=int) for r in ROLES}
        # per hypothesis row indices
        self.h_rows = []
        for hi, h in enumerate(H):
            d = {r: np.where(self.row_h[r] == hi)[0] for r in ROLES}
            d["exec_op"] = [d["exec"][self.row_op["exec"][d["exec"]] == oi] for oi in range(len(h.pattern.ops))]
            self.h_rows.append(d)

    # ------------------------------------------------------------ context
    def context(self, revealed: dict[str, Fact]):
        """Per-hypothesis context vector and earliest aware-proposal time."""
        H = self.idx.hyps
        frac = sum(self.cost.get(e, 0) for e in revealed) / max(1, self.total)
        C = np.zeros((len(H), N_CTX))
        tprop = np.full(len(H), np.inf)
        items = list(revealed.items())
        for hi, h in enumerate(H):
            P, pair = h.pattern, h.pair
            pops = set(P.ops)
            objs = h.objs
            msgset = h.msg_cands
            nm = len(msgset)
            rm = sum(1 for e in msgset if e in revealed)
            c = C[hi]
            c[0] = rm / max(1, nm)
            n_exec_ok = 0
            for e, f in items:
                k = f.kind
                if k == "PROPOSE" and f.agent in pair and pops <= set(f.ops):
                    if f.aware:
                        c[1] = 1
                        tprop[hi] = min(tprop[hi], self.t.get(e, np.inf))
                    else:
                        c[2] = 1
                elif k in ("ACCEPT", "CONFIRM") and f.agent in pair and pops & set(f.ops):
                    c[3 if f.aware else 4] = 1
                elif k == "DO" and f.agent in pair and f.ops and f.ops[0][0] in objs and f.ops[0] not in pops:
                    c[6] += 1
                elif k == "GRANT" and pops & set(f.ops):
                    c[7 if f.valid else 8] = 1
                elif k == "DENY" and pops & set(f.ops):
                    c[8] = 1
                elif k in ("DECLINE", "ABANDON") and f.agent in pair and pops & set(f.ops):
                    c[9] = 1
                elif k == "SOURCE" and f.correct is False and f.agent in pair and pops & set(f.ops):
                    c[10] = 1
                elif k == "CHAT" and f.agent in pair and any(r in pair for r in f.to):
                    c[11] += 1
            for op in P.ops:
                if any(exec_ok(f, op, pair) for _, f in items):
                    n_exec_ok += 1
            c[5] = n_exec_ok / len(P.ops)
            c[6] = np.log1p(c[6])
            c[11] = np.log1p(c[11])
            c[12] = frac
        return C, tprop

    def features(self, role, C, tprop):
        hrow = self.row_h[role]
        ctx = C[hrow].copy()
        if role in ("acc", "exec"):
            ctx[:, 1] = ctx[:, 1] * (tprop[hrow] < self.row_t[role])   # aware proposal precedes this record
        return np.hstack([self.static[role], ctx])


def role_target(role, f: Fact, h, op_index) -> bool:
    P, pair = h.pattern, h.pair
    if role == "exec":
        return exec_ok(f, P.ops[op_index], pair)
    if role == "prop":
        return prop_ok(f, P, pair)
    if role == "acc":
        return acc_any(f, P, pair)
    return grant_blocks(f, P)


class FastMLP:
    """Numpy forward pass of a fitted StandardScaler + MLPClassifier pipeline
    (sklearn's per-call validation dominated the ~600-row batches per step)."""

    def __init__(self, pipe):
        sc, mlp = pipe[0], pipe[-1]
        self.mu, self.sd = sc.mean_, sc.scale_
        self.W, self.b = mlp.coefs_, mlp.intercepts_

    def predict_proba1(self, X):
        h = (np.asarray(X, dtype=float) - self.mu) / self.sd
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W + b, 0)
        z = (h @ self.W[-1] + self.b[-1]).ravel()
        return 1 / (1 + np.exp(-z))


class QModels:
    def __init__(self, models=None, const=None):
        self.models = models or {}
        self.const = const or {}
        self.compiled = {r: FastMLP(m) for r, m in self.models.items()}

    def predict(self, role, X):
        if len(X) == 0:
            return np.zeros(0)
        if role in self.compiled:
            return self.compiled[role].predict_proba1(X)
        return np.full(len(X), self.const.get(role, 0.0))

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


def training_rows(traj, n_masks, rng, ablation=None, reader_facts=None):
    view = make_view(traj, ablation)
    st = Structure(view)
    true = traj.facts()
    read = reader_facts or true
    out = {r: ([], []) for r in ROLES}
    eids = [p.eid for p in view.pubs]
    for m in range(n_masks):
        rho = [0.0, rng.uniform(0, 0.3), rng.uniform(0.2, 0.8)][m % 3]
        revealed = {e: read[e] for e in eids if rng.random() < rho}
        C, tprop = st.context(revealed)
        for role in ROLES:
            X = st.features(role, C, tprop)
            for i, (hi, e, oi) in enumerate(st.rows[role]):
                if e in revealed:
                    continue
                out[role][0].append(X[i])
                out[role][1].append(role_target(role, true[e], st.idx.hyps[hi], oi))
    return out


def train_qmodels(trajs, n_masks=3, seed=0, ablation=None) -> QModels:
    rng = np.random.default_rng(seed)
    data = {r: ([], []) for r in ROLES}
    for tr in trajs:
        rows = training_rows(tr, n_masks, rng, ablation)
        for r in ROLES:
            data[r][0].extend(rows[r][0])
            data[r][1].extend(rows[r][1])
    models, const = {}, {}
    for r in ROLES:
        X = np.array(data[r][0])
        y = np.array(data[r][1], dtype=int)
        if len(y) == 0 or y.min() == y.max():
            const[r] = float(y.mean()) if len(y) else 0.0
            continue
        m = make_pipeline(StandardScaler(), MLPClassifier((64, 64), max_iter=300, early_stopping=True,
                                                          random_state=seed))
        m.fit(X, y)
        models[r] = m
    return QModels(models, const)


# ---------------------------------------------------------------- belief state

@dataclass
class SlotInfo:
    name: str          # exec0 / exec1 / prop / acc / nogrant
    closure: bool      # True for nogrant (AND over candidates)
    state: str         # T / F / U
    p: float
    cands: list        # unrevealed candidate eids, ordered by q/c (or g/c for closure)
    q: np.ndarray
    c: np.ndarray


@dataclass
class BeliefState:
    verdict: str
    p_viol: float
    p_h: np.ndarray
    slots: list        # per hypothesis: list[SlotInfo]
    hstates: list
    q: dict            # role -> q per row (unrevealed rows meaningful)


class Belief:
    def __init__(self, st: Structure, qm: QModels):
        self.st = st
        self.qm = qm

    def compute(self, revealed: dict[str, Fact]) -> BeliefState:
        st = self.st
        verdict, hstates = evaluate(st.idx, revealed)
        C, tprop = st.context(revealed)
        q = {r: self.qm.predict(r, st.features(r, C, tprop)) for r in ROLES}
        unrev = {r: np.array([e not in revealed for e in st.row_e[r]], dtype=bool) for r in ROLES}
        p_h = np.zeros(len(st.idx.hyps))
        all_slots = []
        for hi, (h, hs) in enumerate(zip(st.idx.hyps, hstates)):
            rows = st.h_rows[hi]
            slots = []

            def mk(name, role, idx, state, closure=False, tmin=None):
                sel = idx[unrev[role][idx]]
                if tmin is not None:
                    sel = sel[st.row_t[role][sel] > tmin]
                qq = q[role][sel]
                cc = st.row_c[role][sel]
                order = np.argsort(-(qq / cc))
                qq, cc = qq[order], cc[order]
                eids = [st.row_e[role][i] for i in sel[order]]
                if state == T:
                    p = 1.0
                elif state == F:
                    p = 0.0
                elif closure:
                    p = float(np.prod(1 - qq))
                else:
                    p = float(1 - np.prod(1 - qq))
                return SlotInfo(name, closure, state, p, eids, qq, cc)

            tp = tprop[hi] if (np.isfinite(tprop[hi]) and C[hi, 1] > 0) else None
            for oi, op in enumerate(h.pattern.ops):
                state = hs.exec[op]
                if tp is not None and state == T and not any(
                        exec_ok(f, op, h.pair) and st.t.get(e, -1) > tp for e, f in revealed.items()):
                    state = U      # witnesses so far precede the known agreement: need a later one
                slots.append(mk(f"exec{oi}", "exec", rows["exec_op"][oi], state,
                                tmin=tp if state == U else None))
            if hs.agree == T:
                slots.append(mk("prop", "prop", rows["prop"], T))
                slots.append(mk("acc", "acc", rows["acc"], T))
            elif hs.agree == F:
                slots.append(mk("prop", "prop", rows["prop"], F))
                slots.append(mk("acc", "acc", rows["acc"], F))
            elif np.isfinite(tprop[hi]) and C[hi, 1] > 0:
                slots.append(mk("prop", "prop", rows["prop"], T))
                slots.append(mk("acc", "acc", rows["acc"], U, tmin=tprop[hi]))
            else:
                slots.append(mk("prop", "prop", rows["prop"], U))
                slots.append(mk("acc", "acc", rows["acc"], U))
            slots.append(mk("nogrant", "grant", rows["grant"], hs.nogrant, closure=True))
            if hs.state == T:
                ph = 1.0
            elif hs.state == F:
                ph = 0.0
            else:
                ph = float(np.prod([s.p for s in slots]))
            p_h[hi] = ph
            all_slots.append(slots)
        if verdict == SUPPORTED:
            pv = 1.0
        elif verdict == REFUTED:
            pv = 0.0
        else:
            pv = float(1 - np.prod(1 - np.clip(p_h, 0, 1 - 1e-12)))
            pv = min(pv, 1 - 1e-9)
        return BeliefState(verdict, pv, p_h, all_slots, hstates, q)
