"""Disclosure-boundary tests: a selector's request sequence may depend only on public
metadata and on records it has revealed.  Perturbing the hidden content of records it
never revealed must leave the whole sequence unchanged."""
import copy

import numpy as np
import pytest

from certaudit.belief import train_qmodels
from certaudit.generator import make_split
from certaudit.oracle import BudgetExceeded, EvidenceOracle, PerfectReader, make_view
from certaudit.selectors import SELECTORS, run_audit
from certaudit.world import Fact

QM = None


def qm():
    global QM
    if QM is None:
        QM = train_qmodels(make_split(0, 2), n_masks=2)
    return QM


def perturb_unrevealed(traj, revealed, rng):
    t2 = copy.deepcopy(traj)
    for e in t2.events:
        if e.eid not in revealed:
            e.fact = Fact("CHAT" if e.pub.etype == "msg" else "DENY", e.fact.agent, e.fact.to, e.fact.ops[:1])
            e.text = "perturbed " + str(rng.integers(1e9))
    return t2


@pytest.mark.parametrize("sel", list(SELECTORS))
def test_requests_independent_of_unrevealed_content(sel):
    rng = np.random.default_rng(0)
    trajs = make_split(2, 1)[::4]
    for tr in trajs:
        for frac in (0.1, 0.3):
            r1 = run_audit(tr, SELECTORS[sel](), qm(), PerfectReader(), budget_frac=frac, seed=3)
            seen = {x[0] for x in r1["log"]}
            r2 = run_audit(perturb_unrevealed(tr, seen, rng), SELECTORS[sel](), qm(), PerfectReader(),
                           budget_frac=frac, seed=3)
            assert [x[0] for x in r1["log"]] == [x[0] for x in r2["log"]], (sel, tr.tid)


def test_oracle_charges_and_enforces_budget():
    tr = make_split(2, 1)[0]
    view = make_view(tr)
    o = EvidenceOracle(tr, view, PerfectReader(), budget=100)
    first = view.pubs[0]
    o.reveal(first.eid)
    assert o.spent == first.cost and o.log[0][0] == first.eid
    o.reveal(first.eid)                       # repeated reveal is free and not re-logged
    assert len(o.log) == 1
    big = [p for p in view.pubs if p.cost > 100 - o.spent]
    with pytest.raises(BudgetExceeded):
        o.reveal(big[0].eid)
    with pytest.raises(KeyError):
        o.reveal("e999")


def test_ablated_records_not_disclosable():
    tr = make_split(2, 1)[0]
    view = make_view(tr, "no_comms")
    o = EvidenceOracle(tr, view, PerfectReader())
    msg = next(e for e in tr.events if e.pub.etype == "msg")
    with pytest.raises(KeyError):
        o.reveal(msg.eid)
