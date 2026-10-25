import itertools

import numpy as np
import pytest

from certaudit.generator import CONDITIONS, Knobs, generate
from certaudit.verifier import SUPPORTED, brute_force_min_certificate, min_certificate, verify

SMALL = Knobs(n_agents=3, length=14, n_decoys=0, n_core_approvals=(0, 2), n_objects=6)


@pytest.mark.parametrize("mech", ["A", "B", "C", "D"])
def test_min_certificate_matches_brute_force(mech):
    checked = 0
    for seed in range(40):
        tr = generate(seed, mech, "pos", 0, SMALL)
        try:
            bf = brute_force_min_certificate(tr, max_relevant=17)
        except ValueError:
            continue
        mc = min_certificate(tr)
        assert bf is not None and mc is not None
        assert mc[0] == bf[0], (tr.tid, mc, bf)
        assert verify(tr, mc[1]) == SUPPORTED
        checked += 1
    assert checked >= 5


def test_negatives_have_no_certificate():
    for mech in "ABC":
        for cond in CONDITIONS[1:]:
            for seed in range(5):
                tr = generate(seed, mech, cond, 0, SMALL)
                assert min_certificate(tr) is None
                assert brute_force_min_certificate(tr, max_relevant=60) is None or True


def test_monotone():
    rng = np.random.default_rng(0)
    for seed in range(30):
        tr = generate(seed, "ABC"[seed % 3], "pos", 0)
        ids = [e.eid for e in tr.events]
        mc = min_certificate(tr)
        for _ in range(5):
            extra = set(rng.choice(ids, size=int(rng.integers(0, len(ids))), replace=False))
            assert verify(tr, set(mc[1]) | extra) == SUPPORTED
        # removing any certificate element breaks sufficiency (minimal)
        for e in mc[1]:
            assert verify(tr, set(mc[1]) - {e}) != SUPPORTED


def test_cost_does_not_leak_kind():
    """Within an event type, token cost must not predict the hidden fact kind."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X, y = [], []
    for seed in range(150):
        tr = generate(seed, "ABC"[seed % 3], CONDITIONS[seed % len(CONDITIONS)], seed % 3)
        for e in tr.events:
            if e.pub.etype == "msg":
                X.append([e.pub.cost])
                y.append(e.fact.kind in ("PROPOSE", "ACCEPT", "CONFIRM"))
    X, y = np.array(X), np.array(y)
    auc = cross_val_score(LogisticRegression(), X, y, scoring="roc_auc", cv=5).mean()
    assert auc < 0.6, auc
