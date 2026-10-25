"""With perfect reading and an unlimited budget, every belief-driven selector must reach the
certificate of every positive episode (a hypothesis the verifier has not refuted must never be
abandoned by the belief), and must never report 'supported' on a negative."""
import numpy as np

from certaudit.belief import train_qmodels
from certaudit.generator import Knobs, generate_controlled, make_split
from certaudit.oracle import PerfectReader
from certaudit.selectors import SELECTORS, run_audit

QM = None


def qm():
    global QM
    if QM is None:
        QM = train_qmodels(make_split(0, 2), n_masks=2)
    return QM


def test_all_selectors_complete_on_positives():
    trajs = [generate_controlled(c, 900 + c, m, "pos", 2, Knobs(n_decoys=4)) for c in range(6) for m in "ABC"]
    trajs += make_split(2, 1)[::5]
    for sel in ("suspicion", "voi", "comp", "combined"):
        for tr in trajs:
            r = run_audit(tr, SELECTORS[sel](), qm(), PerfectReader())
            if tr.labels["coord"]:
                assert r["trace"][-1][3] == "supported", (sel, tr.tid, len(r["log"]))
            else:
                assert r["trace"][-1][3] != "supported", (sel, tr.tid)
