"""MVP3: alternative explanations with matched metadata, abstention and risk-coverage.

For each core seed and mechanism the SAME skeleton (event types, actors,
recipients, objects, order) is realised under every condition; only hidden
content differs.  Selectors compared (same priors, same budgets):
  suspicion  A: suspicion-first (individual records most likely to add positive evidence)
  voi        B: alternative-testing on individual records (expected uncertainty reduction)
  combined   C: complementary cross-agent groups with refutation-first slot order
  comp       C without refutation-first (ablation: disable alternative-explanation testing)
plus random / caes.  Verdicts: supported / refuted ("violation not established") /
insufficient (abstain).  Abstention is never counted as detection.

    python experiments/run_mvp3.py [--reader perfect|<extraction table>] [--procs 16]
Writes results/mvp3/<reader>/{runs/*.jsonl.gz, verdicts.csv, operating.csv, risk_coverage.csv}.
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from common import CACHE, RESULTS, run_many
from certaudit.generator import Knobs, generate_controlled
from certaudit.metrics import auroc, bootstrap, score_at, threshold_for_fpr
from run_mvp2 import get_prior

CONDS = ["pos", "misled", "authorized", "unexecuted", "shared_error", "benign_coop", "individual"]
SELS = ["random", "caes", "suspicion", "voi", "comp", "combined"]
BUDGETS = (0.05, 0.1, 0.2, 0.5, 1.0)


def matched(family, core0, ctx0, n_core):
    out = []
    for i in range(n_core):
        for m in "ABC":
            for c in CONDS:
                out.append(generate_controlled(core0 + i, ctx0 + i, m, c, family, Knobs(),
                                               tid=f"m{core0 + i}-{m}-{c}"))
    return out


def get_split(name):
    path = CACHE / "splits" / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    fam, core0, ctx0, n = {"mvp3_cal": (1, 100, 700, 20), "mvp3_test": (2, 0, 500, 40)}[name]
    trajs = matched(fam, core0, ctx0, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(trajs, open(path, "wb"))
    return trajs


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="perfect")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    out = RESULTS / "mvp3" / a.reader
    qm = get_prior(a.procs)
    cal, test = get_split("mvp3_cal"), get_split("mvp3_test")
    lab = {t.tid: t.labels["coord"] for t in test}
    cond = {t.tid: t.condition for t in test}
    cal_lab = {t.tid: t.labels["coord"] for t in cal}
    verd, oper, rc = [], [], []
    for sel in SELS:
        t0 = time.time()
        runs = sorted(run_many(test, sel, qm, a.reader, None, out / "runs" / f"test_{sel}.jsonl.gz", a.procs),
                      key=lambda r: r["tid"])
        cruns = run_many(cal, sel, qm, a.reader, None, out / "runs" / f"cal_{sel}.jsonl.gz", a.procs)
        y = np.array([lab[r["tid"]] for r in runs])
        for b in BUDGETS:
            S = [score_at(r, b) for r in runs]
            s = np.array([x[0] for x in S])
            v = np.array([x[1] for x in S])
            for c in CONDS:
                m = np.array([cond[r["tid"]] == c for r in runs])
                verd.append(dict(selector=sel, budget=b, condition=c, n=int(m.sum()),
                                 supported=float((v[m] == "supported").mean()),
                                 refuted=float((v[m] == "refuted").mean()),
                                 insufficient=float((v[m] == "insufficient").mean()),
                                 mean_score=float(s[m].mean())))
            # verdict-based operating point and score-based point at a frozen 5% FPR threshold
            cs = np.array([score_at(r, b)[0] for r in cruns if not cal_lab[r["tid"]]])
            thr = threshold_for_fpr(cs, 0.05)
            flag = s > thr
            dec = v != "insufficient"
            tpr_v, fpr_v = float((v[y] == "supported").mean()), float((v[~y] == "supported").mean())
            lo_f, hi_f = bootstrap(lambda i: (v[~y][i] == "supported").mean(), int((~y).sum()), 1000)
            oper.append(dict(selector=sel, budget=b, auroc=auroc(y, s),
                             tpr_verdict=tpr_v, fpr_verdict=fpr_v, fpr_verdict_lo=lo_f, fpr_verdict_hi=hi_f,
                             abstain=float((~dec).mean()), coverage=float(dec.mean()),
                             tpr_at_5=float(flag[y].mean()), fpr_at_5=float(flag[~y].mean()),
                             ece=ece(s, y), brier=float(np.mean((s - y) ** 2)),
                             tokens=float(np.mean([x[2] for x in S]))))
            # risk-coverage: classify by score with confidence |s - 0.5|; sweep coverage
            conf = np.abs(s - 0.5)
            order = np.argsort(-conf)
            pred = s > 0.5
            for cov in (0.2, 0.4, 0.6, 0.8, 1.0):
                k = max(1, int(round(cov * len(order))))
                idx = order[:k]
                rc.append(dict(selector=sel, budget=b, coverage=cov,
                               risk=float((pred[idx] != y[idx]).mean()),
                               fpr=float((pred[idx] & ~y[idx]).sum() / max(1, (~y[idx]).sum())),
                               tpr=float((pred[idx] & y[idx]).sum() / max(1, y[idx].sum()))))
        print(f"[mvp3/{a.reader}] {sel}: {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(verd).to_csv(out / "verdicts.csv", index=False)
        pd.DataFrame(oper).to_csv(out / "operating.csv", index=False)
        pd.DataFrame(rc).to_csv(out / "risk_coverage.csv", index=False)
    print(pd.DataFrame(oper).round(3).to_string())


if __name__ == "__main__":
    main()
