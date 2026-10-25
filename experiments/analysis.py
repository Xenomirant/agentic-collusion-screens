"""Budget-curve metrics, frozen-threshold operating points and paired comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import cert_costs  # noqa: F401  (re-exported for scripts)
from certaudit.metrics import auprc, auroc, bootstrap, disc_within, score_at, threshold_for_fpr

BUDGETS = (0.05, 0.10, 0.20, 0.50, 1.00)


def table(runs_by_sel, cal_by_sel, trajs, certs, budgets=BUDGETS, target_fpr=0.05, reps=1000, ci=True):
    lab = {t.tid: t.labels["coord"] for t in trajs}
    mech = {t.tid: t.mechanism for t in trajs}
    rows = []
    for sel, runs in runs_by_sel.items():
        runs = sorted(runs, key=lambda r: r["tid"])
        y = np.array([lab[r["tid"]] for r in runs])
        cal = cal_by_sel.get(sel)
        for b in budgets:
            S = [score_at(r, b) for r in runs]
            s = np.array([x[0] for x in S])
            row = dict(selector=sel, budget=b, n=len(runs), auroc=auroc(y, s), auprc=auprc(y, s),
                       tokens=np.mean([x[2] for x in S]), records=np.mean([x[3] for x in S]),
                       frac_spent=np.mean([x[2] / r["total_cost"] for x, r in zip(S, runs)]),
                       cert_found=np.mean([disc_within(r, b) for r, yy in zip(runs, y) if yy]),
                       supported=np.mean([x[1] == "supported" for x in S]),
                       refuted=np.mean([x[1] == "refuted" for x in S]),
                       supported_on_neg=np.mean([x[1] == "supported" for x, yy in zip(S, y) if not yy]))
            if ci:
                row["auroc_lo"], row["auroc_hi"] = bootstrap(lambda i: auroc(y[i], s[i]), len(y), reps)
            if cal is not None:
                cl = {t: v for t, v in cal["labels"].items()}
                cs = np.array([score_at(r, b)[0] for r in cal["runs"] if not cl[r["tid"]]])
                thr = threshold_for_fpr(cs, target_fpr)
                flag = s > thr
                row["tpr@5"] = flag[y].mean()
                row["fpr@5"] = flag[~y].mean()
            for m in sorted(set(mech.values())):
                idx = np.array([mech[r["tid"]] in (m, "null") for r in runs])
                row[f"auroc_{m}"] = auroc(y[idx], s[idx])
            rows.append(row)
        # search overhead on positives (certificate discovered at any budget)
        ov = [r["disc_cost"] / certs[r["tid"]] for r, yy in zip(runs, y) if yy and r["disc_cost"] and certs[r["tid"]]]
        for row in rows:
            if row["selector"] == sel:
                row["overhead_median"] = float(np.median(ov)) if ov else np.nan
                row["overhead_mean"] = float(np.mean(ov)) if ov else np.nan
    return pd.DataFrame(rows)


def paired(runs_by_sel, trajs, ref, budget, reps=2000, metric="auroc"):
    """Paired bootstrap of metric(ref) - metric(other) on the same trajectories."""
    lab = {t.tid: t.labels["coord"] for t in trajs}
    base = {r["tid"]: score_at(r, budget)[0] for r in runs_by_sel[ref]}
    tids = sorted(base)
    y = np.array([lab[t] for t in tids])
    s0 = np.array([base[t] for t in tids])
    out = []
    for sel, runs in runs_by_sel.items():
        if sel == ref:
            continue
        o = {r["tid"]: score_at(r, budget)[0] for r in runs}
        s1 = np.array([o[t] for t in tids])
        d = auroc(y, s0) - auroc(y, s1)
        lo, hi = bootstrap(lambda i: auroc(y[i], s0[i]) - auroc(y[i], s1[i]), len(y), reps)
        out.append(dict(ref=ref, other=sel, budget=budget, diff=d, lo=lo, hi=hi))
    return pd.DataFrame(out)
