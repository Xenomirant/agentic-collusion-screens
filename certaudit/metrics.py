"""Metrics: budgeted scores from audit traces, frozen thresholds, paired bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def score_at(run, frac):
    """(score, verdict, spent, n_records) after the last reveal affordable at budget `frac`."""
    lim = frac * run["total_cost"] + 1e-9
    best = run["trace"][0]
    for step in run["trace"]:
        if step[1] <= lim:
            best = step
        else:
            break
    n, spent, score, verdict = best
    return score, verdict, spent, n


def disc_within(run, frac):
    return run["disc_cost"] is not None and run["disc_cost"] <= frac * run["total_cost"] + 1e-9


def threshold_for_fpr(scores_neg, target=0.05):
    """Smallest threshold t with FPR = P(score > t | neg) <= target (scores > t are flagged)."""
    s = np.sort(np.asarray(scores_neg))[::-1]
    k = int(np.floor(target * len(s)))       # number of negatives allowed above t
    if k >= len(s):
        return -np.inf
    return float(s[k])


def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auprc(y, s):
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def bootstrap(fn, n, reps=1000, seed=0, strata=None):
    """Percentile CI of fn(idx) over resampled index sets (stratified if strata given)."""
    rng = np.random.default_rng(seed)
    vals = []
    if strata is None:
        for _ in range(reps):
            vals.append(fn(rng.integers(0, n, n)))
    else:
        strata = np.asarray(strata)
        groups = [np.where(strata == g)[0] for g in np.unique(strata)]
        for _ in range(reps):
            vals.append(fn(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])))
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
