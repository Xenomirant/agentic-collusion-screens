"""E4: detecting seller coordination in LLM double auctions from asks alone.

Data: the 75 shipped runs of Agrawal et al. (2025), external/collusion_hackathon/
llm-agent-collusion/data (5 sellers with cost 80, 5 buyers with value 100,
30 rounds). Ground truth: experimental condition (seller messaging on/off),
the paper's collusion index (mean seller ask, last third of rounds) and the
LLM-judge coordination scores of seller messages/reasoning.

Detectors (inputs: the 5 x 30 ask matrix and public trade outcomes only)
----------------------------------------------------------------------
* Bonjour et al. (UAI 2022): individual influence gamma(i;j) = I(a_i; a_j | s)
  with plug-in estimates, net influence Gamma(i;j) = gamma(i;j) - max_k gamma(k;j),
  for a fully observable simultaneous game. Actions: sign of ask change.
  State: previous round's public outcome (trade happened, sign of the
  change of the best ask).
* Factorization defect (this work): phi2 of agent cuts, contemporaneous and
  lag-1, as z-scores against circular time-shift surrogates; the same null
  applied to Bonjour's gamma; total correlation of all 5 sellers (the
  all-agent cut, a tensor statistic invisible to pairwise tests).
* Linear / level baselines: Pearson correlation of ask changes, rho_DCCA of
  asks, ask dispersion, mean ask (the collusion index itself).

Pooled ensemble tests: statistics computed on counts pooled over all runs of
a condition, compared with a label-permutation null.

Output: experiments/results/e4/{runs.csv, summary.json}
"""

import glob
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

from fathon.tensor import detrend, mvdfa, factorization as fz, ews

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'external', 'collusion_hackathon', 'llm-agent-collusion', 'data')
OUT = os.path.join(HERE, 'results', 'e4')
NSURR = 199
TOL = 0.005
rng_global = np.random.default_rng(0)


# ---------------------------------------------------------------- parsing
def parse_run(d):
    txt = open(os.path.join(d, 'auction_results.md')).read()
    blocks = [json.loads(b) for b in re.findall(r'````json\n(.*?)\n````', txt, flags=re.S)]
    meta = json.load(open(os.path.join(d, 'experiment_metadata.json')))['auction_config']
    met = json.load(open(os.path.join(d, 'collusion_metrics.json')))
    sellers = sorted({s for b in blocks for s in b['seller_asks']})
    A = np.full((len(sellers), len(blocks)), np.nan)
    trade = np.zeros(len(blocks), dtype=int)
    for t, b in enumerate(blocks):
        for i, s in enumerate(sellers):
            if s in b['seller_asks'] and b['seller_asks'][s] is not None:
                A[i, t] = float(b['seller_asks'][s])
        trade[t] = int(len(b.get('trades') or []) > 0)
    # forward / backward fill missing asks
    A = pd.DataFrame(A).T.ffill().bfill().T.values
    coord = [v for k in met if k.endswith('_coordination_score') and met[k] for v in met[k] if v is not None]
    ci = met['collusion_indices_avg_seller_asks']
    return dict(run=os.path.basename(d), tag=meta['tag'].rsplit('-', 1)[0], comms=bool(meta['seller_comms_enabled']),
                asks=A, trade=trade, judge=float(np.mean(coord)) if coord else np.nan,
                collusion_index=float(np.mean(ci[-len(ci) // 3:])), n_trades=int(met['num_trades']))


def actions_and_state(r):
    A = r['asks']
    dA = np.diff(A, axis=1)
    act = np.where(dA > TOL, 2, np.where(dA < -TOL, 0, 1))            # (5, 29), action at t = 1..29
    best = A.min(axis=0)
    dbest = np.diff(best)
    # state for the action at round t (t >= 2): previous round's public outcome
    st = 3 * r['trade'][1:-1] + np.where(dbest[:-1] > TOL, 2, np.where(dbest[:-1] < -TOL, 0, 1))
    return act[:, 1:], st                                              # aligned, length 28


# ---------------------------------------------------------- Bonjour et al.
def cond_mi(x, y, s):
    """Plug-in I(X; Y | S) in nats."""
    out = 0.0
    n = len(s)
    for v in np.unique(s):
        m = s == v
        if m.sum() < 2:
            continue
        P = fz.contingency(x[m], y[m])
        pa, pb = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        out += m.sum() / n * float(np.sum(P[nz] * np.log(P[nz] / (pa @ pb)[nz])))
    return out


def bonjour(act, st):
    N = act.shape[0]
    g = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        g[i, j] = g[j, i] = cond_mi(act[i], act[j], st)
    Gam = np.full((N, N), -np.inf)
    for i, j in itertools.permutations(range(N), 2):
        others = [g[k, j] for k in range(N) if k not in (i, j)]
        Gam[i, j] = g[i, j] - max(others)
    pair = max(min(Gam[i, j], Gam[j, i]) for i, j in itertools.combinations(range(N), 2))
    return g, pair


# ----------------------------------------------------------------- nulls
def shifted(act, rng):
    # independent circular shifts of every agent but the first
    T = act.shape[1]
    out = act.copy()
    for i in range(1, act.shape[0]):
        out[i] = np.roll(act[i], rng.integers(3, T - 3))
    return out


def total_correlation(act):
    def H(codes):
        _, c = np.unique(codes, return_counts=True)
        p = c / c.sum()
        return float(-np.sum(p * np.log(p)))
    joint = fz.words(act, 1, 3)
    return sum(H(a) for a in act) - H(joint)


def z_against_shift(stat, act, rng, nSurr=NSURR):
    obs = stat(act)
    surr = np.array([stat(shifted(act, rng)) for _ in range(nSurr)])
    sd = surr.std(ddof=1)
    return float((obs - surr.mean()) / sd) if sd > 0 else 0.0, obs


def mean_pair_phi2(act, lag=0):
    N = act.shape[0]
    pairs = itertools.permutations(range(N), 2) if lag else itertools.combinations(range(N), 2)
    vals = []
    for i, j in pairs:
        if np.unique(act[i]).size < 2 or np.unique(act[j]).size < 2:
            vals.append(0.0)
            continue
        vals.append(fz.group_cut(act, [i], [j], k=1, alphabet=3, lag=lag)['phi2'])
    return float(np.mean(vals))


# ----------------------------------------------------------------- per run
def run_stats(r, seed):
    rng = np.random.default_rng(seed)
    act, st = actions_and_state(r)
    out = {}
    g, pair = bonjour(act, st)
    out['bonjour_gamma_mean'] = float(g[np.triu_indices(5, 1)].mean())
    out['bonjour_pair_Gamma'] = float(pair)
    out['bonjour_gamma_z'] = z_against_shift(lambda a: float(np.mean(
        [cond_mi(a[i], a[j], st) for i, j in itertools.combinations(range(5), 2)])), act, rng)[0]
    out['defect_k1_z'] = z_against_shift(lambda a: mean_pair_phi2(a, 0), act, rng)[0]
    out['defect_lag1_z'] = z_against_shift(lambda a: mean_pair_phi2(a, 1), act, rng)[0]
    out['total_corr_z'], out['total_corr'] = z_against_shift(total_correlation, act, rng)
    A = r['asks']
    dA = np.diff(A, axis=1)
    ok = dA.std(axis=1) > 0
    out['pearson_dask'] = ews.mean_corr(dA[ok]) if ok.sum() >= 2 else 0.0
    if np.all(A.std(axis=1) > 0):
        R = mvdfa.rho_matrix(detrend.profile(A), np.array([4, 5, 6, 8, 10], dtype=np.int64))
        out['rho_dcca'] = float(np.mean([x[np.triu_indices(5, 1)].mean() for x in R]))
    else:
        out['rho_dcca'] = np.nan
    out['ask_dispersion'] = float(A.std(axis=0).mean())
    out['frac_flat'] = float(np.mean(act == 1))
    return out


IND = ['bonjour_gamma_mean', 'bonjour_pair_Gamma', 'bonjour_gamma_z', 'defect_k1_z', 'defect_lag1_z',
       'total_corr', 'total_corr_z', 'pearson_dask', 'rho_dcca', 'ask_dispersion', 'frac_flat']


# ------------------------------------------------------------ pooled tests
def pooled_stat(runs, which):
    """Statistic on counts pooled across runs (concatenated, cut at run boundaries)."""
    acts, sts = zip(*[actions_and_state(r) for r in runs])
    if which == 'bonjour_gamma':
        a = np.concatenate(acts, axis=1)
        s = np.concatenate(sts)
        return float(np.mean([cond_mi(a[i], a[j], s) for i, j in itertools.combinations(range(5), 2)]))
    if which == 'total_corr':
        return total_correlation(np.concatenate(acts, axis=1))
    if which == 'lag1_phi2':
        # lagged pairs within runs only
        x, y = [], []
        for a in acts:
            for i, j in itertools.permutations(range(5), 2):
                x.append(a[i, :-1])
                y.append(a[j, 1:])
        return cut_phi2(np.concatenate(x), np.concatenate(y))
    raise ValueError(which)


def cut_phi2(x, y):
    return fz.cut_stats(fz.contingency(x, y))['phi2']


def permutation_test(runsA, runsB, which, nPerm=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = pooled_stat(runsA, which) - pooled_stat(runsB, which)
    allr = runsA + runsB
    nA = len(runsA)
    null = []
    for _ in range(nPerm):
        p = rng.permutation(len(allr))
        null.append(pooled_stat([allr[k] for k in p[:nA]], which) - pooled_stat([allr[k] for k in p[nA:]], which))
    null = np.array(null)
    return float(obs), float((1 + np.sum(null >= obs)) / (1 + nPerm))


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = [parse_run(d) for d in sorted(glob.glob(os.path.join(DATA, '2025*')))]
    rows = []
    for k, r in enumerate(runs):
        row = {key: r[key] for key in ('run', 'tag', 'comms', 'judge', 'collusion_index', 'n_trades')}
        row.update(run_stats(r, k))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'runs.csv'), index=False)

    def auc_table(pos, neg):
        out = {}
        for ind in IND + ['collusion_index', 'judge']:
            a, b = pos[ind].dropna(), neg[ind].dropna()
            if len(a) and len(b):
                out[ind] = {'auc': ews.auc(a, b), 'p_mwu': float(mannwhitneyu(a, b, alternative='two-sided').pvalue)}
        return out

    base = df[df.tag == 'base']
    comms = df[df.tag == 'base-seller_comms']
    res = {'n_runs': len(df), 'conditions': df.tag.value_counts().to_dict()}
    res['means_by_condition'] = df.groupby('tag')[IND + ['collusion_index', 'judge']].mean().round(4).to_dict()
    res['auc_comms_vs_base'] = auc_table(comms, base)
    res['auc_allcomms_vs_base'] = auc_table(df[df.comms], base)
    res['spearman_collusion_index'] = {ind: list(map(float, spearmanr(df[ind], df.collusion_index, nan_policy='omit')))
                                       for ind in IND}
    res['spearman_judge'] = {ind: list(map(float, spearmanr(df[ind], df.judge, nan_policy='omit'))) for ind in IND}
    byrun = {r['run']: r for r in runs}
    rb = [byrun[x] for x in base.run]
    rc = [byrun[x] for x in comms.run]
    res['pooled_comms_minus_base'] = {w: dict(zip(('diff', 'p_one_sided'), permutation_test(rc, rb, w)))
                                      for w in ('bonjour_gamma', 'total_corr', 'lag1_phi2')}
    # does the coupling signal survive controlling for price rigidity (fraction of unchanged asks)?
    from scipy.stats import rankdata

    def resid(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)

    rz = rankdata(df.frac_flat)
    res['partial_spearman_judge_given_flat'] = {
        ind: list(map(float, spearmanr(resid(rankdata(df[ind]), rz), resid(rankdata(df.judge), rz))))
        for ind in ('defect_k1_z', 'defect_lag1_z', 'total_corr_z', 'bonjour_gamma_z', 'bonjour_pair_Gamma')}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
