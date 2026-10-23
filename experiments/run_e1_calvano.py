"""E1: factorization-defect EWS in Calvano Q-learning markets.

Conditions
----------
collusive  : delta = 0.95, memory 1 (Calvano baseline)
myopic     : delta = 0,    memory 1 (state-contingent but short-sighted)
memoryless : delta = 0.95, no state (cannot condition on the rival)

Outputs (experiments/results/e1/)
---------------------------------
runs/<cond>_<seed>.npz   actions, profits, benchmarks
indicators.csv           rolling indicators per run and window
summary.json             lead times, AUCs, convergence profit gains

Every number reported from E1 comes from these files.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sims'))
import calvano_q as cq  # noqa: E402
from fathon.tensor import detrend, mvdfa, factorization as fz, ews  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e1')
CONDITIONS = {
    'collusive': dict(delta=0.95, memory=True),
    'myopic': dict(delta=0.0, memory=True),
    'memoryless': dict(delta=0.95, memory=False),
}
WINDOW = 10_000
DCCA_SCALES = np.unique(np.round(np.logspace(np.log10(8), np.log10(1000), 12)).astype(np.int64))


def run_one(args):
    cond, seed, T, beta = args
    path = os.path.join(OUT, 'runs', '{}_{}.npz'.format(cond, seed))
    if os.path.exists(path):
        return path
    t0 = time.time()
    r = cq.simulate(T=T, beta=beta, seed=seed, **CONDITIONS[cond])
    np.savez_compressed(path, actions=r['actions'], profit=r['profit'], grid=r['grid'],
                        pN=r['pN'], pM=r['pM'], piN=r['piN'], piM=r['piM'], beta=beta)
    print('{} seed {} done in {:.0f}s'.format(cond, seed, time.time() - t0), flush=True)
    return path


def window_indicators(acts, prices, profit, piN, piM):
    """All indicators on one window. `acts` are grid indices, (n, W)."""
    out = {}
    # ground truth (needs costs and demand, i.e. NOT observable by a regulator)
    out['Delta'] = float((profit.mean() - piN) / (piM - piN))
    out['price_mean'] = float(prices.mean())
    # classic EWS on prices
    out['variance'] = ews.variance(prices)
    out['ar1'] = ews.ar1(prices)
    out['pearson'] = ews.mean_corr(prices) if np.all(prices.std(axis=1) > 0) else 0.0
    # factorization defect on symbols (grid indices binned into 5 levels)
    S = (acts.astype(np.int64) * 5) // 15
    n = S.shape[0]
    for k in (1, 2):
        v = fz.pairwise_matrix(S, stat='phi2', k=k, alphabet=5, debias=True)
        out['phi2_k{}'.format(k)] = float(v[np.triu_indices(n, 1)].mean())
    lagged = fz.pairwise_matrix(S, stat='phi2', k=1, alphabet=5, lag=1, debias=True)
    out['phi2_lag1'] = float(lagged[~np.eye(n, dtype=bool)].mean())
    hg = fz.pairwise_matrix(S, stat='hgr', k=1, alphabet=5, lag=1, debias=False)
    out['hgr_lag1'] = float(hg[~np.eye(n, dtype=bool)].mean())
    # multiscale linear coupling: rho_DCCA averaged over scales, and at large scales
    if np.all(prices.std(axis=1) > 0):
        R = mvdfa.rho_matrix(detrend.profile(prices), DCCA_SCALES)
        off = R[:, 0, 1] if n == 2 else np.array([r[np.triu_indices(n, 1)].mean() for r in R])
        out['rho_dcca'] = float(np.mean(off))
        out['rho_dcca_large'] = float(np.mean(off[DCCA_SCALES >= 100]))
    else:
        out['rho_dcca'] = out['rho_dcca_large'] = 0.0
    return out


def indicators_for_run(path):
    d = np.load(path)
    acts, prof, grid = d['actions'], d['profit'], d['grid']
    name = os.path.basename(path)[:-4]
    cond, seed = name.rsplit('_', 1)
    rows = []
    T = acts.shape[1]
    for end in range(WINDOW, T + 1, WINDOW):
        a = acts[:, end - WINDOW:end]
        row = window_indicators(a, grid[a], prof[:, end - WINDOW:end], float(d['piN']), float(d['piM']))
        row.update(cond=cond, seed=int(seed), t=end, eps=float(np.exp(-float(d['beta']) * end)))
        rows.append(row)
    return rows


INDICATORS = ['variance', 'ar1', 'pearson', 'phi2_k1', 'phi2_k2', 'phi2_lag1', 'hgr_lag1',
              'rho_dcca', 'rho_dcca_large', 'price_mean']


def summarise(df):
    res = {}
    # converged-phase profit gain per condition (last 10 windows)
    last = df[df.t > df.t.max() - 10 * WINDOW]
    res['Delta_final'] = last.groupby('cond').Delta.agg(['mean', 'std', 'min', 'max']).to_dict()
    # discrimination: collusive vs each control, run-level means over the last 10 windows
    runlvl = last.groupby(['cond', 'seed'])[INDICATORS + ['Delta']].mean().reset_index()
    res['auc_vs_control'] = {}
    for ctrl in ('myopic', 'memoryless'):
        if ctrl not in set(runlvl.cond):
            continue
        res['auc_vs_control'][ctrl] = {
            ind: ews.auc(runlvl[runlvl.cond == 'collusive'][ind], runlvl[runlvl.cond == ctrl][ind])
            for ind in INDICATORS + ['Delta']}
    # early warning inside collusive runs: alarm times relative to Delta
    lead = {ind: [] for ind in INDICATORS}
    onset = []
    for seed, g in df[df.cond == 'collusive'].groupby('seed'):
        g = g.sort_values('t')
        t = g.t.values
        nb = 5   # baseline: first 5 windows (eps ~ 1, random play)
        t_star = ews.first_crossing(t, g.Delta.values, 0.5, persist=3)
        if t_star is None:
            continue
        onset.append(int(t_star))
        for ind in INDICATORS:
            v = g[ind].values
            thr = ews.baseline_threshold(v, nb, 3.0)
            ta = ews.first_crossing(t, v, thr, persist=3)
            lead[ind].append(np.nan if ta is None else float(t_star - ta))
    res['onset_Delta_0.5'] = onset
    res['lead_time'] = {ind: {'median': float(np.nanmedian(v)) if len(v) else None,
                              'frac_before': float(np.mean(np.array(v) > 0)) if len(v) else None,
                              'values': v} for ind, v in lead.items()}
    # monotone trend before onset (Kendall tau over windows with t < onset)
    taus = {ind: [] for ind in INDICATORS}
    for seed, g in df[df.cond == 'collusive'].groupby('seed'):
        g = g.sort_values('t')
        t_star = ews.first_crossing(g.t.values, g.Delta.values, 0.5, persist=3)
        if t_star is None:
            continue
        pre = g[g.t < t_star]
        if len(pre) < 5:
            continue
        for ind in INDICATORS:
            taus[ind].append(ews.kendall_trend(pre.t.values, pre[ind].values)[0])
    res['kendall_pre_onset'] = {ind: float(np.nanmedian(v)) if v else None for ind, v in taus.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--T', type=int, default=2_000_000)
    ap.add_argument('--beta', type=float, default=4e-6)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--conds', default='collusive,myopic,memoryless')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
    jobs = [(c, s, args.T, args.beta) for c in args.conds.split(',') for s in range(args.seeds)]
    with Pool(args.procs) as pool:
        paths = pool.map(run_one, jobs)
        rows = sum(pool.map(indicators_for_run, paths), [])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'indicators.csv'), index=False)
    res = summarise(df)
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'lead_time'}, indent=1, default=float))
    print({k: (v['median'], v['frac_before']) for k, v in res['lead_time'].items()})


if __name__ == '__main__':
    main()
