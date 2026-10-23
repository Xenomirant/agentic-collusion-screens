"""E2: detection power vs coupling strength, coalition recovery, XOR blindness.

Outputs (experiments/results/e2/): power.csv, recovery.csv, xor.csv, summary.json
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sims'))
import coupled_agents as ca  # noqa: E402
from fathon.tensor import detrend, mvdfa, factorization as fz, ews  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e2')
EPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
SEEDS = 30
T = 20_000
NSURR = 99
SCALES = np.array([8, 16, 32, 64, 128, 256], dtype=np.int64)


def pearson_p(S, i, j, rng, nSurr=NSURR):
    # contemporaneous correlation with the same circular-shift null
    def stat(X):
        return abs(np.corrcoef(X[0], X[1])[0, 1])
    X = S[[i, j]].astype(float)
    return ews.surrogate_test(stat, X, ews.shift_surrogate, nSurr, seed=int(rng.integers(1 << 31)))[3]


def dcca_p(S, i, j, rng, nSurr=NSURR):
    def stat(X):
        return abs(mvdfa.rho_matrix(detrend.profile(X), SCALES)[:, 0, 1]).mean()
    X = S[[i, j]].astype(float)
    return ews.surrogate_test(stat, X, ews.shift_surrogate, nSurr, seed=int(rng.integers(1 << 31)))[3]


def power_job(args):
    eps, seed, kind = args
    rng = np.random.default_rng(1000 + seed)
    # coalition {0, 1}, plus an uncoupled agent 2 as the across-pair control
    S = ca.simulate([[0, 1], [2]], T=T, eps=eps, kind=kind, seed=seed)
    row = dict(eps=eps, seed=seed, kind=kind)
    for name, (i, j) in (('within', (0, 1)), ('across', (0, 2))):
        row['p_phi2_k2_' + name] = fz.shift_null(S, [i], [j], 'phi2', k=2, nSurr=NSURR, seed=seed)[3]
        row['p_pearson_' + name] = pearson_p(S, i, j, rng)
        row['p_dcca_' + name] = dcca_p(S, i, j, rng)
    return row


def recovery_job(args):
    eps, seed, kind = args
    coal = [[0, 1, 2], [3, 4, 5]]
    S = ca.simulate(coal, T=T, eps=eps, kind=kind, seed=seed)
    truth = np.array([0, 0, 0, 1, 1, 1])
    row = dict(eps=eps, seed=seed, kind=kind)
    for name, M in (('phi2_k2', fz.pairwise_matrix(S, 'phi2', k=2, alphabet=2)),
                    ('pearson', np.abs(np.corrcoef(S.astype(float))))):
        lab = fcluster(fz.coalition_tree(M), 2, 'maxclust')
        row['ok_' + name] = bool(len(set(lab[:3])) == 1 and len(set(lab[3:])) == 1 and lab[0] != lab[3])
    best = fz.best_bipartition(S, k=2, alphabet=2)[0]
    row['ok_bipartition'] = sorted(best[1]) in ([0, 1, 2], [3, 4, 5])
    return row


def xor_job(args):
    eps, seed = args
    S = ca.xor_triad(T=T, eps=eps, seed=seed)
    row = dict(eps=eps, seed=seed)
    row['p_pair_02'] = fz.shift_null(S, [0], [2], 'phi2', k=2, nSurr=NSURR, seed=seed)[3]
    row['p_pair_12'] = fz.shift_null(S, [1], [2], 'phi2', k=2, nSurr=NSURR, seed=seed)[3]
    row['p_group_01_2'] = fz.shift_null(S, [0, 1], [2], 'phi2', k=2, nSurr=NSURR, seed=seed)[3]
    rng = np.random.default_rng(seed)
    row['p_pearson_02'] = pearson_p(S, 0, 2, rng)
    row['p_dcca_02'] = dcca_p(S, 0, 2, rng)
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    with Pool(8) as pool:
        power = pd.DataFrame(pool.map(power_job, [(e, s, k) for k in ('copy',) for e in EPS
                                                  for s in range(SEEDS)]))
        rec = pd.DataFrame(pool.map(recovery_job, [(e, s, k) for k in ('copy', 'xor') for e in EPS[1:]
                                                   for s in range(SEEDS)]))
        xor = pd.DataFrame(pool.map(xor_job, [(e, s) for e in (0.0, 0.1, 0.3) for s in range(SEEDS)]))
    power.to_csv(os.path.join(OUT, 'power.csv'), index=False)
    rec.to_csv(os.path.join(OUT, 'recovery.csv'), index=False)
    xor.to_csv(os.path.join(OUT, 'xor.csv'), index=False)
    alpha = 0.05
    res = {
        'rejection_rate_copy': power.groupby('eps')[[c for c in power if c.startswith('p_')]]
            .agg(lambda p: float(np.mean(p <= alpha))).to_dict(),
        'recovery_rate': rec.groupby(['kind', 'eps'])[[c for c in rec if c.startswith('ok_')]]
            .mean().reset_index().to_dict(orient='records'),
        'xor_rejection_rate': xor.groupby('eps')[[c for c in xor if c.startswith('p_')]]
            .agg(lambda p: float(np.mean(p <= alpha))).to_dict(),
        'settings': dict(T=T, seeds=SEEDS, n_surrogates=NSURR, alpha=alpha),
    }
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
