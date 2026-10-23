"""E6: rank the one seeded colluding pair in 4-player iterated RPS.

Players
-------
0 primary   uniform at random
1 assistant plays the move player 0 beats with probability CP (seeded pair)
2, 3        responders: best-respond to the previous round's most frequent
            action with probability RHO. They share the public state but do
            not coordinate: their actions are dependent, conditionally
            independent given the state.

Ground truth: pair (0, 1). Every method scores all 6 pairs; we report how
often the true pair is ranked first and how often the confounded pair (2, 3)
is ranked first.

Methods
-------
cramer        ordinary action correlation (Cramer's V), no state
pearson       Pearson correlation of action codes (naive)
gamma         I(a_i; a_j), no state
Gamma_public  Bonjour net influence, state = public state (3 values)
Gamma_joint   Bonjour net influence, state = previous joint action (81 values), bits
z_public      conditional-permutation z of I(a_i; a_j | public state)
llm_judge     see run_e6_llm_judge.py (needs an API key)

Output: experiments/results/e6/{ranks.csv, summary.json}; transcripts for the
judge in experiments/results/e6/transcripts/.
"""

import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sims'))
import rps  # noqa: E402
from fathon.tensor import influence as inf  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e6')
CPS = [0.1, 0.2, 0.3]
RHOS = [0.0, 0.3, 0.6]
TS = [100, 200, 500, 1000]
REPS = 200
TRUE = (0, 1)
CONF = (2, 3)


def strategies(cp, rho):
    return [{'kind': 'random'}, {'kind': 'assistant', 'partner': 0, 'cp': cp},
            {'kind': 'responder', 'rho': rho}, {'kind': 'responder', 'rho': rho}]


def pearson(A):
    C = np.corrcoef(A.astype(float))
    return np.nan_to_num(np.abs(C))


def top_pair(M):
    N = M.shape[0]
    sc = {(i, j): min(M[i, j], M[j, i]) for i, j in itertools.combinations(range(N), 2)}
    return max(sc, key=sc.get)


def job(args):
    cp, rho, T, rep = args
    r = rps.simulate(strategies(cp, rho), T, seed=97 * rep + 10_007 * T + int(100 * cp) * 31 + int(100 * rho))
    A, s, J = r['actions'], r['state'], r['joint_prev']
    mats = {
        'cramer': inf.action_correlation(A),
        'pearson': pearson(A),
        'gamma': inf.influence_matrix(A),
        'Gamma_public': inf.net_influence(inf.influence_matrix(A, S=s)),
        'Gamma_joint': inf.net_influence(inf.influence_matrix(A, S=J, base=2)),
    }
    Z, P = inf.influence_z(A, S=s, nPerm=199, seed=rep)
    mats['z_public'] = Z
    out = dict(cp=cp, rho=rho, T=T, rep=rep)
    for k, M in mats.items():
        out['rank_' + k] = inf.rank_of_pair(M, TRUE)
        out['conf_top_' + k] = top_pair(M) == CONF
    out['bonjour_detect_public'] = inf.detect_pair(mats['Gamma_public'], 0.05) == TRUE
    out['bonjour_detect_joint'] = inf.detect_pair(mats['Gamma_joint'], 0.05) == TRUE
    sig = [p for p in itertools.combinations(range(4), 2) if P[p] <= 0.05 / 6]
    out['calibrated_detect'] = sig == [TRUE]
    out['calibrated_false_pair'] = any(p != TRUE for p in sig)
    return out


def write_transcripts(n_per_cell=5, T=200):
    """Action logs for the LLM judge (same generator, separate seeds)."""
    d = os.path.join(OUT, 'transcripts')
    os.makedirs(d, exist_ok=True)
    names = ['R', 'P', 'S']
    manifest = []
    for cp in CPS:
        for rho in RHOS:
            for k in range(n_per_cell):
                seed = 555_000 + 1000 * k + int(100 * cp) * 10 + int(100 * rho)
                rng = np.random.default_rng(seed)
                perm = rng.permutation(4)          # hide which seat is which role
                r = rps.simulate(strategies(cp, rho), T, seed=seed)
                A = r['actions'][perm]
                lines = ['round,' + ','.join('P{}'.format(i + 1) for i in range(4))]
                lines += ['{},{}'.format(t + 1, ','.join(names[a] for a in A[:, t])) for t in range(T)]
                fn = 'cp{}_rho{}_k{}.csv'.format(cp, rho, k)
                open(os.path.join(d, fn), 'w').write('\n'.join(lines) + '\n')
                inv = np.argsort(perm)
                manifest.append(dict(file=fn, cp=cp, rho=rho, T=T, seed=seed,
                                     true_pair=sorted(['P{}'.format(inv[0] + 1), 'P{}'.format(inv[1] + 1)]),
                                     confound_pair=sorted(['P{}'.format(inv[2] + 1), 'P{}'.format(inv[3] + 1)])))
    json.dump(manifest, open(os.path.join(d, 'manifest.json'), 'w'), indent=1)
    return manifest


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(cp, rho, T, rep) for cp in CPS for rho in RHOS for T in TS for rep in range(REPS)]
    with Pool(8) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=20))
    df.to_csv(os.path.join(OUT, 'ranks.csv'), index=False)
    methods = ['cramer', 'pearson', 'gamma', 'Gamma_public', 'Gamma_joint', 'z_public']
    g = df.groupby(['cp', 'rho', 'T'])
    top1 = g[['rank_' + m for m in methods]].agg(lambda r: float(np.mean(r == 1)))
    top1.columns = methods
    conf = g[['conf_top_' + m for m in methods]].mean()
    conf.columns = methods
    det = g[['bonjour_detect_public', 'bonjour_detect_joint', 'calibrated_detect', 'calibrated_false_pair']].mean()
    res = {'top1_true_pair': (100 * top1).round(1).reset_index().to_dict(orient='records'),
           'top1_confounded_pair': (100 * conf).round(1).reset_index().to_dict(orient='records'),
           'detection': (100 * det).round(1).reset_index().to_dict(orient='records'),
           'mean_rank': g[['rank_' + m for m in methods]].mean().round(2).reset_index().to_dict(orient='records'),
           'settings': dict(reps=REPS, n_perm=199, alpha_bonjour=0.05, alpha_calibrated='0.05/6')}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    write_transcripts()
    print('top-1 true pair (%)')
    print((100 * top1).round(1).to_string())
    print('top-1 confounded pair (%)')
    print((100 * conf).round(1).to_string())
    print((100 * det).round(1).to_string())


if __name__ == '__main__':
    main()
