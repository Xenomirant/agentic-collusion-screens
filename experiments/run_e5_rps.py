"""E5: replicate Bonjour et al. (UAI 2022) on scripted 3-player iterated RPS.

Players: A random (primary), B assistant of A with collusion probability CP,
C random. Bonjour's rule: collusion is reported for pair (i, j) when exactly
one pair has net influence >= alpha = 0.05 in both directions; the collusion
detection accuracy (CDA) is the share of iterations returning (A, B).

Their Table 2 (manual case): CP = 0.3 -> 37.1, 48.3, 71.0, 82.8, 89.6, 94.6,
97.8, 98.8, 99.4, 100 % for 100..1000 games; CP = 0.4 -> 58.4, 79.2, 96.2,
99.3, 99.7, 99.9, 100, 100, 100, 100 %. Also CP = 0.2 -> 24.3 % at 1000 games.

The log base of their mutual information and the state used for the
iterated game are not stated in the main text, so both nats / bits and both
a constant state / the previous joint action (27 states) are run. The
previous joint action in bits reproduces their Table 2 most closely. A calibrated rule (conditional permutation p-value <= 0.05/3 for
exactly one pair) is run alongside on fewer iterations.

Output: experiments/results/e5/{cda.csv, summary.json}
"""

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

OUT = os.path.join(HERE, 'results', 'e5')
GAMES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
CPS = [0.1, 0.2, 0.3, 0.4, 1.0]
ITERS = 1000
ITERS_CAL = 200
PAPER = {0.3: [37.1, 48.3, 71.0, 82.8, 89.6, 94.6, 97.8, 98.8, 99.4, 100.0],
         0.4: [58.4, 79.2, 96.2, 99.3, 99.7, 99.9, 100.0, 100.0, 100.0, 100.0]}
STRATS = lambda cp: [{'kind': 'random'}, {'kind': 'assistant', 'partner': 0, 'cp': cp}, {'kind': 'random'}]


def job(args):
    cp, n, it = args
    r = rps.simulate(STRATS(cp), n, seed=1_000_003 * it + int(cp * 1000) * 7919 + n)
    A = r['actions']
    out = dict(cp=cp, games=n, it=it)
    for base, name in ((np.e, 'nats'), (2.0, 'bits')):
        out['detect_' + name] = inf.detect_pair(inf.net_influence(inf.influence_matrix(A, base=base)), 0.05) == (0, 1)
        out['detect_prevjoint_' + name] = inf.detect_pair(inf.net_influence(
            inf.influence_matrix(A, S=r['joint_prev'], base=base)), 0.05) == (0, 1)
    out['top1_gamma'] = inf.rank_of_pair(inf.influence_matrix(A), (0, 1)) == 1
    out['top1_Gamma'] = inf.rank_of_pair(inf.net_influence(inf.influence_matrix(A)), (0, 1)) == 1
    out['top1_cramer'] = inf.rank_of_pair(inf.action_correlation(A), (0, 1)) == 1
    if it < ITERS_CAL:
        _, P = inf.influence_z(A, nPerm=199, seed=it)
        sig = [(i, j) for i, j in ((0, 1), (0, 2), (1, 2)) if P[i, j] <= 0.05 / 3]
        out['detect_calibrated'] = sig == [(0, 1)]
        out['false_pos_calibrated'] = any(p != (0, 1) for p in sig)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(cp, n, it) for cp in CPS for n in GAMES for it in range(ITERS)]
    with Pool(8) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=200))
    df.to_csv(os.path.join(OUT, 'cda.csv'), index=False)
    tab = df.groupby(['cp', 'games']).agg(
        cda_nats=('detect_nats', 'mean'), cda_bits=('detect_bits', 'mean'),
        cda_prevjoint_nats=('detect_prevjoint_nats', 'mean'), cda_prevjoint_bits=('detect_prevjoint_bits', 'mean'),
        cda_calibrated=('detect_calibrated', 'mean'), fp_calibrated=('false_pos_calibrated', 'mean'),
        top1_Gamma=('top1_Gamma', 'mean'), top1_gamma=('top1_gamma', 'mean'), top1_cramer=('top1_cramer', 'mean'))
    tab = (100 * tab).round(1).reset_index()
    for cp, vals in PAPER.items():
        tab.loc[tab.cp == cp, 'paper_cda'] = vals
    res = {'table': tab.to_dict(orient='records')}
    for cp in PAPER:
        t = tab[tab.cp == cp]
        res['mae_vs_paper_cp{}'.format(cp)] = {u: float(np.mean(np.abs(t['cda_' + u].values - t.paper_cda.values)))
                                               for u in ('nats', 'bits', 'prevjoint_nats', 'prevjoint_bits')}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(tab.to_string())
    print({k: v for k, v in res.items() if k.startswith('mae')})


if __name__ == '__main__':
    main()
