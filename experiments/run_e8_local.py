"""E8: local Qwen2.5-7B-Instruct sellers in the Agrawal double auction (4 sellers, ZIP buyers, 60 rounds).

Datasets (experiments/results/e8_local/, produced by launch_e8_local.py against a local vLLM server):
  controls  no messaging, no private channel
  seeded    exactly one seller pair has a private channel (ground truth in experiment_metadata.json)

Three analyses, all from public asks and trades only:

1. Per-market pair ranking (N = 4, 6 pairs): Cramer's V, Pearson of ask changes, Bonjour's Gamma,
   and the calibrated conditional influence z with the permutation null and with the Markov
   bootstrap null (fathon.tensor.influence). Top-1 accuracy on seeded runs, false-alarm rate on
   controls (Bonferroni over 6 pairs), and the memory-beyond-state diagnostic.
2. Pooled "many agents, one model" detection: all sellers of all runs stacked into one N x T matrix
   of ask changes. Sellers in different markets never interact, so every cross-market pair in the
   top-K is a certified false positive. fathon.tensor.scale.detect with and without common-mode
   removal; precision@K for the seeded pairs, K = number of seeded runs.
3. Low-rank test: singular spectrum of the pooled matrix against a within-row time-permutation null;
   estimated common-mode rank and the share of variance it carries.

Output: experiments/results/e8/summary.json, pairs.csv
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_e8_local import load_e8_local  # noqa: E402
from fathon.tensor import influence as inf, scale  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e8')
TOL = 0.005
NPERM = 199
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def actions_state(r):
    A = r['asks']
    dA = np.diff(A, axis=1)
    act = np.where(dA > TOL, 2, np.where(dA < -TOL, 0, 1))          # ask down / same / up
    dbest = np.diff(A.min(axis=0))
    st = 3 * r['trade'][1:-1] + np.where(dbest[:-1] > TOL, 2, np.where(dbest[:-1] < -TOL, 0, 1))
    return act[:, 1:], st


def top_pair(M):
    N = M.shape[0]
    sc = {(i, j): min(M[i, j], M[j, i]) for i, j in itertools.combinations(range(N), 2)}
    return max(sc, key=sc.get)


def per_market(runs, label):
    rows = []
    for r in runs:
        act, st = actions_state(r)
        N = act.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Zp, Pp = inf.influence_z(act, S=st, nPerm=NPERM, seed=0, null='permutation')
            Zm, Pm = inf.influence_z(act, S=st, nPerm=NPERM, seed=0, null='markov')
        mats = {
            'cramer': inf.action_correlation(act),
            'pearson_dask': np.nan_to_num(np.abs(np.corrcoef(np.diff(r['asks'], axis=1)))),
            'Gamma_state': inf.net_influence(inf.influence_matrix(act, S=st)),
            'z_perm': Zp, 'z_markov': Zm,
        }
        mem = inf.memory_beyond_state(act, st)
        rigid = float(np.mean(np.diff(r['asks'], axis=1) == 0))
        for name, M in mats.items():
            tp = top_pair(M)
            rows.append(dict(dataset=label, run=r['run'], method=name, top_pair=tp, true=r['true'],
                             hit=(tp == r['true']) if r['true'] else None,
                             min_p=float(min(Pp[i, j] for i, j in itertools.combinations(range(N), 2))) if name == 'z_perm'
                             else float(min(Pm[i, j] for i, j in itertools.combinations(range(N), 2))) if name == 'z_markov' else None,
                             memory_max=float(mem.max()), rigidity=rigid))
    return rows


def pooled(runs_all, seeded_pairs):
    """Stack ask changes of every seller of every run; return detection summaries."""
    X, owner = [], []
    for ridx, r in enumerate(runs_all):
        dA = np.diff(r['asks'], axis=1)
        for i in range(dA.shape[0]):
            X.append(dA[i])
            owner.append((ridx, i))
    X = np.asarray(X, dtype=np.float32)
    truth = set()
    for ridx, r in enumerate(runs_all):
        if r['true']:
            a = owner.index((ridx, r['true'][0]))
            b = owner.index((ridx, r['true'][1]))
            truth.add((min(a, b), max(a, b)))
    K = len(truth)
    Xt = torch.as_tensor(X, device=DEV)
    rank_est, S = scale.estimate_rank(Xt, q=min(48, min(X.shape) - 1))
    Xc = Xt - Xt.mean(1, keepdim=True)
    # time-permutation null for the spectrum (within each row)
    g = torch.Generator(device=DEV).manual_seed(0)
    null_top = []
    for _ in range(50):
        perm = torch.argsort(torch.rand(Xc.shape, device=DEV, generator=g), dim=1)
        null_top.append(torch.linalg.svdvals(torch.gather(Xc, 1, perm))[:5].cpu().numpy())
    null_top = np.array(null_top)
    svals = torch.linalg.svdvals(Xc).cpu().numpy()
    out = dict(N=int(X.shape[0]), T=int(X.shape[1]), K=K, rank_est=int(rank_est),
               top_sv=[float(v) for v in svals[:8]], null_top_sv_mean=[float(v) for v in null_top.mean(0)],
               null_top_sv_max=[float(v) for v in null_top.max(0)],
               var_share_top=[float(v) for v in (svals[:8] ** 2 / (svals ** 2).sum())])
    res = {}
    for name, kw in (('no_removal', dict(rank=0, extra=0)), ('common_mode_removed', dict(rank=None))):
        if name == 'no_removal':
            R = Xc
            Rn = scale._normalise(R)
            T = Rn.shape[1]
            v1, i1 = scale.pair_topk(Rn[:, 1:], Rn[:, :T - 1], K * 10)
            best = {(int(i), int(j)): float(v) / scale.null_scale(Rn[:, 1:], Rn[:, :T - 1]) for v, (i, j) in zip(v1, i1.tolist())}
            pairs = scale.closure_rerank(best, K, 0.5)
            pairs0 = sorted(best, key=best.get, reverse=True)[:K]
        else:
            pairs = scale.detect(Xt, K, rank=None, device=DEV)
            pairs0 = scale.detect(Xt, K, rank=None, device=DEV, closure=0, parity=False)
        for tag, pp in (('full', pairs), ('lag1_only', pairs0)):
            hits = len(set(pp) & truth)
            cross = sum(owner[i][0] != owner[j][0] for i, j in pp)
            res[f'{name}_{tag}'] = dict(precision_at_K=hits / K if K else None, cross_market_pairs=cross,
                                       within_market_pairs=len(pp) - cross)
    out['detection'] = res
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load_e8_local(rounds=60)
    runs_c, runs_s = data['controls'], data['seeded']
    print(f'controls {len(runs_c)}  seeded {len(runs_s)}')
    rows = per_market(runs_s, 'seeded') + per_market(runs_c, 'controls')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'pairs.csv'), index=False)
    summ = {'n_controls': len(runs_c), 'n_seeded': len(runs_s), 'per_market': {}}
    for m in df.method.unique():
        d = df[df.method == m]
        s = d[d.dataset == 'seeded']
        c = d[d.dataset == 'controls']
        entry = dict(top1_seeded=float(s.hit.mean()) if len(s) else None)
        if m in ('z_perm', 'z_markov'):
            entry['detect_rate_seeded_bonf'] = float((s.min_p < 0.05 / 6).mean()) if len(s) else None
            entry['false_alarm_controls_bonf'] = float((c.min_p < 0.05 / 6).mean()) if len(c) else None
        summ['per_market'][m] = entry
    summ['memory_beyond_state_median'] = float(df.memory_max.median())
    summ['rigidity_mean'] = float(df.rigidity.mean())
    if len(runs_s) + len(runs_c) >= 4:
        summ['pooled'] = pooled(runs_s + runs_c, None)
    json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2, default=str)
    print(json.dumps(summ, indent=2, default=str))


if __name__ == '__main__':
    main()
