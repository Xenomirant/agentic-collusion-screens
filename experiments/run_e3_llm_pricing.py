"""E3: price-only indicators on LLM pricing agents (Fish et al. 2024 setup).

Data: external/collusion_hackathon/algorithmic-pricing-llms/data/results/
all_experiments.parquet (216 runs x 300 rounds, mistral-large-2411 firms,
2-5 firms, prompt prefixes P1 / P2, logit demand with Nash and monopoly
benchmarks).

Ground truth per run: profit gain Delta over the last 100 rounds (needs
costs and demand). Indicators use prices only, on rounds 100-299.

Output: experiments/results/e3/{runs.csv, summary.json}
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fathon.tensor import detrend, mvdfa, factorization as fz, ews

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'external', 'collusion_hackathon', 'algorithmic-pricing-llms',
                    'data', 'results', 'all_experiments.parquet')
OUT = os.path.join(HERE, 'results', 'e3')
NSURR = 99
START = 100
SCALES = np.array([6, 8, 12, 16, 24, 32, 48, 64], dtype=np.int64)


def load_runs():
    df = pd.read_parquet(DATA)
    df = df[df.num_agents >= 2]
    runs = []
    for ts, g in df.groupby('experiment_timestamp'):
        P = g.pivot(index='agent', columns='round', values='chosen_price').sort_index()
        Pi = g.pivot(index='agent', columns='round', values='profit').sort_index()
        first = g.iloc[0]
        runs.append(dict(run=ts, name=first.experiment_name, N=int(first.num_agents),
                         prompt=first.agent_prefix_type, alpha=float(first.alpha),
                         symmetric=bool(first.is_symmetric), prices=P.values.astype(float),
                         profits=Pi.values.astype(float), piN=float(first.nash_profits),
                         piM=float(first.monopoly_profits), pN=float(first.nash_prices),
                         pM=float(first.monopoly_prices)))
    return runs


def pair_z(S, i, j, stat, k, lag, seed):
    if np.unique(S[i]).size < 2 or np.unique(S[j]).size < 2:
        return 0.0
    obs, surr, z, p = fz.shift_null(S, [i], [j], stat, k=k, nSurr=NSURR, seed=seed, lag=lag, minShift=10)
    return float(np.clip(z, -10, 10)) if np.isfinite(z) else 0.0


def indicators(r, seed=0):
    P = r['prices'][:, START:]
    N, T = P.shape
    out = {}
    # ground truth and oracle level (both need demand / cost knowledge)
    out['Delta'] = float((r['profits'][:, -100:].mean() - r['piN']) / (r['piM'] - r['piN']))
    out['price_rel'] = float((P[:, -100:].mean() - r['pN']) / (r['pM'] - r['pN']))
    # scale-free classic indicators
    dP = np.diff(P, axis=1)
    out['cv'] = float(np.mean(P.std(axis=1) / P.mean(axis=1)))
    out['ar1'] = ews.ar1(P) if np.all(P.std(axis=1) > 0) else 1.0
    out['pearson'] = ews.mean_corr(P) if np.all(P.std(axis=1) > 0) else 0.0
    out['pearson_dp'] = ews.mean_corr(dP) if np.all(dP.std(axis=1) > 0) else 0.0
    out['frac_flat'] = float(np.mean(np.abs(dP) < 1e-9))
    # factorization defect: z-scores vs circular-shift null, averaged over pairs
    S_lvl = fz.discretize(P, 3, 'quantile')
    S_chg = fz.discretize(dP, 3, 'sign')
    pairs = list(itertools.combinations(range(N), 2))
    out['z_lvl_k1'] = np.mean([pair_z(S_lvl, i, j, 'phi2', 1, 0, seed) for i, j in pairs])
    out['z_lvl_lag1'] = np.mean([pair_z(S_lvl, i, j, 'phi2', 1, 1, seed) for i, j in itertools.permutations(range(N), 2)])
    out['z_chg_k1'] = np.mean([pair_z(S_chg, i, j, 'phi2', 1, 0, seed) for i, j in pairs])
    out['z_chg_lag1'] = np.mean([pair_z(S_chg, i, j, 'phi2', 1, 1, seed) for i, j in itertools.permutations(range(N), 2)])
    out['z_chg_k2'] = np.mean([pair_z(S_chg, i, j, 'phi2', 2, 0, seed) for i, j in pairs])
    # multiscale linear coupling of prices
    if np.all(P.std(axis=1) > 0):
        R = mvdfa.rho_matrix(detrend.profile(P), SCALES)
        out['rho_dcca'] = float(np.mean([r_[np.triu_indices(N, 1)].mean() for r_ in R]))
        out['absorption'] = float(mvdfa.common_mode(R)['absorption'].mean())
    else:
        out['rho_dcca'] = out['absorption'] = np.nan
    return out


IND = ['price_rel', 'cv', 'ar1', 'pearson', 'pearson_dp', 'frac_flat', 'z_lvl_k1', 'z_lvl_lag1',
       'z_chg_k1', 'z_chg_lag1', 'z_chg_k2', 'rho_dcca', 'absorption']


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_runs()
    rows = []
    for k, r in enumerate(runs):
        row = {key: r[key] for key in ('run', 'name', 'N', 'prompt', 'alpha', 'symmetric')}
        row.update(indicators(r, seed=k))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'runs.csv'), index=False)
    agg = df.groupby('prompt')[['Delta', 'price_rel']].agg(['mean', 'std'])
    res = {'n_runs': len(df), 'by_prompt': {'_'.join(c): agg[c].round(3).to_dict() for c in agg.columns}}
    # (a) P1 vs P2, pooled and per number of firms
    res['auc_P1_vs_P2'] = {ind: ews.auc(df[df.prompt == 'P1'][ind].dropna(), df[df.prompt == 'P2'][ind].dropna())
                           for ind in IND + ['Delta']}
    res['auc_P1_vs_P2_by_N'] = {int(n): {ind: ews.auc(g[g.prompt == 'P1'][ind].dropna(), g[g.prompt == 'P2'][ind].dropna())
                                         for ind in IND + ['Delta']} for n, g in df.groupby('N')}
    # (b) rank association with Delta
    res['spearman_with_Delta'] = {ind: list(map(float, spearmanr(df[ind], df['Delta'], nan_policy='omit')))
                                  for ind in IND}
    # (c) high vs low Delta (median split within N)
    hi = df.groupby('N').Delta.transform(lambda d: d > d.median())
    res['auc_highDelta'] = {ind: ews.auc(df[hi][ind].dropna(), df[~hi][ind].dropna()) for ind in IND}
    # fraction of runs with significant coupling (mean pair z > 2)
    res['frac_z_gt2'] = {ind: df.groupby('prompt')[ind].apply(lambda z: float(np.mean(z > 2))).to_dict()
                         for ind in IND if ind.startswith('z_')}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
