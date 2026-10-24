"""E8 with the scale pipeline: the population detector on real LLM agents (local Qwen2.5-7B sellers).

All 4 x 60 ask series of the 120 markets (40 controls, 40 seeded, 40 seeded_monitored) are stacked
into one population. The common mode (same-model sellers share price trajectories across markets,
see E8) is estimated over the whole population (fathon.tensor.scale.estimate_rank, shift-surrogate
null) and removed; optionally the market mean too (with m = 4 it biases lag-0 correlations by -1/3).
Per market, pairs are scored by:
  coupling   multi-lag (0..3) two-sided coupling of residual ask changes, mutual / one-way
  levels     the same on residual ask levels
  coinc      exact-price coincidence z (scale.coincidence_pairs, within-round null) per market
  combined   max of the robust-z of coupling(levels, mutual) and coinc
Metrics: top-1 accuracy on seeded / monitored markets (chance 1/6) and AUC of the market's max
score, seeded (or monitored) vs controls. Output: experiments/results/e8/scale_pipeline.json
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_e8_local import load_e8_local  # noqa: E402
from fathon.tensor import scale, ews  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e8')


def main():
    data = load_e8_local(rounds=60)
    runs = [(k, r) for k in ('controls', 'seeded', 'seeded_monitored') for r in data[k]]
    m = 4
    iu = torch.triu_indices(m, m, 1)
    pairs = [(int(a), int(b)) for a, b in zip(iu[0], iu[1])]
    asks = np.stack([r['asks'] for _, r in runs]).astype(np.float64)          # (R, 4, 60)
    R = len(runs)
    lev = torch.as_tensor(asks.reshape(R * m, -1))
    chg = torch.as_tensor(np.diff(asks, axis=2).reshape(R * m, -1))
    res = {}
    for mm in (False, True):
        out = {}
        for name, X in (('changes', chg), ('levels', lev)):
            Xc = X - X.mean(1, keepdim=True)
            k, _ = scale.estimate_rank(Xc.float(), q=min(24, Xc.shape[1] - 1))
            Xr = Xc
            if k > 0:
                U, S, V = torch.linalg.svd(Xc, full_matrices=False)
                Xr = Xc - (U[:, :k] * S[:k]) @ V[:k]
            C = scale.market_lag_channels(Xr, m, lags=(0, 1, 2, 3), market_mean=mm)
            for mode in ('mutual', 'oneway'):
                out[f'{name}_{mode}'] = scale.coupling_scores(C, iu, mode).numpy()     # (R, 6)
            out[f'{name}_rank'] = int(k)
        cz = np.zeros((R, len(pairs)))
        for ri, (_, r) in enumerate(runs):
            top, info = scale.coincidence_pairs(r['asks'], 6, null='round', nShift=200, min_count=1)
            for (i, j), (c, mu) in info.items():
                cz[ri, pairs.index((i, j))] = (c - mu) / np.sqrt(mu + 1.0)
        out['coinc'] = cz
        rz = lambda a: (a - np.median(a)) / (1.4826 * np.median(np.abs(a - np.median(a))) + 1e-12)
        out['combined'] = np.maximum(rz(out['levels_mutual']), rz(cz))
        out['combined_changes'] = np.maximum(rz(out['changes_mutual']), rz(cz))
        summ = {}
        lab = [k for k, _ in runs]
        for key, Sc in out.items():
            if not isinstance(Sc, np.ndarray):
                summ[key] = Sc
                continue
            row = {}
            mx = Sc.max(1)
            ctrl = mx[[i for i, l in enumerate(lab) if l == 'controls']]
            for ds in ('seeded', 'seeded_monitored'):
                idx = [i for i, l in enumerate(lab) if l == ds]
                hits = [pairs[int(np.argmax(Sc[i]))] == runs[i][1]['true'] for i in idx]
                row[ds] = dict(top1=float(np.mean(hits)), auc_vs_controls=ews.auc(mx[idx], ctrl))
            summ[key] = row
        res['market_mean_removed' if mm else 'no_market_mean'] = summ
        print(('market mean removed' if mm else 'no market mean'), file=sys.stderr)
        for key, row in summ.items():
            if isinstance(row, dict):
                print(f"  {key:18s} seeded top1 {row['seeded']['top1']:.2f} AUC {row['seeded']['auc_vs_controls']:.2f} | "
                      f"monitored top1 {row['seeded_monitored']['top1']:.2f} AUC {row['seeded_monitored']['auc_vs_controls']:.2f}", file=sys.stderr)
            else:
                print(f'  {key}: {row}', file=sys.stderr)
    json.dump(res, open(os.path.join(OUT, 'scale_pipeline.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
