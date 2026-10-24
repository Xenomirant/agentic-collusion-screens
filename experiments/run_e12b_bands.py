"""E12b: unknown band of coordination scales. Multiscale log-energy coupling vs the strongest
classical baseline (signed q = 2 rho_DCCA on cleaned log|x|), short and long scale sets.

Timing world (planted_markets.chunk_activity, layout random) with coalitions sharing the levels in
`band` (1-based dyadic levels of T = 1024: (1,4) slow = blocks 512..64, (3,6) medium = 256..16,
(5,8) fast = 64..4), aligned or offset by 37 steps. All methods see the same cleaned log|x|
(market mean, then global low-rank removal with the shift-surrogate rank) except where noted.
Output: experiments/results/e12/bands_<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sims'))
sys.path.insert(0, HERE)
import planted_markets as pm  # noqa: E402
from run_e11_scale import triu  # noqa: E402
from run_e12_activity import clean  # noqa: E402
import baselines_mf as bmf  # noqa: E402
from fathon.tensor import scale, ews  # noqa: E402

DEV = 'cuda'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=int, default=2000)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()
    M, m, T = args.markets, args.m, args.T
    iu = triu(m)
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    results = {}
    for band in ((1, 4), (3, 6), (5, 8)):
        for offset in (0, 37):
            G = pm.shared_activity(2 * T if offset else T, seed=0)
            sc = {k: [] for k in ('ours_levels', 'ours_levels_coarse', 'ours_levels_max', 'rho2_cleaned_8_128', 'rho2_cleaned_8_512',
                                  'rho2_cleaned_4_512_kwapien_q1')}
            labs, per_level = [], []
            for M0 in range(0, M, args.chunk):
                M1 = min(M, M0 + args.chunk)
                X, mk, truth, tt = pm.chunk_activity(M0, M1, m, T, G, band=band, offset=offset, seed=0, device=DEV)
                lab = torch.zeros(M1 - M0, iu.shape[1], dtype=torch.int8)
                for (a, b) in truth:
                    lab[(a - M0 * m) // m, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = 1
                labs.append(lab.reshape(-1))
                C, levels = scale.market_level_coupling(X, m)
                cells = [2 ** l for l in levels]
                sc['ours_levels'].append(scale.combine_levels(C, iu, cells).reshape(-1).cpu())
                coarse_idx = [i for i, l in enumerate(levels) if l <= 5]
                sc['ours_levels_coarse'].append(scale.combine_levels(C[coarse_idx], iu, [cells[i] for i in coarse_idx]).reshape(-1).cpu())
                per_level.append(C[:, :, iu[0], iu[1]].reshape(len(levels), -1).cpu())
                Zl = torch.atanh(C[:, :, iu[0], iu[1]].clamp(-0.999, 0.999)) * torch.sqrt(torch.tensor(cells, dtype=C.dtype, device=C.device))[:, None, None]
                Zl = torch.stack([scale.robust_z(z.reshape(-1)) for z in Zl])          # per-level robust z
                sc['ours_levels_max'].append(Zl.max(0).values.cpu())
                La = clean(torch.log(X.abs() + 1e-12), m)
                r = bmf.rho_q(La, m, scales=(8, 16, 32, 64, 128), qs=(2,))[0]
                sc['rho2_cleaned_8_128'].append(r[:, iu[0], iu[1]].reshape(-1).float().cpu())
                r = bmf.rho_q(La, m, scales=(8, 16, 32, 64, 128, 256, 512), qs=(2,))[0]
                sc['rho2_cleaned_8_512'].append(r[:, iu[0], iu[1]].reshape(-1).float().cpu())
                r = bmf.rho_q(La, m, scales=(4, 8, 16, 32, 64, 128, 256, 512), qs=(1,), form='kwapien')[0]
                sc['rho2_cleaned_4_512_kwapien_q1'].append(r[:, iu[0], iu[1]].reshape(-1).float().cpu())
                del X, La
            L = torch.cat(labs).numpy()
            rz = lambda v: scale.robust_z(torch.cat(v).double())
            sc['union_levelsmax_rho2'] = [torch.maximum(rz(sc['ours_levels_max']), rz(sc['rho2_cleaned_8_128']))]
            row = {}
            for k, v in sc.items():
                s = torch.cat(v).numpy().astype(np.float64)
                row[k] = ews.auc(s[L == 1], s[L == 0])
            PL = torch.cat(per_level, 1).numpy()
            row['ours_by_level'] = {int(l): ews.auc(PL[i][L == 1], PL[i][L == 0]) for i, l in enumerate(levels)}
            results[f'band{band[0]}-{band[1]}_off{offset}'] = row
            print(f'band {band} offset {offset}: ' + '  '.join(f'{k} {v:.3f}' for k, v in row.items() if not isinstance(v, dict))
                  + '  | by level ' + ' '.join(f'{l}:{a:.2f}' for l, a in row['ours_by_level'].items()), file=sys.stderr, flush=True)
    os.makedirs(os.path.join(HERE, 'results', 'e12'), exist_ok=True)
    json.dump(results, open(os.path.join(HERE, 'results', 'e12', f'bands{args.tag}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
