"""E12c: timing coordination at an unknown band of scales, at scale, with the integrated pipeline.

Same world as E12b (planted_markets.chunk_activity, layout random, offset 37 = misaligned), bands
slow (1,4), medium (3,6), fast (5,8). Two streamed passes over all markets:
  pass 1  population common modes, one streamed Gram + parallel-analysis rank each:
          (a) base-level log energy (cells of 4 steps) for the QTT pyramid,
          (b) per-step log|x| for the rho_DCCA baselines (same treatment: market mean, then global);
  pass 2  scores per chunk.
Methods: ours = market_level_coupling_qtt + combine_levels (one decomposition, dyadic averaging);
rho2_8_128 / rho2_8_512 = signed q = 2 rho_DCCA on cleaned log|x| (the strongest classical
configuration found); union = max of the robust-z of ours and rho2_8_128.
Output: experiments/results/e12/scale_<tag>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sims'))
sys.path.insert(0, HERE)
import planted_markets as pm  # noqa: E402
from run_e11_scale import triu  # noqa: E402
import baselines_mf as bmf  # noqa: E402
from fathon.tensor import scale, ews  # noqa: E402

DEV = 'cuda'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=int, default=50000)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=2500)
    ap.add_argument('--offset', type=int, default=37)
    ap.add_argument('--bands', type=str, default='1-4,3-6,5-8')
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()
    M, m, T = args.markets, args.m, args.T
    iu = triu(m)
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    G = pm.shared_activity(2 * T if args.offset else T, seed=0)
    out = {}
    for bs in args.bands.split(','):
        band = tuple(int(x) for x in bs.split('-'))
        kw = dict(band=band, offset=args.offset, seed=0, device=DEV)

        def sim(M0):
            return pm.chunk_activity(M0, min(M, M0 + args.chunk), m, T, G, **kw)

        def base_chunks():
            for M0 in range(0, M, args.chunk):
                X = sim(M0)[0]
                E = scale.base_log_energy(X)
                yield scale.market_mean_removed(E - E.mean(1, keepdim=True), m)

        def logabs_chunks():
            for M0 in range(0, M, args.chunk):
                La = torch.log(sim(M0)[0].abs() + 1e-12)
                yield scale.market_mean_removed(La - La.mean(1, keepdim=True), m)

        torch.cuda.synchronize(); t0 = time.time()
        Vb, kb = scale.fit_common_mode(base_chunks, extra=4, device=DEV)
        torch.cuda.synchronize(); t_fit_b = time.time() - t0; t0 = time.time()
        Vl, kl = scale.fit_common_mode(logabs_chunks, extra=8, device=DEV)
        torch.cuda.synchronize(); t_fit_l = time.time() - t0
        Vb, Vl = Vb.float(), Vl.float()
        sc = {'ours': [], 'rho2_8_128': [], 'rho2_8_512': [], 'rho2_8_128_rankonly': [], 'rho2_8_512_rankonly': []}
        labs = []
        t_ours = t_rho = 0.0
        for M0 in range(0, M, args.chunk):
            X, mk, truth, tt = sim(M0)
            lab = torch.zeros(min(M, M0 + args.chunk) - M0, iu.shape[1], dtype=torch.int8)
            for (a, b) in truth:
                lab[(a - M0 * m) // m, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = 1
            labs.append(lab.reshape(-1))
            torch.cuda.synchronize(); t0 = time.time()
            C, cells = scale.market_level_coupling_qtt(X, m, V=Vb, rank=kb, extra=4)
            sc['ours'].append(scale.combine_levels(C, iu, cells).reshape(-1).cpu())
            torch.cuda.synchronize(); t_ours += time.time() - t0; t0 = time.time()
            La0 = torch.log(X.abs() + 1e-12)
            La0 = scale.market_mean_removed(La0 - La0.mean(1, keepdim=True), m)
            for suffix, Vx in (('', Vl), ('_rankonly', Vl[:, :kl])):
                La = La0 - (La0 @ Vx) @ Vx.T
                r = bmf.rho_q(La, m, scales=(8, 16, 32, 64, 128, 256, 512), qs=(2,), agg='none')[0]   # (scales, M, m, m)
                sc['rho2_8_128' + suffix].append(r[:5].mean(0)[:, iu[0], iu[1]].reshape(-1).float().cpu())
                sc['rho2_8_512' + suffix].append(r.mean(0)[:, iu[0], iu[1]].reshape(-1).float().cpu())
            torch.cuda.synchronize(); t_rho += time.time() - t0
            del X, La
        L = torch.cat(labs).numpy()
        S = {k: torch.cat(v).double() for k, v in sc.items()}
        # union with the baseline variant that is best on this band (an oracle choice in the baseline's favour)
        best_rho = max(('rho2_8_128', 'rho2_8_512', 'rho2_8_128_rankonly', 'rho2_8_512_rankonly'),
                       key=lambda k: ews.auc(S[k].numpy()[L == 1], S[k].numpy()[L == 0]))
        S['union_oracle'] = torch.maximum(scale.robust_z(S['ours']), scale.robust_z(S[best_rho]))
        # a-priori union: the baseline with the same removal rule as the pyramid (rank only), scales 8..512
        S['union'] = torch.maximum(scale.robust_z(S['ours']), scale.robust_z(S['rho2_8_512_rankonly']))
        row = dict(rank_base=kb, rank_logabs=kl, best_rho=best_rho, seconds=dict(fit_base=t_fit_b, fit_logabs=t_fit_l, ours=t_ours, rho=t_rho))
        for k, v in S.items():
            v = v.numpy()
            row[k] = dict(auc=ews.auc(v[L == 1], v[L == 0]))
            null = v[L == 0]
            for fpr in (1e-3, 1e-4):
                row[k][f'recall_fpr{fpr:g}'] = float((v[L == 1] > np.quantile(null, 1 - fpr)).mean())
        out[f'band{bs}'] = row
        print(f"band {band}: " + '  '.join(f"{k} AUC {row[k]['auc']:.3f} R@1e-3 {row[k]['recall_fpr0.001']:.3f}" for k in S)
              + f"  | ranks {kb}/{kl}  t ours {t_ours:.1f}s rho {t_rho:.1f}s fit {t_fit_b:.1f}+{t_fit_l:.1f}s", file=sys.stderr, flush=True)
    os.makedirs(os.path.join(HERE, 'results', 'e12'), exist_ok=True)
    json.dump(dict(args=vars(args), N=M * m, results=out), open(os.path.join(HERE, 'results', 'e12', f'scale{args.tag}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
