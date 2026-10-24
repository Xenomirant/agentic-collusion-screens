"""E14: threat-model coverage, specificity against benign interaction, and calibration at scale.

World (experiments/sims/planted_markets.py `chunk`, layout='random'): M markets x m agents, T rounds,
2 base models (global rank-4 latents, strength w_global), one market state, sticky agents. Coalition
types, each in a fraction `prev` of markets: copy_pair, copy_triad, xor_triad, copy_onset (on at T/2),
anti_pair (negative lag-1 coupling), lag_pair (copy at lag 2), match_pair (shared private signal,
lag-0 co-movement). Benign interaction: in `benign_frac` of all markets every non-member follows one
random rival one-way (logit eps_benign): coupled but not colluding.
A second, coalition-free world with the same parameters calibrates the null.

Detectors (all within market, global common mode removed, market mean removed):
  v1           E11 method: lag-1 best-direction channel + parity channel, robust-z max
  v2_oneway    lags 0-3, two-sided, max(|C_ij|, |C_ji|) + parity
  v2_mutual    lags 0-3, two-sided, |C_ij + C_ji| / sqrt 2 (reciprocal) + parity
  raw_lag1     lag-1 correlation of raw actions (no removal), best direction
  rho_dcca     rho_DCCA (lag 0 and lag 1, max), raw
Metrics: AUC per coalition type vs clean pairs and vs benign pairs; recall per type and false
discovery rate (benign + clean flags) at thresholds set on the coalition-free world (FPR 1e-3,
1e-4, family-wise max); PPV extrapolated to prevalences 1/10, 1/100, 1/1000 of markets.
Output: experiments/results/e14/summary_<tag>.json
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
from run_e11_scale import rho_dcca_markets, triu  # noqa: E402
from run_e12_activity import rho_dcca_lag  # noqa: E402
from fathon.tensor import scale, ews  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e14')
DEV = 'cuda'
TYPES = ['copy_pair', 'copy_triad', 'xor_triad', 'copy_onset', 'anti_pair', 'lag_pair', 'match_pair']
CODE = {t: i + 1 for i, t in enumerate(TYPES)}
CODE['benign'] = -1


def detectors(A, V, m, iu, raw=True):
    R = scale.remove_global(A, V)
    out = {}
    s1 = scale.market_pair_scores(R, m, lag=1, parity=True)
    zl = scale.robust_z(s1['lag'][:, iu[0], iu[1]].reshape(-1))
    zp = scale.robust_z(s1['parity'][:, iu[0], iu[1]].reshape(-1))
    out['v1'] = torch.maximum(zl, zp)
    C = scale.market_lag_channels(R, m, lags=(0, 1, 2, 3))
    out['v2_oneway'] = torch.maximum(scale.coupling_scores(C, iu, 'oneway').reshape(-1), zp)
    out['v2_mutual'] = torch.maximum(scale.coupling_scores(C, iu, 'mutual').reshape(-1), zp)
    if raw:
        r1 = scale.market_pair_scores(A, m, lag=1, parity=False, market_mean=False)['lag'][:, iu[0], iu[1]]
        out['raw_lag1'] = r1.reshape(-1)
        out['rho_dcca'] = torch.maximum(rho_dcca_markets(A, m), rho_dcca_lag(A, m, 1))[:, iu[0], iu[1]].reshape(-1)
    return {k: v.float().cpu() for k, v in out.items()}


def run_world(args, mix, benign_frac, tag):
    M, m, T = args.markets, args.m, args.T
    iu = triu(m)
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    zg = pm.global_latents(T, 4, 2, seed=args.seed, device=DEV)
    kw = dict(eps_copy=args.eps_copy, eps_xor=args.eps_xor, w_global=args.w_global, c=args.c, mix=mix, copy_lag=2,
              eps_match=args.eps_match, eps_benign=args.eps_benign, benign_frac=benign_frac, seed=args.seed + (0 if mix else 7),
              device=DEV)
    A0, *_ = pm.chunk(0, min(args.chunk, M), m, T, zg, **kw)
    V = scale.global_basis(A0[:50000], rank=8)
    del A0
    scores, labels = {}, []
    t_det = 0.0
    for ci, M0 in enumerate(range(0, M, args.chunk)):
        M1 = min(M, M0 + args.chunk)
        A, mk, truth, ttype = pm.chunk(M0, M1, m, T, zg, **kw)
        lab = torch.zeros(M1 - M0, iu.shape[1], dtype=torch.int8)
        for (a, b), tp in ttype.items():
            lab[(a - M0 * m) // m, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = CODE[tp]
        labels.append(lab.reshape(-1))
        torch.cuda.synchronize(); t0 = time.time()
        d = detectors(A, V, m, iu, raw=ci < args.baseline_chunks)
        torch.cuda.synchronize(); t_det += time.time() - t0
        for k, v in d.items():
            scores.setdefault(k, []).append(v)
        print(f'[{tag}] chunk {ci}', file=sys.stderr, flush=True)
        del A
    L = torch.cat(labels).numpy()
    S = {k: torch.cat(v).numpy().astype(np.float64) for k, v in scores.items()}
    return S, L, t_det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=int, default=20000)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=2500)
    ap.add_argument('--baseline_chunks', type=int, default=4)
    ap.add_argument('--prev', type=float, default=0.02, help='fraction of markets per coalition type')
    ap.add_argument('--benign_frac', type=float, default=0.5)
    ap.add_argument('--eps_benign', type=float, default=0.5)
    ap.add_argument('--eps_copy', type=float, default=0.75)
    ap.add_argument('--eps_xor', type=float, default=2.0)
    ap.add_argument('--eps_match', type=float, default=1.0)
    ap.add_argument('--w_global', type=float, default=2.0)
    ap.add_argument('--c', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mix = {t: args.prev for t in TYPES}
    S, L, t_det = run_world(args, mix, args.benign_frac, 'coalitions')
    S0, L0, _ = run_world(args, None if False else {}, args.benign_frac, 'null')   # same benign structure, no coalitions
    summ = dict(args=vars(args), seconds_detect=t_det, detectors={})
    for k in S:
        s, s0 = S[k], S0[k]
        n = min(len(s), len(L))
        s, lab = s[:n], L[:n]
        n0 = min(len(s0), len(L0))
        s0, lab0 = s0[:n0], L0[:n0]
        clean0 = s0[lab0 == 0]                       # null world, non-benign pairs
        benign = s[lab == -1]
        clean = s[lab == 0]
        r = dict(n_pairs=int(n), n_benign=int((lab == -1).sum()))
        for tp, c in CODE.items():
            if c > 0 and (lab == c).any():
                pos = s[lab == c]
                r[f'auc_vs_clean_{tp}'] = ews.auc(pos, clean)
                r[f'auc_vs_benign_{tp}'] = ews.auc(pos, benign) if benign.size else None
        pos_all = s[lab > 0]
        r['auc_vs_clean_all'] = ews.auc(pos_all, clean)
        r['auc_vs_benign_all'] = ews.auc(pos_all, benign) if benign.size else None
        # null calibration: the coalition-free world sets thresholds
        z = (clean0 - np.median(clean0)) / (1.4826 * np.median(np.abs(clean0 - np.median(clean0))))
        r['null_tail'] = {f'frac_z>{q}': float((z > q).mean()) for q in (3, 4, 5)}
        for name, thr in (('fpr1e-3', np.quantile(clean0, 1 - 1e-3)), ('fpr1e-4', np.quantile(clean0, 1 - 1e-4)),
                          ('fwer', clean0.max())):
            tp_ = int((pos_all > thr).sum())
            fp_clean = int((clean > thr).sum())
            fp_benign = int((benign > thr).sum())
            rec = {f'recall_{tp}': float((s[lab == c] > thr).mean()) for tp, c in CODE.items() if c > 0 and (lab == c).any()}
            fpr_c = fp_clean / max(1, clean.size)
            fpr_b = fp_benign / max(1, benign.size)
            ppv = {}
            n_pos_per_market = pos_all.size / max(1, (lab > 0).sum() and len(TYPES) * args.prev * args.markets)
            # extrapolate: pairs per coalition market ~ pos_all / (#coalition markets); clean/benign pairs as observed per market
            n_coal_mk = len(TYPES) * args.prev * args.markets
            tp_per_coal_mk = tp_ / max(1.0, n_coal_mk)
            fp_per_mk = (fp_clean + fp_benign) / args.markets
            for pv in (0.1, 0.01, 0.001):
                ppv[str(pv)] = float(pv * tp_per_coal_mk / (pv * tp_per_coal_mk + fp_per_mk)) if tp_ else 0.0
            r[name] = dict(threshold=float(thr), true_flags=tp_, false_flags_clean=fp_clean, false_flags_benign=fp_benign,
                           fdr=float((fp_clean + fp_benign) / max(1, tp_ + fp_clean + fp_benign)),
                           benign_share_of_false=float(fp_benign / max(1, fp_clean + fp_benign)), fpr_benign=fpr_b,
                           ppv_by_prevalence=ppv, **rec)
        # market-level screen: market score = max pair score; coalition markets vs coalition-free markets
        npair = (args.m * (args.m - 1)) // 2
        nm = n // npair
        ms = s[:nm * npair].reshape(nm, npair).max(1)
        ml = lab[:nm * npair].reshape(nm, npair)
        coal_mk = (ml > 0).any(1)
        benign_mk = (ml == -1).any(1) & ~coal_mk
        nm0 = n0 // npair
        ms0 = s0[:nm0 * npair].reshape(nm0, npair).max(1)          # null world: every market is coalition-free
        r['market_auc_all'] = ews.auc(ms[coal_mk], ms[~coal_mk])
        r['market_auc_vs_benign_markets'] = ews.auc(ms[coal_mk], ms[benign_mk]) if benign_mk.any() else None
        for tp, c in CODE.items():
            if c > 0:
                mk_t = (ml == c).any(1)
                if mk_t.any():
                    r[f'market_auc_{tp}'] = ews.auc(ms[mk_t], ms[~coal_mk])
        mrow = {}
        for name, q in (('mfpr1e-2', 1e-2), ('mfpr1e-3', 1e-3)):
            thr = np.quantile(ms0, 1 - q)
            rec = float((ms[coal_mk] > thr).mean())
            fp = float((ms[~coal_mk] > thr).mean())
            ppv = {str(pv): float(pv * rec / (pv * rec + (1 - pv) * fp)) if rec else 0.0 for pv in (0.1, 0.01, 0.001)}
            mrow[name] = dict(threshold=float(thr), recall=rec, fpr_observed=fp,
                              recall_by_type={tp: float((ms[(ml == c).any(1)] > thr).mean()) for tp, c in CODE.items() if c > 0 and (ml == c).any()},
                              ppv_by_prevalence=ppv)
        r['market_screen'] = mrow
        summ['detectors'][k] = r
        print(f"{k:10s} AUC clean {r['auc_vs_clean_all']:.3f} benign {r['auc_vs_benign_all'] or float('nan'):.3f} | "
              + ' '.join(f"{tp[:6]}:{r.get('auc_vs_clean_' + tp, float('nan')):.2f}/{r.get('auc_vs_benign_' + tp) or float('nan'):.2f}" for tp in TYPES)
              + f" | @1e-3 FDR {r['fpr1e-3']['fdr']:.2f} benign-share {r['fpr1e-3']['benign_share_of_false']:.2f}"
              + f" PPV(1/100) {r['fpr1e-3']['ppv_by_prevalence']['0.01']:.3f} | null z>4 {r['null_tail']['frac_z>4']:.1e}"
              + f" || MARKET AUC {r['market_auc_all']:.3f} vs-benign {r['market_auc_vs_benign_markets'] or float('nan'):.3f}"
              + f" recall@mFPR1e-2 {r['market_screen']['mfpr1e-2']['recall']:.2f} PPV(1/100) {r['market_screen']['mfpr1e-2']['ppv_by_prevalence']['0.01']:.3f}",
              file=sys.stderr, flush=True)
    json.dump(summ, open(os.path.join(OUT, f'summary{args.tag}.json'), 'w'), indent=2, default=float)


if __name__ == '__main__':
    main()
