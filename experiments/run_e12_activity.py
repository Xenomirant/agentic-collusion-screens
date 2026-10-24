"""E12: coordination of *timing and intensity* across scales, which classical multifractal tools
cannot see, at a million agents.

World (experiments/sims/planted_markets.py: chunk_activity): M markets x m agents, T rounds.
Every agent's move is x_i(t) = sigma_i(t) xi_i(t) with independent directions xi, so every signed
statistic has zero expectation for every pair. Activity log sigma is a multiplicative (multifractal)
cascade: a shared-model component (2 base models, heterogeneous exposure: the dense low-rank common
mode), a market component and the agent's own cascade. Coalitions (40% of markets: pairs and triads,
full or half share) replace the coarse levels of their own cascades by a shared one: they agree
*when* to be active, on scales of 16 rounds and up. Each agent's marginal law is unchanged.

Methods, all on the same chunks, scored within market:
  signed corr lag 0 / lag 1            classical correlation of moves
  rho_DCCA(x), lag 0 and lag 1         fathon's multiscale detrended cross-correlation (q = 2)
  per-agent MFDFA width                single-series multifractal spectrum (agent-level screen)
  corr |x|                             single-scale activity correlation
  rho_DCCA(|x|)                        multiscale cross-correlation of activity (strongest classical)
  rho_DCCA(|x|), common mode removed   the same after our low-rank + market-mean removal
  cross-replica level coupling         ours: dyadic log-energy pyramid (QTT levels), low-rank
                                       common mode removed per level, Stouffer over levels
  ... without removal                  ablation
Output: experiments/results/e12/summary_N<N>.json
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
from run_e11_scale import rho_dcca_markets, mfdfa_width, triu  # noqa: E402
from fathon.tensor import scale, ews  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e12')
DEV = 'cuda'
CLEAN_RANKS = []
CODES = {'none': 0, 'timing_pair': 1, 'timing_pair_half': 2, 'timing_triad': 3, 'timing_triad_half': 4}


def rho_dcca_lag(X, m, lag=1):
    """rho_DCCA between x_i(t) and x_j(t - lag), symmetrised by max over directions."""
    A = torch.cat([X[:, lag:], X[:, :-lag]], 0)
    n = X.shape[0]
    M = n // m
    # interleave so that market blocks hold [later_i ..., earlier_j ...]
    Xl, Xe = X[:, lag:], X[:, :-lag]
    out = torch.zeros(M, m, m, device=X.device)
    from run_e11_scale import SCALES, projector
    T = Xl.shape[1]
    Yl = (Xl - Xl.mean(1, keepdim=True)).cumsum(1).reshape(M, m, T)
    Ye = (Xe - Xe.mean(1, keepdim=True)).cumsum(1).reshape(M, m, T)
    for s in SCALES:
        nw = T // s
        P = projector(s)
        El = (Yl[:, :, :nw * s].reshape(M, m, nw, s) @ P).reshape(M, m, nw * s)
        Ee = (Ye[:, :, :nw * s].reshape(M, m, nw, s) @ P).reshape(M, m, nw * s)
        F = El @ Ee.transpose(1, 2)
        dl = El.norm(dim=2)
        de = Ee.norm(dim=2)
        R = F / (dl[:, :, None] * de[:, None, :] + 1e-12)
        out += torch.maximum(R.abs(), R.transpose(1, 2).abs())
    return out / len(SCALES)


def clean(X, m, rank=None, extra=8):
    """Market mean removed first, then the global low-rank common mode (estimated rank + extra)
    of the residual: the same order and estimator as the method."""
    M = X.shape[0] // m
    Xc = X - X.mean(1, keepdim=True)
    R = Xc.reshape(M, m, -1)
    R = (R - R.mean(1, keepdim=True)).reshape(X.shape[0], -1)
    r_ = scale.estimate_rank(R[:50000])[0] if rank is None else rank
    CLEAN_RANKS.append(int(r_))
    V = scale.global_basis(R[:50000], r_, extra=extra)
    return R - (R @ V) @ V.T


def evaluate(score, label, name, log=True):
    score = score.numpy().astype(np.float64)
    label = label.numpy()
    pos = label > 0
    null = score[~pos]
    nul = null
    res = dict(n_pairs=int(score.size), n_true=int(pos.sum()), auc_all=float(ews.auc(score[pos], nul)))
    for tp, c in CODES.items():
        if c and (label == c).any():
            res[f'auc_{tp}'] = float(ews.auc(score[label == c], nul))
    for fpr in (1e-2, 1e-3, 1e-4):
        thr = np.quantile(null, 1 - fpr)
        res[f'recall_fpr{fpr:g}'] = float((score[pos] > thr).mean())
    K = int(pos.sum())
    top = np.argpartition(-score, K)[:K]
    res['precision_at_K'] = float(pos[top].mean())
    if log:
        print(f'{name:40s} AUC {res["auc_all"]:.3f}  ' + ' '.join(f'{tp}:{res.get("auc_" + tp, float("nan")):.3f}' for tp in CODES if tp != 'none')
              + f'  P@K {res["precision_at_K"]:.3f}  R@1e-3 {res["recall_fpr0.001"]:.3f}', file=sys.stderr, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=int, default=50000)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=2500)
    ap.add_argument('--baseline_chunks', type=int, default=4)
    ap.add_argument('--lam', type=float, default=0.35)
    ap.add_argument('--w_global_sd', type=float, default=0.8)
    ap.add_argument('--share', type=float, default=1.0)
    ap.add_argument('--coarse', type=int, default=6)
    ap.add_argument('--offset', type=int, default=0, help='misalign cascade blocks and detector cells (e.g. 37)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()
    M, m, T = args.markets, args.m, args.T
    N = M * m
    os.makedirs(OUT, exist_ok=True)
    iu = triu(m)
    npairs = iu.shape[1]
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    G = pm.shared_activity(2 * T if args.offset else T, seed=args.seed, lam=args.lam)
    kw = dict(lam=args.lam, w_global_sd=args.w_global_sd, share=args.share, coarse=args.coarse, offset=args.offset,
              seed=args.seed, device=DEV)
    t_sim = t_ours = t_base = 0.0
    ours, ours_raw, labels, per_level = [], [], [], []
    base = {k: [] for k in ('signed_corr_lag0', 'signed_corr_lag1', 'rho_dcca_x_lag0', 'rho_dcca_x_lag1', 'corr_abs',
                            'rho_dcca_abs', 'rho_dcca_abs_cleaned', 'rho_dcca_logabs', 'rho_dcca_logabs_cleaned')}
    blabels, width_v, width_l = [], [], []
    for ci, M0 in enumerate(range(0, M, args.chunk)):
        M1 = min(M, M0 + args.chunk)
        torch.cuda.synchronize(); t0 = time.time()
        X, mk, truth, ttype = pm.chunk_activity(M0, M1, m, T, G, **kw)
        torch.cuda.synchronize(); t_sim += time.time() - t0
        Mc = M1 - M0
        lab = torch.zeros(Mc, npairs, dtype=torch.int8)
        for (a, b), tp in ttype.items():
            lab[(a - M0 * m) // m, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = CODES[tp]
        labels.append(lab)
        t0 = time.time()
        C, levels = scale.market_level_coupling(X, m)
        cells = [2 ** l for l in levels]
        S = scale.combine_levels(C, iu, cells)
        torch.cuda.synchronize(); t_ours += time.time() - t0
        ours.append(S.cpu())
        per_level.append(C[:, :, iu[0], iu[1]].cpu())
        if ci < args.baseline_chunks:
            Craw, _ = scale.market_level_coupling(X, m, rank=0)
            ours_raw.append(scale.combine_levels(Craw, iu, cells).cpu())
            t0 = time.time()
            base['signed_corr_lag0'].append(scale.market_pair_scores(X, m, lag=0, parity=False, market_mean=False)['lag'][:, iu[0], iu[1]].cpu())
            base['signed_corr_lag1'].append(scale.market_pair_scores(X, m, lag=1, parity=False, market_mean=False)['lag'][:, iu[0], iu[1]].cpu())
            base['rho_dcca_x_lag0'].append(rho_dcca_markets(X, m)[:, iu[0], iu[1]].cpu())
            base['rho_dcca_x_lag1'].append(rho_dcca_lag(X, m, 1)[:, iu[0], iu[1]].cpu())
            Ab = X.abs()
            base['corr_abs'].append(scale.market_pair_scores(Ab, m, lag=0, parity=False, market_mean=False)['lag'][:, iu[0], iu[1]].cpu())
            base['rho_dcca_abs'].append(rho_dcca_markets(Ab, m)[:, iu[0], iu[1]].cpu())
            base['rho_dcca_abs_cleaned'].append(rho_dcca_markets(clean(Ab, m), m)[:, iu[0], iu[1]].cpu())
            La = torch.log(Ab + 1e-12)
            base['rho_dcca_logabs'].append(rho_dcca_markets(La, m)[:, iu[0], iu[1]].cpu())
            base['rho_dcca_logabs_cleaned'].append(rho_dcca_markets(clean(La, m), m)[:, iu[0], iu[1]].cpu())
            w = mfdfa_width(X).cpu()
            member = torch.zeros(X.shape[0], dtype=torch.bool)
            for (a, b) in truth:
                member[a - M0 * m] = True
                member[b - M0 * m] = True
            width_v.append(w); width_l.append(member)
            blabels.append(lab)
            torch.cuda.synchronize(); t_base += time.time() - t0
        print(f'chunk {ci}: sim {t_sim:.1f}s ours {t_ours:.1f}s baselines {t_base:.1f}s', file=sys.stderr, flush=True)
        del X, C
    L = torch.cat(labels).reshape(-1)
    summ = dict(N=N, markets=M, m=m, T=T, args=vars(args), levels=levels, seconds=dict(simulate=t_sim, ours=t_ours, baselines=t_base),
                clean_rank_estimates=CLEAN_RANKS,
                ours=evaluate(torch.cat(ours).reshape(-1), L, 'cross-replica level coupling (ours)'))
    PL = torch.cat(per_level, 1)                      # (levels, markets, pairs)
    summ['ours_auc_by_level'] = {int(l): float(ews.auc(PL[k].reshape(-1)[L > 0].numpy(), PL[k].reshape(-1)[L == 0].numpy()))
                                 for k, l in enumerate(levels)}
    print('per-level AUC (ours):', {l: round(v, 3) for l, v in summ['ours_auc_by_level'].items()}, file=sys.stderr)
    if blabels:
        BL = torch.cat(blabels).reshape(-1)
        summ['ours_no_removal'] = evaluate(torch.cat(ours_raw).reshape(-1), BL, 'ours without common-mode removal')
        for k, v in base.items():
            summ[k] = evaluate(torch.cat(v).reshape(-1), BL, k)
        wv, wl = torch.cat(width_v).numpy(), torch.cat(width_l).numpy()
        summ['mfdfa_width_agent_auc'] = float(ews.auc(wv[wl], wv[~wl]))
        print(f'per-agent MFDFA width: AUC member vs non-member = {summ["mfdfa_width_agent_auc"]:.3f}', file=sys.stderr)
    json.dump(summ, open(os.path.join(OUT, f'summary_N{N}{args.tag}.json'), 'w'), indent=2)
    print(json.dumps(summ['seconds']))


if __name__ == '__main__':
    main()
