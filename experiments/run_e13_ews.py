"""E13: early warning of coalition onset at scale. Who starts colluding, and how soon can we tell?

World: experiments/sims/planted_markets.py `chunk` (copy coupling) with T rounds and onset at T/2 in
the onset markets (index mod 10 == 3); the other coalition types are present from the start and
are excluded from the evaluation (they are neither "onset" nor "clean"). Positives: the onset pair
of every onset market. Negatives: all pairs of clean markets (mod 10 in 4..9).

Common mode, causally: agent-space loadings (rank 8 + 4) are learned on the burn-in window [0, W)
only and projected out of every later window (no look-ahead); the market mean is removed per step.

Indicators computed on sliding windows of W rounds ending at t (grid of t values):
  ours            within-market lag-1 residual coupling after that causal removal (raw score;
                  thresholds are taken from the null pairs of the same window)
  ours_expanding  the same on the expanding window [0, t]
  rho_dcca        within-market rho_DCCA (lag 0 and lag 1, max) on the raw window
  rho_dcca_cleaned  the same on the causally cleaned window (same preprocessing as ours)
  ews_hurst       classical single-series EWS: change of the agent's DFA Hurst exponent between the
                  window and the first W rounds; pair score = min over the two agents
  ews_width       the same with the MFDFA spectrum width h(-3) - h(3)
For each indicator and t: detection rate of positives at false-positive rates 1e-2 / 1e-3 (threshold
from negatives at the same t) and AUC. Output: experiments/results/e13/summary_N<N>.json
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
from run_e11_scale import rho_dcca_markets, triu, projector, SCALES  # noqa: E402
from run_e12_activity import rho_dcca_lag  # noqa: E402
from fathon.tensor import scale, ews  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e13')
DEV = 'cuda'
QS = torch.tensor([-3., -2., -1., 1., 2., 3.])


def hurst_width(X):
    """Per-agent DFA Hurst exponent (q = 2) and MFDFA width h(-3) - h(3) over SCALES."""
    Y = (X - X.mean(1, keepdim=True)).cumsum(1)
    N, T = Y.shape
    q = QS.to(DEV)
    sc = [s for s in SCALES if s <= T // 4]
    logF, logF2 = [], []
    for s in sc:
        nw = T // s
        W = Y[:, :nw * s].reshape(N, nw, s) @ projector(s)
        f2 = (W ** 2).mean(2)
        lf = torch.log(f2 + 1e-30)
        Fq = torch.exp(torch.logsumexp(0.5 * q[None, None, :] * lf[:, :, None], 1) - np.log(nw)) ** (1.0 / q[None, :])
        logF.append(torch.log(Fq))
        logF2.append(0.5 * torch.log(f2.mean(1)))
    ls = torch.log(torch.tensor(sc, dtype=torch.float32, device=DEV))
    ls = ls - ls.mean()
    h = (torch.stack(logF, 1) * ls[None, :, None]).sum(1) / (ls ** 2).sum()
    H = (torch.stack(logF2, 1) * ls[None, :]).sum(1) / (ls ** 2).sum()
    return H, h[:, 0] - h[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=int, default=20000)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--T', type=int, default=2048)
    ap.add_argument('--W', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=2500)
    ap.add_argument('--eps_copy', type=float, default=0.75)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()
    M, m, T, W = args.markets, args.m, args.T, args.W
    N = M * m
    onset = T // 2
    os.makedirs(OUT, exist_ok=True)
    iu = triu(m)
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    grid = [onset - W // 2, onset, onset + W // 4, onset + W // 2, onset + 3 * W // 4, onset + W, onset + 3 * W // 2, onset + 2 * W]
    grid = [t for t in grid if W <= t <= T]
    zg = pm.global_latents(T, 4, 2, seed=args.seed, device=DEV)
    names = ('ours', 'ours_expanding', 'rho_dcca', 'rho_dcca_cleaned', 'ews_hurst', 'ews_width')
    BURN = W                                            # causal: the common mode is learned on [0, BURN) only
    pos = {(t, k): [] for t in grid for k in names}
    neg = {(t, k): [] for t in grid for k in names}
    t_ours = 0.0
    for ci, M0 in enumerate(range(0, M, args.chunk)):
        M1 = min(M, M0 + args.chunk)
        A, mk, truth, ttype = pm.chunk(M0, M1, m, T, zg, seed=args.seed, device=DEV, eps_copy=args.eps_copy, onset=onset)
        mtype = torch.arange(M0, M1) % 10
        clean_mk = torch.nonzero(mtype >= 4).squeeze(1)
        # onset pairs from the labels: (market, pair index)
        onset_rows, onset_cols = [], []
        for (a, b), tp in ttype.items():
            if tp == 'copy_onset':
                la, lb = (a - M0 * m) % m, (b - M0 * m) % m
                onset_rows.append((a - M0 * m) // m)
                onset_cols.append(pidx[(la, lb)])
        onset_rows, onset_cols = torch.tensor(onset_rows), torch.tensor(onset_cols)
        # causal common mode: agent-space loadings from the burn-in window, projected out of every window
        Xc = A - A[:, :BURN].mean(1, keepdim=True)
        Rm = Xc.reshape(-1, m, T)
        Rm = Rm - Rm.mean(1, keepdim=True)                       # market mean (causal: per time step)
        Xm = Rm.reshape(A.shape[0], T)
        Ub, Sb, Vb = torch.svd_lowrank(Xm[:, :BURN], q=16, niter=4)
        Uk = Ub[:, :12]                                          # rank 8 common mode + 4 extra
        R = Xm - Uk @ (Uk.T @ Xm)
        H0, W0 = hurst_width(A[:, :W])
        for t in grid:
            torch.cuda.synchronize(); t0 = time.time()
            sw = scale.market_pair_scores(R[:, t - W:t], m, lag=1, parity=False)['lag'][:, iu[0], iu[1]]
            torch.cuda.synchronize(); t_ours += time.time() - t0
            se = scale.market_pair_scores(R[:, :t], m, lag=1, parity=False)['lag'][:, iu[0], iu[1]]
            Xw = A[:, t - W:t]
            sd = torch.maximum(rho_dcca_markets(Xw, m), rho_dcca_lag(Xw, m, 1))[:, iu[0], iu[1]]
            Rw = R[:, t - W:t]
            sdc = torch.maximum(rho_dcca_markets(Rw, m), rho_dcca_lag(Rw, m, 1))[:, iu[0], iu[1]]
            Ht, Wt = hurst_width(Xw)
            dH = (Ht - H0).abs().reshape(-1, m)
            dW = (Wt - W0).abs().reshape(-1, m)
            sh = torch.minimum(dH[:, iu[0]], dH[:, iu[1]])
            swd = torch.minimum(dW[:, iu[0]], dW[:, iu[1]])
            for k, S in (('ours', sw), ('ours_expanding', se), ('rho_dcca', sd), ('rho_dcca_cleaned', sdc),
                         ('ews_hurst', sh), ('ews_width', swd)):
                S = S.cpu()
                pos[(t, k)].append(S[onset_rows, onset_cols])
                neg[(t, k)].append(S[clean_mk].reshape(-1))
        print(f'chunk {ci} done', file=sys.stderr, flush=True)
        del A, R
    res = {}
    for k in names:
        rows = {}
        for t in grid:
            p_ = torch.cat(pos[(t, k)]).numpy().astype(np.float64)
            n_ = torch.cat(neg[(t, k)]).numpy().astype(np.float64)
            r = dict(auc=float(ews.auc(p_, n_)))
            for fpr in (1e-2, 1e-3):
                r[f'detect_fpr{fpr:g}'] = float((p_ > np.quantile(n_, 1 - fpr)).mean())
            rows[str(t - onset)] = r
        res[k] = rows
        print(f'{k:16s} ' + '  '.join(f'{int(t) - onset:+5d}: AUC {rows[str(t - onset)]["auc"]:.2f} D@1e-2 {rows[str(t - onset)]["detect_fpr0.01"]:.2f}'
                                       for t in grid), file=sys.stderr, flush=True)
    summ = dict(N=N, markets=M, m=m, T=T, W=W, onset=onset, eps_copy=args.eps_copy,
                n_onset_pairs=int(sum(x.numel() for x in pos[(grid[0], 'ours')])),
                n_null_pairs=int(sum(x.numel() for x in neg[(grid[0], 'ours')])),
                seconds_ours_per_window=t_ours / len(grid), results_by_offset_from_onset=res)
    json.dump(summ, open(os.path.join(OUT, f'summary_N{N}{args.tag}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
