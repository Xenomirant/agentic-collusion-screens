"""E11: the threat model at scale. Millions of agents, tens of thousands of markets, a few base
models; coalitions of every planted type; one streaming pass; multifractal baselines on the same data.

World (experiments/sims/planted_markets.py): M markets x m agents, T rounds. Each agent is built
on one of 2 base models (dense loading on a global rank-4 latent per model), responds to its
market's own public state (rank-1 latent per market) and is sticky (c = 0.5). 10% of markets hold
a copy pair, 10% a copy triad, 10% an XOR triad (pairwise invisible), 10% a copy pair whose coupling
switches on at T/2 (early warning); 60% are clean.

Method (fathon.tensor.scale): sketch the global common mode from a 50k-agent subsample
(global_basis), then per chunk of markets: project it out, remove the market mean, score every
within-market pair on the lag-1 residual channel and the parity channel, robust-z each channel
over all pairs, combine by max. No N x N object, no per-pair loop.

Baselines on the same chunks: plain within-market lag-1 correlation (no common-mode handling);
rho_DCCA across scales within market (the multifractal cross-correlation of fathon, batched on the
GPU here; its C implementation is timed per pair and extrapolated); per-agent MFDFA spectrum width
as an agent-level screen (single-series multifractal analysis).

Metrics over all within-market pairs: AUC by coalition type, recall at per-pair false-positive
rates 1e-3 / 1e-4 / 1e-5, precision@K (K = planted pairs), detection delay for onset pairs.

Usage: python run_e11_scale.py --markets 50000 --m 20 --T 1024 --chunk 2500
Output: experiments/results/e11/summary_<N>.json
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
import planted_markets as pm  # noqa: E402
from fathon.tensor import scale, ews  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e11')
DEV = 'cuda'
SCALES = [8, 16, 32, 64, 128]
QS = torch.tensor([-3., -2., -1., 1., 2., 3.])


def sync():
    torch.cuda.synchronize()


def projector(s, dtype=torch.float32):
    t = torch.arange(s, dtype=dtype, device=DEV)
    Vd = torch.stack([torch.ones_like(t), t], 1)
    return torch.eye(s, dtype=dtype, device=DEV) - Vd @ torch.linalg.solve(Vd.T @ Vd, Vd.T)


def rho_dcca_markets(X, m):
    """Mean |rho_DCCA| over SCALES for every within-market pair; X (M*m, T) raw or cleaned."""
    M, T = X.shape[0] // m, X.shape[1]
    Y = (X - X.mean(1, keepdim=True)).cumsum(1).reshape(M, m, T)
    acc = torch.zeros(M, m, m, device=DEV)
    for s in SCALES:
        nw = T // s
        W = Y[:, :, :nw * s].reshape(M, m, nw, s) @ projector(s)
        E = W.reshape(M, m, nw * s)
        F2 = E @ E.transpose(1, 2) / (nw * s)
        d = torch.sqrt(torch.diagonal(F2, dim1=1, dim2=2))
        acc += (F2 / (d[:, :, None] * d[:, None, :] + 1e-12)).abs()
    return acc / len(SCALES)


def mfdfa_width(X):
    """Per-agent MFDFA spectrum width h(-3) - h(3) over SCALES (batched, GPU)."""
    Y = (X - X.mean(1, keepdim=True)).cumsum(1)
    N, T = Y.shape
    q = QS.to(DEV)
    logF = []
    for s in SCALES:
        nw = T // s
        W = Y[:, :nw * s].reshape(N, nw, s) @ projector(s)
        f2 = (W ** 2).mean(2)                                   # (N, nw)
        lf = torch.log(f2 + 1e-30)
        Fq = torch.exp(torch.logsumexp(0.5 * q[None, None, :] * lf[:, :, None], 1) - np.log(nw)) ** (1.0 / q[None, :])
        logF.append(torch.log(Fq))
    logF = torch.stack(logF, 1)                                 # (N, scales, q)
    ls = torch.log(torch.tensor(SCALES, dtype=torch.float32, device=DEV))
    ls = ls - ls.mean()
    h = (logF * ls[None, :, None]).sum(1) / (ls ** 2).sum()    # slope per q
    return h[:, 0] - h[:, -1]


def triu(m):
    return torch.triu_indices(m, m, 1, device=DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=int, default=50000)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=2500)
    ap.add_argument('--baseline_chunks', type=int, default=4, help='chunks on which the baselines run')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps_copy', type=float, default=0.5)
    ap.add_argument('--eps_xor', type=float, default=2.0)
    ap.add_argument('--c', type=float, default=0.5)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()
    simkw = dict(eps_copy=args.eps_copy, eps_xor=args.eps_xor, c=args.c)
    M, m, T = args.markets, args.m, args.T
    N = M * m
    os.makedirs(OUT, exist_ok=True)
    iu = triu(m)
    npairs = iu.shape[1]
    # local pair index lookup for truth
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}

    zg = pm.global_latents(T, 4, 2, seed=args.seed, device=DEV)
    t_sim = t_det = t_base = 0.0
    # --- sketch of the global common mode from the first chunk (<= 50k agents)
    sync(); t0 = time.time()
    A0, mk0, tr0, tt0 = pm.chunk(0, min(args.chunk, M), m, T, zg, seed=args.seed, device=DEV, **simkw)
    sync(); t_sim += time.time() - t0
    t0 = time.time()
    V = scale.global_basis(A0[:50000], rank=8)
    sync(); t_det += time.time() - t0
    del A0

    lag_all, par_all, raw_all, label_all, type_all = [], [], [], [], []
    dcca_raw, dcca_clean, dcca_label, width_vals, width_label = [], [], [], [], []
    onset_curves = {}
    type_code = {'none': 0, 'copy_pair': 1, 'copy_triad': 2, 'xor_triad': 3, 'copy_onset': 4}
    grid = [256, 384, 512, 576, 640, 704, 768, 896, T]
    for ci, M0 in enumerate(range(0, M, args.chunk)):
        M1 = min(M, M0 + args.chunk)
        sync(); t0 = time.time()
        A, mk, truth, ttype = pm.chunk(M0, M1, m, T, zg, seed=args.seed, device=DEV, **simkw)
        sync(); t_sim += time.time() - t0
        Mc = M1 - M0
        # labels
        lab = torch.zeros(Mc, npairs, dtype=torch.int8)
        for (a, b), tp in ttype.items():
            k = (a - M0 * m) // m
            lab[k, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = type_code[tp]
        # --- method
        sync(); t0 = time.time()
        R = scale.remove_global(A, V)
        sc = scale.market_pair_scores(R, m, lag=1, parity=True)
        sync(); t_det += time.time() - t0
        lag_all.append(sc['lag'][:, iu[0], iu[1]].cpu())
        par_all.append(sc['parity'][:, iu[0], iu[1]].cpu())
        type_all.append(lab)
        # --- baselines
        if ci < args.baseline_chunks:
            sync(); t0 = time.time()
            raw = scale.market_pair_scores(A, m, lag=1, parity=False, market_mean=False)['lag']
            raw_all.append(raw[:, iu[0], iu[1]].cpu())
            dr = rho_dcca_markets(A, m)[:, iu[0], iu[1]].cpu()
            dc = rho_dcca_markets(R, m)[:, iu[0], iu[1]].cpu()
            dcca_raw.append(dr); dcca_clean.append(dc); dcca_label.append(lab)
            w = mfdfa_width(A).cpu()
            member = torch.zeros(A.shape[0], dtype=torch.bool)
            for (a, b) in truth:
                member[a - M0 * m] = True; member[b - M0 * m] = True
            width_vals.append(w); width_label.append(member)
            sync(); t_base += time.time() - t0
            # early warning on onset pairs: expanding windows
            if ci < 2:
                onset_rows = torch.nonzero(lab == 4, as_tuple=False)
                null_rows = torch.nonzero(lab == 0, as_tuple=False)
                for t in grid:
                    sct = scale.market_pair_scores(R[:, :t], m, lag=1, parity=False)['lag'][:, iu[0], iu[1]].cpu()
                    z = scale.robust_z(sct.reshape(-1)).reshape(sct.shape)
                    zn = z[null_rows[:, 0], null_rows[:, 1]]
                    zo = z[onset_rows[:, 0], onset_rows[:, 1]]
                    for fpr in (1e-2, 1e-3):
                        thr = torch.quantile(zn[torch.randperm(zn.numel())[:1_000_000]], 1 - fpr).item()
                        onset_curves.setdefault(f'{t}@fpr{fpr:g}', []).append(float((zo > thr).float().mean()))
        print(f'chunk {ci}: markets {M0}-{M1}  sim {t_sim:.1f}s  detect {t_det:.1f}s  baselines {t_base:.1f}s', file=sys.stderr, flush=True)
        del A, R, sc
    # --- evaluation
    lag = torch.cat(lag_all).reshape(-1)
    par = torch.cat(par_all).reshape(-1)
    typ = torch.cat(type_all).reshape(-1)
    z = torch.maximum(scale.robust_z(lag), scale.robust_z(par))
    zl = scale.robust_z(lag)

    def evaluate(score, label, name):
        score = score.numpy().astype(np.float64); label = label.numpy()
        pos_all = label > 0
        null = score[~pos_all]
        res = dict(n_pairs=int(score.size), n_true=int(pos_all.sum()))
        res['auc_all'] = float(ews.auc(score[pos_all], null[:2_000_000] if null.size > 2_000_000 else null))
        for tp, code in type_code.items():
            if code == 0:
                continue
            p = score[label == code]
            if p.size:
                res[f'auc_{tp}'] = float(ews.auc(p, null[:2_000_000] if null.size > 2_000_000 else null))
        for fpr in (1e-3, 1e-4, 1e-5):
            if null.size * fpr >= 1:
                thr = np.quantile(null, 1 - fpr)
                res[f'recall_fpr{fpr:g}'] = float((score[pos_all] > thr).mean())
                for tp, code in type_code.items():
                    if code and (label == code).any():
                        res[f'recall_fpr{fpr:g}_{tp}'] = float((score[label == code] > thr).mean())
        K = int(pos_all.sum())
        top = np.argpartition(-score, K)[:K]
        res['precision_at_K'] = float(pos_all[top].mean())
        print(f'{name:28s} AUC {res["auc_all"]:.3f}  ' + ' '.join(f'{tp}:{res.get(f"auc_{tp}", float("nan")):.3f}' for tp in type_code if tp != 'none')
              + f'  P@K {res["precision_at_K"]:.3f}  R@1e-4 {res.get("recall_fpr0.0001", float("nan")):.3f}', file=sys.stderr, flush=True)
        return res

    summ = dict(N=N, markets=M, m=m, T=T, pairs=int(z.numel()), eps_copy=args.eps_copy, eps_xor=args.eps_xor, c=args.c,
                seconds=dict(simulate=t_sim, detect=t_det, baselines=t_base),
                method=evaluate(z, typ, 'method (lag+parity, z-max)'),
                method_lag_only=evaluate(zl, typ, 'method lag channel only'))
    if raw_all:
        lab_b = torch.cat(dcca_label).reshape(-1)
        summ['baseline_raw_corr'] = evaluate(torch.cat(raw_all).reshape(-1), lab_b, 'raw lag-1 correlation')
        summ['baseline_rho_dcca_raw'] = evaluate(torch.cat(dcca_raw).reshape(-1), lab_b, 'rho_DCCA raw')
        summ['baseline_rho_dcca_cleaned'] = evaluate(torch.cat(dcca_clean).reshape(-1), lab_b, 'rho_DCCA after common-mode removal')
        w = torch.cat(width_vals).numpy(); wl = torch.cat(width_label).numpy()
        summ['baseline_mfdfa_width_agent_auc'] = float(ews.auc(w[wl], w[~wl]))
        print(f'per-agent MFDFA width: AUC colluder vs not = {summ["baseline_mfdfa_width_agent_auc"]:.3f}', file=sys.stderr)
    summ['onset_detection_vs_window'] = {str(t): float(np.mean(v)) for t, v in onset_curves.items()}
    print('onset pairs detected at FPR 1e-3 vs window end:', summ['onset_detection_vs_window'], file=sys.stderr)
    json.dump(summ, open(os.path.join(OUT, f'summary_N{N}{args.tag}.json'), 'w'), indent=2)
    print(json.dumps(summ['seconds']))


if __name__ == '__main__':
    main()
