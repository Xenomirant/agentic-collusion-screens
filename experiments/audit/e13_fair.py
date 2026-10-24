"""Audit: E13 early warning (small world, T=1024), classical rho_DCCA with appropriate scales/lags."""
import os, sys, json
import numpy as np
import torch
REPO = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(REPO, 'sims')); sys.path.insert(0, REPO)
import planted_markets as pm
import baselines_mf as bmf
from run_e11_scale import rho_dcca_markets, triu
from run_e12_activity import rho_dcca_lag
from fathon.tensor import scale, ews

DEV = 'cuda'
M = int(sys.argv[1]); SEED = int(sys.argv[2])
m, T, W = 20, 1024, 256
onset = T // 2
iu = triu(m)
pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
zg = pm.global_latents(T, 4, 2, seed=SEED, device=DEV)
A, mk, truth, ttype = pm.chunk(0, M, m, T, zg, seed=SEED, device=DEV, eps_copy=0.75, onset=onset)
mtype = torch.arange(0, M) % 10
clean_mk = torch.nonzero(mtype >= 4).squeeze(1)
orow, ocol = [], []
for (a, b), tp in ttype.items():
    if tp == 'copy_onset':
        orow.append(a // m); ocol.append(pidx[(a % m, b % m)])
orow, ocol = torch.tensor(orow), torch.tensor(ocol)
BURN = W
Xc = A - A[:, :BURN].mean(1, keepdim=True)
Rm = Xc.reshape(-1, m, T); Rm = Rm - Rm.mean(1, keepdim=True)
Xm = Rm.reshape(A.shape[0], T)
Ub, Sb, Vb = torch.svd_lowrank(Xm[:, :BURN], q=16, niter=4)
Uk = Ub[:, :12]
R = Xm - Uk @ (Uk.T @ Xm)


def lag_rho(X, lag, scales, q=2):
    Mk = X.shape[0] // m
    if lag == 0:
        return bmf.rho_q(X, m, scales=scales, qs=(q,))[0].float()
    Z = torch.cat([X[:, lag:].reshape(Mk, m, -1), X[:, :-lag].reshape(Mk, m, -1)], 1).reshape(Mk * 2 * m, -1)
    return bmf.rho_q(Z, 2 * m, scales=scales, qs=(q,))[0][:, :m, m:].float()


def comb_signed(C):
    Ct = C.transpose(1, 2)
    return torch.maximum(torch.maximum(C, Ct), (C + Ct) * 0.7071067811865476)


def multilag_abs(X, scales, q=2):
    best = None
    for lag in (0, 1, 2, 3):
        C = lag_rho(X, lag, scales, q)
        S = C.abs() if lag == 0 else torch.maximum(C.abs(), C.transpose(1, 2).abs())
        S = S[:, iu[0], iu[1]]
        Z = scale.robust_z(S.reshape(-1)).reshape(S.shape)
        best = Z if best is None else torch.maximum(best, Z)
    return best


grid = [onset - W // 2, onset, onset + W // 4, onset + W // 2, onset + 3 * W // 4, onset + W, onset + 3 * W // 2, onset + 2 * W]
res = {}
for t in grid:
    Rw, Xw = R[:, t - W:t], A[:, t - W:t]
    sc = {}
    sc['ours'] = scale.market_pair_scores(Rw, m, lag=1, parity=False)['lag'][:, iu[0], iu[1]]
    sc['dcca_clean_E13cfg'] = torch.maximum(rho_dcca_markets(Rw, m), rho_dcca_lag(Rw, m, 1))[:, iu[0], iu[1]]
    sc['dcca_raw_E13cfg'] = torch.maximum(rho_dcca_markets(Xw, m), rho_dcca_lag(Xw, m, 1))[:, iu[0], iu[1]]
    sc['dcca_clean_lag1_s4_16_signed'] = comb_signed(lag_rho(Rw, 1, (4, 8, 16)))[:, iu[0], iu[1]]
    sc['dcca_clean_lag1_s4'] = comb_signed(lag_rho(Rw, 1, (4,)))[:, iu[0], iu[1]]
    sc['dcca_clean_ml_s4_16_abs'] = multilag_abs(Rw, (4, 8, 16))
    sc['dcca_raw_lag1_s4_16_signed'] = comb_signed(lag_rho(Xw, 1, (4, 8, 16)))[:, iu[0], iu[1]]
    sc['rhoq4_clean_lag1_s4_16_signed'] = comb_signed(lag_rho(Rw, 1, (4, 8, 16), q=4))[:, iu[0], iu[1]]
    for k, S in sc.items():
        S = S.float().cpu()
        p = S[orow, ocol].numpy().astype(np.float64); n = S[clean_mk].reshape(-1).numpy().astype(np.float64)
        res.setdefault(k, {})[str(t - onset)] = dict(auc=ews.auc(p, n), d1e2=float((p > np.quantile(n, 0.99)).mean()))
for k, rows in res.items():
    print(f'{k:32s} ' + ' '.join(f"{o:>5s}:{r['d1e2']:.2f}({r['auc']:.2f})" for o, r in rows.items()), flush=True)
print('n_onset', len(orow), 'n_null', int(clean_mk.numel() * iu.shape[1]))
json.dump(res, open(f'/tmp/claude-0/-workspace-projects-TenDeFlAn/1e56e37b-5122-4cce-965e-97f9c396d96b/scratchpad/audit/e13_fair_M{M}_s{SEED}.json', 'w'), indent=1)
