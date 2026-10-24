"""Audit: E11 world (small, strong drive w=2, eps_copy .75), classical baselines with the same cleaning."""
import os, sys, json
import numpy as np
import torch
AUD = '/tmp/claude-0/-workspace-projects-TenDeFlAn/1e56e37b-5122-4cce-965e-97f9c396d96b/scratchpad/audit'
REPO = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(REPO, 'sims')); sys.path.insert(0, REPO); sys.path.insert(0, AUD)
import planted_markets as pm
import baselines_mf as bmf
from run_e11_scale import rho_dcca_markets, triu
from fathon.tensor import scale, ews

DEV = 'cuda'
M = int(sys.argv[1]); SEED = int(sys.argv[2])
m, T = 20, 1024
iu = triu(m)
pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
zg = pm.global_latents(T, 4, 2, seed=SEED, device=DEV)
A, mk, truth, ttype = pm.chunk(0, M, m, T, zg, seed=SEED, device=DEV, eps_copy=0.75, eps_xor=2.0, c=0.5, w_global=2.0)
code = {'copy_pair': 1, 'copy_triad': 2, 'xor_triad': 3, 'copy_onset': 4}
lab = torch.zeros(M, iu.shape[1], dtype=torch.int8)
for (a, b), tp in ttype.items():
    lab[a // m, pidx[(a % m, b % m)]] = code[tp]
L = lab.reshape(-1).numpy()
V = scale.global_basis(A[:50000], rank=8)
R0 = scale.remove_global(A, V)
sc = scale.market_pair_scores(R0, m, lag=1, parity=True)
zl = scale.robust_z(sc['lag'][:, iu[0], iu[1]].reshape(-1)); zp = scale.robust_z(sc['parity'][:, iu[0], iu[1]].reshape(-1))
R = scale.market_mean_removed(R0, m)


def lag_rho(X, lag, scales, q=2):
    Mk = X.shape[0] // m
    if lag == 0:
        return bmf.rho_q(X, m, scales=scales, qs=(q,))[0].float()
    Z = torch.cat([X[:, lag:].reshape(Mk, m, -1), X[:, :-lag].reshape(Mk, m, -1)], 1).reshape(Mk * 2 * m, -1)
    return bmf.rho_q(Z, 2 * m, scales=scales, qs=(q,))[0][:, :m, m:].float()


def comb_signed(C):
    Ct = C.transpose(1, 2)
    return torch.maximum(torch.maximum(C, Ct), (C + Ct) * 0.7071067811865476)


def vol_channel(A, lags=(0, 1, 2), horizons=(1, 2), scales=(4, 8, 16)):
    Vv = {h: (A[:, h:] - A[:, :-h]).abs() for h in horizons}
    Ln = T - max(horizons)
    Vv = {h: scale.market_mean_removed(v[:, v.shape[1] - Ln:], m) for h, v in Vv.items()}
    best = None
    for hi in horizons:
        for hj in horizons:
            for lag in lags:
                Mk = M
                Xl = (Vv[hi][:, lag:] if lag else Vv[hi]).reshape(Mk, m, -1)
                Xe = (Vv[hj][:, :-lag] if lag else Vv[hj]).reshape(Mk, m, -1)
                C = bmf.rho_q(torch.cat([Xl, Xe], 1).reshape(Mk * 2 * m, -1), 2 * m, scales=scales, qs=(2,))[0][:, :m, m:].float()
                S = torch.maximum(C.abs(), C.transpose(1, 2).abs())[:, iu[0], iu[1]]
                Zs = scale.robust_z(S.reshape(-1)).reshape(S.shape)
                best = Zs if best is None else torch.maximum(best, Zs)
    return best.reshape(-1)


out = {}
out['method'] = torch.maximum(zl, zp)
out['method_lag_only'] = zl
out['dcca_clean_E11cfg'] = rho_dcca_markets(R0, m)[:, iu[0], iu[1]].reshape(-1)
d1 = comb_signed(lag_rho(R, 1, (4, 8, 16)))[:, iu[0], iu[1]].reshape(-1)
out['dcca_clean_lag1_s4_16'] = d1
vol = vol_channel(A)
out['vol_dcca'] = vol
out['dcca_lag1_s4_16+vol'] = torch.maximum(scale.robust_z(d1), vol)
res = {}
for k, v in out.items():
    s = v.float().cpu().numpy().astype(np.float64)
    pos, null = s[L > 0], s[L == 0]
    r = dict(auc_all=ews.auc(pos, null))
    for tp, c in code.items():
        r[tp] = ews.auc(s[L == c], null)
    K = int((L > 0).sum()); top = np.argpartition(-s, K)[:K]
    r['prec_at_K'] = float((L[top] > 0).mean())
    r['recall_fpr1e-3'] = float((pos > np.quantile(null, 1 - 1e-3)).mean())
    res[k] = r
    print(f"{k:24s} AUC {r['auc_all']:.3f} " + ' '.join(f"{tp}:{r[tp]:.3f}" for tp in code) + f" P@K {r['prec_at_K']:.3f} R@1e-3 {r['recall_fpr1e-3']:.3f}", flush=True)
json.dump(res, open(f'{AUD}/e11_fair_M{M}_s{SEED}.json', 'w'), indent=1)
