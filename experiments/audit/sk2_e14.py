"""Audit: E14 world (small), classical baselines with the same cleaning, multi-lag two-sided."""
import os, sys, json
import numpy as np
import torch
REPO = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(REPO, 'sims')); sys.path.insert(0, REPO)
import planted_markets as pm
import baselines_mf as bmf
from run_e14_threats import detectors, TYPES, CODE
from run_e11_scale import triu
from fathon.tensor import scale, ews

DEV = 'cuda'
M = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m, T = 20, 1024
iu = triu(m)
pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
zg = pm.global_latents(T, 4, 2, seed=SEED, device=DEV)


def lag_rho(X, m, lag, scales, q=2):
    """rho_q (Kristoufek signed, q>0) between x_i(t) (i later) and x_j(t - lag); (M, m, m), signed, mean over scales."""
    if lag == 0:
        return bmf.rho_q(X, m, scales=scales, qs=(q,))[0].float()
    Mk = X.shape[0] // m
    Xl = X[:, lag:].reshape(Mk, m, -1)
    Xe = X[:, :-lag].reshape(Mk, m, -1)
    Z = torch.cat([Xl, Xe], 1).reshape(Mk * 2 * m, -1)
    R = bmf.rho_q(Z, 2 * m, scales=scales, qs=(q,))[0]
    return R[:, :m, m:].float()


def classical_multilag(R, scales, q=2, lags=(0, 1, 2, 3)):
    best = None
    for lag in lags:
        C = lag_rho(R, m, lag, scales, q)
        S = C.abs() if lag == 0 else torch.maximum(C.abs(), C.transpose(1, 2).abs())
        S = S[:, iu[0], iu[1]]
        Z = scale.robust_z(S.reshape(-1)).reshape(S.shape)
        best = Z if best is None else torch.maximum(best, Z)
    return best.reshape(-1)


def vol_channel(A, lags=(0, 1, 2), horizons=(1, 2), scales=(4, 8, 16)):
    """Classical volatility-spillover analysis: rho_DCCA between absolute h-step changes |a(t)-a(t-h)|
    of the two agents, horizons (1,2) x (1,2), lags 0..2, both directions; robust z, max."""
    V = {h: (A[:, h:] - A[:, :-h]).abs() for h in horizons}
    L = T - max(horizons)
    V = {h: v[:, v.shape[1] - L:] for h, v in V.items()}          # align on t = max(h) .. T-1
    best = None
    for hi in horizons:
        for hj in horizons:
            for lag in lags:
                Vi = scale.market_mean_removed(V[hi], m)
                Vj = scale.market_mean_removed(V[hj], m)
                Mk = Vi.shape[0] // m
                if lag:
                    Xl, Xe = Vi[:, lag:].reshape(Mk, m, -1), Vj[:, :-lag].reshape(Mk, m, -1)
                else:
                    Xl, Xe = Vi.reshape(Mk, m, -1), Vj.reshape(Mk, m, -1)
                Z = torch.cat([Xl, Xe], 1).reshape(Mk * 2 * m, -1)
                C = bmf.rho_q(Z, 2 * m, scales=scales, qs=(2,))[0][:, :m, m:].float()
                S = torch.maximum(C.abs(), C.transpose(1, 2).abs())[:, iu[0], iu[1]]
                Zs = scale.robust_z(S.reshape(-1)).reshape(S.shape)
                best = Zs if best is None else torch.maximum(best, Zs)
    return best.reshape(-1)


def world(mix):
    kw = dict(eps_copy=0.75, eps_xor=2.0, w_global=2.0, c=0.5, mix=mix, copy_lag=2, eps_match=1.0, eps_benign=0.5,
              benign_frac=0.5, seed=SEED + (0 if mix else 7), device=DEV)
    A, mk, truth, ttype = pm.chunk(0, M, m, T, zg, **kw)
    V = scale.global_basis(A[:50000], rank=8)
    lab = torch.zeros(M, iu.shape[1], dtype=torch.int8)
    for (a, b), tp in ttype.items():
        lab[a // m, pidx[(a % m, b % m)]] = CODE[tp]
    out = detectors(A, V, m, iu, raw=True)
    R = scale.market_mean_removed(scale.remove_global(A, V), m)
    out['dcca_clean_l01_absmean'] = None
    # E14's rho_dcca config, but on the cleaned data
    from run_e11_scale import rho_dcca_markets
    from run_e12_activity import rho_dcca_lag
    out['dcca_clean_l01_absmean'] = torch.maximum(rho_dcca_markets(R, m), rho_dcca_lag(R, m, 1))[:, iu[0], iu[1]].reshape(-1)
    out['dcca_clean_ml_s8_128'] = classical_multilag(R, (8, 16, 32, 64, 128))
    out['dcca_clean_ml_s4_16'] = classical_multilag(R, (4, 8, 16))
    out['rhoq4_clean_ml_s4_16'] = classical_multilag(R, (4, 8, 16), q=4)
    out['vol_dcca'] = vol_channel(A)
    out['dcca_ml_s4_16+vol'] = torch.maximum(out['dcca_clean_ml_s4_16'], out['vol_dcca'])
    S = {k: v.float().cpu().numpy().astype(np.float64) for k, v in out.items()}
    return S, lab.reshape(-1).numpy()


S, L = world({t: 0.02 for t in TYPES})
S0, L0 = world({})
res = {}
for k in S:
    s, s0 = S[k], S0[k]
    clean, benign, pos = s[L == 0], s[L == -1], s[L > 0]
    clean0 = s0[L0 == 0]
    r = dict(auc_clean=ews.auc(pos, clean), auc_benign=ews.auc(pos, benign))
    for tp in TYPES:
        r[tp] = ews.auc(s[L == CODE[tp]], clean)
    thr3 = np.quantile(clean0, 1 - 1e-3)
    fp = (clean > thr3).sum() + (benign > thr3).sum()
    r['fdr1e-3'] = float(fp / max(1, fp + (pos > thr3).sum()))
    thr = clean0.max()
    r['fwer_true'] = int((pos > thr).sum()); r['fwer_false'] = int((clean > thr).sum() + (benign > thr).sum())
    r['fwer_recall_by_type'] = {tp: float((s[L == CODE[tp]] > thr).mean()) for tp in TYPES}
    npair = iu.shape[1]
    ms = s.reshape(-1, npair).max(1); ml = L.reshape(-1, npair)
    coal = (ml > 0).any(1)
    r['market_auc'] = ews.auc(ms[coal], ms[~coal])
    res[k] = r
    print(f"{k:26s} AUCc {r['auc_clean']:.3f} AUCb {r['auc_benign']:.3f} " +
          ' '.join(f"{tp[:5]}:{r[tp]:.2f}" for tp in TYPES) +
          f" FDR {r['fdr1e-3']:.2f} FW {r['fwer_true']}/{r['fwer_false']} mAUC {r['market_auc']:.3f}", flush=True)
print('n_pos', int((L > 0).sum()), 'n_clean', int((L == 0).sum()), 'n_null', int((L0 == 0).sum()))
json.dump(res, open(f'/tmp/claude-0/-workspace-projects-TenDeFlAn/1e56e37b-5122-4cce-965e-97f9c396d96b/scratchpad/audit/sk2_e14_M{M}_s{SEED}.json', 'w'), indent=1)
