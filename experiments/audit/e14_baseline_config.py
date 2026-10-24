"""Small E14 replica (1000 markets x 20 x 1024, w_global 2, 7 types at 8% of markets each, benign in half):
classical baselines given the same cleaning and lags as v2 (lags 0-3, two-sided)."""
import sys, os, torch, numpy as np
EXP = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(EXP, 'sims')); sys.path.insert(0, EXP)
import planted_markets as pm
from run_e11_scale import triu, projector
from run_e14_threats import detectors, TYPES, CODE
from fathon.tensor import scale, ews
DEV = 'cuda'
M, m, T, CH = 1000, 20, 1024, 250
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
iu = triu(m)
pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
zg = pm.global_latents(T, 4, 2, seed=seed, device=DEV)
kw = dict(eps_copy=0.75, eps_xor=2.0, w_global=2.0, c=0.5, mix={t: 0.08 for t in TYPES}, copy_lag=2, eps_match=1.0,
          eps_benign=0.5, benign_frac=0.5, seed=seed, device=DEV)
A0, *_ = pm.chunk(0, CH, m, T, zg, **kw)
V = scale.global_basis(A0, rank=8)

def dcca_dir(X, lag, scales):
    """signed rho_DCCA of x_i(t) with x_j(t - lag), (M, m, m), j leads."""
    Xl, Xe = (X[:, lag:], X[:, :-lag]) if lag else (X, X)
    Mm = X.shape[0] // m; Tn = Xl.shape[1]
    Yl = (Xl - Xl.mean(1, keepdim=True)).cumsum(1).reshape(Mm, m, Tn)
    Ye = (Xe - Xe.mean(1, keepdim=True)).cumsum(1).reshape(Mm, m, Tn)
    out = 0
    for s in scales:
        nw = Tn // s; P = projector(s)
        El = (Yl[:, :, :nw * s].reshape(Mm, m, nw, s) @ P).reshape(Mm, m, nw * s)
        Ee = (Ye[:, :, :nw * s].reshape(Mm, m, nw, s) @ P).reshape(Mm, m, nw * s)
        out = out + (El @ Ee.transpose(1, 2)) / (El.norm(dim=2)[:, :, None] * Ee.norm(dim=2)[:, None, :] + 1e-12)
    return out / len(scales)

S = {}; labs = []
for M0 in range(0, M, CH):
    A, mk, truth, ttype = pm.chunk(M0, M0 + CH, m, T, zg, **kw)
    lab = torch.zeros(CH, iu.shape[1], dtype=torch.int8)
    for (a, b), tp in ttype.items():
        lab[(a - M0 * m) // m, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = CODE[tp]
    labs.append(lab.reshape(-1))
    d = detectors(A, V, m, iu, raw=True)
    R = scale.market_mean_removed(scale.remove_global(A, V), m)
    C = {lag: dcca_dir(R, lag, (4, 8)) for lag in (0, 1, 2, 3)}
    d['rho_dcca_clean_lags0-3_oneway'] = scale.coupling_scores(C, iu, 'oneway').reshape(-1).cpu()
    d['rho_dcca_clean_lags0-3_mutual'] = scale.coupling_scores(C, iu, 'mutual').reshape(-1).cpu()
    for k, v in d.items():
        S.setdefault(k, []).append(v.reshape(-1).cpu())
L = torch.cat(labs).numpy()
S = {k: torch.cat(v).double() for k, v in S.items()}
# parity channel z from v1 is max(zl, zp); recover zp-equivalent by recomputing is costly: use v2_oneway - coupling
for k in ['v1', 'v2_oneway', 'v2_mutual', 'rho_dcca', 'rho_q_absmax', 'rho_dcca_clean_lags0-3_oneway', 'rho_dcca_clean_lags0-3_mutual']:
    v = S[k].numpy(); clean = v[L == 0]; ben = v[L == -1]
    per = ' '.join(f"{t[:6]}:{ews.auc(v[L == CODE[t]], clean):.2f}" for t in TYPES if (L == CODE[t]).any())
    print(f'seed {seed} {k:32s} vs clean {ews.auc(v[L > 0], clean):.3f} vs benign {ews.auc(v[L > 0], ben):.3f} | {per}')
