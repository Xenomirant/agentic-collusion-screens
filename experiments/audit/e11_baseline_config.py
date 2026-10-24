"""Small E11 replica (1000 markets x 20 x 1024, w_global 2, eps_copy .75): decompose the 0.982 vs 0.714 gap."""
import sys, os, torch, numpy as np
EXP = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(EXP, 'sims')); sys.path.insert(0, EXP)
import planted_markets as pm
import run_e11_scale as e11
from run_e11_scale import triu, projector
from fathon.tensor import scale, ews
DEV = 'cuda'
M, m, T, CH = 1000, 20, 1024, 250
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
iu = triu(m)
pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
kw = dict(eps_copy=0.75, eps_xor=2.0, c=0.5, w_global=2.0)
zg = pm.global_latents(T, 4, 2, seed=seed, device=DEV)
A0, *_ = pm.chunk(0, CH, m, T, zg, seed=seed, device=DEV, **kw)
V = scale.global_basis(A0, rank=8)
code = {'copy_pair': 1, 'copy_triad': 2, 'xor_triad': 3, 'copy_onset': 4}

def dcca_lag_signed(X, lag, scales):
    Xl, Xe = X[:, lag:], X[:, :-lag]
    Mm = X.shape[0] // m; Tn = Xl.shape[1]
    Yl = (Xl - Xl.mean(1, keepdim=True)).cumsum(1).reshape(Mm, m, Tn)
    Ye = (Xe - Xe.mean(1, keepdim=True)).cumsum(1).reshape(Mm, m, Tn)
    out = 0
    for s in scales:
        nw = Tn // s; P = projector(s)
        El = (Yl[:, :, :nw * s].reshape(Mm, m, nw, s) @ P).reshape(Mm, m, nw * s)
        Ee = (Ye[:, :, :nw * s].reshape(Mm, m, nw, s) @ P).reshape(Mm, m, nw * s)
        R = (El @ Ee.transpose(1, 2)) / (El.norm(dim=2)[:, :, None] * Ee.norm(dim=2)[:, None, :] + 1e-12)
        out = out + torch.maximum(torch.maximum(R, R.transpose(1, 2)), (R + R.transpose(1, 2)) * 0.7071)
    return out / len(scales)

S = {}; labs = []
for M0 in range(0, M, CH):
    A, mk, truth, ttype = pm.chunk(M0, M0 + CH, m, T, zg, seed=seed, device=DEV, **kw)
    lab = torch.zeros(CH, iu.shape[1], dtype=torch.int8)
    for (a, b), tp in ttype.items():
        lab[(a - M0 * m) // m, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = code[tp]
    labs.append(lab.reshape(-1))
    R = scale.remove_global(A, V)
    sc = scale.market_pair_scores(R, m, lag=1, parity=True)
    Rmm = scale.market_mean_removed(R, m)
    d = {'method_lag': sc['lag'], 'parity': sc['parity'],
         'rho_dcca_cleaned_published': e11.rho_dcca_markets(R, m),
         'rho_dcca_clean_mm_lag1_s4_8': dcca_lag_signed(Rmm, 1, (4, 8)),
         'rho_dcca_clean_mm_lag1_s8_128': dcca_lag_signed(Rmm, 1, (8, 16, 32, 64, 128))}
    for k, v in d.items():
        S.setdefault(k, []).append(v[:, iu[0], iu[1]].reshape(-1).cpu())
L = torch.cat(labs).numpy()
S = {k: torch.cat(v).double() for k, v in S.items()}
zp = scale.robust_z(S['parity'])
S['method'] = torch.maximum(scale.robust_z(S['method_lag']), zp)
S['rho_dcca_clean_mm_lag1_s4_8+parity'] = torch.maximum(scale.robust_z(S['rho_dcca_clean_mm_lag1_s4_8']), zp)
S['rho_dcca_cleaned_published+parity'] = torch.maximum(scale.robust_z(S['rho_dcca_cleaned_published']), zp)
for k in ['method', 'method_lag', 'rho_dcca_cleaned_published', 'rho_dcca_cleaned_published+parity',
          'rho_dcca_clean_mm_lag1_s8_128', 'rho_dcca_clean_mm_lag1_s4_8', 'rho_dcca_clean_mm_lag1_s4_8+parity']:
    v = S[k].numpy(); neg = v[L == 0]
    per = ' '.join(f"{t}:{ews.auc(v[L == c], neg):.3f}" for t, c in code.items())
    print(f'seed {seed} {k:36s} AUC all {ews.auc(v[L > 0], neg):.3f} | {per}')
