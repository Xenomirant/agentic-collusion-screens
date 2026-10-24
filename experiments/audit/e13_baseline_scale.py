"""Small E13 replica (T=1024, onset 512, W=256, 1000 markets): does rho_DCCA restricted to the lag
and short scales a lag-1 copy needs close the gap to 'ours' (windowed lag-1 residual correlation)?"""
import sys, os, torch, numpy as np
EXP = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(EXP, 'sims')); sys.path.insert(0, EXP)
import planted_markets as pm
import run_e11_scale as e11
from run_e11_scale import triu, projector
from fathon.tensor import scale, ews
DEV = 'cuda'
M, m, T, W, CH = 1000, 20, 1024, 256, 250
onset = T // 2
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
iu = triu(m)
pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
zg = pm.global_latents(T, 4, 2, seed=seed, device=DEV)

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
        out = out + torch.maximum(R, R.transpose(1, 2))
    return out / len(scales)

ts = [onset + 128, onset + 256]
names = ['ours', 'rho_dcca_8_128_lag01_raw', 'rho_dcca_8_128_lag01_clean', 'rho_dcca_lag1_s4_clean', 'rho_dcca_lag1_s4_8_clean', 'rho_dcca_lag1_s4_raw']
pos = {(t, k): [] for t in ts for k in names}; neg = {(t, k): [] for t in ts for k in names}
from run_e12_activity import rho_dcca_lag
for M0 in range(0, M, CH):
    M1 = M0 + CH
    A, mk, truth, ttype = pm.chunk(M0, M1, m, T, zg, seed=seed, device=DEV, eps_copy=0.75, onset=onset)
    mtype = torch.arange(M0, M1) % 10
    clean_mk = torch.nonzero(mtype >= 4).squeeze(1)
    rows, cols = [], []
    for (a, b), tp in ttype.items():
        if tp == 'copy_onset':
            rows.append((a - M0 * m) // m); cols.append(pidx[((a - M0 * m) % m, (b - M0 * m) % m)])
    rows, cols = torch.tensor(rows), torch.tensor(cols)
    Xc = A - A[:, :W].mean(1, keepdim=True)
    Rm = Xc.reshape(-1, m, T); Rm = Rm - Rm.mean(1, keepdim=True); Xm = Rm.reshape(A.shape[0], T)
    Ub, Sb, Vb = torch.svd_lowrank(Xm[:, :W], q=16, niter=4); Uk = Ub[:, :12]
    R = Xm - Uk @ (Uk.T @ Xm)
    for t in ts:
        Rw, Xw = R[:, t - W:t], A[:, t - W:t]
        S = {'ours': scale.market_pair_scores(Rw, m, lag=1, parity=False)['lag'],
             'rho_dcca_8_128_lag01_raw': torch.maximum(e11.rho_dcca_markets(Xw, m), rho_dcca_lag(Xw, m, 1)),
             'rho_dcca_8_128_lag01_clean': torch.maximum(e11.rho_dcca_markets(Rw, m), rho_dcca_lag(Rw, m, 1)),
             'rho_dcca_lag1_s4_clean': dcca_lag_signed(Rw, 1, (4,)),
             'rho_dcca_lag1_s4_8_clean': dcca_lag_signed(Rw, 1, (4, 8)),
             'rho_dcca_lag1_s4_raw': dcca_lag_signed(Xw, 1, (4,))}
        for k, v in S.items():
            v = v[:, iu[0], iu[1]].cpu()
            pos[(t, k)].append(v[rows, cols]); neg[(t, k)].append(v[clean_mk].reshape(-1))
for t in ts:
    print(f'seed {seed} t = onset + {t - onset}')
    for k in names:
        p = torch.cat(pos[(t, k)]).double().numpy(); n = torch.cat(neg[(t, k)]).double().numpy()
        print(f'  {k:28s} AUC {ews.auc(p, n):.3f} D@1e-2 {(p > np.quantile(n, 0.99)).mean():.3f}  (npos {p.size}, nneg {n.size})')
