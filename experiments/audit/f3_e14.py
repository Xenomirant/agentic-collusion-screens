import sys, types, numpy as np, torch
sys.path.insert(0,'/workspace/projects/TenDeFlAn/experiments')
import run_e14_threats as e
from fathon.tensor import scale
orig = e.detectors
def det(A, V, m, iu, raw=True):
    out = orig(A, V, m, iu, raw=True)
    R = scale.remove_global(A, V)
    R = R.reshape(-1, m, R.shape[-1]); R = (R - R.mean(1, keepdim=True)).reshape(-1, R.shape[-1])
    out['rho_dcca_clean'] = torch.maximum(e.rho_dcca_markets(R, m), e.rho_dcca_lag(R, m, 1))[:, iu[0], iu[1]].reshape(-1).float().cpu()
    rq = e.bmf.rho_q(R, m, qs=(1, 2, 4))
    out['rho_q4_clean'] = rq[2][:, iu[0], iu[1]].reshape(-1).float().cpu()
    out['rho_q_absmax_clean'] = rq.abs().max(0).values[:, iu[0], iu[1]].reshape(-1).float().cpu()
    return out
e.detectors = det
a = types.SimpleNamespace(markets=1000, m=20, T=1024, chunk=500, baseline_chunks=99, prev=0.02, benign_frac=0.5,
    eps_benign=0.5, eps_copy=0.75, eps_xor=2.0, eps_match=1.0, w_global=2.0, c=0.5, seed=0, tag='')
mix = {t: a.prev for t in e.TYPES}
S, L, _ = e.run_world(a, mix, a.benign_frac, 'c')
S0, L0, _ = e.run_world(a, {}, a.benign_frac, 'n')
for k in S:
    s, s0 = S[k], S0[k]; clean0 = s0[L0 == 0]; pos = s[L > 0]; thr = clean0.max(); q4 = np.quantile(clean0, 1-1e-3)
    rec = {t: round(float((s[L == c] > thr).mean()), 3) for t, c in e.CODE.items() if c > 0}
    print(f"{k:20s} n={len(s)} fw true={int((pos>thr).sum())} false={int((s[L==0]>thr).sum()+(s[L==-1]>thr).sum())} @1e-3 true={int((pos>q4).sum())} auc={e.ews.auc(pos, s[L==0]):.3f}", rec)
