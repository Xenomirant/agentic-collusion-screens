import sys, os, numpy as np, torch
E='/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0,E); sys.path.insert(0,E+'/sims')
import run_e14_threats as e
from fathon.tensor import scale
orig = e.detectors
def det(A, V, m, iu, raw=True):
    out = orig(A, V, m, iu, raw=True)
    R = scale.remove_global(A, V)
    Rm = scale.market_mean_removed(R, m) if hasattr(scale,'market_mean_removed') else R
    out['rho_dcca_clean'] = torch.maximum(e.rho_dcca_markets(R, m), e.rho_dcca_lag(R, m, 1))[:, iu[0], iu[1]].reshape(-1).float().cpu()
    out['rho_dcca_clean_mm'] = torch.maximum(e.rho_dcca_markets(Rm, m), e.rho_dcca_lag(Rm, m, 1))[:, iu[0], iu[1]].reshape(-1).float().cpu()
    rq = e.bmf.rho_q(Rm, m, qs=(1,2,4))
    out['rho_q_absmax_clean'] = rq.abs().max(0).values[:, iu[0], iu[1]].reshape(-1).float().cpu()
    return out
e.detectors = det
class A: pass
a=A(); a.markets=int(sys.argv[1]); a.m=20; a.T=1024; a.chunk=500; a.baseline_chunks=99
a.prev=float(sys.argv[2]); a.benign_frac=0.5; a.eps_benign=0.5; a.eps_copy=0.75; a.eps_xor=2.0; a.eps_match=1.0; a.w_global=2.0; a.c=0.5; a.seed=int(sys.argv[3])
mix={t:a.prev for t in e.TYPES}
S,L,_=e.run_world(a,mix,a.benign_frac,'c'); S0,L0,_=e.run_world(a,{},a.benign_frac,'n')
for k in S:
    s,s0=S[k],S0[k]; clean0=s0[L0==0]; thr=clean0.max(); thr3=np.quantile(clean0,1-1e-3)
    pos=s[L>0]; lab=L[L>0]
    rec={t:round(float((pos[lab==c]>thr).mean()),3) for t,c in e.CODE.items() if c>0}
    fp=int((s[L<=0]>thr).sum())
    print(f"{k:20s} npos={pos.size} fwer TP={int((pos>thr).sum())} FP={fp} rec@1e-3={float((pos>thr3).mean()):.3f} {rec}")
