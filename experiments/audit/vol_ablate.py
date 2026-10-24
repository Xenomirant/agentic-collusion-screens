import os, sys
import numpy as np, torch
REPO = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(REPO, 'sims')); sys.path.insert(0, REPO)
import planted_markets as pm, baselines_mf as bmf
from run_e11_scale import triu
from fathon.tensor import scale, ews
DEV='cuda'; M=1000; m=20; T=1024
iu=triu(m); pidx={(int(a),int(b)):k for k,(a,b) in enumerate(zip(iu[0].tolist(),iu[1].tolist()))}
zg=pm.global_latents(T,4,2,seed=0,device=DEV)
A,_,_,tt=pm.chunk(0,M,m,T,zg,seed=0,device=DEV,eps_copy=0.75,eps_xor=2.0,w_global=2.0)
lab=torch.zeros(M,iu.shape[1],dtype=torch.int8)
for (a,b),tp in tt.items(): lab[a//m,pidx[(a%m,b%m)]]={'copy_pair':1,'copy_triad':2,'xor_triad':3,'copy_onset':4}[tp]
L=lab.reshape(-1).numpy()
def rho(Vi,Vj,lag,scales=(4,8,16)):
    Xl=(Vi[:,lag:] if lag else Vi).reshape(M,m,-1); Xe=(Vj[:,:-lag] if lag else Vj).reshape(M,m,-1)
    C=bmf.rho_q(torch.cat([Xl,Xe],1).reshape(M*2*m,-1),2*m,scales=scales,qs=(2,))[0][:,:m,m:].float()
    return torch.maximum(C.abs(),C.transpose(1,2).abs())[:,iu[0],iu[1]].reshape(-1).cpu().numpy()
V={h:scale.market_mean_removed((A[:,h:]-A[:,:-h]).abs()[:,2-h:],m) for h in (1,2)}
for hi,hj,lag in [(1,1,0),(1,1,1),(1,1,2),(2,2,0),(2,2,1),(2,1,1),(1,2,1),(2,1,0),(2,1,2)]:
    s=rho(V[hi],V[hj],lag)
    print(f'|d{hi}a_i(t)| vs |d{hj}a_j(t-{lag})|: XOR AUC {ews.auc(s[L==3],s[L==0]):.3f} copy_pair {ews.auc(s[L==1],s[L==0]):.3f}')
