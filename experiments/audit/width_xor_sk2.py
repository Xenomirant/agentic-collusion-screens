import os, sys
import numpy as np, torch
REPO = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(REPO, 'sims')); sys.path.insert(0, REPO)
import planted_markets as pm, baselines_mf as bmf
from run_e11_scale import triu
from run_e14_threats import TYPES, CODE
from fathon.tensor import scale, ews
DEV='cuda'; M=1000; m=20; T=1024
iu=triu(m); pidx={(int(a),int(b)):k for k,(a,b) in enumerate(zip(iu[0].tolist(),iu[1].tolist()))}
for seed in (0,):
    zg=pm.global_latents(T,4,2,seed=seed,device=DEV)
    for wname, kw, codes in [('E11', dict(eps_copy=0.75, eps_xor=2.0, w_global=2.0), {'copy_pair':1,'copy_triad':2,'xor_triad':3,'copy_onset':4}),
                             ('E14', dict(eps_copy=0.75, eps_xor=2.0, w_global=2.0, mix={t:0.02 for t in TYPES}, copy_lag=2, eps_match=1.0, eps_benign=0.5, benign_frac=0.5), dict(CODE))]:
        A,_,_,tt=pm.chunk(0,M,m,T,zg,seed=seed,device=DEV,**kw)
        lab=torch.zeros(M,iu.shape[1],dtype=torch.int8)
        for (a,b),tp in tt.items(): lab[a//m,pidx[(a%m,b%m)]]=codes[tp]
        L=lab.reshape(-1).numpy()
        V=scale.global_basis(A[:50000],rank=8)
        R=scale.market_mean_removed(scale.remove_global(A,V),m)
        w=bmf.hxy_width(R,m,mode='fathon')[:,iu[0],iu[1]].reshape(-1).double().cpu().numpy()
        x=codes['xor_triad']
        print(seed, wname, 'MFDCCA width (fathon, cleaned): XOR vs clean AUC %.3f' % ews.auc(w[L==x], w[L==0]),
              ' all-pos AUC %.3f' % ews.auc(w[L>0], w[L==0]), ' xor vs benign %s' % (('%.3f' % ews.auc(w[L==x], w[L==-1])) if (L==-1).any() else '-'), flush=True)
