import sys, os, itertools
import numpy as np, torch
E='/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, E+'/sims'); sys.path.insert(0, E)
import planted_markets as pm
from run_e11_scale import triu
import baselines_mf as bmf
from fathon.tensor import scale, ews
DEV='cuda'
M, m, T, off, chunk = 1000, 20, 1024, 37, 500
band=(1,4)
iu=triu(m)
pidx={(int(a),int(b)):k for k,(a,b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
def run(gseed, wseed):
    G=pm.shared_activity(2*T, seed=gseed)
    kw=dict(band=band, offset=off, seed=wseed, device=DEV)
    sim=lambda M0: pm.chunk_activity(M0, min(M,M0+chunk), m, T, G, **kw)
    def bc():
        for M0 in range(0,M,chunk):
            Eb=scale.base_log_energy(sim(M0)[0]); yield scale.market_mean_removed(Eb-Eb.mean(1,keepdim=True),m)
    def lc():
        for M0 in range(0,M,chunk):
            La=torch.log(sim(M0)[0].abs()+1e-12); yield scale.market_mean_removed(La-La.mean(1,keepdim=True),m)
    Vb,kb=scale.fit_common_mode(bc,extra=4,device=DEV)
    Vl,kl=scale.fit_common_mode(lc,extra=8,device=DEV)
    Vb,Vl=Vb.float(),Vl.float()
    sc={'ours':[],'r512r':[],'r128r':[],'r128':[]}; labs=[]
    for M0 in range(0,M,chunk):
        X,mk,truth,tt=sim(M0)
        lab=torch.zeros(min(M,M0+chunk)-M0, iu.shape[1], dtype=torch.int8)
        for (a,b) in truth: lab[(a-M0*m)//m, pidx[((a-M0*m)%m,(b-M0*m)%m)]]=1
        labs.append(lab.reshape(-1))
        C,cells=scale.market_level_coupling_qtt(X,m,V=Vb,rank=kb,extra=4)
        sc['ours'].append(scale.combine_levels(C,iu,cells).reshape(-1).cpu())
        La0=torch.log(X.abs()+1e-12); La0=scale.market_mean_removed(La0-La0.mean(1,keepdim=True),m)
        for suf,Vx in (('',Vl),('r',Vl[:,:kl])):
            La=La0-(La0@Vx)@Vx.T
            r=bmf.rho_q(La,m,scales=(8,16,32,64,128,256,512),qs=(2,),agg='none')[0]
            if suf=='r':
                sc['r512r'].append(r.mean(0)[:,iu[0],iu[1]].reshape(-1).float().cpu())
                sc['r128r'].append(r[:5].mean(0)[:,iu[0],iu[1]].reshape(-1).float().cpu())
            else:
                sc['r128'].append(r[:5].mean(0)[:,iu[0],iu[1]].reshape(-1).float().cpu())
    L=torch.cat(labs).numpy()
    a={k:ews.auc(torch.cat(v).double().numpy()[L==1], torch.cat(v).double().numpy()[L==0]) for k,v in sc.items()}
    best=max(a['r512r'],a['r128r'],a['r128'])
    print(f"G{gseed} W{wseed} ranks {kb}/{kl} npos {int(L.sum())} ours {a['ours']:.3f} r512r {a['r512r']:.3f} r128r {a['r128r']:.3f} r128 {a['r128']:.3f} | ours-r512r {a['ours']-a['r512r']:+.3f} ours-best {a['ours']-best:+.3f}", flush=True)
pairs=[tuple(map(int,s.split(':'))) for s in sys.argv[1:]]
for g,w in pairs: run(g,w)
