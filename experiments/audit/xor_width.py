import sys, torch, numpy as np
E='/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, E+'/sims'); sys.path.insert(0, E)
import planted_markets as pm, run_e12_activity as e12, baselines_mf as bmf
from fathon.tensor import ews
M,m,T=300,20,1024
for wg in (0.8, 2.0):
    X,_,truth,ttype = pm.chunk(0,M,m,T,pm.global_latents(T,4,2,seed=1,device='cuda'),seed=1,device='cuda',w_global=wg)
    iu=torch.triu_indices(m,m,1,device='cuda'); pidx={(int(a),int(b)):k for k,(a,b) in enumerate(zip(iu[0].tolist(),iu[1].tolist()))}
    codes={'copy_pair':1,'copy_triad':2,'xor_triad':3,'copy_onset':4}
    lab=torch.zeros(M,iu.shape[1],dtype=torch.int8)
    for (a,b),tp in ttype.items(): lab[a//m,pidx[(a%m,b%m)]]=codes[tp]
    lab=lab.reshape(-1).numpy()
    Z=e12.clean(X.float(),m)
    wf=bmf.hxy_width(Z,m,bmf.SCALES,[-3.,-2.,-1.,1.,2.,3.],mode='fathon')
    s=wf[:,iu[0],iu[1]].reshape(-1).double().cpu().numpy()
    print('w_global',wg,'clean hxy_width_fathon AUC xor', round(ews.auc(s[lab==3],s[lab==0]),3), 'copy_pair', round(ews.auc(s[lab==1],s[lab==0]),3), 'n_xor', int((lab==3).sum()))
