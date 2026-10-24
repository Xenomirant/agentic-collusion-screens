import sys,numpy as np
sys.path.insert(0,'/workspace/projects/TenDeFlAn/experiments')
from load_e8_local import load_e8_local
from fathon.tensor import scale
d=load_e8_local(rounds=60)
runs=d['seeded']+d['controls']
XL=[];owner=[]
for ri,r in enumerate(runs):
    for i in range(r['asks'].shape[0]): XL.append(r['asks'][i,1:]); owner.append((ri,i))
XL=np.asarray(XL,float); print(XL.shape)
for nl in ('round','shift'):
    top,info=scale.coincidence_pairs(XL,40,null=nl)
    cr=[(owner[i],owner[j],info[(i,j)]) for i,j in top if owner[i][0]!=owner[j][0]]
    wi=[(owner[i],owner[j],info[(i,j)]) for i,j in top if owner[i][0]==owner[j][0]]
    print(nl,len(cr),len(wi)); print(' top5',[(owner[i],owner[j],info[(i,j)]) for i,j in top[:5]])
    print(' cross counts',sorted([c[2][0] for c in cr],reverse=True)); print(' cross null',sorted([round(c[2][1],2) for c in cr],reverse=True)[:10])
    print(' within counts',sorted([c[2][0] for c in wi],reverse=True))
    print(' within truth?',[ (c[0][0], runs[c[0][0]]['true'], (c[0][1],c[1][1])) for c in wi][:17])
