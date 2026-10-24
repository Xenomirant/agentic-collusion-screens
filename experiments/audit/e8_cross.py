import sys, numpy as np
sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments')
from load_e8_local import load_e8_local
from fathon.tensor import scale
data = load_e8_local(rounds=60)
runs = data['seeded'] + data['controls']
XL, owner = [], []
for ridx, r in enumerate(runs):
    for i in range(r['asks'].shape[0]):
        XL.append(r['asks'][i, 1:]); owner.append((ridx, i))
XL = np.asarray(XL, float)
print('shape', XL.shape)
truth=set()
for ridx,r in enumerate(runs):
    if r['true']:
        truth.add(tuple(sorted((owner.index((ridx,r['true'][0])), owner.index((ridx,r['true'][1]))))))
top, info = scale.coincidence_pairs(XL, 40, null='round')
cross=[(owner[i],owner[j],info[(i,j)]) for i,j in top if owner[i][0]!=owner[j][0]]
within=[(owner[i],owner[j],info[(i,j)],(i,j) in truth) for i,j in top if owner[i][0]==owner[j][0]]
print('cross', len(cross)); [print(' ',c) for c in cross]
print('within', len(within)); [print(' ',c) for c in within]
print('hits', len(set(top)&truth))
