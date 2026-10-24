import sys, numpy as np, collections
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
print(runs[0].get('models'))
top, info = scale.coincidence_pairs(XL, 40, null='round')
def m(o):
    ms = runs[o[0]].get('models')
    if ms is None: return None
    return ms[o[1]] if isinstance(ms, list) else ms
c = collections.Counter()
for i,j in top:
    a,b=owner[i],owner[j]
    if a[0]!=b[0]: c[(m(a)==m(b), str(m(a)), str(m(b)))]+=1
print(c)
