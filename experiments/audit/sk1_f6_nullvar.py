import sys, torch
sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments/sims'); sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments')
import planted_markets as pm
from run_e11_scale import triu
from fathon.tensor import scale
DEV='cuda'; M, m, T = 250, 20, 1024
iu = triu(m); pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
G = pm.shared_activity(2*T, seed=0)
X, mk, truth, tt = pm.chunk_activity(0, M, m, T, G, band=(1,4), offset=37, seed=0, device=DEV)
lab = torch.zeros(M, iu.shape[1], dtype=torch.bool)
for (a,b) in truth: lab[a//m, pidx[(a%m, b%m)]] = True
C, cells = scale.market_level_coupling_qtt(X, m, extra=4)
P = C[:, :, iu[0], iu[1]].clamp(-0.999,0.999)
for l,c in enumerate(cells):
    z = torch.atanh(P[l])[~lab]*c**0.5
    print(c, 'null var of atanh(r)*sqrt(c):', round(z.var().item(),3), 'mean', round(z.mean().item(),3))
