import sys, time, json
import numpy as np, torch
E = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, E + '/sims'); sys.path.insert(0, E)
import planted_markets as pm
import run_e14_threats as e14
from run_e11_scale import triu
from fathon.tensor import scale, ews
T = 1024
zg = pm.global_latents(T, 4, 2, seed=0, device='cuda')
def scores(M, m, mix, copy_lag=2, seed=0):
    A, mk, truth, ttype = pm.chunk(0, M, m, T, zg, eps_copy=0.75, w_global=2.0, c=0.5, mix=mix, copy_lag=copy_lag,
                                   eps_benign=0.5, benign_frac=0.5, seed=seed, device='cuda')
    iu = triu(m); V = scale.global_basis(A[:50000], rank=8)
    d = e14.detectors(A, V, m, iu, raw=True)
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    lab = torch.zeros(M, iu.shape[1], dtype=torch.int8)
    for (a, b), tp in ttype.items():
        lab[a // m, pidx[(a % m, b % m)]] = -1 if tp == 'benign' else 1
    return {k: v.numpy() for k, v in d.items()}, lab.reshape(-1).numpy(), A
# 1) delayed copying beyond the detector's lag window
for L in (2, 3, 4, 6, 8):
    d, lab, _ = scores(1000, 20, {'lag_pair': 0.2}, copy_lag=L)
    print('lag_pair copy_lag', L, {k: round(float(ews.auc(v[lab == 1], v[lab == 0])), 3) for k, v in d.items()}, flush=True)
# 2) market size: null quantiles of v1 / v2 by m at fixed N=20000, and timing at N fixed
res = {}
for m in (5, 10, 20, 50, 200):
    M = 20000 // m
    d, lab, A = scores(M, m, {}, seed=7)
    q = {k: [float(np.quantile(v[lab == 0], 1 - 1e-3)), float(np.quantile(v[lab == 0], 1 - 1e-4))] for k, v in d.items() if k in ('v1', 'v2_oneway', 'raw_lag1')}
    iu = triu(m); V = scale.global_basis(A[:50000], rank=8)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3): e14.detectors(A, V, m, iu, raw=False)
    torch.cuda.synchronize(); dt = (time.time() - t0) / 3
    print('m', m, 'pairs', int((lab == 0).sum()), 'q(1-1e-3,1-1e-4)', {k: [round(x, 3) for x in v] for k, v in q.items()}, 'detect s', round(dt, 3), flush=True)
