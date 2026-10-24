import sys, json
import numpy as np, torch
E = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, E + '/sims'); sys.path.insert(0, E)
import planted_markets as pm
import run_e14_threats as e14
from run_e11_scale import triu
from fathon.tensor import scale, ews
T, M, m = 1024, 1000, 20
for groups in (2, 10, 25):
    zg = pm.global_latents(T, 4, groups, seed=0, device='cuda')
    mix = {'copy_pair': 0.1, 'lag_pair': 0.1, 'anti_pair': 0.1}
    A, mk, truth, ttype = pm.chunk(0, M, m, T, zg, groups=groups, eps_copy=0.75, w_global=2.0, c=0.5, mix=mix,
                                   eps_benign=0.5, benign_frac=0.5, seed=0, device='cuda')
    iu = triu(m)
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    lab = torch.zeros(M, iu.shape[1], dtype=torch.int8)
    for (a, b), tp in ttype.items():
        lab[a // m, pidx[(a % m, b % m)]] = -1 if tp == 'benign' else 1
    lab = lab.reshape(-1).numpy()
    Vpa, kpa = scale.fit_common_mode(list(A.split(5000)))
    row = {'true_rank': 4 * groups, 'pa_rank': kpa}
    for name, V in (('fixed rank 8 (+8 extra)', scale.global_basis(A, rank=8)), ('PA rank', Vpa.float()),
                    ('oracle rank', scale.global_basis(A, rank=4 * groups))):
        d = e14.detectors(A, V.to(A.dtype), m, iu, raw=(name.startswith('fixed')))
        row[name] = {k: round(float(ews.auc(v.numpy()[lab == 1], v.numpy()[lab == 0])), 3) for k, v in d.items()}
    print(groups, json.dumps(row), flush=True)
