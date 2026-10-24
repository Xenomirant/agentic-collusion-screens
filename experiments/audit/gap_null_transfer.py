# Does the E14 null-world threshold transfer when the coalition-free calibration world is not an exact twin?
import sys, os, json
import numpy as np, torch
E = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, E + '/sims'); sys.path.insert(0, E)
import planted_markets as pm
import run_e14_threats as e14
from run_e11_scale import triu
from fathon.tensor import scale
M, T = 1000, 1024
def world(m=20, zseed=0, cseed=7, c=0.5, w_global=2.0, eps_benign=0.5, benign_frac=0.5, c_het=None):
    zg = pm.global_latents(T, 4, 2, seed=zseed, device='cuda')
    n = M * m
    cc = c
    if c_het is not None:
        g = torch.Generator(device='cuda').manual_seed(123 + cseed)
        cc = c_het[0] + (c_het[1] - c_het[0]) * torch.rand(n, device='cuda', generator=g)
    A, mk, truth, ttype = pm.chunk(0, M, m, T, zg, eps_copy=0.75, eps_xor=2.0, w_global=w_global, c=cc, mix={},
                                   copy_lag=2, eps_match=1.0, eps_benign=eps_benign, benign_frac=benign_frac,
                                   seed=cseed, device='cuda')
    iu = triu(m)
    V = scale.global_basis(A[:50000], rank=8)
    d = e14.detectors(A, V, m, iu, raw=True)
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    lab = torch.zeros(M, iu.shape[1], dtype=torch.int8)
    for (a, b), tp in ttype.items():
        lab[a // m, pidx[(a % m, b % m)]] = -1
    lab = lab.reshape(-1).numpy()
    return {k: v.numpy().astype(np.float64)[lab == 0] for k, v in d.items()}
cal = world()
thr = {k: (np.quantile(v, 1 - 1e-3), np.quantile(v, 1 - 1e-4), v.max()) for k, v in cal.items()}
tests = {
  'twin (new chunk seed, same zg)': dict(cseed=11),
  'new shared-drive draw (zg seed 1)': dict(zseed=1, cseed=11),
  'stickiness c 0.6': dict(cseed=11, c=0.6),
  'heterogeneous c ~ U(0,1)': dict(cseed=11, c_het=(0.0, 1.0)),
  'w_global 2.5': dict(cseed=11, w_global=2.5),
  'no benign followers in null': dict(cseed=11, benign_frac=0.0),
}
out = {'n_clean_cal': int(len(cal['v1']))}
for name, kw in tests.items():
    s = world(**kw)
    out[name] = {k: dict(fpr_at_1e3=float((s[k] > thr[k][0]).mean()), fpr_at_1e4=float((s[k] > thr[k][1]).mean()),
                         flags_above_fwer=int((s[k] > thr[k][2]).sum()), n=int(len(s[k]))) for k in ('v1', 'v2_oneway', 'v2_mutual', 'rho_dcca')}
    print(name, json.dumps({k: [round(v['fpr_at_1e3'], 5), round(v['fpr_at_1e4'], 6), v['flags_above_fwer']] for k, v in out[name].items()}), flush=True)
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + '/gap_null_transfer.json', 'w'), indent=1)
