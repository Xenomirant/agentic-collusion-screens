import sys, torch, numpy as np
sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments/sims'); sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments')
import planted_markets as pm
from run_e11_scale import triu
from fathon.tensor import scale, ews
DEV='cuda'; M, m, T, CH = 1000, 20, 1024, 250
TYPES = ['copy_pair', 'copy_triad', 'xor_triad', 'copy_onset', 'anti_pair', 'lag_pair', 'match_pair']
iu = triu(m); pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
zg = pm.global_latents(T, 4, 2, seed=0, device=DEV)
kw = dict(eps_copy=0.75, eps_xor=2.0, w_global=2.0, c=0.5, mix={t: 0.02 for t in TYPES}, copy_lag=2, eps_match=1.0,
          eps_benign=0.5, benign_frac=0.5, seed=0, device=DEV)
def sim(M0): return pm.chunk(M0, min(M, M0+CH), m, T, zg, **kw)
Vgb = scale.global_basis(sim(0)[0][:50000], rank=8)
Vpa, k = scale.fit_common_mode(lambda: (sim(M0)[0] for M0 in range(0, M, CH)), extra=8, device=DEV)
Vpa_mm, kmm = scale.fit_common_mode(lambda: (sim(M0)[0] for M0 in range(0, M, CH)), extra=8, device=DEV,
                                    transform=lambda c: scale.market_mean_removed(c - c.mean(1, keepdim=True), m))
print('PA rank raw', k, ' PA rank after market-mean removal', kmm, ' E14 uses fixed rank 8 + 8 extra from first chunk')
print('subspace overlap global_basis vs fit_common_mode (min cos):', float(torch.linalg.svdvals(Vgb.double().T @ Vpa.double()).min()))
sc = {}; labs = []
for M0 in range(0, M, CH):
    A, mk, truth, tt = sim(M0)
    lab = torch.zeros(min(M, M0+CH)-M0, iu.shape[1], dtype=torch.int8)
    for (a, b), tp in tt.items():
        lab[(a-M0*m)//m, pidx[((a-M0*m)%m, (b-M0*m)%m)]] = -1 if tp == 'benign' else 1
    labs.append(lab.reshape(-1))
    for name, V in (('global_basis_r8', Vgb), ('fit_common_mode_PA', Vpa.float()), ('fit_common_mode_PA_mm', Vpa_mm.float())):
        R = scale.remove_global(A, V)
        s1 = scale.market_pair_scores(R, m, lag=1, parity=True)
        zl = scale.robust_z(s1['lag'][:, iu[0], iu[1]].reshape(-1)); zp = scale.robust_z(s1['parity'][:, iu[0], iu[1]].reshape(-1))
        C = scale.market_lag_channels(R, m, lags=(0, 1, 2, 3))
        sc.setdefault(name+' v1', []).append(torch.maximum(zl, zp).cpu())
        sc.setdefault(name+' v2_oneway', []).append(torch.maximum(scale.coupling_scores(C, iu, 'oneway').reshape(-1), zp).cpu())
L = torch.cat(labs).numpy()
for k_, v in sc.items():
    v = torch.cat(v).double().numpy()
    print(f'{k_:34s} AUC vs clean {ews.auc(v[L==1], v[L==0]):.3f}  vs benign {ews.auc(v[L==1], v[L==-1]):.3f}')
