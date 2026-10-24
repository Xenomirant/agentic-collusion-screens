import sys, torch
sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments/sims'); sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments')
import planted_markets as pm
from run_e11_scale import triu
from fathon.tensor import scale, ews
DEV='cuda'; M, m, T, CH = 1000, 20, 1024, 250
iu = triu(m); pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
G = pm.shared_activity(2*T, seed=0)
for band in ((1,4),(5,8)):
    def sim(M0): return pm.chunk_activity(M0, min(M, M0+CH), m, T, G, band=band, offset=37, seed=0, device=DEV)
    def base_chunks():
        for M0 in range(0, M, CH):
            E = scale.base_log_energy(sim(M0)[0]); yield scale.market_mean_removed(E - E.mean(1, keepdim=True), m)
    Vb, kb = scale.fit_common_mode(base_chunks, extra=4, device=DEV); Vb = Vb.float()
    sc = {}; labs = []
    Xall = []
    for M0 in range(0, M, CH):
        X, mk, truth, tt = sim(M0)
        lab = torch.zeros(min(M, M0+CH)-M0, iu.shape[1], dtype=torch.int8)
        for (a, b) in truth: lab[(a-M0*m)//m, pidx[((a-M0*m)%m, (b-M0*m)%m)]] = 1
        labs.append(lab.reshape(-1)); Xall.append(X)
        for name, df in (('qtt_rule(dof_factor=8)', 8), ('qtt_extra_everywhere', 0), ('qtt_rank_only_everywhere', 10**9)):
            C, cells = scale.market_level_coupling_qtt(X, m, V=Vb, rank=kb, extra=4, dof_factor=df)
            sc.setdefault(name, []).append(scale.combine_levels(C, iu, cells).reshape(-1).cpu())
    X = torch.cat(Xall)
    C, levels = scale.market_level_coupling(X, m)          # old per-level SVD path, population-wide here
    w = [2 ** l for l in levels]
    sc['per_level_svd(market_level_coupling)'] = [scale.combine_levels(C, iu, w).reshape(-1).cpu()]
    L = torch.cat(labs).numpy()
    print('band', band, 'PA base rank', kb)
    for k, v in sc.items():
        v = torch.cat(v).double().numpy(); print(f'   {k:40s} AUC {ews.auc(v[L==1], v[L==0]):.3f}')
