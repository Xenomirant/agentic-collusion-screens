import sys, torch, numpy as np
sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments/sims'); sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments')
import planted_markets as pm
from run_e11_scale import triu
import baselines_mf as bmf
from fathon.tensor import scale, ews, linalg
DEV='cuda'; M, m, T, CH = 1000, 20, 1024, 250
iu = triu(m); pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
G = pm.shared_activity(2*T, seed=1)
for band in ((1,4),):
    def sim(M0): return pm.chunk_activity(M0, min(M, M0+CH), m, T, G, band=band, offset=37, seed=1, device=DEV)
    def base_chunks():
        for M0 in range(0, M, CH):
            E = scale.base_log_energy(sim(M0)[0]); yield scale.market_mean_removed(E - E.mean(1, keepdim=True), m)
    def logabs_chunks():
        for M0 in range(0, M, CH):
            La = torch.log(sim(M0)[0].abs()+1e-12); yield scale.market_mean_removed(La - La.mean(1, keepdim=True), m)
    res = {}
    labs=[]; sc={}
    fits = {}
    for r in ('pa', 2, 0):
        Vb, kb = scale.fit_common_mode(base_chunks, rank=None if r=='pa' else r, extra=4, device=DEV)
        Vl, kl = scale.fit_common_mode(logabs_chunks, rank=None if r=='pa' else r, extra=8, device=DEV)
        fits[r] = (Vb.float(), kb, Vl.float(), kl)
    for M0 in range(0, M, CH):
        X, mk, truth, tt = sim(M0)
        lab = torch.zeros(min(M, M0+CH)-M0, iu.shape[1], dtype=torch.int8)
        for (a, b) in truth: lab[(a-M0*m)//m, pidx[((a-M0*m)%m, (b-M0*m)%m)]] = 1
        labs.append(lab.reshape(-1))
        La0 = torch.log(X.abs()+1e-12); La0 = scale.market_mean_removed(La0 - La0.mean(1, keepdim=True), m)
        for r, (Vb, kb, Vl, kl) in fits.items():
            C, cells = scale.market_level_coupling_qtt(X, m, V=Vb, rank=kb, extra=4)
            sc.setdefault(f'ours_rank{kb}', []).append(scale.combine_levels(C, iu, cells).reshape(-1).cpu())
            Vx = Vl[:, :kl]
            La = La0 - (La0 @ Vx) @ Vx.T
            rr = bmf.rho_q(La, m, scales=(8,16,32,64,128,256,512), qs=(2,), agg='none')[0]
            sc.setdefault(f'rho2_8_512_rankonly_rank{kl}', []).append(rr.mean(0)[:, iu[0], iu[1]].reshape(-1).float().cpu())
            sc.setdefault(f'rho2_8_128_rankonly_rank{kl}', []).append(rr[:5].mean(0)[:, iu[0], iu[1]].reshape(-1).float().cpu())
    L = torch.cat(labs).numpy()
    print('band', band, 'PA ranks base/logabs', fits['pa'][1], fits['pa'][3])
    for k, v in sc.items():
        v = torch.cat(v).double().numpy()
        print(f'   {k:32s} AUC {ews.auc(v[L==1], v[L==0]):.3f}')
