"""Audit: E12c timing world (small), rho_DCCA with scales matched to the band, and on the same base-level log energy."""
import os, sys, json
import numpy as np
import torch
AUD = '/tmp/claude-0/-workspace-projects-TenDeFlAn/1e56e37b-5122-4cce-965e-97f9c396d96b/scratchpad/audit'
REPO = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(REPO, 'sims')); sys.path.insert(0, REPO)
import planted_markets as pm
import baselines_mf as bmf
from run_e11_scale import triu
from fathon.tensor import scale, ews

DEV = 'cuda'
M = int(sys.argv[1]); SEED = int(sys.argv[2]); bands = sys.argv[3] if len(sys.argv) > 3 else '1-4,3-6,5-8'
m, T, off = 20, 1024, 37
iu = triu(m)
pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
G = pm.shared_activity(2 * T, seed=SEED)
CH = 250
allres = {}
for bs in bands.split(','):
    band = tuple(int(x) for x in bs.split('-'))
    kw = dict(band=band, offset=off, seed=SEED, device=DEV)
    sims = [pm.chunk_activity(M0, min(M, M0 + CH), m, T, G, **kw) for M0 in range(0, M, CH)]

    def base_chunks():
        for X, *_ in sims:
            E = scale.base_log_energy(X)
            yield scale.market_mean_removed(E - E.mean(1, keepdim=True), m)

    def logabs_chunks():
        for X, *_ in sims:
            La = torch.log(X.abs() + 1e-12)
            yield scale.market_mean_removed(La - La.mean(1, keepdim=True), m)

    Vb, kb = scale.fit_common_mode(base_chunks, extra=4, device=DEV)
    Vl, kl = scale.fit_common_mode(logabs_chunks, extra=8, device=DEV)
    Vb, Vl = Vb.float(), Vl.float()
    sc, labs = {}, []
    for ci, (X, mk, truth, tt) in enumerate(sims):
        M0 = ci * CH
        lab = torch.zeros(X.shape[0] // m, iu.shape[1], dtype=torch.int8)
        for (a, b) in truth:
            lab[(a - M0 * m) // m, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = 1
        labs.append(lab.reshape(-1))
        add = lambda k, v: sc.setdefault(k, []).append(v.reshape(-1).float().cpu())
        C, cells = scale.market_level_coupling_qtt(X, m, V=Vb, rank=kb, extra=4)
        add('ours', scale.combine_levels(C, iu, cells))
        La0 = torch.log(X.abs() + 1e-12)
        La0 = scale.market_mean_removed(La0 - La0.mean(1, keepdim=True), m)
        for suffix, Vx in (('', Vl), ('_rankonly', Vl[:, :kl])):
            La = La0 - (La0 @ Vx) @ Vx.T
            r = bmf.rho_q(La, m, scales=(8, 16, 32, 64, 128, 256, 512), qs=(2,), agg='none')[0]
            add('rho2_8_128' + suffix, r[:5].mean(0)[:, iu[0], iu[1]])
            add('rho2_8_512' + suffix, r.mean(0)[:, iu[0], iu[1]])
            add('rho2_64_512' + suffix, r[3:].mean(0)[:, iu[0], iu[1]])
        # rho_DCCA on the SAME cleaned base-level log energy (cells of 4 rounds) that the pyramid uses
        E = scale.base_log_energy(X)
        E = scale.market_mean_removed(E - E.mean(1, keepdim=True), m)
        for suffix, Vx in (('', Vb), ('_rankonly', Vb[:, :kb])):
            Rb = E - (E @ Vx) @ Vx.T
            rb = bmf.rho_q(Rb, m, scales=(4, 8, 16, 32, 64, 128), qs=(2,), agg='none')[0]   # 16..512 rounds
            add('rho2_baseE_all' + suffix, rb.mean(0)[:, iu[0], iu[1]])
            add('rho2_baseE_s16_128c' + suffix, rb[2:].mean(0)[:, iu[0], iu[1]])                # 64..512 rounds
            add('rho2_baseE_s4_16c' + suffix, rb[:3].mean(0)[:, iu[0], iu[1]])                  # 16..64 rounds
            # plain correlation of block-averaged cleaned log energy at the band's block size (classical "realized vol" corr)
    L = torch.cat(labs).numpy()
    S = {k: torch.cat(v).double().numpy() for k, v in sc.items()}
    res = dict(rank_base=int(kb), rank_logabs=int(kl))
    for k, v in S.items():
        null = v[L == 0]
        res[k] = dict(auc=ews.auc(v[L == 1], null), r1e3=float((v[L == 1] > np.quantile(null, 1 - 1e-3)).mean()),
                      r1e2=float((v[L == 1] > np.quantile(null, 1 - 1e-2)).mean()))
    allres[bs] = res
    print(f'band {bs} ranks {kb}/{kl} npos {int(L.sum())}')
    for k in S:
        print(f"   {k:28s} AUC {res[k]['auc']:.3f} R@1e-2 {res[k]['r1e2']:.3f} R@1e-3 {res[k]['r1e3']:.3f}", flush=True)
    del sims
json.dump(allres, open(f'{AUD}/sk1_e12_M{M}_s{SEED}.json', 'w'), indent=1)
