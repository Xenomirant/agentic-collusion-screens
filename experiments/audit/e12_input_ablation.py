"""Small E12 ablation: is the slow-band lead of the 'QTT pyramid' a property of the input
representation (log energy on 4-step cells) rather than of the pyramid? Same world as E12c."""
import sys, os, torch, numpy as np
EXP = '/workspace/projects/TenDeFlAn/experiments'
sys.path.insert(0, os.path.join(EXP, 'sims')); sys.path.insert(0, EXP)
import planted_markets as pm
from run_e11_scale import triu
import baselines_mf as bmf
from fathon.tensor import scale, ews
DEV = 'cuda'
M, m, T, CH = int(sys.argv[1]) if len(sys.argv) > 1 else 1000, 20, 1024, 250
iu = triu(m)
pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
G = pm.shared_activity(2 * T, seed=SEED)
for band in [(1, 4)]:
    kw = dict(band=band, offset=37, seed=SEED, device=DEV)
    sim = lambda M0: pm.chunk_activity(M0, min(M, M0 + CH), m, T, G, **kw)
    def base_chunks():
        for M0 in range(0, M, CH):
            E = scale.base_log_energy(sim(M0)[0]); yield scale.market_mean_removed(E - E.mean(1, keepdim=True), m)
    def logabs_chunks():
        for M0 in range(0, M, CH):
            La = torch.log(sim(M0)[0].abs() + 1e-12); yield scale.market_mean_removed(La - La.mean(1, keepdim=True), m)
    Vb, kb = scale.fit_common_mode(base_chunks, extra=4, device=DEV); Vb = Vb.float()
    Vl, kl = scale.fit_common_mode(logabs_chunks, extra=8, device=DEV); Vl = Vl.float()
    S = {k: [] for k in ['ours', 'ours_equalw', 'rho_logabs_8_512_rankonly', 'rho_logabs_8_128',
                         'rho_cells_4_128_rankonly', 'rho_cells_4_128_full', 'rho_cells_best_wsum']}
    labs = []
    for M0 in range(0, M, CH):
        X, mk, truth, tt = sim(M0)
        lab = torch.zeros(min(M, M0 + CH) - M0, iu.shape[1], dtype=torch.int8)
        for (a, b) in truth:
            lab[(a - M0 * m) // m, pidx[((a - M0 * m) % m, (b - M0 * m) % m)]] = 1
        labs.append(lab.reshape(-1))
        C, cells = scale.market_level_coupling_qtt(X, m, V=Vb, rank=kb, extra=4)
        S['ours'].append(scale.combine_levels(C, iu, cells).reshape(-1).cpu())
        S['ours_equalw'].append(scale.combine_levels(C, iu, None).reshape(-1).cpu())
        La0 = torch.log(X.abs() + 1e-12); La0 = scale.market_mean_removed(La0 - La0.mean(1, keepdim=True), m)
        for suf, Vx in (('', Vl), ('_rankonly', Vl[:, :kl])):
            La = La0 - (La0 @ Vx) @ Vx.T
            r = bmf.rho_q(La, m, scales=(8, 16, 32, 64, 128, 256, 512), qs=(2,), agg='none')[0]
            if suf == '_rankonly':
                S['rho_logabs_8_512_rankonly'].append(r.mean(0)[:, iu[0], iu[1]].reshape(-1).float().cpu())
            else:
                S['rho_logabs_8_128'].append(r[:5].mean(0)[:, iu[0], iu[1]].reshape(-1).float().cpu())
        # same input as the pyramid: base-cell log energy, market mean, population common mode removed
        E = scale.base_log_energy(X); E = scale.market_mean_removed(E - E.mean(1, keepdim=True), m)
        for suf, Vx in (('_rankonly', Vb[:, :kb]), ('_full', Vb)):
            Ec = E - (E @ Vx) @ Vx.T
            r = bmf.rho_q(Ec, m, scales=(4, 8, 16, 32, 64, 128), qs=(2,), agg='none')[0]   # cells of 4 -> 8..512 steps
            S['rho_cells_4_128' + suf].append(r.mean(0)[:, iu[0], iu[1]].reshape(-1).float().cpu())
            if suf == '_rankonly':
                # Fisher-z, weighted by number of windows, like combine_levels
                w = torch.tensor([Ec.shape[1] // s for s in (4, 8, 16, 32, 64, 128)], dtype=r.dtype, device=r.device)
                Z = torch.atanh(r.clamp(-.999, .999)) * w.sqrt()[:, None, None, None]
                S['rho_cells_best_wsum'].append(Z.sum(0)[:, iu[0], iu[1]].reshape(-1).float().cpu())
    L = torch.cat(labs).numpy()
    out = []
    for k, v in S.items():
        v = torch.cat(v).double().numpy(); null = v[L == 0]
        out.append(f"{k} AUC {ews.auc(v[L == 1], null):.3f} R@1e-3 {(v[L == 1] > np.quantile(null, 1 - 1e-3)).mean():.3f}")
    print(f'band {band} ranks {kb}/{kl} npos {(L==1).sum()} nneg {(L==0).sum()}:\n  ' + '\n  '.join(out), flush=True)
