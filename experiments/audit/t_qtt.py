import sys, torch
sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments/sims')
import planted_markets as pm
from fathon.tensor import scale, linalg
torch.manual_seed(0)
DEV='cuda'; M, m, T = 1000, 20, 1024; CH=250
for zeroG in (False, True):
    G = pm.shared_activity(2*T, seed=0)
    if zeroG: G = G*0
    def sim(M0): return pm.chunk_activity(M0, min(M, M0+CH), m, T, G, band=(1,4), offset=37, seed=0, device=DEV)
    def base_chunks():
        for M0 in range(0, M, CH):
            E = scale.base_log_energy(sim(M0)[0]); yield scale.market_mean_removed(E - E.mean(1, keepdim=True), m)
    def logabs_chunks():
        for M0 in range(0, M, CH):
            La = torch.log(sim(M0)[0].abs()+1e-12); yield scale.market_mean_removed(La - La.mean(1, keepdim=True), m)
    Vb, kb = scale.fit_common_mode(base_chunks, extra=4, device=DEV)
    Vl, kl = scale.fit_common_mode(logabs_chunks, extra=8, device=DEV)
    print(f'zeroG={zeroG}: PA rank base={kb} logabs={kl}')
    if zeroG: break
    # identity checks on the full base matrix
    E = torch.cat(list(base_chunks())).double()
    n0 = E.shape[1]
    S_, Vfull = linalg.tall_svd(E, k=kb+4, method='gram', center=True)
    V = Vb.double()
    print('basis agrees with direct SVD of the full matrix: max |cos| deficit', float(1 - torch.linalg.svdvals(V.T @ Vfull[:, :kb+4]).min()))
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    k = kb+4
    right = E - (E @ V) @ V.T
    left = E - U[:, :k] @ (U[:, :k].T @ E)
    print('right-side (cell) projection == agent-side projection of the same matrix: rel diff', float((right-left).norm()/right.norm()))
    for c in (128, 32, 8):
        A = torch.zeros(n0, c, dtype=E.dtype, device=E.device)
        A[torch.arange(n0), torch.arange(n0)//(n0//c)] = c/n0
        RA = right @ A
        # residual energy left along the coarse level's own top-k_l components
        EA = E @ A
        Ul, Sl, Vl_ = torch.linalg.svd(EA, full_matrices=False)
        kl_, _, _ = linalg.estimate_rank_pa(EA, q=min(48, c-1))
        frac = float(((RA @ Vl_[:int(kl_)].T)**2).sum() / (RA**2).sum())
        null = int(kl_)/c
        print(f'cells={c}: own PA rank of averaged E = {int(kl_)}; energy share of inherited residual along those comps = {frac:.4f} (uniform share {null:.4f})')
    # QTT path with V given vs fitted inside
    X = sim(0)[0]
    C1, cells = scale.market_level_coupling_qtt(X, m, V=Vb.float(), rank=kb, extra=4)
    print('cells', cells, 'levels using extra comps:', [c for c in cells if c >= 8*(kb+4)])
