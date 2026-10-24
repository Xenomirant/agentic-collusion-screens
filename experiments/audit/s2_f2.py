import sys, time, torch
sys.path.insert(0, '../experiments'); sys.path.insert(0, '../experiments/sims')
from fathon.tensor import scale
import planted_markets as pm
M, m, T = 1000, 20, 1024
zg = pm.global_latents(T, 4, 2, seed=0, device='cuda')
A, mk, tr, tt = pm.chunk(0, M, m, T, zg, seed=0, device='cuda', eps_copy=0.75, eps_xor=2.0, c=0.5, w_global=2.0)
chunks = list(A.split(5000))
torch.cuda.synchronize(); t0=time.time()
V, r = scale.fit_common_mode(chunks)
torch.cuda.synchronize(); print('PA raw rank', r, 'time', time.time()-t0)
t0=time.time()
V2, r2 = scale.fit_common_mode(chunks, transform=lambda c: scale.market_mean_removed(c, m))
torch.cuda.synchronize(); print('PA mm rank', r2, 'time', time.time()-t0)
t0=time.time(); Vg = scale.global_basis(A[:50000], rank=8); torch.cuda.synchronize(); print('global_basis cols', Vg.shape, time.time()-t0)
# subspace overlap
Vf = V.float().to(Vg.device)
s = torch.linalg.svdvals(Vg[:, :8].T @ Vf[:, :r]); print('principal cosines (first 8 of gb vs PA):', s.cpu().numpy().round(3))
for rep in range(2):
    torch.cuda.synchronize(); t0=time.time(); V, r = scale.fit_common_mode(chunks); torch.cuda.synchronize(); print('repeat PA raw', r, time.time()-t0)
    torch.cuda.synchronize(); t0=time.time(); V, r = scale.fit_common_mode(chunks, rank=8); torch.cuda.synchronize(); print('repeat fixed-rank gram', r, time.time()-t0)
