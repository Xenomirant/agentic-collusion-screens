import sys, torch, numpy as np
sys.path.insert(0, '/workspace/projects/TenDeFlAn/experiments/sims')
import planted_markets as pm
from fathon.tensor import scale, linalg
DEV='cuda'
# 1. lag direction: agent 1 copies agent 0 two steps later (0 leads)
g = torch.Generator(device=DEV).manual_seed(0)
X = torch.randn(4*5, 512, device=DEV, generator=g)
X[1, 2:] = X[0, :-2]
C = scale.market_lag_channels(X, 5, lags=(0, 1, 2, 3), market_mean=False)
print('C[2][0,1,0] (i=1 later, j=0 earlier) =', float(C[2][0, 1, 0]), ' C[2][0,0,1] =', float(C[2][0, 0, 1]))
iu = torch.triu_indices(5, 5, 1, device=DEV)
for mode in ('oneway', 'mutual'):
    z = scale.coupling_scores(C, iu, mode)
    print(mode, 'pair(0,1) z =', float(z[0, 0]), ' median other =', float(z.reshape(-1)[1:].median()))
# 2. fit_common_mode input types
Y = torch.randn(4000, 64, device=DEV, generator=g)
Y += torch.randn(4000, 2, device=DEV, generator=g) @ torch.randn(2, 64, device=DEV, generator=g) * 3
V1, k1 = scale.fit_common_mode(lambda: iter(Y.split(1000)), extra=0)
print('callable chunks -> rank', k1, V1.shape)
V2, k2 = scale.fit_common_mode(list(Y.split(1000)), extra=0)
print('list chunks -> rank', k2)
try:
    V3, k3 = scale.fit_common_mode(Y, extra=0)
    print('2-D tensor -> rank', k3, 'subspace agrees:', float(torch.linalg.svdvals(V3.T @ V1).min()))
except Exception as e:
    print('2-D tensor -> error', type(e).__name__, str(e)[:120])
try:
    V4, k4 = scale.fit_common_mode(Y, extra=0, transform=lambda c: scale.market_mean_removed(c, 20))
    print('2-D tensor + transform -> rank', k4)
except Exception as e:
    print('2-D tensor + transform -> error', type(e).__name__, str(e)[:120])
try:
    V5, k5 = scale.fit_common_mode((c for c in Y.split(1000)), extra=0)
    print('generator -> rank', k5)
except Exception as e:
    print('generator -> error', type(e).__name__, str(e)[:160])
# 3. market reshape when N is not a multiple of m
try:
    out = scale.market_level_coupling_qtt(torch.randn(21*20+7, 1024, device=DEV), 20)
    print('qtt with N % m != 0: ran, shape', out[0].shape)
except Exception as e:
    print('qtt with N % m != 0 -> error', type(e).__name__, str(e)[:100])
try:
    out = scale.market_lag_channels(torch.randn(21*20+7, 256, device=DEV), 20)
    print('lag channels with N % m != 0: ran')
except Exception as e:
    print('lag channels with N % m != 0 -> error', type(e).__name__, str(e)[:100])
# 4. coincidence_scores n-gram precision
rng = np.random.default_rng(0)
N, T = 200, 60
P = np.round(rng.uniform(0, 100, size=(N, T)), 2)          # ~10,000 distinct values
P[5] = P[9]                                                   # one exact copier pair
pairs, sc = scale.coincidence_scores(P, 5, decimals=2, ngram=4)
print('ngram=4 top pairs', pairs[:3])
V = np.round(P, 2); _, codes = np.unique(V, return_inverse=True); base = codes.max() + 2
print('distinct values', base - 1, ' max shingle code ~ base**4 =', float(base)**4, ' > 2**53 =', 2.0**53)
# count spurious coincidences from float rounding of the int64 code
Pr = np.round(rng.uniform(0, 100, size=(N, T)), 2)
pairs2, sc2 = scale.coincidence_scores(Pr, 20, decimals=2, ngram=4, min_count=1)
print('ngram=4 on independent agents: pairs with any coincidence', len(pairs2), pairs2[:3])
pairs3, _ = scale.coincidence_scores(Pr, 20, decimals=2, ngram=3, min_count=1)
print('ngram=3 on independent agents: pairs with any coincidence', len(pairs3))
