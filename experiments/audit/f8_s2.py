import torch, functools
from fathon.tensor import scale
torch.manual_seed(0)
N,T=600,64
Y=torch.randn(N,2)@torch.randn(2,T)*3+torch.randn(N,T)
V,r=scale.fit_common_mode(Y); print('tensor ok', V.shape, r)
V,r=scale.fit_common_mode(list(Y.split(100))); print('list ok', V.shape, r)
try:
    scale.fit_common_mode(Y, transform=functools.partial(scale.market_mean_removed, m=20)); print('tensor+transform ok')
except Exception as e: print('tensor+transform', type(e).__name__, e)
V,r=scale.fit_common_mode(list(Y.split(100)), transform=functools.partial(scale.market_mean_removed, m=20)); print('list+transform ok', r)
try:
    scale.fit_common_mode((c for c in Y.split(100)))
except Exception as e: print('gen', type(e).__name__, e)
