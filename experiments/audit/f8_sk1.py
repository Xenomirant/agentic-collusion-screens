import torch, functools
from fathon.tensor import scale
torch.manual_seed(0)
T=64; N=4000
F=torch.randn(2,T); L=torch.randn(N,2)
Y=(L@F+0.5*torch.randn(N,T)).double()
n=[0]
orig=scale.linalg if hasattr(scale,'linalg') else None
V,r=scale.fit_common_mode(Y); print('tensor ok rank',r,V.shape)
try:
    V,r=scale.fit_common_mode(Y, transform=functools.partial(scale.market_mean_removed,m=20)); print('tensor+transform ok',r)
except Exception as e: print('tensor+transform', type(e).__name__, e)
V,r=scale.fit_common_mode(list(Y.split(1000)), transform=functools.partial(scale.market_mean_removed,m=20)); print('list+transform ok',r)
try:
    V,r=scale.fit_common_mode((c for c in Y.split(1000))); print('gen ok',r)
except Exception as e: print('gen', type(e).__name__, e)
# chunk count for tensor input
from fathon.tensor import linalg
cnt=[0]
orig_iter=linalg.iter_chunks
def wrap(ch,*a,**k):
    for c in orig_iter(ch,*a,**k):
        cnt[0]+=1; yield c
linalg.iter_chunks=wrap
scale.fit_common_mode(Y, rank=2); print('chunks seen in gram pass (tensor):',cnt[0])
