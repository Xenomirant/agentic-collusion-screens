import numpy as np
from fathon.tensor import scale
rng=np.random.default_rng(0)
N,T=50,64
X=rng.integers(0,1000,(N,T))/100.0
for t0 in (10,30):
    X[3,t0]=X[7,t0]=123.45
    X[3,t0+1:t0+4]=np.nan; X[7,t0+1:t0+4]=np.nan
p,s=scale.coincidence_scores(X,5,ngram=4)
print(p,s)
