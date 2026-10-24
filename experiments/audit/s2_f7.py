import numpy as np
from fathon.tensor import scale
rng=np.random.default_rng(0)
X=rng.uniform(0,100,(50,200)).round(2)
for t in (10,100):
    X[3,t]=X[7,t]=55.55
    X[3,t+1:t+4]=np.nan; X[7,t+1:t+4]=np.nan
p,s=scale.coincidence_scores(X,3,ngram=4); print(p,s)
# with full mask
Y=X.copy(); Y[3,10:14]=np.nan;  # nothing; just report
