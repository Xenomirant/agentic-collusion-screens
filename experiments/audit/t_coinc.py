import numpy as np
from fathon.tensor import scale
rng = np.random.default_rng(1)
N, T = 50, 40
P = np.round(rng.uniform(0, 100, size=(N, T)), 2)
# agents 3 and 7: independent prices, but both missing (NaN) in rounds 11-13 and 21-23, equal price in rounds 10 and 20
for a in (3, 7):
    P[a, 11:14] = np.nan; P[a, 21:24] = np.nan
P[7, 10] = P[3, 10]; P[7, 20] = P[3, 20]
pairs, sc = scale.coincidence_scores(P, 3, ngram=4, min_count=2)
print('ngram=4 top pairs:', pairs, sc)
