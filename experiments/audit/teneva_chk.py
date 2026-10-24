import numpy as np, teneva
Y = teneva.rand([4]*6, 6, seed=0)
X = teneva.full(Y); nx = np.linalg.norm(X); print('norm', nx)
for lab, A, e in (('abs e=0.3|X|/sqrt5', X, 0.3*nx/np.sqrt(5)), ('normalised e=0.3', X/nx, 0.3), ('unnormalised e=0.3', X, 0.3)):
    T = teneva.svd(A, e=e)
    err = np.linalg.norm(teneva.full(T)-A)/np.linalg.norm(A)
    print(lab, 'ranks', teneva.props(T) if False else [c.shape[2] for c in T][:-1], 'relerr', round(err,3))
