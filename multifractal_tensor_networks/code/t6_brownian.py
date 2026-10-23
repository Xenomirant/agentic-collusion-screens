"""Test 6: the Brownian layer.
(a) QTT ranks (interleaved s,t bits) of Brownian covariance min(s,t), fBm covariance, |s-t|^{2H}.
(b) singular-value decay of mid-unfolding of a BM / fBm sample path (pathwise incompressibility).
(c) MRW increments dX(w)=sqrt(mu(w)) xi_w: exact 2- and 4-point moment tensors -> interleaved TT ranks <= B_q.
"""
import numpy as np, itertools
rng = np.random.default_rng(3)
def interleave(C, n):  # C[s,t] on 2^n grid, s,t bits MSB first -> tensor with sites (s_k,t_k)
    T = C.reshape((2,) * n + (2,) * n)
    perm = [x for k in range(n) for x in (k, n + k)]
    return T.transpose(perm).reshape((4,) * n)
def ranks(T, tol):
    n = T.ndim; d = T.shape[0]; out = []
    for k in range(1, n):
        s = np.linalg.svd(T.reshape(d ** k, -1), compute_uv=False); out.append(int(np.sum(s > tol * s[0])))
    return out
n = 10; t = (np.arange(2 ** n) + 1) / 2 ** n
S, Tt = np.meshgrid(t, t, indexing='ij')
print("(a) min(s,t)            ranks:", ranks(interleave(np.minimum(S, Tt), n), 1e-12))
for H in [0.3, 0.7]:
    C = 0.5 * (S ** (2 * H) + Tt ** (2 * H) - np.abs(S - Tt) ** (2 * H))
    print(f"    fBm cov H={H}  eps=1e-6 ranks:", ranks(interleave(C, n), 1e-6), " eps=1e-10:", ranks(interleave(C, n), 1e-10))

n = 20
def fbm(H, N):  # Davies-Harte circulant embedding
    k = np.arange(N + 1); g = 0.5 * ((k + 1.0) ** (2 * H) - 2 * k ** (2.0 * H) + np.abs(k - 1.0) ** (2 * H))
    c = np.concatenate([g, g[-2:0:-1]]); lam = np.fft.fft(c).real; lam[lam < 0] = 0
    z = rng.normal(size=2 * N) + 1j * rng.normal(size=2 * N)
    inc = np.fft.fft(np.sqrt(lam / (2 * N)) * z)[:N].real
    return np.cumsum(inc) * N ** (-H)
print("(b) singular values of the 2^10 x 2^10 unfolding of a sample path (normalised), sigma_j at j=1,4,16,64,256:")
for H in [0.3, 0.5, 0.7]:
    x = fbm(H, 2 ** n); s = np.linalg.svd(x.reshape(2 ** 10, 2 ** 10), compute_uv=False); s /= s[0]
    j = np.array([1, 4, 16, 64, 256]); slope = np.polyfit(np.log(np.arange(8, 257)), np.log(s[7:256]), 1)[0]
    print(f"    H={H}: {np.round(s[j - 1], 5)}  fitted decay exponent {slope:.3f}  (KL prediction -(H+1/2) = {-(H + .5):.2f})")

# (c) MRW increments: E[prod_a sqrt(mu(w^a)) xi_{w^a}]
K = 4; atoms = rng.uniform(0.2, 1.8, size=(K, 2)); probs = rng.dirichlet(np.ones(K))
mm = lambda m: float(np.sum(probs * np.prod(atoms ** (np.array(m)[None, :] / 2.0), 1)))  # half-moments
gauss = lambda m: 0.0 if m % 2 else float(np.prod(np.arange(m - 1, 0, -2))) if m else 1.0
def mrw_tensor(n, q):
    T = np.zeros((2 ** q,) * n)
    for idx in itertools.product(range(2 ** q), repeat=n):
        L = np.array([[(s >> a) & 1 for a in range(q)] for s in idx]); val = 1.0
        for k in range(n):
            groups = {}
            for a in range(q): groups.setdefault(tuple(L[:k, a]), []).append(a)
            for g in groups.values():
                m = [0, 0]
                for a in g: m[L[k, a]] += 1
                val *= mm(m)
        leaves = {}
        for a in range(q): leaves[tuple(L[:, a])] = leaves.get(tuple(L[:, a]), 0) + 1
        for m in leaves.values(): val *= gauss(m)
        T[idx] = val
    return T
print("(c) MRW increment moment tensors, interleaved TT ranks:  q=2,n=7:", ranks(mrw_tensor(7, 2), 1e-10),
      "  q=4,n=4:", ranks(mrw_tensor(4, 4), 1e-10), "(Bell B_2=2, B_4=15)")
