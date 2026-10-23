"""Test 1: q-point moment tensor of a scalar (r=1) Mandelbrot cascade.
Exact enumeration; compute TT ranks in the interleaved (replica-quantized) format.
Claim: ranks bounded by Bell number B_q independent of depth n.
"""
import itertools, numpy as np
rng = np.random.default_rng(0)
b = 2
# random tuple (W_0, W_1): discrete law with K atoms, E[W_0+W_1]=1 normalisation not needed
K = 4
atoms = rng.uniform(0.2, 1.8, size=(K, b))
probs = rng.dirichlet(np.ones(K))

def mixed_moment(m):  # E[prod_c W_c^{m_c}]
    return float(np.sum(probs * np.prod(atoms ** np.array(m)[None, :], axis=1)))

def moment_tensor(n, q):
    # index: for each level k, tuple (c^1..c^q); build full tensor shape (2^q,)*n
    T = np.zeros((b ** q,) * n)
    for idx in itertools.product(range(b ** q), repeat=n):
        # decode letters
        letters = np.array([[ (s >> a) & 1 for a in range(q)] for s in idx])  # n x q
        val = 1.0
        # level k: group replicas by prefix of length k (parent node), count children
        for k in range(n):
            prefixes = [tuple(letters[:k, a]) for a in range(q)]
            groups = {}
            for a in range(q):
                groups.setdefault(prefixes[a], []).append(a)
            for g in groups.values():
                m = [0] * b
                for a in g:
                    m[letters[k, a]] += 1
                val *= mixed_moment(m)
        T[idx] = val
    return T

def tt_ranks(T, tol=1e-10):
    n = T.ndim; d = T.shape[0]; rs = []
    for k in range(1, n):
        M = T.reshape(d ** k, -1)
        s = np.linalg.svd(M, compute_uv=False)
        rs.append(int(np.sum(s > tol * s[0])))
    return rs

bell = {1: 1, 2: 2, 3: 5, 4: 15}
for q, n in [(1, 8), (2, 7), (3, 5), (4, 4)]:
    T = moment_tensor(n, q)
    print(f"q={q} n={n}  TT ranks={tt_ranks(T)}  Bell B_q={bell[q]}")

# Contrast: a single sample of the cascade at depth n=12 -> QTT ranks
n = 12
def sample_cascade(n):
    mu = np.ones(1)
    for k in range(n):
        tup = atoms[rng.choice(K, size=mu.size, p=probs)]  # per node tuple
        mu = (mu[:, None] * tup).reshape(-1)
    return mu
mu = sample_cascade(n)
print("sample cascade QTT ranks:", tt_ranks(mu.reshape((2,) * n)))
print("max possible          :", [min(2 ** k, 2 ** (n - k)) for k in range(1, n)])
# deterministic binomial cascade
w = np.array([0.3, 0.7]); mu_det = w
for k in range(n - 1): mu_det = np.kron(mu_det, w)
print("binomial cascade QTT ranks:", tt_ranks(mu_det.reshape((2,) * n)))
