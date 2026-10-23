"""Test 11: dimension-scalability of the replica MPS.
d-dimensional dyadic cascade on [0,1]^d: 2^d children per node, child label = d bits (one per coordinate).
q-point moment tensor written in the bit-interleaved QTT format: sites ordered (level k, coordinate j),
each site carries bit j of level k for all q replicas (site dim 2^q). Total n*d sites.
We compute TT ranks at ALL cuts (also cuts inside a level) for three tuple laws:
  (a) iid children:        W_c iid over the 2^d children          -> predicted bond <= B_q (dimension-free)
  (b) exchangeable tuple:  W = random permutation of a fixed vector -> predicted <= #{(pi, pi') : pi' <= pi}
  (c) general tuple:       arbitrary joint law on (W_0..W_{2^d-1})  -> intra-level cuts may be larger
and the 'fused' format (one site per level, site dim 2^{dq}) where bond <= B_q for any law."""
import itertools, numpy as np
rng = np.random.default_rng(4)
BELL = {1: 1, 2: 2, 3: 5, 4: 15}
PAIRS = {1: 1, 2: 3, 3: 12, 4: 60}   # number of comparable pairs pi' <= pi in the partition lattice

from functools import lru_cache
def make_law(kind, d, K=5):
    f = _make_law(kind, d, K)
    cached = lru_cache(maxsize=None)(lambda ms: f(list(ms)))
    return lambda ms: cached(tuple(ms))

def _make_law(kind, d, K=5):
    C = 2 ** d
    if kind == 'iid':
        vals = rng.uniform(0.2, 1.8, size=K); p = rng.dirichlet(np.ones(K))
        mom = lambda m: float(np.sum(p * vals ** m))
        return lambda ms: np.prod([mom(m) for m in ms])
    if kind == 'exch':
        base = rng.uniform(0.2, 1.8, size=C)
        perms = np.array(list(itertools.permutations(range(C)))) if C <= 8 else None
        tuples = base[perms]
        return lambda ms: float(np.mean(np.prod(tuples ** np.array(ms)[None, :], 1)))
    if kind == 'general':
        tuples = rng.uniform(0.2, 1.8, size=(K, C)); p = rng.dirichlet(np.ones(K))
        return lambda ms: float(np.sum(p * np.prod(tuples ** np.array(ms)[None, :], 1)))

def moment_tensor(law, d, n, q):
    C = 2 ** d
    words = list(itertools.product(range(C), repeat=n))          # each replica: n child labels
    T = np.zeros([C] * (n * q))                                 # index order: replica-major, fixed below
    vals = {}
    for combo in itertools.product(range(len(words)), repeat=q):
        ws = [words[i] for i in combo]; val = 1.0
        for k in range(n):
            groups = {}
            for a in range(q): groups.setdefault(ws[a][:k], []).append(a)
            for g in groups.values():
                ms = [0] * C
                for a in g: ms[ws[a][k]] += 1
                val *= law(ms)
        idx = tuple(ws[a][k] for a in range(q) for k in range(n))
        T[idx] = val
    return T

def to_interleaved(T, d, n, q, fused):
    # T axes: (a,k) child labels; split each label into d bits (MSB = coordinate 0), reorder to (k, j, a)
    C = 2 ** d
    T = T.reshape([2] * (q * n * d))     # axes (a, k, j)
    axes = [(a * n + k) * d + j for k in range(n) for j in range(d) for a in range(q)]
    T = T.transpose(axes)
    if fused: return T.reshape([2 ** (d * q)] * n)
    return T.reshape([2 ** q] * (n * d))

def ranks(T, tol=1e-10):
    D = T.ndim; dims = T.shape; out = []
    for c in range(1, D):
        M = T.reshape(int(np.prod(dims[:c])), -1); s = np.linalg.svd(M, compute_uv=False)
        out.append(int(np.sum(s > tol * s[0])))
    return out

import sys
cases = [(2, 4, 2), (3, 3, 2), (2, 3, 3), (3, 2, 3), (2, 2, 4)]
for (d, n, q) in cases[int(sys.argv[1]):]:
    for kind in ['iid', 'exch', 'general']:
        law = make_law(kind, d); T = moment_tensor(law, d, n, q)
        r_int = ranks(to_interleaved(T, d, n, q, fused=False)); r_fus = ranks(to_interleaved(T, d, n, q, fused=True))
        print(f"d={d} n={n} q={q} {kind:8s} interleaved max={max(r_int):3d} {r_int} | fused max={max(r_fus)} | B_q={BELL[q]} pairs={PAIRS[q]}", flush=True)
