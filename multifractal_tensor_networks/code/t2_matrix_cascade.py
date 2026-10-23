"""Test 2: matrix-valued Mandelbrot cascade (r=2, b=2).
mu_n(w) = u^T A^{(root)}_{w1} A^{(w1)}_{w2} ... A^{(w1..w_{n-1})}_{wn} v, independent tuples per node.
(a) exact replica-MPS formula for E[sum_w mu^q] (integer q) vs brute-force expectation (small n)
(b) annealed pressure Phi(t) for real t via projective transfer operator
(c) quenched growth (1/n)log2 Z_n(q) by simulation vs conjecture  inf_{0<s<=1} Phi(q s)/s
"""
import itertools, numpy as np
rng = np.random.default_rng(1)
b, r, K = 2, 2, 3
atoms = np.exp(rng.normal(0, 1.1, size=(K, b, r, r)))   # K possible tuples (A_0, A_1)
probs = np.array([0.5, 0.3, 0.2])
M = np.einsum('k,kcij->ij', probs, atoms)                 # E[A_0 + A_1]
ev, V = np.linalg.eig(M); i = np.argmax(ev.real); rho = ev[i].real
atoms /= rho; M /= rho
v = np.abs(V[:, i].real); evl, U = np.linalg.eig(M.T); u = np.abs(U[:, np.argmax(evl.real)].real)
u /= u @ v                                                 # u^T v = 1, M v = v  -> mass martingale

def EAq(q):  # sum_c E[A_c^{(x) q}]
    T = 0
    for k in range(K):
        for c in range(b):
            Aq = atoms[k, c]
            for _ in range(q - 1): Aq = np.kron(Aq, atoms[k, c])
            T = T + probs[k] * Aq
    return T

def kronpow(x, q):
    y = x
    for _ in range(q - 1): y = np.kron(y, x)
    return y

# (a) brute force E[sum_w mu_n(w)^q] for n=3 by enumerating node assignments
n = 3
nodes = [w for k in range(n) for w in itertools.product(range(b), repeat=k)]
for q in [2, 3]:
    tot = 0.0
    for assign in itertools.product(range(K), repeat=len(nodes)):
        p = np.prod(probs[list(assign)]); lab = dict(zip(nodes, assign))
        Z = 0.0
        for w in itertools.product(range(b), repeat=n):
            x = u.copy()
            for k in range(n): x = x @ atoms[lab[w[:k]], w[k]]
            Z += (x @ v) ** q
        tot += p * Z
    T = EAq(q)
    mps = kronpow(u, q) @ np.linalg.matrix_power(T, n) @ kronpow(v, q)
    print(f"(a) q={q}: brute-force E Z_n(q) = {tot:.10f}   replica-transfer formula = {mps:.10f}")

# (b) annealed pressure Phi(t)=lim (1/n) log2 E sum_w ||x A_w||^t, projective operator on the cone
grid = np.linspace(0, 1, 4001)
def Phi(t, iters=400):
    f = np.ones_like(grid)
    X = np.stack([grid, 1 - grid], 1)
    lam = 1.0
    for _ in range(iters):
        g = np.zeros_like(grid)
        for k in range(K):
            for c in range(b):
                Y = X @ atoms[k, c]; nrm = Y.sum(1); p = Y[:, 0] / nrm
                g += probs[k] * nrm ** t * np.interp(p, grid, f)
        lam = g.max(); f = g / lam
    return np.log2(lam)
for q in [2, 3]:
    print(f"(b) check Phi({q}) projective = {Phi(q):.5f}   log2 rho(sum E A^{{(x)q}}) = {np.log2(max(abs(np.linalg.eigvals(EAq(q))))):.5f}")

# (c) quenched simulation
def simulate(n, qs):
    X = u[None, :].copy(); out = {q: [] for q in qs}
    for k in range(n):
        lab = rng.choice(K, size=X.shape[0], p=probs)
        X = np.einsum('ni,ncij->ncj', X, atoms[lab]).reshape(-1, r)
        mu = X @ v
        for q in qs: out[q].append(np.log2(np.sum(mu ** q)))
    return out
qs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
nmax, reps = 21, 6
runs = [simulate(nmax, qs) for _ in range(reps)]
svals = np.linspace(0.02, 1, 197)
phi_cache = {}
def Phic(t):
    t = round(t, 4)
    if t not in phi_cache: phi_cache[t] = Phi(t, iters=150)
    return phi_cache[t]
print("(c)   q   annealed Phi(q)   conj. quenched inf Phi(qs)/s   simulated slope (n=11..21)")
for q in qs:
    ann = Phic(q)
    que = min(Phic(q * s) / s for s in svals)
    slopes = [np.polyfit(np.arange(11, nmax + 1), np.array(run[q][10:]), 1)[0] for run in runs]
    print(f"    {q:4.1f}   {ann:9.4f}        {que:9.4f}                {np.mean(slopes):8.4f} +- {np.std(slopes)/np.sqrt(reps):.4f}")
