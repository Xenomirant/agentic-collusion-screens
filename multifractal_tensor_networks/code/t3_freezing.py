"""Test 3: freezing (replica-sector crossing) for scalar (r=1, known: Derrida-Spohn/Biggins)
and matrix (r=2, conjectured) cascades.  Fit log2 Z_n(q) = a n + c log2 n + d to remove
the log-n correction typical of the frozen phase; compare a with inf_{0<s<=1} Phi(qs)/s."""
import numpy as np, sys
rng = np.random.default_rng(7)
b, K = 2, 3
probs = np.array([0.6, 0.37, 0.03])

def make(r):
    atoms = np.exp(rng.normal(0, 0.5, size=(K, b, r, r))); atoms[2] *= 9.0
    M = np.einsum('k,kcij->ij', probs, atoms)
    ev, V = np.linalg.eig(M); i = np.argmax(ev.real); atoms /= ev[i].real; M /= ev[i].real
    v = np.abs(V[:, i].real); evl, U = np.linalg.eig(M.T); u = np.abs(U[:, np.argmax(evl.real)].real); u /= u @ v
    return atoms, u, v

def Phi_fn(atoms, r):
    if r == 1:
        return lambda t: np.log2(np.sum(probs[:, None] * atoms[:, :, 0, 0] ** t))
    grid = np.linspace(0, 1, 3001); X = np.stack([grid, 1 - grid], 1)
    cache = {}
    def Phi(t):
        t = round(float(t), 4)
        if t in cache: return cache[t]
        f = np.ones_like(grid)
        for _ in range(150):
            g = np.zeros_like(grid)
            for k in range(K):
                for c in range(b):
                    Y = X @ atoms[k, c]; nrm = Y.sum(1)
                    g += probs[k] * nrm ** t * np.interp(Y[:, 0] / nrm, grid, f)
            lam = g.max(); f = g / lam
        cache[t] = np.log2(lam); return cache[t]
    return Phi

def simulate(atoms, u, v, r, n, qs):
    X = u[None, :].copy(); out = {q: [] for q in qs}
    for k in range(n):
        lab = rng.choice(K, size=X.shape[0], p=probs)
        X = np.einsum('ni,ncij->ncj', X, atoms[lab]).reshape(-1, r)
        mu = X @ v
        lmu = np.log2(mu)
        for q in qs:
            m = (q * lmu).max(); out[q].append(m + np.log2(np.sum(2.0 ** (q * lmu - m))))
    return out

qs = [0.5, 1.5, 2.0, 3.0, 4.0]
n, reps, n0 = 22, 10, 8
svals = np.linspace(0.02, 1, 197)
for r in [1, 2]:
    atoms, u, v = make(r); Phi = Phi_fn(atoms, r)
    runs = [simulate(atoms, u, v, r, n, qs) for _ in range(reps)]
    ns = np.arange(n0, n + 1)
    D = np.stack([ns, np.log2(ns), np.ones_like(ns)], 1).astype(float)
    print(f"r={r}:   q   annealed   conjectured-quenched (s*)   fit slope a (+-se)   naive slope")
    for q in qs:
        vals = [min((Phi(q * s) / s, s) for s in svals)]
        que, sstar = vals[0]
        Y = np.mean([np.array(run[q][n0 - 1:]) for run in runs], 0)
        coef = np.linalg.lstsq(D, Y, rcond=None)[0]
        boots = []
        for _ in range(200):
            idx = rng.integers(0, reps, reps)
            Yb = np.mean([np.array(runs[i][q][n0 - 1:]) for i in idx], 0)
            boots.append(np.linalg.lstsq(D, Yb, rcond=None)[0][0])
        naive = np.polyfit(ns, Y, 1)[0]
        print(f"        {q:3.1f}   {Phi(q):8.4f}   {que:8.4f} ({sstar:.2f})          {coef[0]:8.4f} +- {np.std(boots):.4f}    {naive:8.4f}")
    sys.stdout.flush()
