"""Test 4: frozen-phase velocity. Conjecture (Biggins-type for matrix cascades):
  (1/n) max_w log2 mu_n(w) -> gamma* = inf_{t>0} Phi(t)/t   (=> quenched tau linear beyond t_c)
with BRW-type correction -(3/(2 t_c)) log2 n.  r=1 is the known (Biggins/Aidekon) control."""
import numpy as np
rng = np.random.default_rng(11)
b, K, sig = 2, 300, 0.9
probs = np.full(K, 1.0 / K)

def make(r):
    g = rng.normal(0, sig, size=(K, b, 1, 1))
    B = np.exp(rng.normal(0, 0.35, size=(K, b, r, r))) * np.array([[1.0, 0.3], [0.5, 1.2]])[:r, :r]
    atoms = B * np.exp(g)
    M = atoms.mean(0).sum(0)
    ev, V = np.linalg.eig(M); i = np.argmax(ev.real); atoms /= ev[i].real; M /= ev[i].real
    v = np.abs(V[:, i].real); evl, U = np.linalg.eig(M.T); u = np.abs(U[:, np.argmax(evl.real)].real); u /= u @ v
    return atoms, u, v

def Phi_fn(atoms, r):
    if r == 1:
        return lambda t: np.log2(np.mean(np.sum(atoms[:, :, 0, 0] ** t, 1)))
    grid = np.linspace(0, 1, 1501); X = np.stack([grid, 1 - grid], 1)
    Ys = np.einsum('gi,kcij->kcgj', X, atoms); nrm = Ys.sum(-1); P = Ys[..., 0] / nrm
    def Phi(t):
        f = np.ones_like(grid)
        for _ in range(120):
            g = np.mean(np.sum(nrm ** t * np.interp(P, grid, f), 1), 0)
            lam = g.max(); f = g / lam
        return np.log2(lam)
    return Phi

def run(atoms, u, v, r, n):
    X = u[None, :].copy(); mx = []
    for k in range(n):
        lab = rng.integers(0, K, size=X.shape[0])
        X = np.einsum('ni,ncij->ncj', X, atoms[lab]).reshape(-1, r)
        s = X.sum(1, keepdims=True);
        mx.append(np.log2((X @ v).max()))
    return np.array(mx)

n, reps, n0 = 22, 12, 6
ns = np.arange(n0, n + 1)
for r in [1, 2]:
    atoms, u, v = make(r); Phi = Phi_fn(atoms, r)
    ts = np.linspace(0.3, 6, 300); vals = np.array([Phi(t) / t for t in ts])
    j = np.argmin(vals); gstar, tc = vals[j], ts[j]
    Y = np.mean([run(atoms, u, v, r, n)[n0 - 1:] for _ in range(reps)], 0)
    a2 = np.polyfit(ns, Y + 1.5 / tc * np.log2(ns), 1)[0]          # fixed 3/2 correction
    D = np.stack([ns, np.log2(ns), np.ones_like(ns)], 1).astype(float)
    a3, c3, _ = np.linalg.lstsq(D, Y, rcond=None)[0]
    print(f"r={r}: t_c={tc:.3f}  predicted gamma*=inf Phi(t)/t={gstar:.4f} | fit(fixed -3/(2t_c) log n)={a2:.4f} | free 3-param fit a={a3:.4f}, c={c3:.3f} (pred c={-1.5/tc:.3f}) | naive={np.polyfit(ns, Y, 1)[0]:.4f}")
