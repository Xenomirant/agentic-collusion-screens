"""Test 5: frozen-phase velocity with beam-pruned BRW (keep top-N particles) to reach depth 300.
Checks gamma* = inf_t Phi(t)/t for r=1 (known) and r=2 (conjecture)."""
import numpy as np
exec(open('t4_max.py').read().split("def run(")[0])   # reuse make(), Phi_fn(), params

def beam(atoms, u, v, r, n, N):
    X = (u / u.sum())[None, :]; L = np.array([np.log2(u.sum())]); out = []
    for k in range(n):
        lab = rng.integers(0, K, size=X.shape[0])
        Y = np.einsum('ni,ncij->ncj', X, atoms[lab]).reshape(-1, r)
        s = Y.sum(1); L = np.repeat(L, b) + np.log2(s); X = Y / s[:, None]
        score = L + np.log2(X @ v)
        if score.size > N:
            keep = np.argpartition(-score, N)[:N]; X, L, score = X[keep], L[keep], score[keep]
        out.append(score.max())
    return np.array(out)

n, N, reps = 300, 1 << 17, 3
for r in [1, 2]:
    atoms, u, v = make(r); Phi = Phi_fn(atoms, r)
    print(f"r={r}: Phi(1)={Phi(1.0):+.5f} (should be 0)", end="  ")
    if r == 2:
        T = sum(np.kron(atoms[k, c], atoms[k, c]) for k in range(K) for c in range(b)) / K
        print(f"Phi(2) proj={Phi(2.0):.5f} exact={np.log2(max(abs(np.linalg.eigvals(T)))):.5f}", end="  ")
    ts = np.linspace(0.3, 6, 600); vals = np.array([Phi(t) / t for t in ts]); j = np.argmin(vals)
    gstar, tc = vals[j], ts[j]
    Ms = np.mean([beam(atoms, u, v, r, n, N) for _ in range(reps)], 0)
    ns = np.arange(1, n + 1); sel = ns >= 100
    corr = Ms + 1.5 / tc * np.log2(ns)
    print(f"\n   t_c={tc:.3f} predicted gamma*={gstar:.5f} | beam slope (n=100..300) raw={np.polyfit(ns[sel], Ms[sel], 1)[0]:.5f}, log-corrected={np.polyfit(ns[sel], corr[sel], 1)[0]:.5f}")
