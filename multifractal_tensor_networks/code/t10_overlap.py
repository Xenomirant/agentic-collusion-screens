"""Test 10: 1RSB / overlap prediction from the replica-sector picture.
Gibbs measure G_q(w) = mu_n(w)^q / Z_n(q). Participation ratio Y_n = sum_w G_q(w)^2 = Z_n(2q)/Z_n(q)^2
(= prob. that two replicas sit on the same leaf). Replica-sector (1RSB) prediction:
  E[Y_n] -> 0 for q < t_c ;  E[Y_n] -> 1 - t_c/q  for q > t_c   (Derrida-Spohn for r=1).
Tested for r=1 (control) and r=2 (matrix cascade)."""
import numpy as np
exec(open('t4_max.py').read().split("def run(")[0])   # make(), Phi_fn(), K=300, sig=0.9

def leaves(atoms, u, v, r, n):
    X = u[None, :].copy(); L = np.zeros(1)
    for k in range(n):
        lab = rng.integers(0, K, size=X.shape[0])
        Y = np.einsum('ni,ncij->ncj', X, atoms[lab]).reshape(-1, r)
        s = Y.sum(1); L = np.repeat(L, b) + np.log2(s); X = Y / s[:, None]
    return L + np.log2(X @ v)          # log2 mu_n(w)

qs = [0.8, 1.6, 2.0, 3.0, 4.0]
for r in [1, 2]:
    atoms, u, v = make(r); Phi = Phi_fn(atoms, r)
    ts = np.linspace(0.3, 6, 600); vals = np.array([Phi(t) / t for t in ts]); tc = ts[np.argmin(vals)]
    for n in [12, 16, 20]:
        Ys = {q: [] for q in qs}
        for rep in range(24 if n < 20 else 12):
            lm = leaves(atoms, u, v, r, n)
            for q in qs:
                a = q * lm; a -= a.max(); g = 2.0 ** a; g /= g.sum(); Ys[q].append(np.sum(g ** 2))
        print(f"r={r} t_c={tc:.3f} n={n}: " + "  ".join(f"q={q}: E[Y]={np.mean(Ys[q]):.3f}(pred {max(0, 1 - tc / q):.3f})" for q in qs), flush=True)
