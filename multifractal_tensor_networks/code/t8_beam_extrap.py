"""Test 8: frozen-phase speed, beam (N-BRW) at several N, Brunet-Derrida extrapolation in 1/ln^2 N.
r=1 is the rigorous control (Biggins); r=2 tests inf_t Phi(t)/t for matrix cascades."""
import numpy as np
exec(open('t5_beam.py').read().split("n, N, reps")[0])
n, reps = 260, 2
for r in [1, 2]:
    atoms, u, v = make(r); Phi = Phi_fn(atoms, r)
    ts = np.linspace(0.3, 6, 600); vals = np.array([Phi(t) / t for t in ts]); j = np.argmin(vals); gstar, tc = vals[j], ts[j]
    xs, ys = [], []
    for e in [10, 12, 14, 16]:
        N = 1 << e
        Ms = np.mean([beam(atoms, u, v, r, n, N) for _ in range(reps)], 0)
        ns = np.arange(1, n + 1); sel = ns >= 80
        sl = np.polyfit(ns[sel], Ms[sel] + 1.5 / tc * np.log2(ns[sel]), 1)[0]
        xs.append(1 / np.log(N) ** 2); ys.append(sl)
        print(f"r={r} N=2^{e}: slope={sl:.5f}", flush=True)
    a, c = np.polyfit(xs, ys, 1)
    print(f"r={r}: predicted gamma*={gstar:.5f}  extrapolated (N->inf) speed={c:.5f}  diff={c-gstar:+.5f}", flush=True)
