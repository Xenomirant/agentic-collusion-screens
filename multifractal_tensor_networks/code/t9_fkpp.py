"""Test 9: noise-free frozen-phase test via the (type-dependent) F-KPP / max-recursion.
G_n(x, p) = P( max_{|w|=n} log2 ||x_p A_w||_1 <= x ),  x_p = (p, 1-p) start direction.
Recursion (independent tuple per node, product over children of the SAME tuple):
  G_{n+1}(x, p) = E_k  prod_c  G_n( x - log2||x_p A_c^{(k)}||, p'(p, A_c^{(k)}) ).
Front speed = a.s. speed of the max = conjectured gamma* = inf_t Phi(t)/t (Biggins-type theorem).
Bramson correction -(3/(2 t_c)) log2 n is removed before fitting the speed."""
import numpy as np, sys, time
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 21)
b, K, sig = 2, 60, 0.9
probs = np.full(K, 1.0 / K)

def make(r):
    g = rng.normal(0, sig, size=(K, b, 1, 1))
    B = np.exp(rng.normal(0, 0.35, size=(K, b, r, r))) * np.array([[1.0, 0.3], [0.5, 1.2]])[:r, :r]
    atoms = B * np.exp(g)
    M = atoms.mean(0).sum(0); ev = np.linalg.eigvals(M); atoms /= max(ev.real)
    return atoms

def Phi_proj(atoms, t, grid):
    X = np.stack([grid, 1 - grid], 1)
    Ys = np.einsum('gi,kcij->kcgj', X, atoms); nrm = Ys.sum(-1); P = Ys[..., 0] / nrm
    f = np.ones_like(grid)
    for _ in range(300):
        g = np.mean(np.sum(nrm ** t * np.interp(P, grid, f), 1), 0); lam = g.max(); f = g / lam
    return np.log2(lam)

def run(r, nsteps=1500, dx=0.01, width=30.0, Pn=81):
    atoms = make(r)
    if r == 1:
        Phi = lambda t: np.log2(np.mean(np.sum(atoms[:, :, 0, 0] ** t, 1)))
    else:
        pg = np.linspace(0, 1, 801); Phi = lambda t: Phi_proj(atoms, t, pg)
    ts = np.linspace(0.5, 4, 351); vals = np.array([Phi(t) / t for t in ts]); j = np.argmin(vals)
    gstar, tc = vals[j], ts[j]
    xs = np.arange(-width / 2, width / 2, dx); Nx = xs.size
    pgrid = np.linspace(0, 1, Pn) if r == 2 else np.array([0.5])
    if r == 2:
        X = np.stack([pgrid, 1 - pgrid], 1)
        Y = np.einsum('gi,kcij->kcgj', X, atoms); nrm = Y.sum(-1)
        shift = np.log2(nrm)                      # (K,b,P)
        pnew = Y[..., 0] / nrm                    # (K,b,P)
        jlo = np.clip(np.floor(pnew * (Pn - 1)).astype(int), 0, Pn - 2); wp = pnew * (Pn - 1) - jlo
    else:
        shift = np.log2(atoms[:, :, 0, 0])[:, :, None]; jlo = np.zeros_like(shift, dtype=int); wp = np.zeros_like(shift)
    G = np.tile((xs >= 0).astype(float), (pgrid.size, 1))
    offset = 0.0; front = []
    for n in range(1, nsteps + 1):
        Gn = np.zeros_like(G)
        for k in range(K):
            prod = np.ones_like(G)
            for c in range(b):
                if r == 2:
                    rows = (1 - wp[k, c][:, None]) * G[jlo[k, c]] + wp[k, c][:, None] * G[np.minimum(jlo[k, c] + 1, Pn - 1)]
                else:
                    rows = G
                s = shift[k, c][:, None]                       # evaluate rows at x - s
                pos = (xs[None, :] - s - xs[0]) / dx
                i0 = np.floor(pos).astype(int); fr = pos - i0
                i0c = np.clip(i0, 0, Nx - 1); i1c = np.clip(i0 + 1, 0, Nx - 1)
                ri = np.arange(rows.shape[0])[:, None]
                val = (1 - fr) * rows[ri, i0c] + fr * rows[ri, i1c]
                val = np.where(pos < 0, 0.0, np.where(pos > Nx - 1, 1.0, val))
                prod *= val
            Gn += probs[k] * prod
        G = Gn
        # front location (G=1/2) at the middle direction row, then recentre grid
        row = G[G.shape[0] // 2]; idx = np.searchsorted(row, 0.5)
        xf = xs[idx - 1] + (0.5 - row[idx - 1]) / (row[idx] - row[idx - 1]) * dx
        front.append(offset + xf)
        m = int(round(xf / dx))
        if m != 0:
            G = np.roll(G, -m, axis=1)
            if m > 0: G[:, -m:] = 1.0
            else: G[:, :-m] = 0.0
            offset += m * dx
    front = np.array(front); ns = np.arange(1, nsteps + 1); sel = ns >= nsteps // 3
    v_raw = np.polyfit(ns[sel], front[sel], 1)[0]
    v_cor = np.polyfit(ns[sel], front[sel] + 1.5 / tc * np.log2(ns[sel]), 1)[0]
    return gstar, tc, v_raw, v_cor

for r in [1, 2]:
    t0 = time.time(); g, tc, vr, vc = run(r)
    print(f"r={r}: t_c={tc:.3f} predicted gamma*={g:.5f} | F-KPP front speed raw={vr:.5f}  Bramson-corrected={vc:.5f}  diff={vc-g:+.5f}  ({time.time()-t0:.0f}s)", flush=True)
