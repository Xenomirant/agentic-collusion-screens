"""Test 7: is the unfolding singular spectrum of a multifractal random walk path blind to intermittency?
Dyadic log-normal cascade (lambda^2 intermittency), MRW increments sqrt(mu) xi, path = cumsum."""
import numpy as np
rng = np.random.default_rng(5); n = 20
for lam2 in [0.0, 0.05, 0.15, 0.3]:
    exps = []; zeta = []
    for rep in range(4):
        mu = np.ones(1)
        for k in range(n):
            W = np.exp(np.sqrt(lam2 * np.log(2)) * rng.normal(size=(mu.size, 2)) - lam2 * np.log(2) / 2)
            mu = (mu[:, None] * W / 2).reshape(-1)
        x = np.cumsum(np.sqrt(mu) * rng.normal(size=mu.size))
        s = np.linalg.svd(x.reshape(2 ** 10, 2 ** 10), compute_uv=False)
        exps.append(np.polyfit(np.log(np.arange(8, 257)), np.log(s[7:256]), 1)[0])
        # empirical structure-function exponents of the path at q=4 (sample)
        hs = [2 ** j for j in range(2, 12)]
        S4 = [np.mean(np.abs(x[h:] - x[:-h]) ** 4) for h in hs]
        zeta.append(np.polyfit(np.log2(hs), np.log2(S4), 1)[0])
    zt = 2 - 2 * 2 * (2 - 1) * lam2 / 2 * 1  # zeta(4)=4/2 - lam2*(4/2)(4/2-1)... log-normal MRW: zeta(q)=q/2 - lam2 q(q-2)/8
    print(f"lambda^2={lam2:4.2f}: SVD decay exponent {np.mean(exps):.3f} (BM: -1.00)   sample zeta(4)={np.mean(zeta):.3f}  theory zeta(4)={4/2 - lam2*4*2/8:.3f}")
