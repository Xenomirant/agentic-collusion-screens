"""E0: estimators on processes with known scaling.

* binomial p-model cascade: analytic h(q) = 1/q - log2(p^q + (1-p)^q) / q
  (Kantelhardt et al. 2002) and tau(q) = -log2(p^q + (1-p)^q);
* fractional Gaussian noise (Davies-Harte): DFA exponent = H;
* multivariate: N fGn sharing a common component at weight w gives
  rho_DCCA -> w^2 / (w^2 + 1 - w^2) at all scales.

Output: experiments/results/e0/summary.json
"""

import json
import os
import numpy as np
from fathon.tensor import detrend, mvdfa, replica

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'e0')


def binomial_cascade(n, p):
    w = np.array([p, 1 - p])
    mu = w
    for _ in range(n - 1):
        mu = np.kron(mu, w)
    return mu


def fgn(T, H, rng):
    """Exact fractional Gaussian noise by circulant embedding (Davies-Harte)."""
    k = np.arange(T + 1)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.concatenate([g, g[-2:0:-1]])
    lam = np.clip(np.fft.fft(c).real, 0, None)
    M = c.size
    z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    return np.fft.fft(np.sqrt(lam / M) * z).real[:T]


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    # cascade
    p, n = 0.3, 16
    mu = binomial_cascade(n, p)
    qs = np.array([-4, -2, -1, 1, 2, 3, 4], dtype=float)
    scales = np.unique(np.round(2 ** np.linspace(4, 12, 17)).astype(np.int64))
    F = mvdfa.mfdfa(detrend.profile(mu), scales, qs)[0]
    h, _ = mvdfa.fit_exponents(scales, F)
    h_th = 1 / qs - np.log2(p ** qs + (1 - p) ** qs) / qs
    tau = replica.tau_spectrum(mu[None], qs)
    res['cascade'] = {'q': qs.tolist(), 'h_mfdfa': h.tolist(), 'h_theory': h_th.tolist(),
                      'max_abs_err_h': float(np.max(np.abs(h - h_th))),
                      'tau_replica': tau.tolist(), 'tau_theory': (-np.log2(p ** qs + (1 - p) ** qs)).tolist(),
                      'max_abs_err_tau': float(np.max(np.abs(tau + np.log2(p ** qs + (1 - p) ** qs))))}
    # fGn
    rng = np.random.default_rng(0)
    sc = np.unique(np.round(2 ** np.linspace(4, 11, 15)).astype(np.int64))
    out = {}
    for H in (0.3, 0.5, 0.7, 0.9):
        est = []
        for _ in range(10):
            x = fgn(2 ** 15, H, rng)
            est.append(mvdfa.fit_exponents(sc, mvdfa.mfdfa(detrend.profile(x), sc, [2])[0, 0])[0])
        out[str(H)] = {'mean': float(np.mean(est)), 'std': float(np.std(est))}
    res['fgn_dfa'] = out
    # multivariate common component
    N, T, H, w = 5, 2 ** 14, 0.7, 0.6
    common = fgn(T, H, rng)
    X = np.stack([w * common + np.sqrt(1 - w ** 2) * fgn(T, H, rng) for _ in range(N)])
    R = mvdfa.rho_matrix(detrend.profile(X), sc[sc <= 1024])
    off = np.array([r[np.triu_indices(N, 1)].mean() for r in R])
    res['common_component'] = {'w': w, 'rho_theory': w ** 2, 'rho_dcca_mean': float(off.mean()),
                               'rho_dcca_by_scale': off.tolist()}
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (v if k != 'cascade' else {kk: v[kk] for kk in ('max_abs_err_h', 'max_abs_err_tau')})
                      for k, v in res.items()}, indent=1))


if __name__ == '__main__':
    main()
