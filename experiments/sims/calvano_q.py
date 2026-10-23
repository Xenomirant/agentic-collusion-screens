"""Calvano et al. (2020, AER) repeated Bertrand game with Q-learning firms.

Logit demand ``q_i = e^{(a_i - p_i)/mu} / (sum_j e^{(a_j - p_j)/mu} + e^{a_0/mu})``,
marginal cost ``c``, ``m`` prices on a grid spanning the Bertrand-Nash and
monopoly prices widened by ``xi``. State = all firms' prices in the previous
period (memory 1). Exploration ``eps_t = exp(-beta t)``.

Ground truth for collusion is the profit gain
``Delta = (pi - pi_N) / (pi_M - pi_N)``: 0 at Nash, 1 at joint monopoly.
"""

import itertools
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def logit_demand(p, a, a0, mu):
    e = np.exp((a - p) / mu)
    return e / (e.sum(axis=-1, keepdims=True) + np.exp(a0 / mu))


def nash_monopoly(n, a=2.0, a0=0.0, mu=0.25, c=1.0):
    """Symmetric Bertrand-Nash and joint-monopoly prices and per-firm profits."""
    av = np.full(n, a)

    def profit(p_own, p_other):
        p = np.full(n, p_other)
        p[0] = p_own
        return (p_own - c) * logit_demand(p, av, a0, mu)[0]

    def br(p_other):
        return minimize_scalar(lambda x: -profit(x, p_other), bounds=(c, c + 5), method='bounded').x

    pN = brentq(lambda p: br(p) - p, c + 1e-6, c + 5)
    joint = minimize_scalar(lambda x: -(x - c) * logit_demand(np.full(n, x), av, a0, mu)[0],
                            bounds=(c, c + 5), method='bounded').x
    return {'pN': pN, 'pM': joint, 'piN': profit(pN, pN), 'piM': profit(joint, joint)}


def simulate(n=2, m=15, xi=0.1, alpha=0.15, beta=4e-6, delta=0.95, T=1_500_000,
             a=2.0, a0=0.0, mu=0.25, c=1.0, memory=True, seed=0):
    """Run the learning dynamics.

    Parameters
    ----------
    memory : bool
        If False the state is constant (memoryless firms cannot condition
        on the rival, so they cannot sustain reward-punishment schemes).

    Returns
    -------
    dict
        ``actions`` (n, T) int8 grid indices, ``prices`` grid, ``profit``
        (n, T) float32, ``eps`` (T,), and the Nash/monopoly benchmarks.
    """
    rng = np.random.default_rng(seed)
    bench = nash_monopoly(n, a, a0, mu, c)
    span = bench['pM'] - bench['pN']
    grid = np.linspace(bench['pN'] - xi * span, bench['pM'] + xi * span, m)
    av = np.full(n, a)
    # profit tensor PI[i][a_1, ..., a_n]
    PI = np.zeros((n,) + (m,) * n)
    for acts in itertools.product(range(m), repeat=n):
        p = grid[list(acts)]
        PI[(slice(None),) + acts] = (p - c) * logit_demand(p, av, a0, mu)
    nS = m ** n if memory else 1
    # Calvano initialisation: discounted payoff against a uniformly randomising rival
    Q = np.empty((n, nS, m))
    for i in range(n):
        q0 = np.moveaxis(PI[i], i, 0).reshape(m, -1).mean(axis=1) / (1 - delta)
        Q[i] = q0[None, :]
    weights = m ** np.arange(n - 1, -1, -1)
    acts = np.zeros((n, T), dtype=np.int8)
    prof = np.zeros((n, T), dtype=np.float32)
    eps = np.exp(-beta * np.arange(T))
    s = int(rng.integers(nS))
    block = 1 << 16
    a_now = np.zeros(n, dtype=np.int64)
    for t0 in range(0, T, block):
        u = rng.random((min(block, T - t0), n))
        r = rng.integers(0, m, size=(min(block, T - t0), n))
        for k in range(u.shape[0]):
            t = t0 + k
            e = eps[t]
            for i in range(n):
                a_now[i] = r[k, i] if u[k, i] < e else int(np.argmax(Q[i, s]))
            idx = tuple(a_now)
            s_next = int(a_now @ weights) if memory else 0
            for i in range(n):
                pi = PI[(i,) + idx]
                Qi = Q[i, s]
                Qi[a_now[i]] += alpha * (pi + delta * Q[i, s_next].max() - Qi[a_now[i]])
                prof[i, t] = pi
            acts[:, t] = a_now
            s = s_next
    return {'actions': acts, 'grid': grid, 'profit': prof, 'eps': eps, 'Q': Q, **bench}


def profit_gain(profit, piN, piM, window=10_000):
    """Rolling profit gain Delta averaged over firms and a trailing window."""
    p = profit.mean(axis=0).astype(float)
    k = np.ones(window) / window
    return (np.convolve(p, k, mode='valid') - piN) / (piM - piN)
