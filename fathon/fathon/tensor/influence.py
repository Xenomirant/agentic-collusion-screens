#    influence.py - pairwise (net) influence between agents, Bonjour et al. 2022
#    Copyright (C) 2026-  fathon tensor extension contributors
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Individual and net influence of agents on each other.

Bonjour, Aggarwal & Bhargava (UAI 2022) define, for agents i, j acting in a
common observed state s (fully observable, simultaneous moves),

    gamma(i; j) = I(a_i; a_j | s)                       (individual influence)
    Gamma(i; j) = gamma(i; j) - max_{k != i, j} gamma(k; j)    (net influence)

and flag the pair (i, j) as colluding when ``Gamma(i; j) >= alpha`` and
``Gamma(j; i) >= alpha``; collusion is reported only if exactly one pair is
flagged. For sequential games, ``gamma(i; j)`` uses only the cases where j
moves after i, which here is the lagged variant ``I(a_i(t - lag); a_j(t) | s)``.

``gamma`` is the state-conditioned version of the single-agent cut statistic
of `factorization` (window length 1). Two calibrated variants are added:

* `influence_z`: z-scores of ``gamma`` against a *conditional* permutation
  null that shuffles ``a_j`` among the rounds sharing the same state. This
  keeps ``P(a_i | s)`` and ``P(a_j | s)`` and destroys only the extra
  dependence, which is exactly the null "independent policies given s".
  (A circular shift would also destroy dependence created by the common
  state, so it cannot separate collusion from common responses.)
* the plug-in bias of ``gamma`` grows with the number of states, so the
  z-scores are comparable across environments while raw ``gamma`` is not.
"""

import itertools
import numpy as np


def _codes(x):
    _, inv = np.unique(np.asarray(x), return_inverse=True)
    return inv


def conditional_mi(x, y, s=None, base=np.e):
    """Plug-in conditional mutual information ``I(X; Y | S)``.

    Parameters
    ----------
    x, y : array_like
        Discrete samples of equal length.
    s : array_like, optional
        Discrete conditioning variable (default : none).
    base : float, optional
        Logarithm base (default : e, i.e. nats).

    Returns
    -------
    float
        Estimate of the conditional mutual information.
    """
    x, y = _codes(x), _codes(y)
    s = np.zeros_like(x) if s is None else _codes(s)
    nx, ny, ns = x.max() + 1, y.max() + 1, s.max() + 1
    C = np.zeros((ns, nx, ny))
    np.add.at(C, (s, x, y), 1.0)
    n = C.sum()
    ps = C.sum(axis=(1, 2))
    out = 0.0
    for k in range(ns):
        if ps[k] == 0:
            continue
        P = C[k] / ps[k]
        pa = P.sum(1, keepdims=True)
        pb = P.sum(0, keepdims=True)
        nz = P > 0
        out += ps[k] / n * float(np.sum(P[nz] * np.log(P[nz] / (pa @ pb)[nz])))
    return out / np.log(base)


def _aligned(A, S, i, j, lag):
    T = A.shape[1]
    if lag == 0:
        return A[i], A[j], (None if S is None else S)
    xi = A[i, :T - lag]
    yj = A[j, lag:]
    s = None if S is None else S[lag:]
    return xi, yj, s


def influence_matrix(A, S=None, lag=0, base=np.e):
    """Individual influence ``gamma[i, j] = I(a_i(t - lag); a_j(t) | s_t)``.

    Parameters
    ----------
    A : array_like
        Actions, shape ``(N, T)``, any discrete labels.
    S : array_like, optional
        Common observed state per time step, shape ``(T,)`` (default : none).
    lag : int, optional
        0 for simultaneous moves (symmetric matrix); > 0 for the sequential
        variant (influence of i's earlier move on j's later move).
    base : float, optional
        Logarithm base (default : e).

    Returns
    -------
    numpy ndarray
        ``(N, N)`` matrix with zero diagonal.
    """
    A = np.asarray(A)
    N = A.shape[0]
    G = np.zeros((N, N))
    pairs = itertools.combinations(range(N), 2) if lag == 0 else itertools.permutations(range(N), 2)
    for i, j in pairs:
        x, y, s = _aligned(A, S, i, j, lag)
        G[i, j] = conditional_mi(x, y, s, base)
        if lag == 0:
            G[j, i] = G[i, j]
    return G


def net_influence(G):
    """Net influence ``Gamma[i, j] = G[i, j] - max_{k != i, j} G[k, j]``.

    Needs at least three agents. The diagonal is NaN.
    """
    G = np.asarray(G, dtype=float)
    N = G.shape[0]
    if N < 3:
        raise ValueError('Error: net influence needs at least 3 agents.')
    Gam = np.full((N, N), np.nan)
    for i, j in itertools.permutations(range(N), 2):
        Gam[i, j] = G[i, j] - max(G[k, j] for k in range(N) if k not in (i, j))
    return Gam


def pair_scores(Gam):
    """Symmetric pair score ``min(Gamma[i, j], Gamma[j, i])`` for every pair, sorted decreasingly."""
    N = Gam.shape[0]
    sc = [(float(min(Gam[i, j], Gam[j, i])), (i, j)) for i, j in itertools.combinations(range(N), 2)]
    return sorted(sc, key=lambda t: -t[0])


def detect_pair(Gam, alpha=0.05):
    """Bonjour et al.'s rule: return the colluding pair if exactly one pair has
    ``Gamma >= alpha`` in both directions, else None."""
    flagged = [p for v, p in pair_scores(Gam) if v >= alpha]
    return flagged[0] if len(flagged) == 1 else None


def conditional_permutation(y, s, rng):
    """Shuffle `y` within the strata of `s` (all of `y` if `s` is None)."""
    y = np.asarray(y).copy()
    if s is None:
        return rng.permutation(y)
    s = np.asarray(s)
    for v in np.unique(s):
        idx = np.flatnonzero(s == v)
        y[idx] = y[rng.permutation(idx)]
    return y


def influence_z(A, S=None, lag=0, nPerm=199, seed=0):
    """z-scores of ``gamma`` against the conditional permutation null.

    Returns
    -------
    numpy ndarray
        ``(N, N)`` z-scores (zero diagonal; symmetric when ``lag == 0``).
    numpy ndarray
        ``(N, N)`` one-sided permutation p-values.
    """
    A = np.asarray(A)
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    Z = np.zeros((N, N))
    Pv = np.ones((N, N))
    pairs = itertools.combinations(range(N), 2) if lag == 0 else itertools.permutations(range(N), 2)
    for i, j in pairs:
        x, y, s = _aligned(A, S, i, j, lag)
        obs = conditional_mi(x, y, s)
        null = np.array([conditional_mi(x, conditional_permutation(y, s, rng), s) for _ in range(nPerm)])
        sd = null.std(ddof=1)
        z = (obs - null.mean()) / sd if sd > 0 else 0.0
        p = (1 + np.sum(null >= obs)) / (1 + nPerm)
        Z[i, j], Pv[i, j] = z, p
        if lag == 0:
            Z[j, i], Pv[j, i] = z, p
    return Z, Pv


def action_correlation(A):
    """Ordinary action dependence without state: Cramer's V for every pair."""
    A = np.asarray(A)
    N = A.shape[0]
    V = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        x, y = _codes(A[i]), _codes(A[j])
        C = np.zeros((x.max() + 1, y.max() + 1))
        np.add.at(C, (x, y), 1.0)
        n = C.sum()
        E = np.outer(C.sum(1), C.sum(0)) / n
        chi2 = np.sum((C - E) ** 2 / np.where(E > 0, E, 1.0))
        k = min(C.shape) - 1
        V[i, j] = V[j, i] = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
    return V


def rank_of_pair(M, pair):
    """Rank (1 = top) of `pair` among all pairs by the symmetric score of `M`.

    For asymmetric `M` (e.g. net influence) the pair score is the minimum of
    the two directions; NaN diagonals are ignored. Ties count pessimistically.
    """
    M = np.asarray(M, dtype=float)
    N = M.shape[0]
    score = {(i, j): min(M[i, j], M[j, i]) for i, j in itertools.combinations(range(N), 2)}
    target = score[tuple(sorted(pair))]
    return 1 + sum(v >= target for p, v in score.items() if p != tuple(sorted(pair)))
