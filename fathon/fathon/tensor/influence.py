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

* `influence_z`: z-scores of ``gamma`` against a *conditional* null. The
  permutation variant shuffles ``a_j`` among the rounds sharing the same
  state; it keeps ``P(a_i | s)`` and ``P(a_j | s)`` and destroys only the
  extra dependence, but it is exact only when rounds are exchangeable given
  ``s``: an agent whose policy has memory the state does not capture (a sticky
  seller, say) makes it anti-conservative (`memory_beyond_state`). The
  ``'markov'`` variant resamples ``a_j`` from its fitted policy
  ``P(a_j(t) | s_t, a_j(t-1), ...)`` and keeps that memory. (A circular shift
  would also destroy dependence created by the common state, so it cannot
  separate collusion from common responses.)
* the plug-in bias of ``gamma`` grows with the number of states, so the
  z-scores are comparable across environments while raw ``gamma`` is not.
"""

import itertools
import warnings
import numpy as np


def _codes(x):
    _, inv = np.unique(np.asarray(x), return_inverse=True)
    return inv.ravel()


def _counts(x, y, s, nx, ny, ns):
    """Joint counts ``C[s, x, y]`` by one bincount."""
    return np.bincount((s * nx + x) * ny + y, minlength=ns * nx * ny).reshape(ns, nx, ny).astype(float)


def _mi_from_counts(C):
    """Plug-in ``I(X; Y | S)`` in nats from counts of shape ``(..., ns, nx, ny)``."""
    C = np.asarray(C, dtype=float)
    n = C.sum(axis=(-3, -2, -1))
    ps = C.sum(axis=(-2, -1), keepdims=True)
    px = C.sum(axis=-1, keepdims=True)
    py = C.sum(axis=-2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(C > 0, C * np.log(C * ps / (px * py)), 0.0)
    return term.sum(axis=(-3, -2, -1)) / np.where(n > 0, n, 1.0)


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
    return float(_mi_from_counts(_counts(x, y, s, nx, ny, ns))) / np.log(base)


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


def conditional_permutation(y, s, rng, size=None):
    """Shuffle `y` within the strata of `s` (all of `y` if `s` is None).

    With ``size`` given, return ``size`` independent permutations as rows.
    """
    y = np.asarray(y)
    one = size is None
    B = 1 if one else int(size)
    out = np.tile(y, (B, 1))
    if s is None:
        keys = rng.random((B, y.size))
        out = np.take_along_axis(out, np.argsort(keys, axis=1), axis=1)
    else:
        s = np.asarray(s)
        for v in np.unique(s):
            idx = np.flatnonzero(s == v)
            keys = rng.random((B, idx.size))
            out[:, idx] = y[idx][np.argsort(keys, axis=1)]
    return out[0] if one else out


def markov_bootstrap(y, s, rng, size, order=1):
    """Surrogates of `y` from its fitted policy ``P(y_t | s_t, y_{t-1..t-order})``.

    The observed state sequence is kept, each surrogate chain starts from the
    first `order` observed symbols, and unseen ``(s, history)`` cells fall back
    to ``P(y | s)``. This is the null "the agent follows its own (Markov)
    policy, independently of the others given the state": it keeps the agent's
    memory, which a within-stratum permutation destroys.

    Returns
    -------
    numpy ndarray
        Surrogate codes, shape ``(size, T)``.
    """
    y = _codes(y)
    T, ny = y.size, int(y.max()) + 1
    s = np.zeros(T, dtype=np.int64) if s is None else _codes(s)
    ns = int(s.max()) + 1
    order = int(order)
    if order < 0:
        raise ValueError('Error: order must be >= 0.')
    nh = ny ** order
    hist = np.zeros(T, dtype=np.int64)
    for m in range(1, order + 1):
        hist[order:] = hist[order:] * ny + y[order - m:T - m]
    C = np.bincount((s[order:] * nh + hist[order:]) * ny + y[order:], minlength=ns * nh * ny)
    C = C.reshape(ns, nh, ny).astype(float)
    Cs = np.bincount(s * ny + y, minlength=ns * ny).reshape(ns, ny).astype(float)
    Ps = Cs / np.where(Cs.sum(1, keepdims=True) > 0, Cs.sum(1, keepdims=True), 1.0)
    tot = C.sum(-1, keepdims=True)
    P = np.where(tot > 0, C / np.where(tot > 0, tot, 1.0), Ps[:, None, :])
    cum = np.cumsum(P, axis=-1)
    Yb = np.empty((size, T), dtype=np.int64)
    Yb[:, :order] = y[:order]
    U = rng.random((size, T))
    for t in range(order, T):
        h = np.zeros(size, dtype=np.int64)
        for m in range(1, order + 1):
            h = h * ny + Yb[:, t - m]
        c = cum[s[t], h]
        Yb[:, t] = np.minimum((U[:, t, None] > c).sum(-1), ny - 1)
    return Yb


def memory_beyond_state(A, S=None, lag=0):
    """Per-agent plug-in ``I(a(t); a(t - 1) | s_t)`` minus its first-order bias.

    A value well above zero means the agent's policy has memory the state does
    not capture, so rounds are not exchangeable given `S` and the permutation
    null of `influence_z` is anti-conservative; use ``null='markov'`` then.
    """
    A = np.asarray(A)
    N, T = A.shape
    s = None if S is None else np.asarray(S)
    out = np.zeros(N)
    for j in range(N):
        y = _codes(A[j, lag:] if lag else A[j])
        ss = None if s is None else _codes(s[lag:] if lag else s)
        yt, yp = y[1:], y[:-1]
        st = None if ss is None else ss[1:]
        ny = int(y.max()) + 1
        ns = 1 if st is None else int(st.max()) + 1
        bias = ns * (ny - 1) ** 2 / (2.0 * yt.size)
        out[j] = conditional_mi(yt, yp, st) - bias
    return out


def influence_z(A, S=None, lag=0, nPerm=199, seed=0, null='permutation', order=1, warn=True):
    """z-scores of ``gamma`` against a calibrated null.

    Parameters
    ----------
    A : array_like
        Actions, shape ``(N, T)``.
    S : array_like, optional
        Common observed state per time step (default : none).
    lag : int, optional
        See `influence_matrix` (default : 0).
    nPerm : int, optional
        Number of surrogates (default : 199).
    seed : int, optional
        RNG seed (default : 0).
    null : {'permutation', 'markov'}, optional
        ``'permutation'``: shuffle ``a_j`` within the strata of `S`. Exact
        only if rounds are exchangeable given `S`, i.e. the agents have no
        memory beyond the state. ``'markov'``: `markov_bootstrap` of ``a_j``
        given `S`, which keeps each agent's own order-`order` memory and is
        the right null for sticky agents (default : 'permutation').
    order : int, optional
        Markov order of the bootstrap policy (default : 1).
    warn : bool, optional
        With ``null='permutation'``, warn when `memory_beyond_state` suggests
        the exchangeability assumption fails (default : True).

    Returns
    -------
    numpy ndarray
        ``(N, N)`` z-scores (zero diagonal; symmetric when ``lag == 0``).
    numpy ndarray
        ``(N, N)`` one-sided surrogate p-values.
    """
    A = np.asarray(A)
    N = A.shape[0]
    if null not in ('permutation', 'markov'):
        raise ValueError('Error: null must be "permutation" or "markov".')
    if null == 'permutation' and warn:
        mem = memory_beyond_state(A, S, lag)
        if np.any(mem > 0.02):
            warnings.warn('influence_z: agents {} show memory beyond the state (I(a_t; a_t-1 | s) - bias = {}); '
                          'the permutation null is anti-conservative, consider null="markov".'
                          .format(np.flatnonzero(mem > 0.02).tolist(), np.round(mem[mem > 0.02], 3).tolist()),
                          RuntimeWarning, stacklevel=2)
    rng = np.random.default_rng(seed)
    Z = np.zeros((N, N))
    Pv = np.ones((N, N))
    pairs = itertools.combinations(range(N), 2) if lag == 0 else itertools.permutations(range(N), 2)
    for i, j in pairs:
        x, y, s = _aligned(A, S, i, j, lag)
        x, y = _codes(x), _codes(y)
        s = np.zeros_like(x) if s is None else _codes(s)
        nx, ny, ns = x.max() + 1, y.max() + 1, s.max() + 1
        obs = _mi_from_counts(_counts(x, y, s, nx, ny, ns))
        if null == 'permutation':
            Yb = conditional_permutation(y, s, rng, size=nPerm)
        else:
            Yb = markov_bootstrap(y, s, rng, nPerm, order)
        K = ns * nx * ny
        idx = np.arange(nPerm)[:, None] * K + (s * nx + x)[None, :] * ny + Yb
        Cb = np.bincount(idx.ravel(), minlength=nPerm * K).reshape(nPerm, ns, nx, ny).astype(float)
        nullv = _mi_from_counts(Cb)
        sd = nullv.std(ddof=1)
        z = (obs - nullv.mean()) / sd if sd > 0 else 0.0
        p = (1 + np.sum(nullv >= obs)) / (1 + nPerm)
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
