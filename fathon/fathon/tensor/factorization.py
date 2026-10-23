#    factorization.py - factorization defect of joint multi-agent processes
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

"""Does a joint process factorize over a cut of the agents?

Agents that do not interact generate a joint process whose word distribution
is a tensor product, ``P(a_{t:t+k}, b_{t:t+k}) = P_A(a_{t:t+k}) P_B(b_{t:t+k})``
for every window length ``k``. Equivalently the joint transfer operator is a
Kronecker product ``T_A kron T_B``: the Van Loan-Pitsianis rearrangement of the
pair-transition matrix *is* the cut matrix of length-2 windows. So the cut
matrix ``P[a-word, b-word]`` has rank one for every ``k`` if and only if the
agents factorize, and its spectrum measures how far they are from it.

The spectrum used here is that of the whitened cut matrix
``D_A^{-1/2} P D_B^{-1/2}`` (correspondence analysis). Its leading singular
value is always 1. The others are the canonical correlations of the cut:

* ``hgr``: the second singular value, the Hirschfeld-Gebelein-Renyi maximal
  correlation (0 iff independent);
* ``phi2``: the sum of squares of the others, the mean-square contingency;
  ``n * phi2`` is Pearson's chi-square statistic;
* ``entropy``: Renyi-2 entropy of the normalised squared spectrum, a
  classical "entanglement entropy" of the cut (0 iff rank one).

Overlapping windows are dependent, so null distributions come from
circular time-shift surrogates (`shift_null`), not from chi-square tables.
"""

import itertools
import numpy as np
from . import decomp


def discretize(X, nBins=3, method='quantile'):
    """Map real series to symbols ``0 .. nBins - 1``.

    Parameters
    ----------
    X : array_like
        Series, shape ``(N, T)`` or ``(T,)``; each row is binned separately.
    nBins : int, optional
        Alphabet size (default : 3).
    method : {'quantile', 'uniform', 'sign'}, optional
        Per-row quantile bins, equal-width bins, or the sign of the series
        (``nBins`` is then 2 or 3 with a zero band of +-1e-12)
        (default : 'quantile').

    Returns
    -------
    numpy ndarray
        Integer symbols with the shape of `X`.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    S = np.empty(X.shape, dtype=np.int64)
    for i, x in enumerate(X):
        if method == 'quantile':
            edges = np.quantile(x, np.linspace(0, 1, nBins + 1)[1:-1])
            S[i] = np.searchsorted(edges, x, side='right')
        elif method == 'uniform':
            lo, hi = np.min(x), np.max(x)
            edges = np.linspace(lo, hi, nBins + 1)[1:-1]
            S[i] = np.searchsorted(edges, x, side='right')
        elif method == 'sign':
            if nBins == 2:
                S[i] = (x > 0).astype(np.int64)
            else:
                S[i] = np.where(x > 1e-12, 2, np.where(x < -1e-12, 0, 1))
        else:
            raise ValueError('Error: unknown method {}.'.format(method))
    return S


def words(S, k, alphabet):
    """Encode sliding windows of length `k` as integers.

    Parameters
    ----------
    S : numpy ndarray
        Symbols, shape ``(N, T)`` (agents of one group) or ``(T,)``.
    k : int
        Window length.
    alphabet : int
        Alphabet size per agent and time step.

    Returns
    -------
    numpy ndarray
        Codes of the joint word of all rows, shape ``(T - k + 1,)``.
    """
    S = np.atleast_2d(np.asarray(S))
    T = S.shape[1] - k + 1
    if T < 1:
        raise ValueError('Error: window length k exceeds the series length.')
    if S.min() < 0 or S.max() >= alphabet:
        raise ValueError('Error: symbols must lie in 0 .. alphabet - 1 (got {}..{} with alphabet {}).'
                         .format(int(S.min()), int(S.max()), alphabet))
    if alphabet ** (S.shape[0] * k) >= 2 ** 62:
        # integer codes would overflow: label the distinct joint windows instead
        cols = np.stack([row[j:j + T] for row in S for j in range(k)], axis=1)
        _, code = np.unique(cols, axis=0, return_inverse=True)
        return code.ravel().astype(np.int64)
    code = np.zeros(T, dtype=np.int64)
    for row in S:
        for j in range(k):
            code = code * alphabet + row[j:j + T]
    return code


def contingency(a, b):
    """Joint probability matrix of two code sequences on their observed values."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    P = np.zeros((ua.size, ub.size))
    np.add.at(P, (ia, ib), 1.0)
    return P / P.sum()


def cut_spectrum(P):
    """Canonical correlations of a joint probability matrix.

    Returns
    -------
    numpy ndarray
        Singular values of ``D_A^{-1/2} P D_B^{-1/2}`` without the leading 1.
    """
    P = np.asarray(P, dtype=float)
    pa = P.sum(axis=1)
    pb = P.sum(axis=0)
    P = P[pa > 0][:, pb > 0]            # unobserved rows / columns carry no information
    pa, pb = pa[pa > 0], pb[pb > 0]
    if P.size == 0:
        return np.zeros(0)
    W = P / np.sqrt(np.outer(pa, pb))
    s = np.linalg.svd(W, compute_uv=False)
    return s[1:]


def cut_stats(P):
    """Factorization defect statistics of a joint probability matrix.

    Returns
    -------
    dict
        ``hgr``, ``phi2``, ``entropy`` (see module docstring), ``mi``
        (plug-in mutual information, nats) and ``kron_defect``
        (``1 - s_1^2 / sum s^2`` of the raw, unwhitened matrix).
    """
    c = cut_spectrum(P)
    pa = P.sum(axis=1, keepdims=True)
    pb = P.sum(axis=0, keepdims=True)
    nz = P > 0
    mi = float(np.sum(P[nz] * np.log(P[nz] / (pa @ pb)[nz])))
    s = np.linalg.svd(P, compute_uv=False)
    p = np.concatenate([[1.0], c ** 2])
    return {
        'hgr': float(c[0]) if c.size else 0.0,
        'phi2': float(np.sum(c ** 2)),
        'entropy': decomp.renyi_entropy(p / p.sum(), 2.0),
        'mi': mi,
        'kron_defect': float(1.0 - s[0] ** 2 / np.sum(s ** 2)),
    }


def group_cut(S, groupA, groupB, k=2, alphabet=None, lag=0):
    """Factorization statistics across a cut of agents.

    Parameters
    ----------
    S : numpy ndarray
        Symbols, shape ``(N, T)``.
    groupA, groupB : iterable of int
        Disjoint agent indices.
    k : int, optional
        Window length; ``k = 2`` is the Kronecker test of the one-step
        transfer operator (default : 2).
    alphabet : int, optional
        Alphabet size (default : ``S.max() + 1``).
    lag : int, optional
        Shift of B's window relative to A's. ``lag = k`` compares A's past
        with B's future, i.e. the directed influence A -> B (default : 0).

    Returns
    -------
    dict
        See `cut_stats`; also ``n`` (number of windows) and ``n_states``
        (observed A and B words).
    """
    S = np.asarray(S)
    alphabet = int(S.max()) + 1 if alphabet is None else alphabet
    a = words(S[list(groupA)], k, alphabet)
    b = words(S[list(groupB)], k, alphabet)
    if lag > 0:
        a, b = a[:-lag], b[lag:]
    elif lag < 0:
        a, b = a[-lag:], b[:lag]
    P = contingency(a, b)
    out = cut_stats(P)
    out['n'] = a.size
    out['n_states'] = P.shape
    return out


def shift_null(S, groupA, groupB, stat='phi2', k=2, nSurr=200, minShift=None, seed=0, **kw):
    """Circular time-shift null for a cut statistic.

    Shifting group B by a random lag keeps every agent's own dynamics and
    destroys their alignment, which is exactly the factorized null.

    Returns
    -------
    float
        Observed statistic.
    numpy ndarray
        Surrogate statistics.
    float
        z-score of the observed value.
    float
        One-sided p-value with the (1 + count) / (1 + nSurr) correction.
    """
    S = np.asarray(S)
    T = S.shape[1]
    minShift = max(10 * k, T // 20) if minShift is None else minShift
    if T <= 2 * minShift:
        raise ValueError('Error: series too short for a circular-shift null (T = {}, minShift = {}).'
                         .format(T, minShift))
    rng = np.random.default_rng(seed)
    obs = group_cut(S, groupA, groupB, k=k, **kw)[stat]
    surr = np.empty(nSurr)
    gB = list(groupB)
    for r in range(nSurr):
        Sr = S.copy()
        Sr[gB] = np.roll(S[gB], rng.integers(minShift, T - minShift), axis=1)
        surr[r] = group_cut(Sr, groupA, groupB, k=k, **kw)[stat]
    sd = surr.std(ddof=1)
    z = (obs - surr.mean()) / sd if sd > 0 else np.inf
    p = (1 + np.sum(surr >= obs)) / (1 + nSurr)
    return obs, surr, float(z), float(p)


def pairwise_matrix(S, stat='phi2', k=2, alphabet=None, lag=0, debias=True):
    """Matrix of pairwise cut statistics between single agents.

    With ``debias=True`` the statistic of a time-shifted copy (shift of
    ``T // 2``) is subtracted, removing most of the finite-sample bias.
    """
    S = np.asarray(S)
    N, T = S.shape
    alphabet = int(S.max()) + 1 if alphabet is None else alphabet

    def codes(row):
        u, inv = np.unique(words(row, k, alphabet), return_inverse=True)
        return inv.ravel(), u.size

    plain = [codes(row) for row in S]                               # each agent's words, once
    rolled = [codes(np.roll(row, T // 2)) for row in S] if debias else None

    def value(a, b):
        (ia, na), (ib, nb) = a, b
        if lag > 0:
            ia, ib = ia[:-lag], ib[lag:]
        elif lag < 0:
            ia, ib = ia[-lag:], ib[:lag]
        P = np.bincount(ia * nb + ib, minlength=na * nb).reshape(na, nb).astype(float)
        return cut_stats(P / P.sum())[stat]

    M = np.zeros((N, N))
    for i, j in itertools.permutations(range(N), 2) if lag else itertools.combinations(range(N), 2):
        v = value(plain[i], plain[j])
        if debias:
            v -= value(plain[i], rolled[j])
        M[i, j] = v
        if not lag:
            M[j, i] = v
    return M


def coalition_tree(M, method='average'):
    """Hierarchical clustering of agents from a pairwise coupling matrix.

    Parameters
    ----------
    M : numpy ndarray
        Symmetric non-negative coupling (e.g. `pairwise_matrix`).
    method : str, optional
        scipy linkage method (default : 'average').

    Returns
    -------
    numpy ndarray
        scipy linkage matrix; leaves merged first are the strongest coalition.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    D = np.max(M) - M
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(D, checks=False), method=method)


def best_bipartition(S, agents=None, k=1, alphabet=None, stat='phi2', nSurr=4, seed=0):
    """Agent bipartition with the weakest coupling (the natural TTN split).

    The plug-in statistic grows with the number of observed words on each
    side (roughly ``(r_A - 1)(r_B - 1) / n`` for phi2), so cuts of different
    sizes are only comparable after subtracting the mean of `nSurr`
    circular-shift surrogates of group B. Exhaustive over bipartitions, so
    meant for small groups.

    Returns
    -------
    list of tuple
        ``(debiased value, groupA, groupB)`` sorted increasingly.
    """
    S = np.asarray(S)
    T = S.shape[1]
    agents = list(range(S.shape[0])) if agents is None else list(agents)
    first, rest = agents[0], agents[1:]
    rng = np.random.default_rng(seed)
    shifts = rng.integers(T // 10, T - T // 10, size=nSurr)
    out = []
    for r in range(0, len(rest)):
        for comb in itertools.combinations(rest, r):
            A = [first] + list(comb)
            B = [a for a in agents if a not in A]
            v = group_cut(S, A, B, k, alphabet)[stat]
            if nSurr:
                bias = 0.0
                for sh in shifts:
                    Sr = S.copy()
                    Sr[B] = np.roll(S[B], sh, axis=1)
                    bias += group_cut(Sr, A, B, k, alphabet)[stat]
                v -= bias / nSurr
            out.append((v, A, B))
    return sorted(out, key=lambda t: t[0])


def coarse_grain(X, level):
    """Block-average series over non-overlapping blocks of ``2**level`` steps."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    b = 2 ** level
    T = (X.shape[1] // b) * b
    return X[:, :T].reshape(X.shape[0], -1, b).mean(axis=-1)


def scale_profile(X, groupA, groupB, levels, nBins=3, k=2, stat='phi2', method='quantile',
                  nSurr=0, seed=0):
    """Factorization defect as a function of the coarse-graining scale.

    Returns
    -------
    numpy ndarray
        Statistic per level (minus the surrogate mean if ``nSurr > 0``).
    numpy ndarray
        z-scores per level (NaN if ``nSurr == 0``).
    """
    vals, zs = [], []
    for lv in levels:
        S = discretize(coarse_grain(X, lv), nBins, method)
        if nSurr > 0:
            obs, surr, z, _ = shift_null(S, groupA, groupB, stat, k, nSurr, seed=seed)
            vals.append(obs - surr.mean())
            zs.append(z)
        else:
            vals.append(group_cut(S, groupA, groupB, k)[stat])
            zs.append(np.nan)
    return np.array(vals), np.array(zs)


def gaussian_cut(XA, XB, k=2, lag=0, ridge=1e-8):
    """Linear (Gaussian) analogue of the cut statistics, no discretization.

    Delay-embeds both groups with windows of length `k` and returns the
    canonical correlations between the embeddings and the Gaussian mutual
    information ``-0.5 * sum log(1 - rho^2)``.
    """
    XA = np.atleast_2d(np.asarray(XA, dtype=float))
    XB = np.atleast_2d(np.asarray(XB, dtype=float))
    T = XA.shape[1] - k + 1

    def embed(X):
        return np.concatenate([X[:, j:j + T] for j in range(k)], axis=0).T

    A, B = embed(XA), embed(XB)
    if lag > 0:
        A, B = A[:-lag], B[lag:]
    elif lag < 0:
        A, B = A[-lag:], B[:lag]
    A = A - A.mean(0)
    B = B - B.mean(0)

    def inv_sqrt(C):
        w, V = np.linalg.eigh(C + ridge * np.trace(C) / C.shape[0] * np.eye(C.shape[0]))
        return V @ np.diag(w ** -0.5) @ V.T

    n = A.shape[0]
    K = inv_sqrt(A.T @ A / n) @ (A.T @ B / n) @ inv_sqrt(B.T @ B / n)
    rho = np.clip(np.linalg.svd(K, compute_uv=False), 0.0, 1.0 - 1e-12)
    return {'cancorr': rho, 'gauss_mi': float(-0.5 * np.sum(np.log(1.0 - rho ** 2)))}
