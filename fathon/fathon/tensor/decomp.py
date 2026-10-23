#    decomp.py - minimal tensor decompositions (Tucker, TT, CP)
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

"""Small, dependency-free tensor decompositions used by the estimators."""

import numpy as np


def unfold(X, mode):
    """Mode-`mode` matricisation, shape ``(X.shape[mode], -1)``."""
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def fold(M, mode, shape):
    """Inverse of `unfold`."""
    full = (shape[mode],) + tuple(s for k, s in enumerate(shape) if k != mode)
    return np.moveaxis(M.reshape(full), 0, mode)


def mode_product(X, U, mode):
    """Multiply tensor `X` by matrix `U` along `mode` (``U @ X_(mode)``)."""
    shape = list(X.shape)
    shape[mode] = U.shape[0]
    return fold(U @ unfold(X, mode), mode, shape)


def _rank_from_tol(s, tol, max_rank):
    if s.size == 0 or s[0] == 0:
        return 1
    r = int(np.sum(s > tol * s[0])) if tol is not None else s.size
    if max_rank is not None:
        r = min(r, max_rank)
    return max(r, 1)


def tucker(X, ranks=None, tol=None, n_iter=0):
    """Truncated HOSVD, optionally refined by HOOI.

    Parameters
    ----------
    X : numpy ndarray
        Tensor.
    ranks : iterable of int, optional
        Multilinear ranks; inferred from `tol` if None.
    tol : float, optional
        Relative singular value threshold per mode (default : keep all).
    n_iter : int, optional
        HOOI sweeps after the HOSVD initialisation (default : 0).

    Returns
    -------
    numpy ndarray
        Core tensor.
    list of numpy ndarray
        Factor matrices with orthonormal columns.
    list of numpy ndarray
        Mode-wise singular values of `X`.
    """
    X = np.asarray(X, dtype=float)
    factors, spectra = [], []
    for m in range(X.ndim):
        U, s, _ = np.linalg.svd(unfold(X, m), full_matrices=False)
        r = ranks[m] if ranks is not None else _rank_from_tol(s, tol, None)
        factors.append(U[:, :r])
        spectra.append(s)
    for _ in range(n_iter):
        for m in range(X.ndim):
            Y = X
            for k in range(X.ndim):
                if k != m:
                    Y = mode_product(Y, factors[k].T, k)
            U, _, _ = np.linalg.svd(unfold(Y, m), full_matrices=False)
            factors[m] = U[:, :factors[m].shape[1]]
    core = X
    for m in range(X.ndim):
        core = mode_product(core, factors[m].T, m)
    return core, factors, spectra


def tucker_to_full(core, factors):
    """Reconstruct a tensor from its Tucker form."""
    X = core
    for m, U in enumerate(factors):
        X = mode_product(X, U, m)
    return X


def tt_svd(X, tol=1e-10, max_rank=None):
    """Tensor-train (MPS) decomposition by sequential truncated SVDs.

    Parameters
    ----------
    X : numpy ndarray
        Tensor of order d.
    tol : float, optional
        Relative singular value threshold at each cut (default : 1e-10).
    max_rank : int, optional
        Maximum bond dimension.

    Returns
    -------
    list of numpy ndarray
        Cores of shape ``(r_{k-1}, n_k, r_k)``.
    list of numpy ndarray
        Singular values at each of the ``d - 1`` cuts (before truncation).
    """
    X = np.asarray(X, dtype=float)
    shape = X.shape
    cores, spectra = [], []
    r = 1
    M = X.reshape(1, -1)
    for k in range(len(shape) - 1):
        M = M.reshape(r * shape[k], -1)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        spectra.append(s)
        rk = _rank_from_tol(s, tol, max_rank)
        cores.append(U[:, :rk].reshape(r, shape[k], rk))
        M = s[:rk, None] * Vt[:rk]
        r = rk
    cores.append(M.reshape(r, shape[-1], 1))
    return cores, spectra


def tt_to_full(cores):
    """Contract TT cores into the full tensor."""
    X = cores[0]
    for G in cores[1:]:
        X = np.tensordot(X, G, axes=([-1], [0]))
    return X.reshape(X.shape[1:-1])


def tt_ranks(X, tol=1e-10):
    """TT ranks of the exact sequential unfoldings (no error propagation).

    Same definition as ``t1_replica_rank.tt_ranks`` in
    ``multifractal_tensor_networks/code``.
    """
    X = np.asarray(X, dtype=float)
    out = []
    for k in range(1, X.ndim):
        s = np.linalg.svd(X.reshape(int(np.prod(X.shape[:k])), -1), compute_uv=False)
        out.append(_rank_from_tol(s, tol, None))
    return out


def cut_spectra(X):
    """Normalised squared singular values at every sequential cut.

    Returns a list of probability vectors ``p_k = s_k^2 / sum s_k^2``; their
    entropies are the (classical) "entanglement" entropies of the cuts.
    """
    X = np.asarray(X, dtype=float)
    out = []
    for k in range(1, X.ndim):
        s = np.linalg.svd(X.reshape(int(np.prod(X.shape[:k])), -1), compute_uv=False)
        p = s ** 2
        out.append(p / p.sum() if p.sum() > 0 else p)
    return out


def renyi_entropy(p, alpha=2.0):
    """Renyi entropy of a probability vector (alpha = 1 gives Shannon)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-12:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p ** alpha)) / (1.0 - alpha))


def cp_als(X, rank, n_iter=200, tol=1e-9, seed=0, nonneg=False):
    """CANDECOMP/PARAFAC by alternating least squares.

    Parameters
    ----------
    X : numpy ndarray
        Tensor.
    rank : int
        Number of rank-one components.
    n_iter : int, optional
        Maximum number of sweeps (default : 200).
    tol : float, optional
        Stop when the relative fit changes less than `tol` (default : 1e-9).
    seed : int, optional
        Seed of the random initialisation (default : 0).
    nonneg : bool, optional
        Clip factors at zero after each update (default : False).

    Returns
    -------
    numpy ndarray
        Component weights, shape ``(rank,)``, sorted decreasingly.
    list of numpy ndarray
        Unit-norm factor matrices, one per mode, shape ``(X.shape[m], rank)``.
    float
        Relative reconstruction error.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    # HOSVD initialisation is deterministic and usually better than random
    factors = []
    for m in range(X.ndim):
        U, _, _ = np.linalg.svd(unfold(X, m), full_matrices=False)
        A = np.zeros((X.shape[m], rank))
        k = min(rank, U.shape[1])
        A[:, :k] = U[:, :k]
        if k < rank:
            A[:, k:] = rng.standard_normal((X.shape[m], rank - k))
        factors.append(np.abs(A) if nonneg else A)
    normX = np.linalg.norm(X)
    prev = np.inf
    err = np.linalg.norm(X - cp_to_full(np.ones(rank), factors)) / normX
    for _ in range(n_iter):
        for m in range(X.ndim):
            others = [factors[k] for k in range(X.ndim) if k != m]
            # Khatri-Rao product matching the C-order unfolding
            kr = others[0]
            for A in others[1:]:
                kr = (kr[:, None, :] * A[None, :, :]).reshape(-1, rank)
            G = np.ones((rank, rank))
            for A in others:
                G *= A.T @ A
            factors[m] = unfold(X, m) @ kr @ np.linalg.pinv(G)
            if nonneg:
                factors[m] = np.clip(factors[m], 0.0, None)
        err = np.linalg.norm(X - cp_to_full(np.ones(rank), factors)) / normX
        if abs(prev - err) < tol:
            break
        prev = err
    norms = np.ones(rank)
    for m in range(X.ndim):
        n = np.linalg.norm(factors[m], axis=0)
        n[n == 0] = 1.0
        factors[m] = factors[m] / n
        norms *= n
    order = np.argsort(norms)[::-1]
    return norms[order], [A[:, order] for A in factors], float(err)


def cp_to_full(weights, factors):
    """Reconstruct a tensor from CP weights and factors."""
    rank = len(weights)
    X = np.zeros(tuple(A.shape[0] for A in factors))
    for r in range(rank):
        comp = factors[0][:, r] * weights[r]
        for A in factors[1:]:
            comp = np.multiply.outer(comp, A[:, r])
        X += comp
    return X


def kron_rearrange(M, shapeA, shapeB):
    """Van Loan-Pitsianis rearrangement.

    For ``M`` of shape ``(mA * mB, nA * nB)`` with rows indexed ``(a, b)`` and
    columns ``(a', b')``, returns ``R`` of shape ``(mA * nA, mB * nB)`` with
    ``R[(a, a'), (b, b')] = M[(a, b), (a', b')]``. ``M = A kron B`` exactly
    when ``R`` has rank one.
    """
    mA, nA = shapeA
    mB, nB = shapeB
    T = np.asarray(M).reshape(mA, mB, nA, nB)
    return T.transpose(0, 2, 1, 3).reshape(mA * nA, mB * nB)


def nearest_kron(M, shapeA, shapeB):
    """Best Frobenius approximation ``M ~ A kron B`` and the singular values of the rearrangement."""
    R = kron_rearrange(M, shapeA, shapeB)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    A = (np.sqrt(s[0]) * U[:, 0]).reshape(shapeA)
    B = (np.sqrt(s[0]) * Vt[0]).reshape(shapeB)
    return A, B, s
