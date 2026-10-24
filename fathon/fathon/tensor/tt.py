#    tt.py - tensor-train / quantized tensor-train toolbox and TT-cross
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

"""Tensor trains (TT / MPS), quantized tensor trains (QTT) and TT-cross.

A TT is a list of cores ``G_k`` of shape ``(r_{k-1}, n_k, r_k)`` with
``r_0 = r_d = 1``, representing ``A[i_1..i_d] = G_1[:, i_1, :] ... G_d[:, i_d, :]``.
Cores are numpy arrays (float64 by default; float32 is kept if given) or
torch tensors on any device; every routine of the arithmetic layer works with
either and keeps the backend, dtype and device of its input.

Contents

* arithmetic: `tt_from_dense` (TT-SVD), `tt_to_dense`, `tt_ranks`,
  `tt_orthogonalize` (QR sweeps), `tt_round` (Oseledets 2011, Alg. 2),
  `tt_add`, `tt_scale`, `tt_hadamard`, `tt_dot`, `tt_norm`, `tt_eval`,
  `tt_contract_vectors`, `tt_sum_samples`;
* QTT: `qtt_from_vector`, `qtt_coarse_grain`, `qtt_interleave`;
* cross approximation: `maxvol` (Goreinov et al. 2010) and `tt_cross`
  (two-site DMRG-type cross, Savostyanov-Oseledets 2011 / Savostyanov 2014);
* replica (moment) tensors without dense ``2**(n q)`` arrays: `replica_tt`,
  `cross_replica_tt`, `level_cut_spectra`, `level_cut_ranks`, and the exact
  law-level MPS of a scalar random cascade (`cascade_moment_tt`, REPORT.md
  Theorem A) with a sampler `cascade_sample`.

The replica conventions follow `fathon.tensor.replica`: a measure on ``2**n``
dyadic cells is a QTT with the most significant (coarsest) bit first, and the
interleaved q-replica tensor has one site per level with letter
``c = sum_a bit_a 2**a`` (replica ``a`` is bit ``a``).
"""

import math
import numpy as np
from . import decomp

try:  # torch is optional for this module
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------
# backend helpers (numpy / torch)
# ---------------------------------------------------------------------------

def _is_torch(x):
    return torch is not None and isinstance(x, torch.Tensor)


class _NP:
    @staticmethod
    def perm(x, p):
        return np.transpose(x, p)

    tensordot = staticmethod(np.tensordot)
    einsum = staticmethod(np.einsum)

    @staticmethod
    def qr(A):
        return np.linalg.qr(A, mode='reduced')

    @staticmethod
    def svd(A):
        try:
            return np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:  # rare LAPACK gesdd failure: fall back to the Gram route
            w, V = np.linalg.eigh(A.T @ A)
            order = np.argsort(w)[::-1]
            s = np.sqrt(np.clip(w[order], 0, None))
            V = V[:, order]
            U = A @ V / np.where(s > 0, s, 1.0)
            return U, s, V.T

    @staticmethod
    def zeros(shape, like):
        return np.zeros(shape, dtype=like.dtype)

    @staticmethod
    def cat(xs, axis):
        return np.concatenate(xs, axis=axis)

    @staticmethod
    def norm(x):
        return float(np.linalg.norm(x))

    @staticmethod
    def asarray(v, like):
        return np.asarray(v, dtype=like.dtype)

    @staticmethod
    def numpy(x):
        return np.asarray(x)


class _TH:
    @staticmethod
    def perm(x, p):
        return x.permute(*p)

    @staticmethod
    def tensordot(a, b, axes):
        return torch.tensordot(a, b, dims=axes)

    @staticmethod
    def einsum(eq, *ops):
        return torch.einsum(eq, *ops)

    @staticmethod
    def qr(A):
        return torch.linalg.qr(A, mode='reduced')

    @staticmethod
    def svd(A):
        return torch.linalg.svd(A, full_matrices=False)

    @staticmethod
    def zeros(shape, like):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)

    @staticmethod
    def cat(xs, axis):
        return torch.cat(xs, dim=axis)

    @staticmethod
    def norm(x):
        return float(torch.linalg.norm(x))

    @staticmethod
    def asarray(v, like):
        return torch.as_tensor(np.asarray(v) if not _is_torch(v) else v, dtype=like.dtype, device=like.device)

    @staticmethod
    def numpy(x):
        return x.detach().cpu().numpy()


def _xp(x):
    return _TH if _is_torch(x) else _NP


def _prep(cores):
    """Validate a core list; cast numpy integer / float16 cores to float64."""
    out = []
    for G in cores:
        if not _is_torch(G):
            G = np.asarray(G)
            if G.dtype not in (np.float64, np.float32, np.complex128, np.complex64):
                G = G.astype(np.float64)
        if G.ndim != 3:
            raise ValueError('Error: TT cores must have shape (r_prev, n, r_next), got {}.'.format(tuple(G.shape)))
        out.append(G)
    for a, b in zip(out[:-1], out[1:]):
        if a.shape[2] != b.shape[0]:
            raise ValueError('Error: inconsistent TT ranks {} and {}.'.format(tuple(a.shape), tuple(b.shape)))
    if out and (out[0].shape[0] != 1 or out[-1].shape[2] != 1):
        raise ValueError('Error: boundary TT ranks must be 1.')
    return out


def _trunc_rank(s, delta, max_rank=None):
    """Smallest r with ``sqrt(sum_{j >= r} s_j^2) <= delta`` (at least 1)."""
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 1
    tail = np.sqrt(np.cumsum((s ** 2)[::-1])[::-1])  # tail[j] = ||s[j:]||
    ok = np.nonzero(tail <= delta)[0]
    r = int(ok[0]) if ok.size else s.size
    r = max(r, 1)
    if max_rank is not None:
        r = min(r, int(max_rank))
    return r


# ---------------------------------------------------------------------------
# core arithmetic
# ---------------------------------------------------------------------------

def tt_from_dense(X, eps=1e-14, max_rank=None, tol=None):
    """TT-SVD of a dense tensor.

    Implements Oseledets (2011), Algorithm 1 (TT-SVD): sequential truncated
    SVDs of the unfoldings with per-cut Frobenius threshold
    ``delta = eps ||X||_F / sqrt(d - 1)``, so that the result satisfies
    ``||X - TT||_F <= eps ||X||_F``. With ``tol`` given instead, the relative
    singular-value threshold of `decomp.tt_svd` is used.

    Parameters
    ----------
    X : numpy ndarray or torch.Tensor
        Dense tensor of order d.
    eps : float, optional
        Relative Frobenius accuracy (default : 1e-14).
    max_rank : int, optional
        Maximum bond dimension (overrides `eps` where it binds).
    tol : float, optional
        Use `decomp.tt_svd` with this relative singular-value threshold.

    Returns
    -------
    list
        TT cores of shape ``(r_{k-1}, n_k, r_k)``.
    """
    if tol is not None and not _is_torch(X):
        return decomp.tt_svd(np.asarray(X, dtype=float), tol=tol, max_rank=max_rank)[0]
    if not _is_torch(X):
        X = np.asarray(X)
        if X.dtype not in (np.float64, np.float32):
            X = X.astype(np.float64)
    xp = _xp(X)
    shape = tuple(X.shape)
    d = len(shape)
    if d == 1:
        return [X.reshape(1, shape[0], 1)]
    delta = eps * xp.norm(X) / math.sqrt(d - 1)
    cores, r = [], 1
    M = X.reshape(1, -1)
    for k in range(d - 1):
        M = M.reshape(r * shape[k], -1)
        U, s, Vt = xp.svd(M)
        if tol is not None:
            s_np = xp.numpy(s)
            rk = decomp._rank_from_tol(s_np, tol, max_rank)
        else:
            rk = _trunc_rank(xp.numpy(s), delta, max_rank)
        cores.append(U[:, :rk].reshape(r, shape[k], rk))
        M = s[:rk, None] * Vt[:rk]
        r = rk
    cores.append(M.reshape(r, shape[-1], 1))
    return cores


def tt_to_dense(cores):
    """Contract TT cores into the dense tensor of shape ``(n_1, ..., n_d)``."""
    cores = _prep(cores)
    xp = _xp(cores[0])
    X = cores[0]
    for G in cores[1:]:
        X = xp.tensordot(X, G, ([X.ndim - 1], [0]))
    return X.reshape(tuple(X.shape[1:-1]))


def tt_ranks(cores):
    """Internal bond dimensions ``[r_1, ..., r_{d-1}]``."""
    return [int(G.shape[2]) for G in cores[:-1]]


def tt_shape(cores):
    """Mode sizes ``(n_1, ..., n_d)``."""
    return tuple(int(G.shape[1]) for G in cores)


def _orth_left(cores, upto=None):
    """Left-orthogonalise cores ``0 .. upto-1`` by Householder QR sweeps."""
    xp = _xp(cores[0])
    cores = list(cores)
    upto = len(cores) - 1 if upto is None else upto
    for k in range(upto):
        r0, n, r1 = cores[k].shape
        Q, R = xp.qr(cores[k].reshape(r0 * n, r1))
        cores[k] = Q.reshape(r0, n, Q.shape[1])
        cores[k + 1] = xp.tensordot(R, cores[k + 1], ([1], [0]))
    return cores


def _orth_right(cores, downto=0):
    """Right-orthogonalise cores ``d-1 .. downto+1`` by QR of transposed unfoldings."""
    xp = _xp(cores[0])
    cores = list(cores)
    for k in range(len(cores) - 1, downto, -1):
        r0, n, r1 = cores[k].shape
        Q, R = xp.qr(cores[k].reshape(r0, n * r1).T)
        cores[k] = Q.T.reshape(Q.shape[1], n, r1)
        cores[k - 1] = xp.tensordot(cores[k - 1], R.T, ([2], [0]))
    return cores


def tt_orthogonalize(cores, direction='left'):
    """Orthogonalise a TT by QR sweeps (Oseledets 2011, Sec. 3; Holtz-Rohwedder-Schneider 2012).

    Parameters
    ----------
    cores : list
        TT cores.
    direction : {'left', 'right'}, optional
        'left': cores ``1..d-1`` get orthonormal left unfoldings
        (``G^T G = I`` for ``G = core.reshape(r0 n, r1)``) and the norm sits in
        the last core; 'right': cores ``2..d`` get orthonormal right unfoldings and
        the norm sits in the first core (default : 'left').

    Returns
    -------
    list
        Cores of the same tensor; ranks may decrease to ``min(r0 n, r1)``.
    """
    cores = _prep(cores)
    if direction == 'left':
        return _orth_left(cores)
    if direction == 'right':
        return _orth_right(cores)
    raise ValueError("Error: direction must be 'left' or 'right'.")


def tt_round(cores, eps=1e-10, max_rank=None):
    """TT-rounding (recompression) of Oseledets (2011), Algorithm 2.

    Right-to-left QR orthogonalisation followed by a left-to-right sweep of
    truncated SVDs with per-cut threshold ``delta = eps ||A||_F / sqrt(d - 1)``.
    Without `max_rank` the result is quasi-optimal and satisfies
    ``||A - round(A)||_F <= eps ||A||_F``. Cost ``O(d n r^3)``.

    Parameters
    ----------
    cores : list
        TT cores.
    eps : float, optional
        Relative Frobenius accuracy (default : 1e-10).
    max_rank : int, optional
        Rank cap; where it binds the error bound no longer holds.

    Returns
    -------
    list
        Rounded TT cores.
    """
    cores = _prep(cores)
    d = len(cores)
    if d == 1:
        return [G for G in cores]
    xp = _xp(cores[0])
    cores = _orth_right(cores)
    delta = eps * xp.norm(cores[0]) / math.sqrt(d - 1)
    for k in range(d - 1):
        r0, n, r1 = cores[k].shape
        U, s, Vt = xp.svd(cores[k].reshape(r0 * n, r1))
        rk = _trunc_rank(xp.numpy(s), delta, max_rank)
        cores[k] = U[:, :rk].reshape(r0, n, rk)
        cores[k + 1] = xp.tensordot(s[:rk, None] * Vt[:rk], cores[k + 1], ([1], [0]))
    return cores


def tt_scale(cores, alpha):
    """Multiply a TT by a scalar (applied to the first core)."""
    cores = list(_prep(cores))
    cores[0] = cores[0] * alpha
    return cores


def tt_add(*tts, weights=None):
    """Sum of TTs with block-diagonal cores (Oseledets 2011, Sec. 4.1); ranks add.

    Parameters
    ----------
    *tts : lists of cores
        TTs of identical mode sizes.
    weights : sequence of float, optional
        Coefficients ``sum_j w_j A_j`` (default : all 1).

    Returns
    -------
    list
        Cores of the sum (not rounded; call `tt_round`).
    """
    tts = [_prep(t) for t in tts]
    if weights is not None:
        tts = [tt_scale(t, w) for t, w in zip(tts, weights)]
    if len(tts) == 1:
        return list(tts[0])
    d = len(tts[0])
    if any(len(t) != d or tt_shape(t) != tt_shape(tts[0]) for t in tts):
        raise ValueError('Error: tt_add needs TTs of identical shape.')
    xp = _xp(tts[0][0])
    if d == 1:
        G = tts[0][0]
        for t in tts[1:]:
            G = G + t[0]
        return [G]
    out = [xp.cat([t[0] for t in tts], 2)]
    for k in range(1, d - 1):
        R0 = sum(int(t[k].shape[0]) for t in tts)
        R1 = sum(int(t[k].shape[2]) for t in tts)
        G = xp.zeros((R0, tts[0][k].shape[1], R1), tts[0][k])
        a = b = 0
        for t in tts:
            r0, _, r1 = t[k].shape
            G[a:a + r0, :, b:b + r1] = t[k]
            a, b = a + r0, b + r1
        out.append(G)
    out.append(xp.cat([t[-1] for t in tts], 0))
    return out


def tt_hadamard(a, b):
    """Elementwise product of two TTs (slice-wise Kronecker products; ranks multiply)."""
    a, b = _prep(a), _prep(b)
    xp = _xp(a[0])
    out = []
    for G, H in zip(a, b):
        r0, n, r1 = G.shape
        s0, _, s1 = H.shape
        out.append(xp.einsum('inj,knl->iknjl', G, H).reshape(r0 * s0, n, r1 * s1))
    return out


def tt_dot(a, b):
    """Inner product ``<A, B>`` by left-to-right contraction, ``O(d n r^3)``, no dense tensor."""
    a, b = _prep(a), _prep(b)
    xp = _xp(a[0])
    Z = None
    for G, H in zip(a, b):
        if Z is None:
            W = G  # (1, n, r)
        else:
            W = xp.tensordot(Z, G, ([0], [0]))  # (rb, n, ra')
        # Z'[ra', rb'] = sum_{rb, n} W[rb, n, ra'] H[rb, n, rb']
        Z = xp.tensordot(W, H, ([0, 1], [0, 1]))
    return float(Z.reshape(-1)[0])


def tt_norm(cores):
    """Frobenius norm via a left QR sweep (norm of the last core).

    Numerically safer than ``sqrt(tt_dot(A, A))`` for differences of nearly
    equal tensors, where the Gram route loses half the digits.
    """
    cores = _prep(cores)
    xp = _xp(cores[0])
    return xp.norm(_orth_left(cores)[-1])


def tt_eval(cores, idx):
    """Entries ``A[idx[b, 0], ..., idx[b, d-1]]`` for a batch of multi-indices.

    Parameters
    ----------
    cores : list
        TT cores.
    idx : array_like of int
        Shape ``(B, d)``.

    Returns
    -------
    numpy ndarray or torch.Tensor
        Values, shape ``(B,)``; cost ``O(B d r^2)``.
    """
    cores = _prep(cores)
    xp = _xp(cores[0])
    if _is_torch(cores[0]):
        idx = torch.as_tensor(np.asarray(idx) if not _is_torch(idx) else idx, device=cores[0].device).long()
    else:
        idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx.reshape(1, -1)
    V = cores[0][0][idx[:, 0], :]
    for k in range(1, len(cores)):
        V = xp.einsum('br,rbs->bs', V, cores[k][:, idx[:, k], :])
    return V[:, 0]


def tt_contract_vectors(cores, vectors):
    """Contract some modes with vectors (marginalise / weight), staying in TT format.

    Parameters
    ----------
    cores : list
        TT cores.
    vectors : dict
        ``{mode: vector}`` with ``len(vector) == n_mode``.

    Returns
    -------
    list or float
        Cores of the remaining modes, or a scalar if every mode is contracted.
    """
    cores = _prep(cores)
    xp = _xp(cores[0])
    d = len(cores)
    vecs = {int(m) % d: xp.asarray(v, cores[0]) for m, v in vectors.items()}
    out, pending = [], None
    for k, G in enumerate(cores):
        if k in vecs:
            M = xp.einsum('anb,n->ab', G, vecs[k])
            pending = M if pending is None else pending @ M
        else:
            if pending is not None:
                G = xp.tensordot(pending, G, ([1], [0]))
                pending = None
            out.append(G)
    if pending is not None:
        if not out:
            return float(pending.reshape(-1)[0])
        out[-1] = xp.tensordot(out[-1], pending, ([2], [0]))
    return out


def tt_sum_samples(tts, eps=1e-12, max_rank=None, round_at=256):
    """Sum of many TTs with periodic TT-rounding to bound the ranks.

    The TTs are accumulated in a buffer and added at once (block-diagonal
    cores, `tt_add`) whenever the summed bond dimension would exceed
    `round_at`, then recompressed with `tt_round` (Oseledets 2011, Alg. 2).
    Each rounding adds at most ``eps`` times the norm of the current partial
    sum, so for nonnegative terms (sample moment tensors) the total relative
    error is at most ``eps`` times the number of roundings.

    Parameters
    ----------
    tts : iterable of lists of cores
        Terms (a generator is fine; terms are consumed one at a time).
    eps : float, optional
        Relative accuracy of each rounding (default : 1e-12).
    max_rank : int, optional
        Rank cap of each rounding.
    round_at : int, optional
        Bond dimension that triggers a rounding (default : 256).

    Returns
    -------
    list
        Cores of the (rounded) sum.
    """
    acc, buf, width = None, [], 0
    for t in tts:
        t = _prep(t)
        r = max(tt_ranks(t) or [1])
        buf.append(t)
        width += r
        acc_r = max(tt_ranks(acc) or [1]) if acc is not None else 0
        if acc_r + width > round_at:
            acc = tt_round(tt_add(*(([acc] if acc is not None else []) + buf)), eps, max_rank)
            buf, width = [], 0
    if buf:
        acc = tt_add(*(([acc] if acc is not None else []) + buf))
    if acc is None:
        raise ValueError('Error: tt_sum_samples got no terms.')
    return tt_round(acc, eps, max_rank)


# ---------------------------------------------------------------------------
# QTT helpers
# ---------------------------------------------------------------------------

def qtt_from_vector(v, base=2, eps=1e-14, max_rank=None):
    """Quantized TT of a vector of length ``base**n`` (Oseledets 2010; Khoromskij 2011).

    The vector is reshaped to ``(base,) * n`` with the most significant digit
    (coarsest dyadic level) first, matching `replica.interleave`, and
    decomposed by TT-SVD (`tt_from_dense`).

    Parameters
    ----------
    v : array_like or torch.Tensor
        Vector of length ``base**n``.
    base : int, optional
        Quantization base (default : 2).
    eps : float, optional
        Relative Frobenius accuracy (default : 1e-14).
    max_rank : int, optional
        Rank cap.

    Returns
    -------
    list
        ``n`` cores of shape ``(r_{k-1}, base, r_k)``.
    """
    L = int(v.shape[-1]) if hasattr(v, 'shape') else len(v)
    n = int(round(math.log(L, base)))
    if base ** n != L:
        raise ValueError('Error: length {} is not a power of {}.'.format(L, base))
    if not _is_torch(v):
        v = np.asarray(v, dtype=np.float64) if not np.issubdtype(np.asarray(v).dtype, np.floating) else np.asarray(v)
    if n == 0:
        return [v.reshape(1, 1, 1)]
    return tt_from_dense(v.reshape((base,) * n), eps=eps, max_rank=max_rank)


def qtt_coarse_grain(cores, n_keep, average=True):
    """Coarse-grain a (Q)TT measure to its first `n_keep` levels without decompressing.

    The trailing ``d - n_keep`` modes are contracted with ``[1/n_k, ...]``
    (``average=True``) or ``[1, ...]`` (``average=False``, the projective
    coarse-grained measure ``mu_l(w) = sum_c mu_{l+1}(w c)``). For an
    interleaved replica TT (site dimension ``2**q``) this gives the replica
    tensor of the coarse-grained measure (sum over all replicas' children).

    Parameters
    ----------
    cores : list
        (Q)TT cores, coarsest level first.
    n_keep : int
        Number of leading levels kept (``0`` returns the total, a scalar).
    average : bool, optional
        Average (True) or sum (False) over the fine levels (default : True).

    Returns
    -------
    list or float
        Cores of the coarse-grained tensor.
    """
    cores = _prep(cores)
    d = len(cores)
    if not 0 <= n_keep <= d:
        raise ValueError('Error: n_keep must lie in [0, {}].'.format(d))
    vecs = {}
    for k in range(n_keep, d):
        m = int(cores[k].shape[1])
        vecs[k] = np.full(m, 1.0 / m if average else 1.0)
    if not vecs:
        return list(cores)
    return tt_contract_vectors(cores, vecs)


def qtt_interleave(qtts):
    """Interleaved TT of the Kronecker (outer) product of q QTTs.

    Site ``k`` of the result has dimension ``prod_a n_k^(a)`` and letter
    ``c = sum_a c_a prod_{b < a} n_k^(b)``, i.e. for binary QTTs
    ``c = sum_a bit_a 2**a`` with replica ``a`` as bit ``a``, the convention of
    `replica.interleave`. The site cores are slice-wise Kronecker products, so
    the bond dimensions are the products of the factors' ranks (exact, no
    truncation).

    Parameters
    ----------
    qtts : list of lists of cores
        ``q`` TTs with the same number of sites.

    Returns
    -------
    list
        Interleaved cores.
    """
    qtts = [_prep(t) for t in qtts]
    d = len(qtts[0])
    if any(len(t) != d for t in qtts):
        raise ValueError('Error: qtt_interleave needs QTTs with the same number of sites.')
    xp = _xp(qtts[0][0])
    out = []
    for k in range(d):
        G = qtts[-1][k]  # highest replica = most significant digit of the letter
        for t in reversed(qtts[:-1]):
            H = t[k]
            r0, nx, r1 = G.shape
            s0, ny, s1 = H.shape
            G = xp.einsum('ixj,kyl->ikxyjl', G, H).reshape(r0 * s0, nx * ny, r1 * s1)
        out.append(G)
    return out


# ---------------------------------------------------------------------------
# maxvol and TT-cross
# ---------------------------------------------------------------------------

def _lu_pivots(A):
    """Row pivots of Gaussian elimination with partial pivoting (column by column)."""
    B = np.array(A, dtype=float, copy=True)
    N, r = B.shape
    piv = np.empty(r, dtype=np.int64)
    used = np.zeros(N, dtype=bool)
    for j in range(r):
        col = np.abs(B[:, j])
        col[used] = -1.0
        i = int(np.argmax(col))
        piv[j] = i
        used[i] = True
        if B[i, j] != 0:
            B -= np.outer(B[:, j] / B[i, j], B[i])
    return piv


def maxvol(A, tol=1.05, max_iter=500):
    """Quasi-maximal-volume ``r x r`` submatrix of a tall ``N x r`` matrix.

    Implements the maxvol algorithm of Goreinov, Oseledets, Savostyanov,
    Tyrtyshnikov and Zamarashkin, "How to find a good submatrix" (2010):
    start from the pivots of Gaussian elimination with partial pivoting,
    form ``B = A A[piv]^{-1}`` and, while some ``|B_ij| > tol``, swap row ``i``
    into position ``j`` with a rank-one (Sherman-Morrison) update of ``B``.
    On exit every entry of ``A A[piv]^{-1}`` is bounded by `tol` in modulus.

    Parameters
    ----------
    A : array_like
        Matrix of shape ``(N, r)``, ``N >= r``, full column rank.
    tol : float, optional
        Dominance threshold, ``>= 1`` (default : 1.05).
    max_iter : int, optional
        Maximum number of swaps (default : 500).

    Returns
    -------
    numpy ndarray
        Row indices ``piv`` of length ``r``.
    """
    A = np.asarray(A, dtype=float)
    N, r = A.shape
    if N <= r:
        return np.arange(N)
    piv = _lu_pivots(A)
    try:
        B = np.linalg.solve(A[piv].T, A.T).T
    except np.linalg.LinAlgError:
        return piv
    for _ in range(max_iter):
        i, j = np.unravel_index(int(np.argmax(np.abs(B))), B.shape)
        bij = B[i, j]
        if abs(bij) <= tol:
            break
        piv[j] = i
        bj = B[:, j].copy()
        bi = B[i].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / bij)
    return piv


def _superblock_index(Il, nk, nk1, Jr, d, k):
    rl, rr = Il.shape[0], Jr.shape[0]
    idx = np.empty((rl, nk, nk1, rr, d), dtype=np.int64)
    if k > 0:
        idx[..., :k] = Il[:, None, None, None, :]
    idx[..., k] = np.arange(nk)[None, :, None, None]
    idx[..., k + 1] = np.arange(nk1)[None, None, :, None]
    if k + 2 < d:
        idx[..., k + 2:] = Jr[None, None, None, :, :]
    return idx.reshape(-1, d)


def _rel_diff(a, b):
    na = tt_norm(a)
    if na == 0:
        return 0.0 if tt_norm(b) == 0 else np.inf
    return tt_norm(tt_add(a, b, weights=[1.0, -1.0])) / na


def tt_cross(f, dims, max_rank=50, eps=1e-10, n_sweeps=10, seed=0, init_rank=2, maxvol_tol=1.05,
             verbose=False):
    """TT approximation from entry evaluations by two-site (DMRG-type) cross interpolation.

    Implements the rank-adaptive two-site TT-cross of Savostyanov and
    Oseledets, "Fast adaptive interpolation of multi-dimensional arrays in
    tensor train format" (2011) and Savostyanov, "Quasioptimality of maximum-
    volume cross interpolation of tensors" (2014), building on the fixed-rank
    TT-cross of Oseledets and Tyrtyshnikov (2010):

    * nested left / right index sets ``I_k`` / ``J_k`` (random nested start of
      size `init_rank`);
    * at each bond the superblock ``A[I_{k-1} x n_k, n_{k+1} x J_{k+1}]``
      (``r n x n r`` entries) is evaluated, its truncated SVD (Frobenius
      threshold ``eps ||B|| / sqrt(d - 1)``) sets the new rank, and `maxvol`
      on the left (right) singular vectors picks the new nested pivots;
    * left-to-right and right-to-left half-sweeps alternate; each half-sweep
      yields an interpolative TT whose cores contain ``U U[piv]^{-1}``, bounded
      entrywise by `maxvol_tol`, so the representation is well conditioned.

    Iteration stops when two consecutive half-sweep TTs differ by less than
    `eps` in relative Frobenius norm (computed in TT arithmetic). An exactly
    rank-``r`` tensor is recovered to machine precision once the pivots
    capture the ranks (typically 1-2 sweeps). Evaluations per sweep:
    ``2 sum_k r_{k-1} n_k n_{k+1} r_{k+1} = O(d n^2 r^2)``.

    Parameters
    ----------
    f : callable
        Batched evaluator: ``f(idx)`` with ``idx`` an int64 array ``(B, d)``
        returns ``B`` values (numpy or torch).
    dims : sequence of int
        Mode sizes ``(n_1, ..., n_d)``.
    max_rank : int, optional
        Bond-dimension cap (default : 50).
    eps : float, optional
        Relative accuracy of the superblock truncations and of the stopping
        test (default : 1e-10).
    n_sweeps : int, optional
        Maximum number of full (left-right-left) sweeps (default : 10).
    seed : int, optional
        Seed of the random initial right index sets (default : 0).
    init_rank : int, optional
        Size of the initial index sets (default : 2).
    maxvol_tol : float, optional
        `maxvol` dominance threshold (default : 1.05).
    verbose : bool, optional
        Print progress per half-sweep.

    Returns
    -------
    list of numpy ndarray
        TT cores (float64).
    dict
        ``n_evals`` (number of entries requested from `f`), ``sweeps`` (full
        sweeps done, fractional for a stop after a half-sweep), ``rel_change``
        (history), ``ranks`` and ``converged``.
    """
    dims = [int(n) for n in dims]
    d = len(dims)
    info = {'n_evals': 0, 'sweeps': 0.0, 'rel_change': [], 'ranks': None, 'converged': False}

    def F(idx):
        v = f(idx)
        v = v.detach().cpu().numpy() if _is_torch(v) else np.asarray(v)
        info['n_evals'] += idx.shape[0]
        return v.astype(np.float64, copy=False).reshape(-1)

    if d == 1:
        idx = np.arange(dims[0])[:, None]
        info.update(converged=True, ranks=[])
        return [F(idx).reshape(1, dims[0], 1)], info

    rng = np.random.default_rng(seed)
    empty = np.zeros((1, 0), dtype=np.int64)
    # J[k]: right index set at bond k (sites k+1..d-1); I[k]: left set at bond k (sites 0..k)
    J = [None] * d
    J[d - 1] = empty
    for k in range(d - 2, -1, -1):
        cand = np.concatenate([np.repeat(np.arange(dims[k + 1]), J[k + 1].shape[0])[:, None],
                               np.tile(J[k + 1], (dims[k + 1], 1))], axis=1)
        take = rng.choice(cand.shape[0], size=min(init_rank, cand.shape[0]), replace=False)
        J[k] = cand[np.sort(take)]
    I = [None] * d

    def left(k):
        return empty if k < 0 else I[k]

    def superblock(k):
        Il, Jr = left(k - 1), J[k + 1]
        B = F(_superblock_index(Il, dims[k], dims[k + 1], Jr, d, k))
        return Il, Jr, B.reshape(Il.shape[0] * dims[k], dims[k + 1] * Jr.shape[0])

    def truncate(B):
        U, s, Vt = np.linalg.svd(B, full_matrices=False)
        if s.size == 0 or s[0] == 0:
            return U[:, :1], np.ones(1) * 0.0, Vt[:1]
        r = _trunc_rank(s, eps * np.linalg.norm(s) / math.sqrt(d - 1), max_rank)
        return U[:, :r], s[:r], Vt[:r]

    prev = None
    cores = [None] * d
    for half in range(2 * n_sweeps):
        if half % 2 == 0:  # left-to-right
            for k in range(d - 1):
                Il, Jr, B = superblock(k)
                U, s, Vt = truncate(B)
                piv = maxvol(U, maxvol_tol)
                cores[k] = np.linalg.solve(U[piv].T, U.T).T.reshape(Il.shape[0], dims[k], -1)
                I[k] = np.concatenate([Il[piv // dims[k]], (piv % dims[k])[:, None]], axis=1)
                last = ((U[piv] * s) @ Vt).reshape(len(piv), dims[k + 1], Jr.shape[0])
            cores[d - 1] = last
        else:  # right-to-left
            for k in range(d - 2, -1, -1):
                Il, Jr, B = superblock(k)
                U, s, Vt = truncate(B)
                V = Vt.T
                piv = maxvol(V, maxvol_tol)
                rr = Jr.shape[0]
                cores[k + 1] = np.linalg.solve(V[piv].T, V.T).reshape(-1, dims[k + 1], rr)
                J[k] = np.concatenate([(piv // rr)[:, None], Jr[piv % rr]], axis=1)
                first = ((U * s) @ V[piv].T).reshape(Il.shape[0], dims[k], len(piv))
            cores[0] = first
        info['sweeps'] = (half + 1) / 2.0
        cur = [c.copy() for c in cores]
        if prev is not None:
            ch = _rel_diff(cur, prev)
            info['rel_change'].append(float(ch))
            if verbose:
                print('tt_cross half-sweep {}: ranks {} change {:.2e} evals {}'.format(
                    half + 1, tt_ranks(cur), ch, info['n_evals']))
            if ch < max(eps, 1e-13):
                info['converged'] = True
                break
        prev = cur
    info['ranks'] = tt_ranks(cores)
    return cores, info


# ---------------------------------------------------------------------------
# replica (moment) tensors in QTT format
# ---------------------------------------------------------------------------

class _MomentEvaluator:
    """``E_s prod_a mu_a[s, w^a]`` at interleaved letters (batched, numpy or torch)."""

    def __init__(self, mus, chunk_elems=2 ** 24, device=None, dtype=None):
        self.q = len(mus)
        base = mus[0]
        self.torch = _is_torch(base) or device is not None
        cache = {}
        self.cols = []
        for m in mus:
            key = id(m)
            if key not in cache:
                if self.torch:
                    t = m if _is_torch(m) else torch.as_tensor(np.asarray(m))
                    t = t.to(device=device if device is not None else t.device,
                             dtype=dtype if dtype is not None else (t.dtype if t.is_floating_point() else torch.float64))
                    cache[key] = t.T.contiguous()  # (L, S)
                else:
                    a = np.asarray(m)
                    a = a.astype(dtype if dtype is not None else (a.dtype if a.dtype in (np.float32, np.float64) else np.float64))
                    cache[key] = np.ascontiguousarray(a.T)
            self.cols.append(cache[key])
        self.L, self.S = self.cols[0].shape
        self.n = int(round(math.log2(self.L)))
        if 2 ** self.n != self.L:
            raise ValueError('Error: measures need 2**n cells, got {}.'.format(self.L))
        self.rows = max(1, chunk_elems // self.S)

    def cells(self, idx):
        """Per-replica cell indices ``(q, B)`` of interleaved letters ``idx (B, n)``."""
        idx = np.asarray(idx, dtype=np.int64)
        w = np.zeros((self.q, idx.shape[0]), dtype=np.int64)
        for k in range(self.n):
            for a in range(self.q):
                w[a] |= ((idx[:, k] >> a) & 1) << (self.n - 1 - k)
        return w

    def __call__(self, idx):
        w = self.cells(idx)
        B = w.shape[1]
        out = np.empty(B)
        for a0 in range(0, B, self.rows):
            sl = slice(a0, a0 + self.rows)
            if self.torch:
                wi = torch.as_tensor(w[:, sl], device=self.cols[0].device)
                P = self.cols[0][wi[0]]
                for a in range(1, self.q):
                    P = P * self.cols[a][wi[a]]
                out[sl] = P.mean(dim=1).double().cpu().numpy()
            else:
                P = self.cols[0][w[0, sl]]
                for a in range(1, self.q):
                    P = P * self.cols[a][w[a, sl]]
                out[sl] = P.mean(axis=1)
        return out


def cross_replica_tt(mus, eps=1e-6, max_rank=64, method='cross', n_sweeps=6, seed=0, init_rank=2,
                     device=None, dtype=None, round_at=512, max_sample_rank=4096, return_info=False):
    """Cross-replica tensor ``E[mu_1(w^1) ... mu_q(w^q)]`` as an interleaved QTT, no dense arrays.

    Target: `replica.cross_replica_tensor` (shape ``(2**q,) * n``, letter
    ``c = sum_a bit_a 2**a``, replica ``a`` = agent ``a``), whose dense
    construction needs ``2**(n q)`` entries.

    Parameters
    ----------
    mus : list of array_like or torch.Tensor
        Per-agent sample measures, each ``(S, 2**n)``, samples aligned.
    eps : float, optional
        Relative accuracy (cross truncation / rounding) (default : 1e-6). For
        noisy empirical tensors choose it near the sampling noise level.
    max_rank : int, optional
        Bond-dimension cap (default : 64).
    method : {'cross', 'sum'}, optional
        'cross' (default): `tt_cross` on the entry evaluator
        ``M(c_1..c_n) = mean_s prod_a mu_a[s, w^a(c)]`` (each evaluation costs
        ``q S``; on the device of torch inputs, or of `device`).
        'sum': exact construction as ``(1/S) sum_s qtt_interleave(QTT(mu_1[s]),
        ..., QTT(mu_q[s]))`` with periodic rounding (`tt_sum_samples`). A
        sample of a cascade is QTT-incompressible (ranks up to ``2**(n/2)``), so
        each term has ranks up to ``2**(q n / 2)``; this is only feasible for
        small ``n q`` (raises if a term exceeds `max_sample_rank`).
    n_sweeps, seed, init_rank : optional
        Passed to `tt_cross`.
    device : str or torch.device, optional
        Evaluate on this torch device (default : that of torch inputs, else numpy).
    dtype : optional
        Evaluation dtype (default : float64; float32 is faster on GPUs).
    round_at : int, optional
        'sum' only: bond dimension triggering a rounding (default : 512).
    max_sample_rank : int, optional
        'sum' only: largest admissible per-sample interleaved rank.
    return_info : bool, optional
        Also return a dict (evaluation count, sweeps, timings...).

    Returns
    -------
    list
        Interleaved TT cores ``(r_{k-1}, 2**q, r_k)``.
    dict
        Only with ``return_info=True``.
    """
    q = len(mus)
    if method == 'cross':
        ev = _MomentEvaluator(mus, device=device, dtype=dtype)
        cores, info = tt_cross(ev, [2 ** q] * ev.n, max_rank=max_rank, eps=eps, n_sweeps=n_sweeps,
                               seed=seed, init_rank=init_rank)
        info['method'] = 'cross'
    elif method == 'sum':
        S = int(mus[0].shape[0])
        cache = {}

        def terms():
            for s in range(S):
                qs = []
                for m in mus:
                    key = id(m)
                    if key not in cache or cache[key][0] != s:
                        cache[key] = (s, qtt_from_vector(m[s], eps=1e-15))
                    qs.append(cache[key][1])
                r = int(np.prod([max(tt_ranks(t) or [1]) for t in qs]))
                if r > max_sample_rank:
                    raise ValueError('Error: a sample term has interleaved rank {} > max_sample_rank = {}; '
                                     "samples are QTT-incompressible, use method='cross'.".format(r, max_sample_rank))
                yield qtt_interleave(qs)

        cores = tt_scale(tt_sum_samples(terms(), eps=eps, max_rank=max_rank, round_at=round_at), 1.0 / S)
        info = {'method': 'sum', 'n_evals': 0, 'ranks': tt_ranks(cores)}
    else:
        raise ValueError("Error: method must be 'cross' or 'sum'.")
    return (cores, info) if return_info else cores


def replica_tt(mu, q, eps=1e-6, max_rank=64, method='cross', **kw):
    """Empirical q-point moment tensor ``E[mu(w^1) ... mu(w^q)]`` as an interleaved QTT.

    Target: `replica.replica_tensor` (shape ``(2**q,) * n``) without forming
    any ``2**(n q)`` array; see `cross_replica_tt` for the methods (TT-cross,
    Savostyanov-Oseledets 2011, or exact sample sums with Oseledets 2011
    rounding) and the keyword arguments.

    Parameters
    ----------
    mu : array_like or torch.Tensor
        Sample measures, shape ``(S, 2**n)``.
    q : int
        Number of replicas.
    eps, max_rank, method, **kw
        As in `cross_replica_tt`.

    Returns
    -------
    list
        Interleaved TT cores (and an info dict with ``return_info=True``).
    """
    return cross_replica_tt([mu] * q, eps=eps, max_rank=max_rank, method=method, **kw)


def level_cut_spectra(cores):
    """Singular values of every level-cut unfolding, computed from the TT.

    The TT is brought to right-canonical form by QR sweeps; a left-to-right
    sweep of SVDs of the ``(r_{k-1} n_k) x r_k`` cores then yields, at cut
    ``k``, exactly the singular values of the unfolding
    ``A[(i_1..i_k), (i_{k+1}..i_d)]`` (canonical / Vidal form; Oseledets
    2011, Sec. 3; Schollwoeck 2011). Cost ``O(d n r^3)``; replaces the dense
    ``decomp.tt_ranks`` / ``decomp.cut_spectra`` (``p = s**2 / sum s**2``).

    Parameters
    ----------
    cores : list
        TT cores.

    Returns
    -------
    list of numpy ndarray
        ``d - 1`` vectors of singular values (decreasing).
    """
    cores = _prep(cores)
    xp = _xp(cores[0])
    cores = _orth_right(cores)
    out = []
    for k in range(len(cores) - 1):
        r0, n, r1 = cores[k].shape
        U, s, Vt = xp.svd(cores[k].reshape(r0 * n, r1))
        out.append(np.asarray(xp.numpy(s), dtype=float))
        cores[k] = U.reshape(r0, n, -1)
        cores[k + 1] = xp.tensordot(s[:, None] * Vt, cores[k + 1], ([1], [0]))
    return out


def level_cut_ranks(cores, tol=1e-10):
    """Level-cut ranks: number of singular values above ``tol * s_max`` at each cut.

    Same rule as `decomp.tt_ranks` applied to the dense tensor, computed from
    the TT (`level_cut_spectra`).
    """
    return [decomp._rank_from_tol(s, tol, None) for s in level_cut_spectra(cores)]


# ---------------------------------------------------------------------------
# exact replica MPS of a scalar random cascade (REPORT.md, Theorem A, r = 1)
# ---------------------------------------------------------------------------

def _set_partitions(q):
    """Set partitions of ``range(q)`` as restricted growth strings."""
    out = []

    def rec(prefix, m):
        if len(prefix) == q:
            out.append(tuple(prefix))
            return
        for b in range(m + 2):
            rec(prefix + [b], max(m, b))

    rec([0], 0) if q > 0 else out.append(())
    return out


def _canon(labels):
    seen, out = {}, []
    for x in labels:
        out.append(seen.setdefault(x, len(seen)))
    return tuple(out)


def cascade_moment_tt(atoms, probs, q, n):
    """Exact q-point moment tensor of a scalar random cascade as an interleaved uniform MPS.

    REPORT.md Theorem A with ``r = 1``: the bond space is the set partitions
    of the q replicas (dimension Bell(q)), the core is
    ``G[pi, c, pi'] = 1[pi' = pi ^ ker c] prod_{beta in pi} E[prod_{a in beta} W_{c_a}]``,
    the left boundary is the one-block partition and the right boundary sums
    over partitions. Equals ``E[prod_a mu_n(w^a)]`` for
    ``mu_n(w) = prod_k W^{(w_1..w_{k-1})}_{w_k}`` with iid node tuples ``W``.

    Parameters
    ----------
    atoms : array_like
        Atoms of the node law, shape ``(K, b)`` (``b`` children per node).
    probs : array_like
        Atom probabilities, shape ``(K,)``.
    q : int
        Number of replicas.
    n : int
        Depth.

    Returns
    -------
    list of numpy ndarray
        ``n`` cores of shape ``(B_q, b**q, B_q)`` (boundaries of size 1).
    """
    atoms = np.asarray(atoms, dtype=float)
    probs = np.asarray(probs, dtype=float)
    b = atoms.shape[1]
    parts = _set_partitions(q)
    pos = {p: i for i, p in enumerate(parts)}
    Bq = len(parts)
    G = np.zeros((Bq, b ** q, Bq))
    for i, p in enumerate(parts):
        for c in range(b ** q):
            digits = [(c // b ** a) % b for a in range(q)]
            val = 1.0
            for blk in set(p):
                m = np.zeros(b)
                for a in range(q):
                    if p[a] == blk:
                        m[digits[a]] += 1
                val *= float(np.sum(probs * np.prod(atoms ** m[None, :], axis=1)))
            G[i, c, pos[_canon(list(zip(p, digits)))]] = val
    first = G[pos[(0,) * q]][None]
    last = G.sum(axis=2)[:, :, None]
    if n == 1:
        return [G[pos[(0,) * q]].sum(axis=1)[None, :, None]]
    return [first] + [G.copy() for _ in range(n - 2)] + [last]


def cascade_sample(atoms, probs, n, S, seed=0):
    """S independent samples of a scalar random cascade on ``b**n`` cells.

    Parameters
    ----------
    atoms : array_like
        Node-law atoms, shape ``(K, b)``.
    probs : array_like
        Atom probabilities, shape ``(K,)``.
    n : int
        Depth.
    S : int
        Number of samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy ndarray
        Measures, shape ``(S, b**n)`` (not renormalised).
    """
    atoms = np.asarray(atoms, dtype=float)
    rng = np.random.default_rng(seed)
    K, b = atoms.shape
    mu = np.ones((S, 1))
    for _ in range(n):
        tup = atoms[rng.choice(K, size=mu.shape, p=probs)]  # (S, nodes, b)
        mu = (mu[:, :, None] * tup).reshape(S, -1)
    return mu
