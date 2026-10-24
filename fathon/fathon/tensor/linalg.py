#    linalg.py - streaming tall-skinny linear algebra for population matrices (torch backend)
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

"""Streaming tall-skinny numerics for ``N agents x T rounds`` matrices.

The population matrix ``X`` has ``N`` up to several million rows and ``T ~ 1e3``
columns and arrives as an iterable of row chunks that need not fit in memory
together. Everything here touches each chunk once per pass and keeps only
``T x T`` (or ``T x k``) state:

* `tsqr`, `cholqr`, `gram`: the triangular / Gram factor of the whole stack
  (communication-avoiding TSQR, CholeskyQR2, float64 Gram);
* `tall_svd`: singular values and right singular vectors from that factor;
  `left_factors` gives ``U`` for one chunk on demand, so the ``N x k`` left
  factor is never materialised unless asked; `randomized_svd_tall` streams the
  Halko-Martinsson-Tropp range finder when ``T`` is too large for ``T x T``;
* `parallel_analysis_null`, `estimate_rank_pa`: the circular-shift
  parallel-analysis null in closed form (summed periodogram, spread by the
  Marchenko-Pastur deterministic equivalent), without any surrogate SVD;
* `brand_update_cols`, `brand_update_rows`, `incremental_svd`: Brand's
  rank-k thin-SVD updates for causal (online) use;
* `CausalProjector`, `causal_projector`: agent-space projection of a later
  window on loadings learned on a past window.

A *chunk source* is a single 2-D tensor (or array; split into row blocks of
`chunk_rows`), a list/tuple of ``(n_c, T)`` tensors, a zero-argument callable
returning a fresh iterator of chunks, or (single-pass functions only) any
iterator of chunks. Computations run on the device of the chunks.
"""

import os
import sys
import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# chunk sources
# ---------------------------------------------------------------------------

_DEFAULT_ROWS = 1 << 16


def iter_chunks(chunks, chunk_rows=None):
    """Iterate over the row chunks of a chunk source.

    Parameters
    ----------
    chunks : torch.Tensor, numpy ndarray, list, tuple, callable or iterator
        A ``(N, T)`` matrix (split into row blocks), a sequence of ``(n_c, T)``
        chunks, a zero-argument callable returning an iterator of chunks, or an
        iterator of chunks.
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix (default : 65536).

    Returns
    -------
    iterator of torch.Tensor
        Chunks of shape ``(n_c, T)``.
    """
    if callable(chunks) and not torch.is_tensor(chunks):
        chunks = chunks()
    if torch.is_tensor(chunks) or isinstance(chunks, np.ndarray):
        X = torch.as_tensor(chunks)
        if X.ndim != 2:
            raise ValueError('Error: expected a 2-D (N, T) matrix, got shape {}.'.format(tuple(X.shape)))
        step = int(chunk_rows or _DEFAULT_ROWS)
        for s in range(0, X.shape[0], step):
            yield X[s:s + step]
        return
    for c in chunks:
        c = torch.as_tensor(c)
        if c.ndim == 1:
            c = c[None]
        yield c


def _reiterable(chunks, chunk_rows=None):
    """Zero-argument factory of fresh chunk iterators (for multi-pass algorithms)."""
    if torch.is_tensor(chunks) or isinstance(chunks, (np.ndarray, list, tuple)) or callable(chunks):
        return lambda: iter_chunks(chunks, chunk_rows)
    raise TypeError('Error: this algorithm needs several passes over the data; pass a tensor, '
                    'a list of chunks or a zero-argument callable returning a fresh iterator.')


def _prep(X, dtype, center):
    if dtype is not None:
        X = X.to(dtype)
    elif not X.is_floating_point():
        X = X.to(torch.float64)
    if center:
        X = X - X.mean(1, keepdim=True)
    return X


def _acc_dtype(dtype):
    """Accumulation dtype: float64 unless complex/other explicitly asked."""
    return torch.float64 if dtype in (None, torch.float32, torch.float16, torch.bfloat16, torch.float64) else dtype


def _sym_eigh_desc(G):
    lam, V = torch.linalg.eigh(0.5 * (G + G.T))
    return lam.flip(0), V.flip(1)


def subspace_angles(A, B):
    """Principal angles between the column spaces of `A` and `B` (Bjorck-Golub 1973).

    Parameters
    ----------
    A, B : torch.Tensor
        Matrices with the same number of rows.

    Returns
    -------
    torch.Tensor
        Angles in radians, increasing; the last one is the largest angle.
    """
    Qa = torch.linalg.qr(A.to(torch.float64)).Q
    Qb = torch.linalg.qr(B.to(torch.float64).to(A.device)).Q
    c = torch.linalg.svdvals(Qa.T @ Qb).clamp(-1.0, 1.0)
    return torch.arccos(c).sort().values


# ---------------------------------------------------------------------------
# streaming factorizations
# ---------------------------------------------------------------------------

def _qr_r(X):
    return torch.linalg.qr(X, mode='r').R


def _local_cholqr2(X):
    """R factor of an in-memory block by CholeskyQR2; None if the block is (numerically) rank deficient."""
    if X.shape[0] < X.shape[1]:
        return None
    G = X.T @ X
    L, info = torch.linalg.cholesky_ex(G)
    if int(info) != 0:
        return None
    R1 = L.T
    Q = torch.linalg.solve_triangular(R1, X, upper=True, left=False)
    L2, info = torch.linalg.cholesky_ex(Q.T @ Q)
    if int(info) != 0:
        return None
    return L2.T @ R1


def _positive_diag(R):
    d = torch.sign(torch.diagonal(R))
    d = torch.where(d == 0, torch.ones_like(d), d)
    return R * d[:, None]


class _TSQRReducer:
    """Accumulates R factors of row blocks; tree (binary counter) or flat (sequential) reduction."""

    def __init__(self, reduction='tree'):
        if reduction not in ('tree', 'sequential'):
            raise ValueError(reduction)
        self.reduction = reduction
        self.stack = []          # (level, R) for the tree
        self.R = None            # running R for the flat reduction

    def push(self, R):
        if self.reduction == 'sequential':
            self.R = R if self.R is None else _qr_r(torch.cat([self.R, R], 0))
            return
        level = 0
        while self.stack and self.stack[-1][0] == level:
            _, R0 = self.stack.pop()
            R = _qr_r(torch.cat([R0, R], 0))
            level += 1
        self.stack.append((level, R))

    def result(self):
        if self.reduction == 'sequential':
            R = self.R
        else:
            if not self.stack:
                R = None
            else:
                R = self.stack.pop()[1]
                while self.stack:
                    R = _qr_r(torch.cat([self.stack.pop()[1], R], 0))
        if R is None:
            raise ValueError('Error: empty chunk source.')
        return _positive_diag(R)


def tsqr(chunks, dtype=torch.float64, reduction='tree', center=False, local='qr', chunk_rows=None):
    """R factor of the stacked chunks by communication-avoiding TSQR.

    Each chunk ``X_c`` is factored locally, ``X_c = Q_c R_c``, and the small
    ``R_c`` are reduced by QR of stacked pairs, either along a binary tree
    (``O(log #chunks)`` resident R factors, the reduction of
    Demmel-Grigori-Hoemmen-Langou 2012) or sequentially (flat tree). Only R
    is formed; ``X = Q R`` with ``Q`` implicit, ``R^T R = X^T X``.

    Parameters
    ----------
    chunks : chunk source
        See `iter_chunks`; a single pass is made.
    dtype : torch.dtype, optional
        Working dtype of the local factorizations (default : float64; float32
        is faster but resolves singular values only down to ~1e-7 relative).
    reduction : {'tree', 'sequential'}, optional
        Reduction tree (default : 'tree').
    center : bool, optional
        Centre every row first (default : False).
    local : {'qr', 'cholqr2'}, optional
        Leaf factorization: Householder QR, or CholeskyQR2 (Fukaya et al. 2014;
        GEMM-rich, falls back to Householder when the block's Gram is not
        numerically positive definite) (default : 'qr').
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.

    Returns
    -------
    torch.Tensor
        Upper triangular ``R`` of shape ``(min(N, T), T)`` with nonnegative
        diagonal.
    """
    red = _TSQRReducer(reduction)
    for X in iter_chunks(chunks, chunk_rows):
        X = _prep(X, dtype, center)
        R = _local_cholqr2(X) if local == 'cholqr2' else None
        if R is None:
            R = _qr_r(X)
        red.push(R)
    return red.result()


def gram(chunks, dtype=torch.float64, center=False, chunk_rows=None):
    """Streamed Gram matrix ``G = sum_c X_c^T X_c``, accumulated in float64.

    Parameters
    ----------
    chunks : chunk source
        See `iter_chunks`; a single pass is made.
    dtype : torch.dtype, optional
        Dtype of the per-chunk product (default : float64). With float32 the
        chunk GEMM runs in float32 and the partial Grams are summed in float64.
    center : bool, optional
        Centre every row first (default : False).
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.

    Returns
    -------
    torch.Tensor
        ``(T, T)`` float64 Gram matrix.
    """
    G = None
    for X in iter_chunks(chunks, chunk_rows):
        X = _prep(X, dtype, center)
        P = (X.T @ X).to(torch.float64)
        G = P if G is None else G + P
    if G is None:
        raise ValueError('Error: empty chunk source.')
    return G


def _chol_upper(G, shift=None):
    """Upper Cholesky factor of ``G``; adds an escalating diagonal shift when needed.

    Returns ``(R, shift_used)``.
    """
    eps = torch.finfo(G.dtype).eps
    n = G.shape[0]
    I = torch.eye(n, dtype=G.dtype, device=G.device)
    s = 0.0 if shift is None else float(shift)
    tr = float(torch.trace(G)) or 1.0
    for _ in range(12):
        L, info = torch.linalg.cholesky_ex(G + s * I if s > 0 else G)
        if int(info) == 0:
            return L.T, s
        # Fukaya et al. (2020) shift is 11 (m n + n (n + 1)) u ||X||^2; we start lower and escalate
        s = max(s * 100.0, 10.0 * n * eps * tr)
    raise torch.linalg.LinAlgError('Error: Cholesky failed even with shift {:.3g}.'.format(s))


def cholqr(chunks, passes=2, dtype=torch.float64, center=False, shift=None, chunk_rows=None):
    """R factor by (shifted) CholeskyQR2 on a streamed float64 Gram matrix.

    Pass 1 forms ``G = X^T X`` in float64 and ``R_1 = chol(G)``. With
    ``passes >= 2`` the data is streamed again and the Gram of
    ``X R_1^{-1}`` (one triangular solve per chunk) refactored, which restores
    orthogonality of the implicit Q to machine precision for
    ``cond(X) < u^{-1/2}`` (CholeskyQR2, Fukaya-Nakatsukasa-Yanagisawa-Yamamoto
    2014). If a Gram is not numerically positive definite a diagonal shift is
    added (shifted CholeskyQR3, Fukaya-Kannan-Nakatsukasa-Yamamoto-Yanagisawa
    2020) and one extra pass is made; singular values below ``sqrt(shift)``
    are then not resolved (row-centred data are exactly rank deficient).

    Parameters
    ----------
    chunks : chunk source
        See `iter_chunks`; must be re-iterable when ``passes > 1``.
    passes : int, optional
        Number of Cholesky passes (default : 2). ``passes=1`` is plain
        CholeskyQR (single pass, Gram-level accuracy).
    dtype : torch.dtype, optional
        Dtype of the per-chunk products (default : float64).
    center : bool, optional
        Centre every row first (default : False).
    shift : float, optional
        Initial diagonal shift (default : None, only if needed).
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.

    Returns
    -------
    torch.Tensor
        ``(T, T)`` upper triangular float64 ``R`` with ``R^T R ~ X^T X``.
    """
    src = _reiterable(chunks, chunk_rows) if passes > 1 else (lambda: iter_chunks(chunks, chunk_rows))
    R = None
    p = 0
    extra = 0
    while p < passes + extra:
        G = None
        for X in src():
            X = _prep(X, dtype, center)
            if R is not None:
                X = torch.linalg.solve_triangular(R.to(X.dtype), X, upper=True, left=False)
            P = (X.T @ X).to(torch.float64)
            G = P if G is None else G + P
        if G is None:
            raise ValueError('Error: empty chunk source.')
        Rp, s = _chol_upper(G, shift if p == 0 else None)
        if s > 0 and extra == 0 and passes > 1:
            extra = 1
        R = Rp if R is None else Rp @ R
        p += 1
    return R


# ---------------------------------------------------------------------------
# tall SVD
# ---------------------------------------------------------------------------

_CANS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                             'external', 'cans_svd_src', 'cans_svd')


def load_cans_svd(path=None):
    """Import ``cans_svd`` from the external CANS repository at call time.

    The repository (Chebyshev-accelerated Newton-Schulz polar decomposition)
    is not vendored; its directory is put on ``sys.path`` only for the import.

    Parameters
    ----------
    path : str, optional
        Repository root containing ``src_torch`` (default :
        ``$FATHON_CANS_PATH`` or ``<project>/external/cans_svd_src/cans_svd``).

    Returns
    -------
    callable
        ``cans_svd(matrix, degree=3, preprocess=True, preprocess_iters=2,
        delta=0.99, a=1e-3, n=50, eps_qr=1e-5) -> (U, s, Vt)``.
    """
    path = os.path.abspath(path or os.environ.get('FATHON_CANS_PATH', _CANS_DEFAULT))
    added = path not in sys.path
    if added:
        sys.path.append(path)
    try:
        from src_torch.svd.cans_svd import cans_svd
    except Exception as exc:          # ImportError, or errors inside the foreign package
        raise ImportError("Error: method='cans' needs the CANS repository (src_torch.svd.cans_svd) at {}; "
                          "set FATHON_CANS_PATH. Original error: {!r}".format(path, exc)) from exc
    finally:
        if added and path in sys.path:
            sys.path.remove(path)
    return cans_svd


def _svd_of_R(R, k):
    _, S, Vh = torch.linalg.svd(R, full_matrices=False)
    V = Vh.T
    if k is not None:
        S, V = S[:k], V[:, :k]
    return S, V


def tall_svd(chunks, k=None, method='tsqr', center=False, dtype=torch.float64, chunk_rows=None,
             reduction='tree', cans_kwargs=None):
    """Singular values and right singular vectors of the stacked tall matrix.

    ``X = U diag(S) V^T`` with ``X`` the vertical stack of the chunks. Only the
    ``T x T`` factor is formed; ``U`` for any chunk follows from
    `left_factors`.

    Methods
    -------
    * ``'tsqr'``: SVD of the TSQR factor R (`tsqr`); backward stable, the
      R-SVD of Chan (1982) on top of Demmel et al. (2012) TSQR. One pass.
    * ``'cholqr2'``: SVD of the CholeskyQR2 factor (`cholqr`). Two passes,
      GEMM-only.
    * ``'gram'``: symmetric eigendecomposition of the float64 Gram. One pass,
      cheapest; singular values below ``~1e-8 S[0]`` are not resolved.
    * ``'cans'``: TSQR, then the SVD of R by polar decomposition with CANS
      (Chebyshev-accelerated Newton-Schulz, Grishina-Kuznetsov-Oseledets 2025)
      from the optional external repository (`load_cans_svd`).
    * ``'torch'``: materialise the stack and call ``torch.linalg.svd``
      (reference; needs the whole matrix in memory).

    Parameters
    ----------
    chunks : chunk source
        See `iter_chunks`.
    k : int, optional
        Number of leading triplets to keep (default : all ``min(N, T)``).
    method : {'tsqr', 'cholqr2', 'gram', 'cans', 'torch'}, optional
        Factorization (default : 'tsqr').
    center : bool, optional
        Centre every row first (default : False).
    dtype : torch.dtype, optional
        Working dtype (default : float64).
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.
    reduction : {'tree', 'sequential'}, optional
        TSQR reduction tree (default : 'tree').
    cans_kwargs : dict, optional
        Keyword arguments for ``cans_svd`` (method 'cans').

    Returns
    -------
    torch.Tensor
        Singular values ``S``, decreasing, shape ``(r,)``.
    torch.Tensor
        Right singular vectors ``V``, shape ``(T, r)``, ``r = k or min(N, T)``.
    """
    if method == 'tsqr':
        return _svd_of_R(tsqr(chunks, dtype, reduction, center, chunk_rows=chunk_rows), k)
    if method in ('cholqr2', 'cholqr'):
        return _svd_of_R(cholqr(chunks, 2, dtype, center, chunk_rows=chunk_rows), k)
    if method == 'gram':
        G = gram(chunks, dtype, center, chunk_rows)
        lam, V = _sym_eigh_desc(G)
        S = lam.clamp_min(0).sqrt()
        if k is not None:
            S, V = S[:k], V[:, :k]
        return S, V
    if method == 'cans':
        cans_svd = load_cans_svd()
        R = tsqr(chunks, dtype, reduction, center, chunk_rows=chunk_rows)
        _, S, Vt = cans_svd(R, **(cans_kwargs or {}))
        V = Vt.T
        if k is not None:
            S, V = S[:k], V[:, :k]
        return S, V
    if method == 'torch':
        X = torch.cat([_prep(c, dtype, center) for c in iter_chunks(chunks, chunk_rows)], 0)
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.T
        if k is not None:
            S, V = S[:k], V[:, :k]
        return S, V
    raise ValueError("Error: unknown method {!r}.".format(method))


def left_factors(X_chunk, S, V, k=None, center=False, rtol=None):
    """Left singular vectors of one chunk, ``U_c = X_c V_k diag(S_k)^{-1}``.

    Since ``X V = U diag(S)``, the rows of ``U`` belonging to a chunk need
    only that chunk and the ``T x k`` right factor, so ``U`` for millions of
    rows is produced on demand and never stored.

    Parameters
    ----------
    X_chunk : torch.Tensor
        Rows ``(n_c, T)`` (raw; centred here if `center`).
    S, V : torch.Tensor
        Singular values and right singular vectors from `tall_svd`.
    k : int, optional
        Number of components (default : all of `V`).
    center : bool, optional
        Centre the rows first; must match the `tall_svd` call (default : False).
    rtol : float, optional
        Components with ``S < rtol S[0]`` get a zero column (default :
        ``T * eps`` of `V`'s dtype).

    Returns
    -------
    torch.Tensor
        ``(n_c, k)`` block of ``U`` in `V`'s dtype.
    """
    k = V.shape[1] if k is None else k
    Vk, Sk = V[:, :k], S[:k].to(V.dtype)
    X = _prep(torch.as_tensor(X_chunk).to(V.device), V.dtype, center)
    rtol = V.shape[0] * torch.finfo(V.dtype).eps if rtol is None else rtol
    inv = torch.where(Sk > rtol * S[0], 1.0 / Sk, torch.zeros_like(Sk))
    return (X @ Vk) * inv


def randomized_svd_tall(chunks_factory, k, oversample=8, n_power=2, center=False, dtype=torch.float64,
                        seed=0, chunk_rows=None):
    """Top-`k` SVD of a streamed tall matrix by randomized subspace iteration.

    Halko-Martinsson-Tropp (2011) range finder applied to ``X^T`` (whose
    range lives in ``R^T``): ``Y = X^T Omega`` with a Gaussian test matrix
    generated chunk by chunk, then `n_power` subspace iterations
    ``Y <- X^T (X Q)``, ``Q = qr(Y)`` (Algorithm 4.4; both products of one
    iteration are applied while a chunk is resident, so an iteration costs
    one pass), and finally the SVD of the ``N x l`` matrix ``X Q`` by TSQR
    (Algorithm 5.1), ``V = Q W``. Memory is ``O(T l)``; passes:
    ``n_power + 2``. Use when ``T`` is too large for a dense ``T x T``
    factor; for ``T ~ 1e3`` `tall_svd` is exact and usually as fast.

    Parameters
    ----------
    chunks_factory : chunk source
        Re-iterable source (tensor, list, or zero-argument callable returning
        a fresh iterator; chunks must be identical on every pass).
    k : int
        Rank.
    oversample : int, optional
        Extra sketch columns ``p``, ``l = k + p`` (default : 8).
    n_power : int, optional
        Number of power (subspace) iterations (default : 2).
    center : bool, optional
        Centre every row first (default : False).
    dtype : torch.dtype, optional
        Working dtype (default : float64).
    seed : int, optional
        Seed of the Gaussian test matrix (default : 0).
    chunk_rows : int, optional
        Rows per block when the source is a single matrix.

    Returns
    -------
    torch.Tensor
        ``S``, shape ``(k,)``.
    torch.Tensor
        ``V``, shape ``(T, k)``.
    """
    src = _reiterable(chunks_factory, chunk_rows)
    l = k + oversample
    Y = None
    g = None
    for X in src():
        X = _prep(X, dtype, center)
        if Y is None:
            Y = torch.zeros(X.shape[1], l, dtype=X.dtype, device=X.device)
            g = torch.Generator(device=X.device).manual_seed(int(seed))
        Om = torch.randn(X.shape[0], l, generator=g, dtype=X.dtype, device=X.device)
        Y += X.T @ Om
    if Y is None:
        raise ValueError('Error: empty chunk source.')
    l = min(l, Y.shape[0])
    for _ in range(n_power):
        Q = torch.linalg.qr(Y).Q[:, :l]
        Y = torch.zeros_like(Q)
        for X in src():
            X = _prep(X, dtype, center)
            Y += X.T @ (X @ Q)
    Q = torch.linalg.qr(Y).Q[:, :l]
    R = tsqr((_prep(X, dtype, center) @ Q for X in src()), dtype=None)
    _, S, Wh = torch.linalg.svd(R, full_matrices=False)
    V = Q @ Wh.T
    return S[:k], V[:, :k]


# ---------------------------------------------------------------------------
# parallel analysis against circular-shift surrogates, in closed form
# ---------------------------------------------------------------------------

class _Moments:
    """One-pass accumulator: Gram or TSQR factor, summed periodogram, row energies, row subsample."""

    def __init__(self, method, dtype, center, keep_rows=0, seed=0, need_gram=False):
        self.method, self.dtype, self.center = method, dtype, center
        self.need_gram = need_gram or method == 'gram'
        self.red = _TSQRReducer('tree') if method == 'tsqr' else None
        self.G = None
        self.P = None
        self.n = 0
        self.e1 = 0.0
        self.e2 = 0.0
        self.p4 = 0.0
        self.keep_rows = int(keep_rows)
        self.seed = seed
        self.sub = None
        self.keys = None
        self.gen = None

    def add(self, X):
        X = _prep(X, self.dtype, self.center)
        n, T = X.shape
        if self.need_gram:
            P = (X.T @ X).to(torch.float64)
            self.G = P if self.G is None else self.G + P
        if self.red is not None:
            self.red.push(_qr_r(X))
        F = torch.fft.rfft(X, dim=1)
        Pi = (F.real ** 2 + F.imag ** 2).to(torch.float64)          # per-row half periodogram
        p = Pi.sum(0)
        self.P = p if self.P is None else self.P + p
        wk = torch.full((Pi.shape[1],), 2.0, dtype=torch.float64, device=Pi.device)
        wk[0] = 1.0
        if T % 2 == 0:
            wk[-1] = 1.0
        self.p4 += float(((Pi ** 2) * wk).sum())                       # sum_i sum_k |FFT(x_i)_k|^4
        a = (X.to(torch.float64) ** 2).sum(1)
        self.e1 += float(a.sum())
        self.e2 += float((a ** 2).sum())
        self.n += n
        if self.keep_rows > 0:
            # bottom-k sketch on uniform random keys: a uniform row sample without replacement
            if self.gen is None:
                self.gen = torch.Generator(device=X.device).manual_seed(int(self.seed) + 7919)
            key = torch.rand(n, generator=self.gen, device=X.device, dtype=torch.float64)
            sub = X if self.sub is None else torch.cat([self.sub, X], 0)
            keys = key if self.keys is None else torch.cat([self.keys, key], 0)
            if keys.numel() > self.keep_rows:
                idx = torch.topk(keys, self.keep_rows, largest=False).indices
                sub, keys = sub[idx], keys[idx]
            self.sub, self.keys = sub, keys

    def finish(self, T):
        P = self.P
        lam = torch.cat([P, P[1:(T + 1) // 2].flip(0)]) / T
        S = None
        if self.method == 'tsqr':
            S = torch.linalg.svdvals(self.red.result())
        elif self.G is not None:
            S = torch.linalg.eigvalsh(0.5 * (self.G + self.G.T)).flip(0).clamp_min(0).sqrt()
        # second-moment matching of the surrogate spectrum to the Marchenko-Pastur model
        den = self.e2 - self.p4 / T ** 2
        num = self.e1 ** 2 + float((lam ** 2).sum())
        n_eff = {'rows': float(self.n),
                 'energy': self.e1 ** 2 / self.e2 if self.e2 > 0 else float(self.n),
                 'moment': num / den if den > 0 else float(self.n)}
        return lam, S, n_eff


def periodogram_eigenvalues(chunks, center=True, dtype=torch.float64, chunk_rows=None):
    """Eigenvalues of the expected circular-shift-surrogate Gram matrix, ``sum_i |FFT(x_i)|^2 / T``.

    See `parallel_analysis_null` for the derivation.

    Parameters
    ----------
    chunks : chunk source
        See `iter_chunks`; a single pass is made.
    center : bool, optional
        Centre every row first (default : True).
    dtype : torch.dtype, optional
        Dtype of the FFTs (default : float64); sums are float64.
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.

    Returns
    -------
    torch.Tensor
        ``(T,)`` eigenvalues in DFT-frequency order ``k = 0 .. T-1``
        (``lam[k] = lam[T-k]``).
    dict
        Effective row counts for the spreading: ``'rows'`` (``N``),
        ``'energy'`` and ``'moment'`` (see `parallel_analysis_null`).
    int
        Number of rows.
    """
    acc = _Moments('none', dtype, center)
    T = None
    for X in iter_chunks(chunks, chunk_rows):
        T = X.shape[1]
        acc.add(X)
    if T is None:
        raise ValueError('Error: empty chunk source.')
    lam, _, n_eff = acc.finish(T)
    return lam, n_eff, acc.n


def deterministic_equivalent_spectrum(lam, n_rows, q=None, grid=1024, local=96, eta=0.5, newton_iters=30,
                                      fp_iters=20):
    """Expected sorted eigenvalues of a sample Gram matrix with population Gram eigenvalues `lam`.

    Model: ``Y = Z Sigma^{1/2}`` with ``n_rows`` i.i.d. isotropic rows and
    ``n_rows Sigma`` having eigenvalues `lam` (so ``E[Y^T Y]`` has spectrum
    `lam`). For ``T, n_rows -> inf`` the eigenvalue distribution of
    ``Y^T Y / n_rows`` converges to the generalized Marchenko-Pastur law
    (Marchenko-Pastur 1967; Silverstein-Bai 1995), whose companion Stieltjes
    transform ``m`` solves, for ``z`` in the upper half plane,

        ``z = -1/m + (1/n_rows) sum_j tau_j / (1 + tau_j m)``,
        ``tau_j = lam_j / n_rows``,

    with a unique solution in the upper half plane; the density of the
    ``T x T`` sample covariance is ``Im m(x + i0) / (pi T / n_rows)``. The
    equation is solved on a grid (log-spaced, refined with `local` uniform
    points across the interval ``tau_j (1 +- 6 sqrt(w_j / n_rows))`` into
    which each leading population eigenvalue of multiplicity ``w_j`` spreads)
    by a vectorised fixed-point warm start and safeguarded Newton at
    ``z = x (1 + i eta r)``, ``r`` the finest relative grid spacing; the density is integrated from the top and the
    expected k-th largest eigenvalue is read off as the ``k - 1/2`` quantile
    of ``T`` times the counting function. As ``n_rows -> inf`` it reduces to
    `lam` itself.

    Parameters
    ----------
    lam : torch.Tensor
        Population Gram eigenvalues, any order, ``(T,)``.
    n_rows : float
        (Effective) number of rows.
    q : int, optional
        Number of leading eigenvalues wanted (default : 48).
    grid : int, optional
        Points of the global log grid (default : 1024).
    local : int, optional
        Refinement points per leading population eigenvalue (default : 96).
    eta : float, optional
        Imaginary part in units of the finest relative grid spacing
        (default : 0.5).
    newton_iters, fp_iters : int, optional
        Iteration counts (defaults : 30, 20).

    Returns
    -------
    torch.Tensor
        ``(q,)`` expected leading Gram eigenvalues, decreasing (float64, on
        `lam`'s device).
    """
    lam = torch.as_tensor(lam, dtype=torch.float64)
    dev = lam.device
    T = lam.numel()
    q = min(48 if q is None else int(q), T)
    lam_sorted = lam.sort(descending=True).values.clamp_min(0)
    top = float(lam_sorted[0])
    if top <= 0:
        return torch.zeros(q, dtype=torch.float64, device=dev)
    N = float(n_rows)
    gam = T / N
    tau_u, w = torch.unique(lam_sorted / top, return_counts=True)
    w = w.to(torch.float64)
    sg = math.sqrt(gam)
    x_hi = (1.0 + sg) ** 2 * 1.15
    q_need = min(T, 2 * q + 16)
    tau_lo = float(lam_sorted[q_need - 1]) / top
    x_lo = tau_lo * max(1.0 - sg, 0.1) ** 2 * 0.7
    if x_lo <= 0:
        x_lo = x_hi * 1e-6
    pts = [torch.logspace(math.log10(x_lo), math.log10(x_hi), grid, dtype=torch.float64, device=dev)]
    sel = tau_u >= tau_lo
    mean_tau = float((w * tau_u).sum()) / T
    for t_, w_ in zip(tau_u[sel].tolist(), w[sel].tolist()):
        h = t_ * min(0.5, 6.0 * math.sqrt(w_ / N) + 3.0 * gam * mean_tau / t_)
        pts.append(torch.linspace(max(t_ - h, x_lo), min(t_ + h, x_hi), local, dtype=torch.float64, device=dev))
    x = torch.unique(torch.cat(pts))
    x = x[(x >= x_lo) & (x <= x_hi)]
    # one small *relative* imaginary part everywhere: a spacing-proportional eta would let the
    # wide Lorentzians of coarse points soak up mass from nearby narrow intervals
    rel = float((torch.diff(x) / x[1:]).min())
    z = torch.complex(x, eta * rel * x)
    tau_u = tau_u[None, :]
    w = w[None, :]

    def s_of(m):
        return (w * tau_u / (1.0 + tau_u * m[:, None])).sum(1) / N

    m = -1.0 / z
    for _ in range(fp_iters):
        m = 1.0 / (-z + s_of(m))
    for _ in range(newton_iters):
        d = 1.0 + tau_u * m[:, None]
        Fv = -1.0 / m + (w * tau_u / d).sum(1) / N - z
        dF = 1.0 / m ** 2 - (w * tau_u ** 2 / d ** 2).sum(1) / N
        m_new = m - Fv / dF
        bad = ~torch.isfinite(m_new) | (m_new.imag <= 0)
        if bad.any():
            m_fp = 1.0 / (-z + s_of(m))
            m_new = torch.where(bad, m_fp, m_new)
        m = m_new
    dens = m.imag.clamp_min(0) / (math.pi * gam)                 # density of x, integrates to 1
    seg = 0.5 * (dens[1:] + dens[:-1]) * (x[1:] - x[:-1])
    above = torch.zeros_like(x)
    above[:-1] = T * torch.flip(torch.cumsum(torch.flip(seg, [0]), 0), [0])   # count above x_j
    targets = np.arange(q, dtype=np.float64) + 0.5
    # `above` decreases in x: interpolate x as a function of the count
    a_inc = torch.flip(above, [0]).cpu().numpy()
    x_inc = torch.flip(x, [0]).cpu().numpy()
    xs = np.interp(targets, a_inc, x_inc, left=np.nan, right=np.nan)
    out = torch.as_tensor(xs, dtype=torch.float64, device=dev) * top
    # beyond the grid (not enough mass resolved): fall back to the population values
    return torch.where(torch.isfinite(out), out, lam_sorted[:q])


def parallel_analysis_null(chunks, q=None, center=True, spread='mp', exclude_identity=False, n_eff='moment',
                           dtype=torch.float64, chunk_rows=None):
    r"""Expected singular values of row-wise circular-shift surrogates, without any surrogate SVD.

    Derivation. Let ``x_i`` in ``R^T`` be the rows (centred if `center`) and
    ``S`` the cyclic shift ``(S x)(t) = x(t-1 mod T)``. A surrogate draws
    independent uniform ``s_i`` in ``{0, .., T-1}`` and stacks
    ``y_i = S^{s_i} x_i``; its Gram is ``G~ = sum_i y_i y_i^T``. Then

        ``E[y_i y_i^T]_{tu} = (1/T) sum_s x_i(t-s) x_i(u-s) = c_i(u-t) / T``,
        ``c_i(h) = sum_t x_i(t) x_i(t+h mod T)``,

    the *circular* (wrapped) unnormalised autocovariance. Hence
    ``E G~ = C`` is the symmetric circulant with first row
    ``c(h) = (1/T) sum_i c_i(h)``; its diagonal is ``c(0) = tr(G)/T``
    (surrogates keep the total energy: ``tr E G~ = tr G`` exactly). A
    circulant is diagonalised by the DFT, and by the circular Wiener-Khinchin
    identity ``sum_h c_i(h) e^{-2 pi i h k / T} = |FFT(x_i)_k|^2``, so

        ``eig(E G~) = lam_k = sum_i |FFT(x_i)_k|^2 / T``,  ``k = 0 .. T-1``

    (``lam_k = lam_{T-k}``: cosine/sine pairs). Checks: Parseval gives
    ``sum_k lam_k = sum_i ||x_i||^2``. Centring commutes with ``S`` (the mean
    is shift invariant), so it may be done before or after shifting; centred
    rows have ``FFT(x_i)_0 = 0`` and the constant vector is an exact null
    direction, as for the data. The autocovariance must be the circular one
    because the surrogate itself wraps around; the linear (Toeplitz)
    autocovariance describes ``E G`` of a stationary process instead, and the
    two differ by ``O(xi / T)`` for a correlation length ``xi`` (the
    circulant is the periodic wrap of the Toeplitz matrix). Drawing shifts
    from ``{1, .., T-1}`` (as `scale.estimate_rank` does) removes the
    identity: ``E G~ = (T C - G) / (T - 1)``, an ``O(1/T)`` correction that
    `exclude_identity` applies (needs the Gram and one ``T x T`` eigvalsh).

    Expected Gram is not expected spectrum: one surrogate's eigenvalues
    scatter around ``lam`` with random-matrix spreading of relative size
    ``~ sqrt(T / N)``. ``spread='mp'`` maps ``lam`` through the
    Marchenko-Pastur deterministic equivalent
    (`deterministic_equivalent_spectrum`) with ``N_eff`` rows. A surrogate
    row keeps its energy ``a_i = ||x_i||^2`` and its periodogram
    ``P_i(k) = |FFT(x_i)_k|^2`` exactly, unlike a Gaussian row, so ``N_eff``
    is chosen to match the second spectral moment exactly: with independent
    shifts ``E (y_i . y_j)^2 = sum_k P_i(k) P_j(k) / T^2`` (circular
    cross-correlation and Parseval), hence
    ``E tr G~^2 = sum_i a_i^2 + tr Lam^2 - sum_i sum_k P_i(k)^2 / T^2``,
    whereas the Gaussian model with ``n`` rows has
    ``tr Lam^2 + ((tr Lam)^2 + tr Lam^2) / n``; equating them gives
    (``n_eff='moment'``)

        ``N_eff = ((sum_i a_i)^2 + sum_k lam_k^2) / (sum_i a_i^2 - sum_i sum_k P_i(k)^2 / T^2)``,

    which is ``~N`` for homogeneous rows, smaller for unequal energies, and
    accumulated in the same pass. ``'energy'`` uses
    ``(sum a_i)^2 / sum a_i^2`` and ``'rows'`` uses ``N``.

    Parameters
    ----------
    chunks : chunk source
        See `iter_chunks`; a single pass is made.
    q : int, optional
        Number of leading values (default : all ``T`` for ``spread='none'``,
        48 for ``'mp'``).
    center : bool, optional
        Centre every row first (default : True).
    spread : {'mp', 'none'}, optional
        Finite-``N`` spreading (default : 'mp').
    exclude_identity : bool, optional
        Shifts uniform on ``{1, .., T-1}`` instead of ``{0, .., T-1}``
        (default : False).
    n_eff : {'moment', 'energy', 'rows'} or float, optional
        Row count used for the spreading (default : 'moment').
    dtype : torch.dtype, optional
        Dtype of the per-chunk FFT / products (default : float64).
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.

    Returns
    -------
    torch.Tensor
        Null singular values, decreasing, float64.
    """
    acc = _Moments('none', dtype, center, need_gram=exclude_identity)
    T = None
    for X in iter_chunks(chunks, chunk_rows):
        T = X.shape[1]
        acc.add(X)
    if T is None:
        raise ValueError('Error: empty chunk source.')
    lam, _, ne = acc.finish(T)
    return _null_from(lam, acc.G, acc.n, ne, T, q, spread, exclude_identity, n_eff)


def _circulant(lam):
    T = lam.numel()
    c = torch.fft.ifft(lam.to(torch.complex128)).real                 # first column / row of C
    idx = (torch.arange(T, device=lam.device)[None, :] - torch.arange(T, device=lam.device)[:, None]) % T
    return c[idx]


def _null_from(lam, G, n, ne, T, q, spread, exclude_identity, n_eff):
    if exclude_identity:
        if G is None:
            raise ValueError('Error: exclude_identity needs the Gram matrix.')
        pop = torch.linalg.eigvalsh((T * _circulant(lam) - G) / (T - 1)).flip(0).clamp_min(0)
    else:
        pop = lam.sort(descending=True).values.clamp_min(0)
    if spread == 'none':
        out = pop if q is None else pop[:q]
        return out.sqrt()
    if spread != 'mp':
        raise ValueError("Error: unknown spread {!r}.".format(spread))
    nr = ne[n_eff] if isinstance(n_eff, str) else float(n_eff)
    return deterministic_equivalent_spectrum(pop, nr, q=48 if q is None else q).clamp_min(0).sqrt()


def _shift_surrogate(X, gen):
    N, T = X.shape
    sh = torch.randint(0, T, (N,), device=X.device, generator=gen)
    ar = torch.arange(T, device=X.device)
    return torch.gather(X, 1, (ar[None, :] + sh[:, None]) % T)


def shift_invariant_eigenvalues(G, kind='toeplitz'):
    """Eigenvalues of the shift-invariant projection of a Gram matrix.

    ``kind='circulant'``: average every *circular* diagonal of ``G``,
    ``c(h) = (1/T) sum_t G[t, t+h mod T]``; the result is the expected Gram of
    row-wise uniform circular-shift surrogates (see `parallel_analysis_null`),
    with eigenvalues ``FFT(c)``, equal to the summed periodogram ``/ T`` when
    ``G = X^T X``. ``kind='toeplitz'``: average every *linear* diagonal,
    ``g(h) = (1/(T-h)) sum_t G[t, t+h]`` (the pooled unbiased linear
    autocovariances); for mutually independent stationary rows this is an
    unbiased estimate of ``E[G] = sum_i Toeplitz(gamma_i)`` with no
    wrap-around artefact. Both are the orthogonal (Frobenius) projections of
    ``G`` onto the respective subspace of shift-invariant matrices, so any
    cross-row alignment (a common mode) is averaged out except for its own
    autocovariance.

    Parameters
    ----------
    G : torch.Tensor
        Symmetric ``(T, T)`` matrix.
    kind : {'toeplitz', 'circulant'}, optional
        Projection (default : 'toeplitz').

    Returns
    -------
    torch.Tensor
        Eigenvalues, decreasing, clamped at 0 (float64).
    """
    G = G.to(torch.float64)
    T = G.shape[0]
    ar = torch.arange(T, device=G.device)
    if kind == 'circulant':
        idx = (ar[None, :] - ar[:, None]) % T
        c = torch.bincount(idx.reshape(-1), weights=G.reshape(-1), minlength=T) / T
        lam = torch.fft.fft(c.to(torch.complex128)).real
        return lam.sort(descending=True).values.clamp_min(0)
    if kind == 'toeplitz':
        idx = (ar[None, :] - ar[:, None]).abs()
        tot = torch.bincount(idx.reshape(-1), weights=G.reshape(-1), minlength=T)
        cnt = torch.bincount(idx.reshape(-1), minlength=T).to(torch.float64)
        g = tot / cnt
        return torch.linalg.eigvalsh(g[idx]).flip(0).clamp_min(0)
    raise ValueError("Error: unknown kind {!r}.".format(kind))


def estimate_rank_pa(chunks, q=48, margin=1.05, null='toeplitz', deflate=True, spread='mp', calibrate=0,
                     mc_rows=4096, method='gram', center=True, exclude_identity=False, n_eff='moment',
                     dtype=torch.float64, seed=0, chunk_rows=None):
    """Common-mode rank by closed-form (deflated) parallel analysis, in one streaming pass.

    One pass accumulates the float64 Gram ``G`` (or the TSQR factor, ``G = R^T R``)
    together with the per-row periodogram moments needed for ``N_eff``
    (see `parallel_analysis_null`). The null for component ``k + 1`` is the
    leading expected singular value of the shift-invariant projection
    (`shift_invariant_eigenvalues`) of the *deflated* Gram
    ``G_k = G - sum_{j<=k} S_j^2 v_j v_j^T`` (the Gram of
    ``X (I - V_k V_k^T)``), spread by the Marchenko-Pastur deterministic
    equivalent; the rank is the length of the leading run of
    ``S_{k+1} > margin * null_{k+1}``. This is Horn (1965) parallel analysis
    with permutation-type surrogates (Buja-Eyuboglu 1992) evaluated in closed
    form as in deterministic parallel analysis, with the deflation that
    prevents strong components from shadowing weaker ones (Dobriban-Owen
    2019, "deflated DPA"): undeflated PA compares the k-th data value with the
    k-th null value although the data's noise spectrum starts r indices
    lower. For the circulant null the deflated spectrum is also
    ``lam(f) - sum_j S_j^2 |FFT(v_j)(f)|^2 / T``.

    Parameters
    ----------
    chunks : chunk source
        See `iter_chunks`; a single pass is made.
    q : int, optional
        Maximum rank examined (default : 48).
    margin : float, optional
        Relative margin over the null (default : 1.05), covering Tracy-Widom
        fluctuations of the leading null eigenvalue and model misfit.
    null : {'toeplitz', 'circulant'}, optional
        ``'circulant'`` is the exact expectation of circular-shift surrogates
        (`parallel_analysis_null`); ``'toeplitz'`` uses linear
        autocovariances and avoids the wrap-around artefact, which inflates
        the data-to-null ratio of strongly autocorrelated noise by
        ``O(xi / T)`` (e.g. ~8% for AR(1) phi = 0.97 at T = 256, <1% at
        T = 1024) (default : 'toeplitz').
    deflate : bool, optional
        Deflated (sequential) comparison (default : True). False compares
        ``S_k`` with the k-th value of the undeflated null.
    spread : {'mp', 'none'}, optional
        Finite-sample spreading of the null (default : 'mp').
    calibrate : int, optional
        If ``> 0``, that many explicit circular-shift surrogates of a uniform
        random row subsample (``mc_rows`` rows, kept during the same pass by a
        bottom-k sketch) calibrate the margin as
        ``max(margin, max_surrogates s_1 / null_1)`` against the subsample's
        own closed-form null; conservative at full size since fluctuations
        shrink with ``N`` (default : 0).
    mc_rows : int, optional
        Subsample size for `calibrate` (default : 4096).
    method : {'gram', 'tsqr'}, optional
        Factor for ``S`` and ``V`` (default : 'gram').
    center : bool, optional
        Centre every row first (default : True).
    exclude_identity : bool, optional
        Circulant null with shifts on ``{1, .., T-1}`` (default : False).
    n_eff : {'moment', 'energy', 'rows'} or float, optional
        Row count for the spreading (default : 'moment').
    dtype : torch.dtype, optional
        Dtype of per-chunk products (default : float64).
    seed : int, optional
        Seed for the subsample and calibration surrogates (default : 0).
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.

    Returns
    -------
    int
        Estimated rank, at most `q`.
    torch.Tensor
        Leading ``q`` data singular values.
    torch.Tensor
        Thresholds ``margin * null_k`` for the components examined (the run
        stops at the first failure, so its length is ``rank + 1`` unless all
        `q` pass).
    """
    if method not in ('gram', 'tsqr'):
        raise ValueError(method)
    if null not in ('toeplitz', 'circulant'):
        raise ValueError(null)
    acc = _Moments(method, dtype, center, keep_rows=mc_rows if calibrate > 0 else 0, seed=seed, need_gram=True)
    T = None
    for X in iter_chunks(chunks, chunk_rows):
        T = X.shape[1]
        acc.add(X)
    if T is None:
        raise ValueError('Error: empty chunk source.')
    lam_p, _, ne = acc.finish(T)
    G = acc.G
    if method == 'tsqr':
        _, S, Vh = torch.linalg.svd(acc.red.result(), full_matrices=False)
        V = Vh.T
    else:
        ev, V = _sym_eigh_desc(G)
        S = ev.clamp_min(0).sqrt()
    q = min(q, T, acc.n)
    S = S[:q]
    nr = ne[n_eff] if isinstance(n_eff, str) else float(n_eff)

    def pop_of(Gk):
        if null == 'circulant' and exclude_identity:
            C = _circulant_from_gram(Gk)
            return torch.linalg.eigvalsh((T * C - Gk) / (T - 1)).flip(0).clamp_min(0)
        return shift_invariant_eigenvalues(Gk, null)

    def spread_of(pop, qq):
        if spread == 'none':
            return pop[:qq].sqrt()
        if spread != 'mp':
            raise ValueError(spread)
        return deterministic_equivalent_spectrum(pop, nr, q=qq).clamp_min(0).sqrt()

    marg = float(margin)
    if calibrate > 0:
        Xs = acc.sub
        Gs = Xs.T.to(torch.float64) @ Xs.to(torch.float64)
        sub_m = _Moments('none', None, False)
        sub_m.add(Xs)
        _, _, ne_s = sub_m.finish(T)
        nr_s = ne_s[n_eff] if isinstance(n_eff, str) else float(n_eff) * Xs.shape[0] / acc.n
        pop_s = shift_invariant_eigenvalues(Gs, 'circulant')
        null_s = deterministic_equivalent_spectrum(pop_s, nr_s, q=1).clamp_min(0).sqrt()[0] \
            if spread == 'mp' else pop_s[0].sqrt()
        gen = torch.Generator(device=Xs.device).manual_seed(int(seed) + 1)
        for _ in range(int(calibrate)):
            sv = float(torch.linalg.svdvals(_shift_surrogate(Xs, gen))[0])
            marg = max(marg, sv / float(null_s))

    thr = []
    if not deflate:
        nul = spread_of(pop_of(G), q)
        thr = marg * nul
        above = S > thr
        return int(torch.cumprod(above.to(torch.int64), 0).sum()), S, thr
    k = 0
    while k < q:
        Gk = G - (V[:, :k] * S[:k] ** 2) @ V[:, :k].T if k else G
        t = marg * spread_of(pop_of(Gk), 1)[0]
        thr.append(t)
        if not bool(S[k] > t):
            break
        k += 1
    return k, S, torch.stack(thr)


def _circulant_from_gram(G):
    T = G.shape[0]
    ar = torch.arange(T, device=G.device)
    idx = (ar[None, :] - ar[:, None]) % T
    c = torch.bincount(idx.reshape(-1), weights=G.to(torch.float64).reshape(-1), minlength=T) / T
    return c[idx]


def marchenko_pastur_rank(chunks, q=48, margin=1.05, center=True, dtype=torch.float64, chunk_rows=None):
    """White-noise rank: singular values above ``margin sigma (sqrt N + sqrt T)`` (Marchenko-Pastur edge).

    Only valid for temporally white idiosyncratic noise; provided for comparison.

    Parameters
    ----------
    chunks : chunk source
        See `iter_chunks`; a single pass is made.
    q : int, optional
        Number of leading components examined (default : 48).
    margin : float, optional
        Relative margin over the edge (default : 1.05).
    center : bool, optional
        Centre every row first (default : True).
    dtype : torch.dtype, optional
        Dtype of per-chunk products (default : float64).
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.

    Returns
    -------
    int
        Number of singular values above the edge (at most `q`).
    torch.Tensor
        Leading singular values.
    """
    G = None
    n = 0
    for X in iter_chunks(chunks, chunk_rows):
        X = _prep(X, dtype, center)
        n += X.shape[0]
        P = (X.T @ X).to(torch.float64)
        G = P if G is None else G + P
    T = G.shape[0]
    S = torch.linalg.eigvalsh(G).flip(0).clamp_min(0).sqrt()[:min(q, T, n)]
    sigma = math.sqrt(float(torch.trace(G)) / (n * T))
    edge = margin * sigma * (math.sqrt(n) + math.sqrt(T))
    return int((S > edge).sum()), S


# ---------------------------------------------------------------------------
# Brand incremental SVD
# ---------------------------------------------------------------------------

def brand_update_cols(U, S, V, C, k=None, rtol=None):
    """Rank-`k` thin-SVD update for appended columns (Brand 2006).

    Given ``X ~ U diag(S) V^T`` (``U``: ``m x r``, ``V``: ``n x r``) and new
    columns ``C`` (``m x c``), with ``L = U^T C`` and ``H = C - U L = J K``
    (QR, after a second Gram-Schmidt pass, "twice is enough"),

        ``[X, C] = [U, J] [[diag S, L], [0, K]] [[V, 0], [0, I]]^T``,

    so one SVD of the small ``(r + c)`` square core updates the factors;
    the result is truncated to rank `k`. Cost ``O((m + n)(r + c)^2)``.
    Exact (up to rounding) when no truncation occurs.

    Parameters
    ----------
    U, S, V : torch.Tensor
        Current thin SVD (``r`` may be 0).
    C : torch.Tensor
        New columns ``(m, c)``.
    k : int, optional
        Rank kept (default : ``r + c``, no truncation).
    rtol : float, optional
        New directions with ``|K_jj| < rtol * max(S)`` are dropped as
        numerically in ``span(U)`` (default : ``m * eps``).

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor
        Updated ``U`` (``m x k``), ``S`` (``k``), ``V`` (``(n + c) x k``).
    """
    C = torch.as_tensor(C).to(dtype=U.dtype, device=U.device)
    m, c = C.shape
    r = S.numel()
    n = V.shape[0]
    L = U.T @ C
    H = C - U @ L
    L2 = U.T @ H
    H = H - U @ L2
    L = L + L2
    J, K = torch.linalg.qr(H)
    scale = max(float(S.max()) if r else 0.0, float(torch.linalg.matrix_norm(C, 2)) if C.numel() else 0.0)
    rtol = m * torch.finfo(U.dtype).eps if rtol is None else rtol
    keep = torch.diagonal(K).abs() > rtol * max(scale, 1e-300)
    J = J * keep[None, :].to(J.dtype)
    K = K * keep[:, None].to(K.dtype)
    core = torch.zeros(r + c, r + c, dtype=U.dtype, device=U.device)
    if r:
        core[:r, :r] = torch.diag(S.to(U.dtype))
        core[:r, r:] = L
    core[r:, r:] = K
    Uc, Sc, Vch = torch.linalg.svd(core)
    W = Vch.T
    U_new = torch.cat([U, J], 1) @ Uc
    V_new = torch.cat([V @ W[:r], W[r:]], 0) if r else W[r:]
    k = r + c if k is None else min(k, r + c)
    return U_new[:, :k], Sc[:k], V_new[:, :k]


def brand_update_rows(U, S, V, B, k=None, rtol=None):
    """Rank-`k` thin-SVD update for appended rows (Brand 2006, on the transpose).

    Parameters
    ----------
    U, S, V : torch.Tensor
        Current thin SVD ``X ~ U diag(S) V^T``.
    B : torch.Tensor
        New rows ``(b, n)``.
    k : int, optional
        Rank kept (default : no truncation).
    rtol : float, optional
        See `brand_update_cols`.

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor
        Updated ``U`` (``(m + b) x k``), ``S``, ``V`` (``n x k``).
    """
    B = torch.as_tensor(B).to(dtype=V.dtype, device=V.device)
    V2, S2, U2 = brand_update_cols(V, S, U, B.T, k, rtol)
    return U2, S2, V2


def incremental_svd(blocks, k, axis=1, dtype=torch.float64, reorth_every=0):
    """Rank-`k` SVD of a matrix built block by block with Brand updates.

    Parameters
    ----------
    blocks : iterable of torch.Tensor
        Column blocks ``(m, c_j)`` (``axis=1``, e.g. new time steps) or row
        blocks ``(b_j, n)`` (``axis=0``, e.g. new agents).
    k : int
        Rank kept after every update.
    axis : {0, 1}, optional
        Direction of growth (default : 1, columns).
    dtype : torch.dtype, optional
        Working dtype (default : float64).
    reorth_every : int, optional
        Re-orthonormalise ``U`` and ``V`` by QR every that many updates to
        stop the slow loss of orthogonality Brand reports (default : 0, never).

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor
        ``U``, ``S``, ``V`` of the truncated SVD.
    """
    U = S = V = None
    for j, Bk in enumerate(blocks):
        Bk = torch.as_tensor(Bk).to(dtype)
        if U is None:
            U, S, Vh = torch.linalg.svd(Bk, full_matrices=False)
            U, S, V = U[:, :k], S[:k], Vh.T[:, :k]
            continue
        if axis == 1:
            U, S, V = brand_update_cols(U, S, V, Bk, k)
        elif axis == 0:
            U, S, V = brand_update_rows(U, S, V, Bk, k)
        else:
            raise ValueError(axis)
        if reorth_every and j % reorth_every == 0:
            Qu, Ru = torch.linalg.qr(U)
            Qv, Rv = torch.linalg.qr(V)
            Uc, S, Vch = torch.linalg.svd(Ru @ torch.diag(S) @ Rv.T)
            U, V = Qu @ Uc, Qv @ Vch.T
    if U is None:
        raise ValueError('Error: no blocks.')
    return U, S, V


# ---------------------------------------------------------------------------
# causal projection in agent space
# ---------------------------------------------------------------------------

class CausalProjector:
    """Remove a past window's common mode from a later window, in agent space.

    `fit` computes ``S, V`` of the past window ``P`` (``N x T0``) with
    `tall_svd`; the agent loadings ``U = P V S^{-1}`` (``N x k``,
    orthonormal columns) are never stored but recomputed chunk by chunk with
    `left_factors`. A later window ``L`` (``N x T1``, other time indices, so
    the time-space projector ``V V^T`` does not apply) is cleaned as
    ``R = L - U (U^T L)``: one pass accumulates ``W = U^T L = sum_c U_c^T L_c``
    (``k x T1``) and a second yields ``R_c = L_c - U_c W``. Nothing after
    ``T0`` is used to learn ``U``, so the projection is causal.

    Parameters
    ----------
    k : int
        Number of common components removed.
    method : str, optional
        `tall_svd` method (default : 'gram').
    center : bool, optional
        Centre the rows of each window (default : True).
    dtype : torch.dtype, optional
        Working dtype (default : float64).
    """

    def __init__(self, k, method='gram', center=True, dtype=torch.float64):
        self.k, self.method, self.center, self.dtype = int(k), method, center, dtype
        self.S = self.V = None

    def fit(self, past, chunk_rows=None):
        """Learn ``S, V`` on the past window (chunk source). Returns self."""
        self.S, self.V = tall_svd(past, k=self.k, method=self.method, center=self.center, dtype=self.dtype,
                                  chunk_rows=chunk_rows)
        return self

    def loadings(self, past_chunk):
        """Agent loadings ``U_c`` (``n_c x k``) of the rows of one past chunk."""
        return left_factors(past_chunk, self.S, self.V, self.k, center=self.center)

    def coefficients(self, pairs):
        """``W = U^T L`` (``k x T1``) from an iterable of aligned ``(past_chunk, later_chunk)`` pairs."""
        W = None
        for P, L in pairs:
            if P.shape[0] != L.shape[0]:
                raise ValueError('Error: past and later chunks must hold the same rows.')
            Uc = self.loadings(P)
            Lc = _prep(L.to(Uc.device), self.dtype, self.center).to(Uc.dtype)
            w = Uc.T @ Lc
            W = w if W is None else W + w
        return W

    def residual(self, past_chunk, later_chunk, W):
        """``R_c = L_c - U_c W`` for one aligned pair of chunks."""
        Uc = self.loadings(past_chunk)
        Lc = _prep(later_chunk.to(Uc.device), self.dtype, self.center).to(Uc.dtype)
        return Lc - Uc @ W

    def transform(self, pairs_factory):
        """Generator of residual chunks; `pairs_factory()` returns fresh aligned ``(past, later)`` pairs."""
        W = self.coefficients(pairs_factory())
        for P, L in pairs_factory():
            yield self.residual(P, L, W)


def causal_projector(chunks, t0, k, t1=None, method='gram', center=True, dtype=torch.float64, chunk_rows=None):
    """Residual chunks of ``X[:, t0:t1]`` after removing the common mode learned on ``X[:, :t0]``.

    Convenience wrapper of `CausalProjector` for one re-iterable chunk source
    whose columns are split at `t0` (three passes: fit, coefficients,
    residuals).

    Parameters
    ----------
    chunks : chunk source
        Re-iterable source of ``(n_c, T)`` chunks.
    t0 : int
        End of the past (training) window.
    k : int
        Number of common components removed.
    t1 : int, optional
        End of the later window (default : T).
    method, center, dtype
        See `CausalProjector`.
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.

    Returns
    -------
    generator of torch.Tensor
        Residual chunks ``(n_c, t1 - t0)``.
    """
    src = _reiterable(chunks, chunk_rows)
    cp = CausalProjector(k, method, center, dtype).fit(lambda: (X[:, :t0] for X in src()))
    return cp.transform(lambda: ((X[:, :t0], X[:, t0:t1]) for X in src()))
