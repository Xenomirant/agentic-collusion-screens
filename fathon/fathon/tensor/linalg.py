#    linalg.py - streamed tall-skinny linear algebra for population matrices (torch backend)
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

"""Streamed tall-skinny numerics for the ``N agents x T rounds`` detector.

The population matrix ``X`` has ``N`` up to several million rows and
``T ~ 256-2048`` columns and arrives in row chunks. Every routine here makes one
pass per stage over the chunks and keeps only ``T x T`` state, so ``N`` is
unbounded; all dense work is delegated to torch (``torch.linalg.qr / eigh /
svd``, ``torch.fft``). What remains is project-specific glue:

* `tsqr`, `gram`, `tall_svd`, `left_factors`: singular values and top right
  singular subspace of the whole stacked population (TSQR or float64 Gram;
  optional CANS polar SVD of the small factor via `load_cans_svd`);
* `periodogram_eigenvalues`, `parallel_analysis_null`, `estimate_rank_pa`: the
  number of common components against a null that keeps each row's own
  autocorrelation, in closed form (summed periodogram / pooled autocovariance,
  spread by Wishart draws of the matching Gaussian model), optionally
  calibrated by explicit circular-shift surrogates accumulated in the same pass
  (on by default, for unequal row variances); smooth common shapes (trends,
  steps, slow drifts) are indistinguishable from persistent noise under the
  default Toeplitz null and are reported as untestable rather than counted;
* `CausalProjector`, `causal_projector`: agent-space loadings learned on a
  burn-in window, projected out of later windows.

``experiments/bench_linalg.py`` (H100, rank-4 common mode + noise) measured that
`gram` (chunk GEMMs, float64 sum, ``eigh``) is the fastest option at every size
(0.03-1.7 s for N = 1e5-4e6, T = 256-1024, on a GPU shared with other jobs).
`tsqr` is 5-10x slower (Householder QR on the
GPU) but backward stable, so it is the choice for ill-conditioned data.
``torch.svd_lowrank`` needs the least memory and recovers the strong
components, but not the noise ones. ``torch.linalg.svd`` of the float32 matrix
needs the matrix in memory and is float32-accurate only. CANS is faster than
``torch.linalg.svd`` on the small R factor at T = 1024; its own singular values
are only ~1e-9 accurate, so `tall_svd` recomputes them as ``||R v_j||``.

float32 GEMMs on CUDA run in IEEE float32 inside this module (`_ieee_fp32`):
PyTorch may round them to TF32, which NGC containers enable by default
(``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``) and which costs three decades on the
noise singular values of a population Gram (1e-2 instead of 1e-5 relative).

A *chunk source* is a 2-D tensor or array (split into blocks of `chunk_rows`
rows), a list/tuple of ``(n_c, T)`` chunks, a zero-argument callable returning a
fresh iterator of chunks, or (single-pass functions only) any iterator.
Computations run on the device of the chunks.
"""

import contextlib
import math
import os
import sys
import warnings

import numpy as np
import scipy.linalg as scipy_linalg
import torch

__all__ = ['iter_chunks', 'tsqr', 'gram', 'tall_svd', 'left_factors', 'load_cans_svd', 'subspace_angles',
           'periodogram_eigenvalues', 'GaussianNull', 'parallel_analysis_null',
           'auto_margin', 'estimate_rank_pa', 'CausalProjector', 'causal_projector']


# ---------------------------------------------------------------------------
# chunk sources
# ---------------------------------------------------------------------------

_DEFAULT_ROWS = 1 << 16


def iter_chunks(chunks, chunk_rows=None):
    """Iterate over the ``(n_c, T)`` row chunks of a chunk source (see the module docstring)."""
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
        yield c[None] if c.ndim == 1 else c


def _reiterable(chunks, chunk_rows=None):
    """Zero-argument factory of fresh chunk iterators (for multi-pass use)."""
    if torch.is_tensor(chunks) or isinstance(chunks, (np.ndarray, list, tuple)) or callable(chunks):
        return lambda: iter_chunks(chunks, chunk_rows)
    raise TypeError('Error: several passes are needed; pass a tensor, a list of chunks or a '
                    'zero-argument callable returning a fresh iterator.')


def _prep(X, dtype, center, standardize=False):
    X = X.to(dtype if dtype is not None else (X.dtype if X.is_floating_point() else torch.float64))
    X = X - X.mean(1, keepdim=True) if center else X
    if standardize:                       # unit mean square per row; all-zero rows stay zero
        rms = X.pow(2).mean(1, keepdim=True).sqrt()
        X = X / torch.where(rms > 0, rms, torch.ones_like(rms))
    return X


@contextlib.contextmanager
def _ieee_fp32(*tensors):
    """Run the float32 CUDA GEMMs of the block in IEEE float32 rather than TF32.

    TF32 keeps 10 mantissa bits (~5e-4 relative per product). It is on whenever
    ``torch.backends.cuda.matmul.fp32_precision == 'tf32'`` (``allow_tf32``),
    the default in NGC containers through ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``.
    Measured (N = 5e5, T = 1024, rank 4 + noise, float32 `gram`): relative error
    of the noise singular values (~1e-2 S[0]) 1.3e-2 with TF32, 1.3e-5 without.
    The global flag is switched for the block only and then restored, so the
    caller's own GEMMs (chunk generation, say) keep their setting; not
    thread-safe. A no-op unless one of `tensors` is a float32 CUDA tensor.
    """
    if not any(t.is_cuda and t.dtype == torch.float32 for t in tensors):
        yield
        return
    m = torch.backends.cuda.matmul
    try:
        old, attr = m.fp32_precision, 'fp32_precision'
        setattr(m, attr, 'ieee')
    except (AttributeError, RuntimeError):     # older torch, or a caller who set the legacy flag
        old, attr = m.allow_tf32, 'allow_tf32'
        setattr(m, attr, False)
    try:
        yield
    finally:
        setattr(m, attr, old)


def subspace_angles(A, B):
    """Principal angles (radians, increasing) between the column spaces of `A` and `B`.

    Thin wrapper of ``scipy.linalg.subspace_angles`` (Bjorck & Golub 1973, Math.
    Comp. 27: 579, with the sine formula for small angles; Knyazev & Argentati
    2002, SIAM J. Sci. Comput. 23: 2008), evaluated in float64 on the CPU.
    Returns ``min(k_A, k_B)`` angles, on `A`'s device.
    """
    def np64(M):
        return M.detach().to('cpu', torch.float64).numpy() if torch.is_tensor(M) else np.asarray(M, np.float64)
    ang = np.sort(scipy_linalg.subspace_angles(np64(A), np64(B)))
    return torch.as_tensor(ang, dtype=torch.float64, device=A.device if torch.is_tensor(A) else None)


# ---------------------------------------------------------------------------
# streamed factorizations
# ---------------------------------------------------------------------------

def tsqr(chunks, dtype=torch.float64, center=False, chunk_rows=None, fan_in=16):
    """R factor of the stacked chunks by TSQR, ``X = Q R`` with ``Q`` never formed.

    Each chunk is reduced to its ``R`` by ``torch.linalg.qr(mode='r')``; the small
    factors are stacked and re-factored whenever `fan_in` of them are held (a flat
    reduction tree; Demmel, Grigori, Hoemmen & Langou 2012, SIAM J. Sci. Comput.
    34: A206). Householder QR at every level makes the result backward stable
    whatever the conditioning, unlike the Gram route.

    Parameters
    ----------
    chunks : chunk source
        One pass is made.
    dtype : torch.dtype, optional
        Working dtype of the local QRs (default : float64; float32 chunks are
        upcast one chunk at a time).
    center : bool, optional
        Centre every row first (default : False).
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.
    fan_in : int, optional
        Number of stacked R factors per reduction (default : 16).

    Returns
    -------
    torch.Tensor
        Upper triangular ``R``, ``(min(N, T), T)``, nonnegative diagonal,
        ``R^T R = X^T X``.
    """
    Rs = []
    for X in iter_chunks(chunks, chunk_rows):
        Rs.append(torch.linalg.qr(_prep(X, dtype, center), mode='r').R)
        if len(Rs) >= fan_in:
            Rs = [torch.linalg.qr(torch.cat(Rs), mode='r').R]
    if not Rs:
        raise ValueError('Error: empty chunk source.')
    R = torch.linalg.qr(torch.cat(Rs), mode='r').R if len(Rs) > 1 else Rs[0]
    d = torch.sign(torch.diagonal(R))
    return R * torch.where(d == 0, torch.ones_like(d), d)[:, None]


def gram(chunks, dtype=torch.float64, center=False, chunk_rows=None):
    """Streamed Gram matrix ``X^T X = sum_c X_c^T X_c``, summed in float64.

    With ``dtype=float32`` the per-chunk GEMM runs in float32 (fast on GPUs; in
    IEEE float32 even where TF32 is enabled, see `_ieee_fp32`) and only the
    partial sums are float64. Squaring the condition number limits the singular
    values resolved to ``~sqrt(eps) S[0]`` of the GEMM dtype (``1.5e-8`` for
    float64, ``3.5e-4`` for float32).

    Returns
    -------
    torch.Tensor
        ``(T, T)`` float64 Gram matrix.
    """
    return _gram_n(chunks, dtype, center, chunk_rows)[0]


def _gram_n(chunks, dtype=torch.float64, center=False, chunk_rows=None):
    """`gram` plus the number of rows seen."""
    G, n = None, 0
    for X in iter_chunks(chunks, chunk_rows):
        X = _prep(X, dtype, center)
        with _ieee_fp32(X):
            P = (X.T @ X).to(torch.float64)
        G = P if G is None else G + P
        n += X.shape[0]
    if G is None:
        raise ValueError('Error: empty chunk source.')
    return G, n


def _eigh_desc(G):
    lam, V = torch.linalg.eigh(0.5 * (G + G.T))
    return lam.flip(0), V.flip(1)


_CANS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                             'external', 'cans_svd_src', 'cans_svd')


def load_cans_svd(path=None):
    """Import ``cans_svd`` from the external CANS repository (not vendored; it has no licence).

    CANS (Chebyshev-accelerated Newton-Schulz; Grishina, Kuznetsov & Oseledets
    2025, arXiv:2506.10935) computes the polar factor ``W`` of a matrix with
    GEMM-only iterations and the SVD from ``eigh(W^T A)``. The repository root is
    on ``sys.path`` only during the import.

    Parameters
    ----------
    path : str, optional
        Root containing ``src_torch`` (default : ``$FATHON_CANS_PATH`` or
        ``<project>/external/cans_svd_src/cans_svd``).

    Returns
    -------
    callable
        ``cans_svd(matrix, degree=3, preprocess=True, preprocess_iters=2,
        delta=0.99, a=1e-3, n=50, eps_qr=1e-5) -> (U, s, Vt)``.

    Raises
    ------
    ImportError
        If the repository is absent or fails to import.
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


def tall_svd(chunks, k=None, method='tsqr', center=False, dtype=torch.float64, chunk_rows=None,
             cans_kwargs=None):
    """Singular values and right singular vectors of the stacked tall matrix, ``X = U diag(S) V^T``.

    Only a ``T x T`` factor is formed (the R-SVD; Chan 1982, ACM TOMS 8: 72);
    `left_factors` gives the rows of ``U`` of any chunk on demand.

    Methods
    -------
    * ``'tsqr'``: ``torch.linalg.svd`` of the `tsqr` factor; backward stable,
      resolves singular values down to ``~eps S[0]``. One pass.
    * ``'gram'``: ``torch.linalg.eigh`` of the float64 `gram`; cheapest, but
      singular values below ``~1e-8 S[0]`` (float64 GEMM) or ``~3e-4 S[0]``
      (float32 GEMM) are not resolved. One pass.
    * ``'cans'``: TSQR, then the CANS polar SVD of ``R`` (`load_cans_svd`). Its
      right vectors are accurate (top-8 subspace angle 1e-14 at condition
      numbers 1e3-1e8), its singular values are not: with the default
      ``cans_kwargs`` they carry a uniform relative error of ~1e-9 (2.6e-6 for
      the small ones at condition number 1e8), which tightening ``eps_qr``
      makes worse. The singular values returned are therefore recomputed as
      ``S_j = ||R v_j||`` (one ``T x T`` GEMM), which brings them to 3e-15
      (cond 1e3) and 1e-10 (cond 1e8), on a par with ``'tsqr'``.
    * ``'torch'``: concatenate the chunks and call ``torch.linalg.svd``
      (reference; the whole matrix must fit in memory).

    Parameters
    ----------
    chunks : chunk source
    k : int, optional
        Number of leading triplets kept (default : all ``min(N, T)``).
    method : {'tsqr', 'gram', 'cans', 'torch'}, optional
        Default : 'tsqr'.
    center : bool, optional
        Centre every row first (default : False).
    dtype : torch.dtype, optional
        Working dtype (default : float64).
    chunk_rows : int, optional
        Rows per block when `chunks` is a single matrix.
    cans_kwargs : dict, optional
        Keyword arguments of ``cans_svd``.

    Returns
    -------
    torch.Tensor
        ``S``, decreasing, ``(r,)``.
    torch.Tensor
        ``V``, ``(T, r)``, ``r = min(k, N, T)`` for every method (with fewer
        rows than columns, ``'gram'`` drops the ``T - N`` eigenvalues that are
        zero up to rounding).
    """
    if method == 'gram':
        G, n = _gram_n(chunks, dtype, center, chunk_rows)
        lam, V = _eigh_desc(G)
        S, V = lam[:n].clamp_min(0).sqrt(), V[:, :n]
    elif method == 'tsqr':
        _, S, Vh = torch.linalg.svd(tsqr(chunks, dtype, center, chunk_rows), full_matrices=False)
        V = Vh.T
    elif method == 'cans':
        cans_svd = load_cans_svd()
        R = tsqr(chunks, dtype, center, chunk_rows)
        with _ieee_fp32(R):
            V = cans_svd(R, **(cans_kwargs or {}))[2].T
            S = (R @ V).norm(dim=0)                    # CANS' own values are ~1e-9 accurate only
        order = torch.argsort(S, descending=True)
        S, V = S[order], V[:, order]
    elif method == 'torch':
        X = torch.cat([_prep(c, dtype, center) for c in iter_chunks(chunks, chunk_rows)])
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.T
    else:
        raise ValueError('Error: unknown method {!r}.'.format(method))
    r = min(S.numel(), V.shape[1])
    k = r if k is None else min(int(k), r)
    return S[:k], V[:, :k]


# exponent a of the default rtol = eps^a of `left_factors`, per `tall_svd` method
_RESOLUTION = {'tsqr': 0.5, 'torch': 0.5, 'cans': 0.5, 'gram': 0.25}


def _default_rtol(method, dtype):
    if method not in _RESOLUTION:
        raise ValueError('Error: unknown method {!r}.'.format(method))
    return torch.finfo(dtype).eps ** _RESOLUTION[method]


def left_factors(X_chunk, S, V, k=None, center=False, rtol=None, method='tsqr'):
    """Rows of ``U`` belonging to one chunk, ``U_c = X_c V_k diag(S_k)^{-1}``.

    Since ``X V = U diag(S)``, ``U`` for millions of rows is produced chunk by
    chunk and never stored. Only resolved components get a column; the others
    get a zero column, so ``U`` is orthonormal on the resolved components only.
    The error of a column grows as ``eps S[0] / S_j`` for the backward-stable
    R-SVD (``'tsqr'``, ``'cans'``, ``'torch'``) and as ``eps (S[0] / S_j)^2`` for
    ``'gram'`` (squared condition number). Measured on a 6000 x 96 matrix with
    singular values spread over 8-12 decades, ``max |U^T U - I|`` over the kept
    columns is 6e-9 for tsqr with ``S_j / S[0] > 1e-8``, and 5e-9 for gram with
    ``S_j / S[0] > 1e-4`` (0.27 if gram columns down to 1e-8 are kept). Hence
    the default ``rtol``: ``eps^(1/2)`` for the R-SVD methods, ``eps^(1/4)`` for
    ``'gram'`` (``eps`` of `V`'s dtype; pass ``rtol = 0.02`` for a float32-GEMM
    gram factor). With `center`, the direction annihilated by the row centring
    (right singular vector ``~ 1 / sqrt(T)``, singular value pure rounding,
    which can exceed any fixed ``rtol``) is zeroed as well.

    Parameters
    ----------
    X_chunk : tensor or array, ``(n_c, T)``
    S, V : torch.Tensor
        From `tall_svd`.
    k : int, optional
        Number of leading components (default : all columns of `V`).
    center : bool, optional
        Must match the `tall_svd` call (default : False).
    rtol : float, optional
        Components with ``S_j <= rtol S[0]`` get a zero column (default : above).
    method : {'tsqr', 'gram', 'cans', 'torch'}, optional
        `tall_svd` method that produced `S, V`; sets the default `rtol`
        (default : 'tsqr').

    Returns
    -------
    torch.Tensor
        ``(n_c, k)`` in `V`'s dtype and device.
    """
    k = V.shape[1] if k is None else k
    Vk, Sk = V[:, :k], S[:k].to(V.dtype)
    X = _prep(torch.as_tensor(X_chunk).to(V.device), V.dtype, center)
    rtol = _default_rtol(method, V.dtype) if rtol is None else rtol
    keep = Sk > rtol * S[0]
    if center:
        keep &= Vk.sum(0).abs() < 0.5 * math.sqrt(V.shape[0])
    inv = torch.where(keep, 1.0 / Sk, torch.zeros_like(Sk))
    with _ieee_fp32(X):
        return (X @ Vk) * inv


# ---------------------------------------------------------------------------
# parallel analysis: a null that keeps every row's autocorrelation
# ---------------------------------------------------------------------------

class _NullStats:
    """One pass: float64 Gram, summed periodogram, row-energy moments, optional surrogate Grams."""

    def __init__(self, dtype, center, need_gram=True, n_surr=0, seed=0, standardize=False):
        self.dtype, self.center, self.need_gram, self.standardize = dtype, center, need_gram, standardize
        self.n_surr, self.seed = int(n_surr), seed
        self.G = self.P = self.gen = self.T = None
        self.Gs = [None] * self.n_surr
        self.n, self.e1, self.e2, self.p4 = 0, 0.0, 0.0, 0.0

    def add(self, X):
        X = _prep(X, self.dtype, self.center, self.standardize)
        n, T = X.shape
        self.T = T
        if self.need_gram:
            with _ieee_fp32(X):
                P = (X.T @ X).to(torch.float64)
            self.G = P if self.G is None else self.G + P
        F = torch.fft.rfft(X, dim=1)
        Pi = (F.real ** 2 + F.imag ** 2).to(torch.float64)          # per-row half periodogram
        self.P = Pi.sum(0) if self.P is None else self.P + Pi.sum(0)
        w = torch.full((Pi.shape[1],), 2.0, dtype=torch.float64, device=Pi.device)   # rfft -> full DFT
        w[0] = 1.0
        if T % 2 == 0:
            w[-1] = 1.0
        self.p4 += float(((Pi ** 2) * w).sum())
        a = (X.to(torch.float64) ** 2).sum(1)
        self.e1 += float(a.sum())
        self.e2 += float((a ** 2).sum())
        self.n += n
        if self.n_surr:
            if self.gen is None:
                self.gen = torch.Generator(device=X.device).manual_seed(int(self.seed))
            ar = torch.arange(T, device=X.device)
            for j in range(self.n_surr):
                sh = torch.randint(0, T, (n,), device=X.device, generator=self.gen)
                Y = torch.gather(X, 1, (ar[None, :] + sh[:, None]) % T)
                with _ieee_fp32(Y):
                    P = (Y.T @ Y).to(torch.float64)
                self.Gs[j] = P if self.Gs[j] is None else self.Gs[j] + P

    def lam(self):
        """Eigenvalues of the circulant expected surrogate Gram, in DFT order."""
        T = self.T
        return torch.cat([self.P, self.P[1:(T + 1) // 2].flip(0)]) / T

    def n_eff(self, kind='rows'):
        """Row count of the spread model: ``N``, or the second-moment match (`parallel_analysis_null`)."""
        if not isinstance(kind, str):
            return float(kind)
        if kind == 'rows':
            return float(self.n)
        if kind != 'moment':
            raise ValueError('Error: unknown n_eff {!r}.'.format(kind))
        den = self.e2 - self.p4 / self.T ** 2
        return (self.e1 ** 2 + float((self.lam() ** 2).sum())) / den if den > 0 else float(self.n)


def _stream_stats(chunks, chunk_rows=None, **kw):
    acc = _NullStats(**kw)
    for X in iter_chunks(chunks, chunk_rows):
        acc.add(X)
    if acc.T is None:
        raise ValueError('Error: empty chunk source.')
    return acc


def periodogram_eigenvalues(chunks, center=True, dtype=torch.float64, chunk_rows=None):
    """Eigenvalues ``lam_k = sum_i |FFT(x_i)_k|^2 / T`` of the expected circular-shift-surrogate Gram.

    See `parallel_analysis_null` for the derivation. One pass, no ``T x T`` state.

    Returns
    -------
    torch.Tensor
        ``(T,)`` float64 eigenvalues in DFT order (``lam[k] = lam[T - k]``).
    float
        Second-moment-matched effective row count (``n_eff='moment'`` of
        `parallel_analysis_null`).
    int
        Number of rows.
    """
    acc = _stream_stats(chunks, chunk_rows, dtype=dtype, center=center, need_gram=False)
    return acc.lam(), acc.n_eff('moment'), acc.n


class GaussianNull:
    """Expected leading eigenvalues of the Gram of ``n`` Gaussian rows with expected Gram spectrum `lam`.

    Model: ``Y = Z diag(lam / n)^{1/2}`` with ``n`` i.i.d. standard Gaussian rows, so
    ``E Y^T Y = diag(lam)``; only the spectrum of the expected Gram matters. Its
    sample Gram is ``D W D`` with ``W ~ Wishart_T(n, I)``, drawn by the Bartlett
    decomposition ``W = A A^T`` (``A`` lower triangular, ``A_ii^2 ~ chi^2(n - i)``,
    ``A_ij ~ N(0, 1)`` below the diagonal; Bartlett 1933, Proc. R. Soc. Edinb.
    53: 260; Kshirsagar 1959, Sankhya 21: 1), at cost ``O(T^3)`` whatever ``n``,
    non-integer ``n`` allowed. The expected leading eigenvalues are averaged
    over `draws` Wishart draws, which are drawn once and reused for every
    spectrum passed to `__call__` (common random numbers across the deflation
    steps). This is the exact finite-``(n, T)`` version of the generalized
    Marchenko-Pastur law (Silverstein & Bai 1995, J. Multivar. Anal. 54: 175),
    without its asymptotics or a Stieltjes-transform solver.

    Everything runs in torch on `device` (the chunks' device in
    `estimate_rank_pa`), so on a GPU the ``T x T`` work is not CPU-bound.
    ``scipy.stats.wishart`` draws the same matrices but on the CPU only (and its
    import costs 3-9 s on this host); ``torch.distributions.Wishart`` (torch
    2.14) flags every valid sample as singular, warns and redraws it. The draws
    depend on the device (CPU and CUDA generators differ).

    Parameters
    ----------
    T : int
        Dimension.
    n_rows : float
        (Effective) number of rows.
    draws : int, optional
        Number of Wishart draws (default : 4; one draw has a relative sd of the
        leading singular value of ~0.5-1% at N = 1e3, less for larger N).
    seed : int, optional
        Seed of the draws (default : 0).
    device : torch.device or str, optional
        Device of the draws and eigenvalue problems (default : CPU).
    """

    def __init__(self, T, n_rows, draws=4, seed=0, device=None):
        device = torch.device('cpu' if device is None else device)
        rng = np.random.default_rng(seed)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.n, self.T = float(n_rows), int(T)
        W = []
        for _ in range(draws):
            if self.n > T - 1:
                A = torch.randn(T, T, generator=gen, dtype=torch.float64, device=device).tril(-1)
                chi = np.sqrt(rng.chisquare(self.n - np.arange(T)))
                A.diagonal().copy_(torch.as_tensor(chi, dtype=torch.float64))
            else:                                        # fewer rows than columns: plain Gaussian rows
                A = torch.randn(T, max(int(round(self.n)), 1), generator=gen, dtype=torch.float64, device=device)
            W.append(A @ A.T)
        self.W = torch.stack(W)

    def __call__(self, lam, q=1):
        """Expected ``q`` leading eigenvalues (decreasing, float64 torch tensor on `lam`'s device)."""
        lam_t = torch.as_tensor(lam)
        d = (lam_t.detach().to(self.W.device, torch.float64).clamp_min(0) / self.n).sqrt()
        q = min(int(q), d.numel())
        ev = torch.linalg.eigvalsh(d[:, None] * self.W * d[None, :])        # (draws, T), ascending
        return ev[:, -q:].flip(1).mean(0).to(lam_t.device)


def parallel_analysis_null(chunks, q=None, center=True, spread='wishart', n_eff='rows', draws=4, seed=0,
                           dtype=torch.float64, chunk_rows=None):
    r"""Expected singular values of row-wise circular-shift surrogates, without any surrogate SVD.

    Let ``x_i`` be the (centred) rows and ``S`` the cyclic shift. A surrogate
    stacks ``y_i = S^{s_i} x_i`` with independent uniform ``s_i in {0..T-1}``.
    Then ``E[y_i y_i^T]_{tu} = c_i(u - t) / T`` with the *circular* autocovariance
    ``c_i(h) = sum_t x_i(t) x_i(t + h mod T)``, so the expected Gram is the
    symmetric circulant of ``c = sum_i c_i / T``; the DFT diagonalises it, and by
    circular Wiener-Khinchin its eigenvalues are

        ``lam_k = sum_i |FFT(x_i)_k|^2 / T``  (`periodogram_eigenvalues`),

    with ``sum_k lam_k = sum_i ||x_i||^2`` (Parseval). This is Horn's (1965)
    parallel analysis with the permutation-type surrogates of Buja & Eyuboglu
    (1992, Multivar. Behav. Res. 27: 509), evaluated in closed form. Shifts drawn
    from ``{1..T-1}`` (as in `scale.estimate_rank`) change ``E G`` by ``O(1/T)``.

    An expected Gram is not an expected spectrum: one surrogate's eigenvalues
    spread around `lam` by ``~ sqrt(T / N)``. ``spread='wishart'`` takes the
    expected spectrum of ``N_eff`` Gaussian rows with expected Gram spectrum
    `lam` (`GaussianNull`):
    ``n_eff='rows'`` takes ``N_eff = N``; ``n_eff='moment'`` matches the second
    spectral moment of the surrogate Gram exactly. A surrogate row keeps its
    energy ``a_i`` and periodogram ``P_i(k)``; with independent shifts
    ``E (y_i . y_j)^2 = sum_k P_i(k) P_j(k) / T^2``, hence
    ``E tr G~^2 = sum_i a_i^2 + sum_k lam_k^2 - sum_{i,k} P_i(k)^2 / T^2``,
    while ``n`` Gaussian rows give ``tr Lam^2 + ((tr Lam)^2 + tr Lam^2) / n``;
    equating gives ``N_eff = ((sum a_i)^2 + sum lam_k^2) / (sum a_i^2 - sum P_i(k)^2 / T^2)``.

    Validity: deviation of the three leading null values from the mean of 10
    explicit surrogate SVDs, in %, T = 256 (``tests/test_linalg.py`` checks the
    homogeneous cases)::

        case (range lo/hi)         N = 1000            N = 2000            N = 4000
                                   rows     moment     rows     moment     rows     moment
        white                      +0.1     0.0        +0.1/0.3 +0.1/0.2   0/+0.3   0/+0.2
        AR(1) phi = 0.97           -1.5/0.7 -1.5/0.7   -0.9/0.3 -0.9/0.3   -0.7/0.2 -0.8/0.2
        two spectra, 2x energy     -0.2/1.9 +0.9/2.6   -0.2/1.1 +0.4/1.3   -0.5/0.6 -0.2/0.8
        lognormal sd, sigma 0.25   -7/-6    -3/-2      -6/-5    -2/-1      -3.5/-3  -1/-0.7
        lognormal sd, sigma 0.5    -31/-25  -17/-10    -31/-20  -19/-7     -27/-17  -17/-6
        5% of rows at 3x sd        -26/-24  -11/-10    -19/-18  -7/-6      -14/-14  -4/-4
        5% of rows at 10x sd       -49/-48  -4/-3      -43/-41  -3/-2      -36/-35  -2/-1

    With rows of comparable energy both choices are within 2%. With unequal
    row variances the closed form is too low, and more so the smaller N / T
    (the largest rows act as outliers that no homogeneous bulk model
    captures); 'moment' is much closer than 'rows' but still up to 19% low.
    'rows' stays the default because 'moment' is computed from the raw rows,
    where a common mode with unequal loadings also lowers it and inflates the
    whole null; unequal variances are left to the explicit-surrogate
    calibration of `estimate_rank_pa` (on by default there), which corrects
    either choice.

    Parameters
    ----------
    chunks : chunk source
        One pass.
    q : int, optional
        Number of leading values (default : ``T`` for ``spread='none'``, 48 otherwise).
    center : bool, optional
        Centre every row first (default : True).
    spread : {'wishart', 'none'}, optional
        Finite-sample spread, or the bare expected-Gram spectrum (default : 'wishart').
    n_eff : {'rows', 'moment'} or float, optional
        Row count of the spread model (default : 'rows').
    draws, seed : int, optional
        Wishart draws of `GaussianNull` (defaults : 4, 0).
    dtype : torch.dtype, optional
        Dtype of the per-chunk FFTs (default : float64; sums are float64).
    chunk_rows : int, optional

    Returns
    -------
    torch.Tensor
        Null singular values, decreasing, float64.
    """
    acc = _stream_stats(chunks, chunk_rows, dtype=dtype, center=center, need_gram=False)
    pop = acc.lam().sort(descending=True).values.clamp_min(0)
    if spread == 'none':
        return (pop if q is None else pop[:q]).sqrt()
    if spread != 'wishart':
        raise ValueError('Error: unknown spread {!r}.'.format(spread))
    gn = GaussianNull(acc.T, acc.n_eff(n_eff), draws, seed, device=pop.device)
    return gn(pop, 48 if q is None else q).clamp_min(0).sqrt()


def _null_gram(G, kind, center, tol=1e-10, max_iter=500):
    """Expected Gram of the null, estimated from the (deflated) data Gram ``G``; linear in ``G``.

    ``'circulant'``: average every circular diagonal; for ``G = X^T X`` this is
    the expected circular-shift-surrogate Gram (centred rows stay centred).

    ``'toeplitz'``: pool the linear autocovariances, an estimate of
    ``E G = sum_i Gamma_i`` for independent stationary rows without the
    wrap-around of circular shifts. Row centring does not commute with the
    Toeplitz structure: for centred rows ``E G = P Gamma P``
    (``P = I - 1 1^T / T``), whose diagonal means are the mean-corrected
    autocovariances, biased down at every lag by ``~ S(0) / T`` (Percival 1993,
    Am. Stat. 47: 274), which for AR(1) phi = 0.97 at T = 256 puts the null 3%
    low. With `center`, the Toeplitz ``Gamma`` is solved from
    ``diagmeans(P Gamma P) = diagmeans(G)`` by fixed-point iteration (a
    contraction of size ``O(xi / T)``) and ``P Gamma P`` is returned.
    """
    G = G.to(torch.float64)
    T = G.shape[0]
    ar = torch.arange(T, device=G.device)
    circ = kind == 'circulant'
    idx = (((ar[None, :] - ar[:, None]) % T) if circ else (ar[None, :] - ar[:, None]).abs()).reshape(-1)
    cnt = torch.full((T,), float(T), dtype=torch.float64, device=G.device) if circ else 2.0 * (T - ar).double()

    def lag_sums(M):
        """``sum_t M[t, t + h]`` (circularly for the circulant), h = 0..T-1, as a strided view."""
        Z = torch.cat([M, M if circ else torch.zeros_like(M)], 1)          # (T, 2T): Z[t, t + h]
        return Z.as_strided((T, T), (2 * T + 1, 1)).sum(0)

    def diagmeans(M):                       # mean over the (circular) diagonals at lag h, both triangles
        return (lag_sums(M) if circ else lag_sums(M) + lag_sums(M.T.contiguous())) / cnt

    def center2(M):
        return M - M.mean(0, keepdim=True) - M.mean(1, keepdim=True) + M.mean()

    target = diagmeans(G)
    if kind == 'circulant' or not center:
        return target[idx].reshape(T, T)
    # constant lags are annihilated by P (``P 1 1^T P = 0``), so g is determined up to a constant
    # that may drift; convergence is judged on P Gamma P, which the drift does not change
    g, M = target.clone(), center2(target[idx].reshape(T, T))
    for it in range(1, max_iter + 1):
        g = g + target - diagmeans(M)
        M_new = center2(g[idx].reshape(T, T))
        if it % 5 == 0 and float((M_new - M).abs().max()) <= tol * float(M_new.abs().max()):
            return M_new
        M = M_new
    warnings.warn('estimate_rank_pa: the centring correction of the Toeplitz null did not converge '
                  '(rows nearly non-stationary?).')
    return M


def auto_margin(n_eff, z=3.0, floor=1.05):
    """Default margin of `estimate_rank_pa`: ``max(floor, 1 + z / sqrt(2 n_eff))``.

    A leading null eigenvalue that stands out of the bulk (strongly
    autocorrelated rows, e.g. AR(1) phi = 0.97 at T = 256) fluctuates as a
    supercritical spike: Gaussian with relative sd at most ``sqrt(2 / N)``
    (Paul 2007, Stat. Sinica 17: 1617; Bai & Yao 2008, Ann. IHP 44: 447), i.e.
    at most ``1 / sqrt(2 N)`` for the singular value, much more than the
    Tracy-Widom scale of an edge eigenvalue. Measured sd of ``S_1 / null_1``
    for pure AR(1) phi = 0.97 at T = 256: 0.40 / sqrt(N) (N = 150-1000), against
    the bound 0.71 / sqrt(N); ``z = 3`` on the bound is ~5 measured sd. The
    floor covers model misfit at large N.
    """
    return max(float(floor), 1.0 + float(z) / math.sqrt(2.0 * max(float(n_eff), 1.0)))


def estimate_rank_pa(chunks, q=48, margin='auto', null='toeplitz', spread='wishart', n_eff='rows', draws=4,
                     calibrate=2, standardize=False, factor='gram', center=True, dtype=torch.float64, seed=0,
                     chunk_rows=None, return_info=False):
    """Number of common components by closed-form deflated parallel analysis, in one streamed pass.

    The pass accumulates the float64 Gram ``G`` (plus the summed periodogram,
    row-energy moments and `calibrate` explicit surrogate Grams). The closed-form
    spread model (`GaussianNull`) assumes rows of comparable energy; unequal
    row variances, the rule in real populations, would otherwise read as common
    components, so by default two explicit surrogates calibrate it (see
    `calibrate` and `standardize`). Component
    ``i + 1`` counts when ``S_{i+1} > margin * c * null``, ``null`` being the
    leading value of the null Gram ``Pi(G_k)`` of the deflated Gram
    (`_null_gram`; linear in its argument), spread as in
    `parallel_analysis_null`. Deflation (Dobriban & Owen 2019, JRSS-B 81: 163,
    deflated deterministic parallel analysis) stops strong components from
    shadowing weak ones. Here ``G_{k+1} = G_k - e_k v_k v_k^T`` with ``e_k`` chosen
    so that the deflated Gram and its own null agree along ``v_k``,
    ``S_k^2 - e_k = v_k^T Pi(G_{k+1}) v_k``, i.e.
    ``e_k = (S_k^2 - v_k^T Pi(G_k) v_k) / (1 - m_k)``, ``m_k = v_k^T Pi(v_k v_k^T) v_k``
    (clipped to ``[0, S_k^2]``): the common component and the null energy it
    smeared over all frequencies are removed, the noise's own share along
    ``v_k`` is kept. (Removing all of ``S_k^2`` strips that share, which for a
    slow common mode in strongly autocorrelated noise sits at the null's
    spectral peak and lowers the next null; removing only
    ``S_k^2 - v_k^T Pi(G_k) v_k`` leaves the smear and raises it by 5-10% after a
    strong component.)

    Blind spot. The null is estimated from ``G`` itself, so a component also
    feeds the null it is tested against, with weight ``m = v^T Pi(v v^T) v``
    (the share of its own energy that the null puts back along ``v``). When
    ``m >= 1 / margin^2`` it cannot pass at any amplitude. Under the default
    centred Toeplitz null this happens for smooth shapes whose energy sits at
    the lowest frequencies: a linear trend (``m = 1.000``: random slopes are a
    stationary process in the limit of zero frequency), a half cosine (0.985),
    a logistic step of width T/10 (0.94), a random walk with drift (0.93);
    but also for the leading direction of persistent noise (AR(1) phi = 0.97
    at T = 128: 0.91; fGn H = 0.95: 0.92; AR(1) phi = 0.99 at T = 256: 0.93).
    From one window the two cannot be told apart, so such a component is
    *untestable*: if it fails its test it is neither counted nor deflated, a
    warning names it (``info['untestable']``), and the run continues with the
    next component tested against the next null value (so a strong component
    behind a smooth one is still found). To count smooth common shapes, use
    ``null='circulant'`` (Horn's circular-shift surrogates destroy a
    time-aligned trend: ``m = 0.20`` for a linear trend), which in turn
    overcounts persistent noise (``S_1 / null_1`` = 1.06-1.29 for AR(1)
    phi = 0.97-0.99 and fGn H = 0.95, T = 128-512).

    False positives with the defaults (pure noise, 50 replications each, T = 256
    unless stated): none at N = 1000 for white, AR(1) phi = 0.97, fGn
    H = 0.8 / 0.95, a multiplicative cascade, a mixture of white and AR(0.95)
    rows, lognormal row sd with sigma = 0.25 / 0.5 / 1, 5% of rows at 3x or
    10x sd, and AR(0.97) rows with lognormal sd; none at N = 300 and 150 (white,
    AR(0.97), fGn 0.8; auto margin 1.12 / 1.17), and none at N = 1000, T = 128
    for AR(0.97) / AR(0.99) / fGn 0.95, whose leading direction is untestable
    in 45-50 of the 50 runs. ``S_1 / (c null_1)`` had mean 0.96-0.995 and
    max 1.052. With a fixed ``margin = 1.05`` and no calibration, AR(0.97)
    gave 4 of 50 false positives at N = 300 and N = 150 (sd of ``S_1 / null_1``
    1.25% at N = 1000, 2.2% at 300, 3.4% at 150), hence `auto_margin`.

    Parameters
    ----------
    chunks : chunk source
        One pass (two with ``factor='tsqr'``).
    q : int, optional
        Largest number of components examined (default : 48).
    margin : float or 'auto', optional
        Relative margin over the null (default : 'auto', `auto_margin` of the
        spread model's row count: 1.067 at N = 1e3, 1.05 from N = 1800 on).
    null : {'toeplitz', 'circulant', 'mp'}, optional
        ``'circulant'``: exact expectation of circular-shift surrogates.
        ``'toeplitz'``: pooled *linear* autocovariances, the expected Gram of
        independent stationary rows. Circular shifts wrap a non-periodic series,
        which Fejer-smooths its spectrum and lowers the null peak of strongly
        autocorrelated noise by ``O(xi / T)`` (several % for AR(1) phi = 0.97 at
        T = 256), enough to overcount. Centred rows are handled exactly (see
        `_null_gram`). ``'mp'``: the white-noise Marchenko-Pastur
        edge ``margin sigma (sqrt N + sqrt T)`` (invalid for autocorrelated rows;
        for comparison). Default : 'toeplitz' (see the blind spot above).
    spread : {'wishart', 'none'}, optional
        Finite-sample spread of the null (default : 'wishart', `GaussianNull`).
    n_eff : {'rows', 'moment'} or float, optional
        Row count of the spread model (default : 'rows'; 'moment' is computed
        from the raw rows, so a strong common mode with unequal loadings
        lowers it and inflates the null).
    draws : int, optional
        Wishart draws of `GaussianNull` (default : 4).
    calibrate : int, optional
        Number of explicit circular-shift surrogates of the *whole* population
        accumulated in the same pass (one gather and one extra GEMM per chunk
        each). The factor ``c = max(1, max_s S~_1 / null_1)``, ``null_1`` the
        closed-form circulant value, corrects the closed form where the
        Gaussian spread model of homogeneous rows underestimates: unequal row
        variances. With ``calibrate=0``, lognormal row scales with
        sigma = 0.25 give a false positive in 14 of 20 replications (N = 1e3,
        T = 256, white noise) and sigma = 0.5 or 1 in 20 of 20 (ranks 24-44);
        with ``calibrate=2``, none of 20 (also 5% of rows at 3x or 10x sd, and
        AR(1) phi = 0.97 rows with sigma = 0.5). ``c`` can only raise the
        thresholds. The surrogates also carry the common modes, shifted, and
        loadings make them unequal rows too: with two strong common components
        and Gaussian loadings (N = 2000, T = 256) ``c = 1.13``, i.e. 13% higher
        thresholds for the weaker components. Default : 2 (0 gives ``c = 1``).
    standardize : bool, optional
        Scale every (centred) row to unit variance first (Horn's 1965 analysis
        of the correlation matrix; `S` is then the spectrum of the standardized
        rows). It also removes the false positives of unequal row variances
        with ``calibrate=0``, but only when there is no strong common mode:
        rows dominated by a common mode with unequal loadings are scaled down,
        noise included, which makes the residual noise unequal instead
        (5 common factors with lognormal sigma = 0.5 loadings in AR(1) noise:
        rank 9 instead of 5). Default : False.
    factor : {'gram', 'tsqr'}, optional
        Factor giving ``S`` and ``V`` (default : 'gram'; the null always uses ``G``).
    center : bool, optional
        Centre every row first (default : True).
    dtype : torch.dtype, optional
        Dtype of the per-chunk products (default : float64).
    seed : int, optional
        Seed of the calibration surrogates and Wishart draws.
    chunk_rows : int, optional
    return_info : bool, optional
        Also return a dict with ``'untestable'`` (indices of the untestable
        components skipped), ``'self_null'`` (``m`` of every component
        examined), ``'margin'`` and ``'calibration'`` (``c``). Default : False.

    Returns
    -------
    int
        Estimated rank (at most `q`): components that passed their test.
    torch.Tensor
        Leading data singular values.
    torch.Tensor
        Thresholds of the components examined (length ``rank + skipped + 1``
        unless all `q` are examined).
    dict
        Only with `return_info`.

    Notes
    -----
    The ``T x T`` work (centring fixed point, eigenvalues, Wishart draws) runs
    on the chunks' device: about 15 ``T x T`` eigenvalue problems and ~100
    ``T x T`` reductions per call. Measured on an H100 shared with other jobs
    (16 CPU threads): N = 2e6, T = 1024 takes 9.4 s, of which 4.0 s is the
    streamed pass (2 surrogates) and ~5 s the ``T x T`` part (it took 24-56 s
    with the earlier CPU-only null at 96 threads); N = 4e6, T = 1024 takes
    11-13 s against 7-10 s for one bare pass (``null='mp'``). On the CPU the
    ``T x T`` part is sensitive to oversubscription: torch defaults to one
    thread per host core (96 here) even inside a 16-CPU cgroup, so set
    ``torch.set_num_threads`` to the CPU quota (``bench_linalg.py`` does).
    """
    if null not in ('toeplitz', 'circulant', 'mp') or spread not in ('wishart', 'none') or \
            factor not in ('gram', 'tsqr'):
        raise ValueError('Error: unknown null, spread or factor.')
    stats = dict(dtype=dtype, center=center, n_surr=calibrate, seed=seed, standardize=standardize)
    if factor == 'tsqr':
        src = _reiterable(chunks, chunk_rows)
        S, V = tall_svd((_prep(X, dtype, center, standardize) for X in src()), method='tsqr', dtype=dtype)
        acc = _stream_stats(src(), **stats)
    else:
        acc = _stream_stats(chunks, chunk_rows, **stats)
        ev, V = _eigh_desc(acc.G)
        S = ev.clamp_min(0).sqrt()
    G, T, n = acc.G, acc.T, acc.n
    neff = acc.n_eff(n_eff)
    margin = auto_margin(neff) if isinstance(margin, str) and margin == 'auto' else float(margin)
    q = min(q, T, n)
    S, V = S[:q].to(G.device, torch.float64), V.to(G.device, torch.float64)
    info = {'untestable': [], 'self_null': [], 'margin': margin, 'calibration': 1.0}

    def done(k, thr):
        thr = torch.as_tensor(thr, dtype=torch.float64)
        return (k, S, thr, info) if return_info else (k, S, thr)

    if null == 'mp':
        edge = margin * math.sqrt(float(torch.trace(G)) / (n * T)) * (math.sqrt(n) + math.sqrt(T))
        k = int(torch.cumprod((S > edge).to(torch.int64), 0).sum())
        return done(k, [edge] * min(k + 1, q))
    gn = GaussianNull(T, neff, draws, seed, device=G.device) if spread == 'wishart' else None

    def null_value(C, j):
        """j-th (1-based) leading null singular value of the null Gram C."""
        ev = torch.linalg.eigvalsh(C)
        lam = ev.flip(0)[:j] if gn is None else gn(ev, j)
        return float(lam[j - 1].clamp_min(0).sqrt())

    c = 1.0
    if calibrate:
        ref = null_value(_null_gram(G, 'circulant', center), 1)
        c = max(1.0, max(float(torch.linalg.eigvalsh(Gs)[-1].clamp_min(0).sqrt()) for Gs in acc.Gs) / ref)
    info['calibration'] = c
    blind = 1.0 / margin ** 2
    thr, k, skipped, C = [], 0, info['untestable'], _null_gram(G, null, center)
    for i in range(q):
        v = V[:, i]
        Cv = _null_gram(torch.outer(v, v), null, center)
        m = float(v @ Cv @ v)
        info['self_null'].append(m)
        thr.append(margin * c * null_value(C, len(skipped) + 1))
        if S[i] > thr[-1]:
            # deflate G_k -> G_k - e v v^T with the null of the result self-consistent along v:
            # S^2 - e = v^T Pi(G_k - e v v^T) v, Pi linear  =>  e = (S^2 - v^T C v) / (1 - m)
            s2 = float(S[i]) ** 2
            e = (s2 - float(v @ C @ v)) / (1.0 - m) if m < blind else s2
            C = C - min(max(e, 0.0), s2) * Cv
            k += 1
        elif m >= blind:
            skipped.append(i)            # untestable: not counted, not deflated, uses up one null value
        else:
            break
    if skipped:
        warnings.warn('estimate_rank_pa: component(s) {} are untestable under the {} null (v^T Pi(v v^T) v >= '
                      '1/margin^2 = {:.3f}): a smooth common shape (trend, step, slow drift) cannot be told from '
                      'persistent noise there, so they were not counted; null="circulant" tests them against '
                      'circular-shift surrogates but overcounts persistent noise.'.format(skipped, null, blind),
                      stacklevel=2)
    return done(k, thr)


# ---------------------------------------------------------------------------
# causal projection in agent space
# ---------------------------------------------------------------------------

class CausalProjector:
    """Remove a burn-in window's common mode from later windows, in agent space.

    `fit` takes ``S, V`` of the burn-in window ``P`` (``N x T0``) from `tall_svd`;
    the agent loadings ``U = P V S^{-1}`` (``N x k``, orthonormal columns) are
    recomputed chunk by chunk with `left_factors`, never stored. A later window
    ``L`` (``N x T1``, other time indices, so the time-space projector ``V V^T``
    does not apply) is cleaned as ``R = (I - U U^T) L``: one pass accumulates
    ``W = U^T L`` (``k x T1``), a second yields ``R_c = L_c - U_c W``. Nothing
    after ``T0`` is used to learn ``U``.

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
        """Learn ``S, V`` on the burn-in window (a chunk source). Returns self."""
        self.S, self.V = tall_svd(past, self.k, self.method, self.center, self.dtype, chunk_rows)
        return self

    def loadings(self, past_chunk):
        """Agent loadings ``U_c`` (``n_c x k``) of the rows of one burn-in chunk.

        Unresolved components (`left_factors`; ``rtol`` from the method and the
        working dtype) get a zero column, i.e. are not removed.
        """
        return left_factors(past_chunk, self.S, self.V, self.k, center=self.center,
                            rtol=_default_rtol(self.method, self.dtype), method=self.method)

    def _later(self, L, Uc):
        return _prep(torch.as_tensor(L).to(Uc.device), self.dtype, self.center).to(Uc.dtype)

    def coefficients(self, pairs):
        """``W = U^T L`` (``k x T1``) from aligned ``(past_chunk, later_chunk)`` pairs."""
        W = None
        for P, L in pairs:
            if P.shape[0] != L.shape[0]:
                raise ValueError('Error: past and later chunks must hold the same rows.')
            Uc = self.loadings(P)
            Lc = self._later(L, Uc)
            with _ieee_fp32(Lc):
                w = Uc.T @ Lc
            W = w if W is None else W + w
        return W

    def residual(self, past_chunk, later_chunk, W):
        """``R_c = L_c - U_c W`` for one aligned pair of chunks."""
        Uc = self.loadings(past_chunk)
        Lc = self._later(later_chunk, Uc)
        with _ieee_fp32(Lc):
            return Lc - Uc @ W

    def transform(self, pairs_factory):
        """Residual chunks; `pairs_factory()` returns fresh aligned ``(past, later)`` pairs (two passes)."""
        W = self.coefficients(pairs_factory())
        for P, L in pairs_factory():
            yield self.residual(P, L, W)


def causal_projector(chunks, t0, k, t1=None, method='gram', center=True, dtype=torch.float64, chunk_rows=None):
    """Residual chunks of ``X[:, t0:t1]`` after removing the common mode learned on ``X[:, :t0]``.

    Wrapper of `CausalProjector` for one re-iterable chunk source split at column
    `t0` (three passes: fit, coefficients, residuals).

    Returns
    -------
    generator of torch.Tensor
        Residual chunks ``(n_c, t1 - t0)``.
    """
    src = _reiterable(chunks, chunk_rows)
    cp = CausalProjector(k, method, center, dtype).fit(lambda: (X[:, :t0] for X in src()))
    return cp.transform(lambda: ((X[:, :t0], X[:, t0:t1]) for X in src()))
