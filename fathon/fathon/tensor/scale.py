#    scale.py - coalition detection among many agents (torch backend)
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

"""Who colludes, among many agents that share a model?

Agents built on the same underlying model respond to the same public
information in the same way, so their joint action matrix ``X (N x T)`` has
a dense low-rank part (the common mode) that every pairwise statistic
mistakes for coupling. A coalition is a *localised* dependence: a block of
few agents whose residual actions, after the common mode is removed, still
predict one another.

Pipeline (all torch, batched, GPU if available):

1. centre ``X``, estimate the common mode by randomised SVD and subtract it
   (`common_mode_residual`);
2. score pairs by the lagged cross-correlation of the residuals, computed in
   row chunks so the ``N x N`` matrix is never materialised at once
   (`pair_scores`);
3. return the top-``K`` pairs (`top_pairs`) or everything in one call
   (`detect`).
"""

import torch


def _as_tensor(X, device=None, dtype=torch.float32):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    return torch.as_tensor(X, dtype=dtype, device=device)


def common_mode_residual(X, rank, niter=4, oversample=8, extra=8):
    """Remove the top ``rank + extra`` principal components of the centred rows.

    `rank` is the common-mode rank; `extra` further components are removed
    as well. Binary / clipped actions are nonlinear in the shared latent, so
    the common mode leaks weakly into components just below the spectral
    edge; removing a few more than the nominal rank costs nothing (a
    coalition of a few agents is invisible to a global component) and cleans
    the residual noticeably.

    Returns
    -------
    torch.Tensor
        Residual matrix, same shape as `X`.
    torch.Tensor
        Singular values of the fitted components (``rank + extra + oversample``).
    """
    X = X - X.mean(1, keepdim=True)
    k = rank + extra
    if k <= 0:
        return X, torch.zeros(0, device=X.device)
    U, S, V = torch.svd_lowrank(X, q=k + oversample, niter=niter)
    R = X - (U[:, :k] * S[:k]) @ V[:, :k].T
    return R, S


def _normalise(R):
    R = R - R.mean(1, keepdim=True)
    return R / (R.norm(dim=1, keepdim=True) + 1e-12)


def pair_topk(A, B, K, chunk=4096):
    """Top-`K` unordered pairs by the best of one-way and mutual coupling,
    ``max(<A_i,B_j>, <A_j,B_i>, (<A_i,B_j> + <A_j,B_i>) / sqrt 2)``.

    `A` and `B` are row-normalised feature matrices with aligned columns
    (`A` the "later" and `B` the "earlier" features). Computed in row chunks,
    so memory is ``chunk x N`` rather than ``N x N``.

    Returns
    -------
    torch.Tensor
        Scores ``(M,)``, best first.
    torch.Tensor
        Pair indices ``(M, 2)`` with ``i < j``.
    """
    N = A.shape[0]
    vals, idx = [], []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        C = A[s:e] @ B.T                                   # i in chunk later, j earlier
        C2 = B[s:e] @ A.T                                  # j later, i earlier
        C = torch.maximum(torch.maximum(C, C2), (C + C2) * 0.7071067811865476)   # one-way or mutual coupling
        rows = torch.arange(s, e, device=A.device)
        # keep only j > i to count each unordered pair once (also drops the diagonal)
        mask = torch.arange(N, device=A.device)[None, :] <= rows[:, None]
        C = C.masked_fill(mask, -float('inf'))
        flat = C.reshape(-1)
        k = min(K, flat.numel())
        v, f = torch.topk(flat, k)
        keep = torch.isfinite(v)
        v, f = v[keep], f[keep]
        vals.append(v)
        idx.append(torch.stack([s + f // N, f % N], 1))
    vals = torch.cat(vals)
    idx = torch.cat(idx)
    order = torch.argsort(vals, descending=True)[:K]
    return vals[order], idx[order]


def pair_scores(R, lag=1, chunk=4096, K=None):
    """Lagged cross-correlation of row-normalised residuals, best of one-way
    and mutual coupling.

    With `K` given, only the `K` best pairs are returned (`pair_topk`).

    Returns
    -------
    torch.Tensor
        Pair scores, shape ``(N, N)`` if `K` is None, else ``(M,)``.
    torch.Tensor or None
        Pair indices ``(M, 2)`` when `K` is given.
    """
    N, T = R.shape
    R = _normalise(R)
    A = R[:, lag:] if lag else R
    B = R[:, :T - lag] if lag else R
    if K is None:
        C = A @ B.T
        C = torch.maximum(torch.maximum(C, C.T), (C + C.T) * 0.7071067811865476)
        C.fill_diagonal_(-float('inf'))
        return C, None
    return pair_topk(A, B, K, chunk)


def parity_features(X):
    """Second-order features that expose XOR-type (higher-order) coupling.

    If ``a_i(t) = a_j(t-1) XOR a_k(t-1)`` around a ring of three, then
    ``a_i(t) XOR a_i(t-2) = a_k(t-1) XOR a_k(t-2)``: a *pairwise* identity
    between the parity ``f_i(t) = s_i(t) s_i(t-2)`` and
    ``g_k(t) = s_k(t-1) s_k(t-2)`` of the signed actions ``s = 2a - 1``, while
    every plain pairwise correlation in the ring is zero.

    Returns
    -------
    torch.Tensor, torch.Tensor
        ``f`` and ``g``, shape ``(N, T - 2)``, aligned in time.
    """
    s = torch.sign(X - X.mean(1, keepdim=True))
    s = torch.where(s == 0, torch.ones_like(s), s)
    f = s[:, 2:] * s[:, :-2]
    g = s[:, 1:-1] * s[:, :-2]
    return f, g


def null_scale(A, B, n=200000, q=0.999, seed=0):
    """Scale of a channel's null: the `q`-quantile of ``max(<A_i,B_j>, <A_j,B_i>)``
    over `n` random pairs. Channels are made comparable by dividing by it."""
    N = A.shape[0]
    g = torch.Generator(device=A.device).manual_seed(seed)
    i = torch.randint(0, N, (n,), device=A.device, generator=g)
    j = torch.randint(0, N, (n,), device=A.device, generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    v = torch.maximum((A[i] * B[j]).sum(1), (A[j] * B[i]).sum(1))
    return torch.quantile(v, q).item()


def merge_topk(channels, K):
    """Merge ``(vals, idx)`` candidate lists from several channels, best score per pair."""
    best = {}
    for vals, idx in channels:
        for v, (i, j) in zip(vals.tolist(), idx.tolist()):
            if best.get((i, j), -float('inf')) < v:
                best[(i, j)] = v
    return sorted(best, key=best.get, reverse=True)[:K]


def top_pairs(C, K):
    """Top-`K` unordered pairs of a full score matrix."""
    N = C.shape[0]
    iu = torch.triu_indices(N, N, 1, device=C.device)
    v = C[iu[0], iu[1]]
    t = torch.topk(v, min(K, v.numel())).indices
    return [(int(iu[0][i]), int(iu[1][i])) for i in t]


def detect(X, K, rank=4, lag=1, device=None, chunk=4096, parity=True):
    """Top-`K` candidate colluding pairs after common-mode removal.

    Parameters
    ----------
    X : array_like
        Actions or prices, shape ``(N, T)``.
    K : int
        Number of pairs to return.
    rank : int, optional
        Common-mode rank to remove (default : 4).
    lag : int, optional
        Lag of the residual cross-correlation (default : 1).
    parity : bool, optional
        Add the `parity_features` channel for XOR-type coalitions. Each
        channel is scaled by its own `null_scale` and a pair's score is the
        best over channels (default : True).

    Returns
    -------
    list of tuple
        Pairs ``(i, j)`` with ``i < j``, best first.
    """
    X = _as_tensor(X, device)
    R, _ = common_mode_residual(X, rank)
    Rn = _normalise(R)
    T = Rn.shape[1]
    A1, B1 = (Rn[:, lag:], Rn[:, :T - lag]) if lag else (Rn, Rn)
    v1, i1 = pair_topk(A1, B1, K, chunk)
    channels = [(v1 / null_scale(A1, B1), i1)]
    if parity:
        f, g = parity_features(X)
        f, _ = common_mode_residual(f, rank)
        g, _ = common_mode_residual(g, rank)
        fn, gn = _normalise(f), _normalise(g)
        v2, i2 = pair_topk(fn, gn, K, chunk)
        channels.append((v2 / null_scale(fn, gn), i2))
    return merge_topk(channels, K)
