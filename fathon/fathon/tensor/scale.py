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


def common_mode_residual(X, rank, niter=4, oversample=8):
    """Remove the top-`rank` principal components of the centred rows.

    Returns
    -------
    torch.Tensor
        Residual matrix, same shape as `X`.
    torch.Tensor
        Singular values of the fitted components (``rank + oversample``).
    """
    X = X - X.mean(1, keepdim=True)
    if rank <= 0:
        return X, torch.zeros(0, device=X.device)
    U, S, V = torch.svd_lowrank(X, q=rank + oversample, niter=niter)
    R = X - (U[:, :rank] * S[:rank]) @ V[:, :rank].T
    return R, S


def pair_scores(R, lag=1, chunk=4096, K=None):
    """Lagged cross-correlation of row-normalised residuals, best direction.

    With `K` given, only the `K` best pairs are kept per row chunk and then
    merged, so memory is ``chunk x N`` rather than ``N x N``.

    Returns
    -------
    torch.Tensor
        Pair scores, shape ``(N, N)`` if `K` is None, else ``(M,)``.
    torch.Tensor or None
        Pair indices ``(M, 2)`` when `K` is given.
    """
    N, T = R.shape
    R = R - R.mean(1, keepdim=True)
    R = R / (R.norm(dim=1, keepdim=True) + 1e-12)
    A = R[:, lag:] if lag else R
    B = R[:, :T - lag] if lag else R
    if K is None:
        C = A @ B.T
        C = torch.maximum(torch.maximum(C, C.T), (C + C.T) * 0.7071067811865476)
        C.fill_diagonal_(-float('inf'))
        return C, None
    vals, idx = [], []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        C = A[s:e] @ B.T                                   # i in chunk predicts j (i earlier)
        C2 = B[s:e] @ A.T                                  # j earlier, i later
        C = torch.maximum(torch.maximum(C, C2), (C + C2) * 0.7071067811865476)   # one-way or mutual coupling
        rows = torch.arange(s, e, device=R.device)
        # keep only j > i to count each unordered pair once (also drops the diagonal)
        mask = torch.arange(N, device=R.device)[None, :] <= rows[:, None]
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


def top_pairs(C, K):
    """Top-`K` unordered pairs of a full score matrix."""
    N = C.shape[0]
    iu = torch.triu_indices(N, N, 1, device=C.device)
    v = C[iu[0], iu[1]]
    t = torch.topk(v, min(K, v.numel())).indices
    return [(int(iu[0][i]), int(iu[1][i])) for i in t]


def detect(X, K, rank=4, lag=1, device=None, chunk=4096):
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

    Returns
    -------
    list of tuple
        Pairs ``(i, j)`` with ``i < j``, best first.
    """
    X = _as_tensor(X, device)
    R, _ = common_mode_residual(X, rank)
    vals, idx = pair_scores(R, lag=lag, chunk=chunk, K=K)
    return [(int(i), int(j)) for i, j in idx.tolist()]
