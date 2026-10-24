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


def estimate_rank(X, q=48, niter=4, margin=1.05, method='pa', n_surr=2, seed=0, max_rows=20000):
    """Number of principal components shared across rows beyond what chance alignment explains.

    ``method='pa'`` (default): `fathon.tensor.linalg.estimate_rank_pa`, the same null in closed form
    (no surrogate SVDs; centred-Toeplitz expectation, Wishart finite-size spread, deflation).

    ``method='shift'``: parallel analysis against explicit circular-shift surrogates. Every row
    is rolled by its own random offset, which keeps each row's autocorrelation and marginal but
    destroys the alignment across rows; a component counts as common mode when its singular value
    exceeds ``margin`` times the largest corresponding singular value over `n_surr` surrogates.
    This is the right null when the idiosyncratic part is temporally correlated (sticky agents,
    multifractal activity cascades), where the white-noise edge overcounts badly.

    ``method='mp'``: count singular values above ``margin * sigma (sqrt N + sqrt T)``, the
    Marchenko-Pastur edge for i.i.d. entries (``sigma`` = rms entry). Cheap, but only valid for
    temporally white idiosyncratic noise.

    At most `max_rows` rows (an evenly spaced subsample) are used: a common mode shared by all
    rows is just as visible in a subsample, and the surrogate SVDs then cost the same whatever N.

    Returns
    -------
    int
        Estimated rank, at most ``q``.
    torch.Tensor
        The top-``q`` singular values (of the subsample).
    """
    if max_rows and X.shape[0] > max_rows:
        X = X[torch.linspace(0, X.shape[0] - 1, max_rows, device=X.device).long()]
    X = X - X.mean(1, keepdim=True)
    N, T = X.shape
    q = min(q, N, T)
    if method == 'pa':
        # parallel analysis against the closed-form null of row-wise circular-shift surrogates
        # (fathon.tensor.linalg.estimate_rank_pa: Toeplitz null with centring correction, Wishart
        # finite-size spread, deflation, two calibration surrogates), one streamed pass
        from . import linalg
        k, S, _ = linalg.estimate_rank_pa(X.double(), q=q)
        return int(k), S.to(X.device) if torch.is_tensor(S) else torch.as_tensor(S)

    def topsv(Z):
        return torch.linalg.svdvals(Z)[:q] if q >= min(N, T) else torch.svd_lowrank(Z, q=q, niter=niter)[1]

    S = topsv(X)
    if method == 'mp':
        sigma = (X.norm() ** 2 / (N * T)).sqrt()
        edge = margin * sigma * (N ** 0.5 + T ** 0.5)
        return int((S > edge).sum()), S
    if method != 'shift':
        raise ValueError(method)
    g = torch.Generator(device=X.device).manual_seed(int(seed))
    ar = torch.arange(T, device=X.device)
    Smax = torch.zeros_like(S)
    for _ in range(n_surr):
        sh = torch.randint(1, max(T, 2), (N,), device=X.device, generator=g)
        Xs = torch.gather(X, 1, (ar[None, :] + sh[:, None]) % T)
        Smax = torch.maximum(Smax, topsv(Xs))
    above = S > margin * Smax
    # the rank is the length of the leading run of components above the null
    k = int(torch.cumprod(above.to(torch.int64), 0).sum())
    return k, S


def common_mode_residual(X, rank=None, niter=4, oversample=8, extra=8):
    """Remove the top ``rank + extra`` principal components of the centred rows.

    `rank` is the common-mode rank (`estimate_rank` when None); `extra`
    further components are removed as well. Binary / clipped actions are nonlinear in the shared latent, so
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
    if rank is None:
        rank, _ = estimate_rank(X)
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


def coincidence_pairs(X, K, decimals=2, nShift=20, seed=0, min_count=2, null='round'):
    """Pairs of agents that post *identical* values in the same round far more often than chance.

    LLM sellers that coordinate tend to match each other's price to the cent, a static
    signature that lagged correlation misses. Exact coincidences are found by hashing
    ``(round, rounded value)`` so that only agents sharing a value in a round are paired:
    no ``N x N`` matrix, cost proportional to the number of coincidences. Each pair's count is
    compared with the counts obtained after shifting every agent's series by an independent
    random circular shift (which keeps each agent's own value distribution and run lengths).

    Parameters
    ----------
    X : array_like
        Values, shape ``(N, T)`` (prices, asks, ...).
    K : int
        Number of pairs to return.
    decimals : int, optional
        Rounding before comparing (default : 2, i.e. cents).
    nShift : int, optional
        Number of surrogate passes for the null counts (default : 20).
    null : {'round', 'shift'}, optional
        ``'round'``: permute the agents within every round, which keeps each
        round's multiset of values (a common price trajectory, focal prices
        shared by same-model agents) and asks whether *these two* agents
        coincide more than any two agents at the same time. ``'shift'``:
        independent circular shifts per agent, which keeps each agent's own
        value distribution and run lengths but destroys the common
        trajectory (default : 'round').

    Returns
    -------
    list of tuple
        Pairs ``(i, j)`` with ``i < j``, best first.
    dict
        ``{(i, j): (count, null_mean)}`` for the returned pairs.
    """
    import collections
    import numpy as np
    X = np.asarray(X, dtype=float)
    N, T = X.shape
    V = np.round(X, decimals)

    def counts(V):
        c = collections.Counter()
        for t in range(V.shape[1]):
            groups = collections.defaultdict(list)
            col = V[:, t]
            for i in np.flatnonzero(np.isfinite(col)):
                groups[col[i]].append(i)
            for members in groups.values():
                if len(members) > 1:
                    for a in range(len(members)):
                        for b in range(a + 1, len(members)):
                            c[(members[a], members[b])] += 1
        return c

    obs = counts(V)
    rng = np.random.default_rng(seed)
    nullc = collections.Counter()
    for _ in range(nShift):
        if null == 'shift':
            sh = rng.integers(1, T, size=N)
            Vs = np.stack([np.roll(V[i], int(sh[i])) for i in range(N)])
        elif null == 'round':
            Vs = np.stack([V[rng.permutation(N), t] for t in range(T)], axis=1)
        else:
            raise ValueError(null)
        nullc.update(counts(Vs))
    scores = {}
    for pair, c in obs.items():
        if c >= min_count:
            mu = nullc.get(pair, 0) / nShift
            scores[pair] = (c - mu) / np.sqrt(mu + 1.0)        # Poisson-like excess
    top = sorted(scores, key=scores.get, reverse=True)[:K]
    return top, {p_: (obs[p_], nullc.get(p_, 0) / nShift) for p_ in top}


def coincidence_scores(X, K, decimals=2, cap=64, min_count=2, ngram=1):
    """Scalable exact-coincidence scoring: pairs sharing a value in the same round, weighted by
    how rare that value is in that round.

    For each round the agents are sorted by value; two agents at sorted distance ``d < cap``
    with equal values coincide, and the coincidence is worth ``log(N / g)`` where ``g`` is the
    number of agents posting that value (a focal price shared by thousands of same-model agents
    is worth almost nothing, a rare value shared by two is worth a lot). Values shared by more
    than `cap` agents are skipped. When a dense population covers the whole price grid, single
    values are never rare; ``ngram > 1`` then hashes runs of `ngram` consecutive values
    (shingles), which are shared by far fewer agents. Under the within-round permutation null
    the expected score is the same for every pair, so the raw weighted score ranks pairs
    directly. Cost is ``O(T * N log N + cap * T * N)`` with no ``N x N`` object.

    Returns
    -------
    list of tuple
        Pairs ``(i, j)`` with ``i < j``, best first.
    numpy ndarray
        Their scores.
    """
    import numpy as np
    X = np.asarray(X, dtype=float)
    N, T = X.shape
    V = np.round(X, decimals)
    if ngram > 1:
        # integer codes of values, then a code per run of `ngram` consecutive values
        _, codes = np.unique(V, return_inverse=True)
        codes = codes.reshape(V.shape).astype(np.int64) + 1
        codes[~np.isfinite(V)] = 0
        base = int(codes.max()) + 1
        G = np.zeros((N, T - ngram + 1), dtype=np.int64)
        for k in range(ngram):
            G = G * base + codes[:, k:T - ngram + 1 + k]
        G = np.where((codes[:, :T - ngram + 1] == 0), -1, G)
        V = G.astype(float)
        V[V < 0] = np.nan
        T = V.shape[1]
    keys, wts = [], []
    for t in range(T):
        col = V[:, t]
        ok = np.isfinite(col)
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(col[idx], kind='stable')]
        vals = col[order]
        # group sizes via run boundaries
        change = np.r_[True, vals[1:] != vals[:-1]]
        gid = np.cumsum(change) - 1
        gsize = np.bincount(gid)
        g_of = gsize[gid]
        w_of = np.log(N / g_of)
        keep = (g_of >= 2) & (g_of <= cap)
        for d in range(1, min(cap, order.size)):
            same = (gid[d:] == gid[:-d]) & keep[:-d]
            if not same.any():
                continue
            a = order[:-d][same]
            b = order[d:][same]
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            keys.append(lo.astype(np.int64) * N + hi)
            wts.append(w_of[:-d][same])
    if not keys:
        return [], np.zeros(0)
    keys = np.concatenate(keys)
    wts = np.concatenate(wts)
    uk, inv = np.unique(keys, return_inverse=True)
    score = np.bincount(inv, weights=wts)
    cnt = np.bincount(inv)
    score[cnt < min_count] = -np.inf
    top = np.argsort(-score)[:K]
    top = top[np.isfinite(score[top])]
    pairs = [(int(k // N), int(k % N)) for k in uk[top]]
    return pairs, score[top]


def merge_topk(channels, K):
    """Merge ``(vals, idx)`` candidate lists from several channels, best score per pair."""
    best = {}
    for vals, idx in channels:
        for v, (i, j) in zip(vals.tolist(), idx.tolist()):
            if best.get((i, j), -float('inf')) < v:
                best[(i, j)] = v
    return sorted(best, key=best.get, reverse=True)[:K]


def closure_rerank(scores, K, weight=0.5):
    """Re-rank candidate pairs by coalition closure.

    A coalition is a block, so a true pair usually has a common partner that
    is itself coupled to both members, whereas chance pairs among the
    candidates rarely close a triangle. Each candidate edge ``(i, j)`` gets
    ``score + weight * max_k min(score(i, k), score(k, j))`` over common
    neighbours ``k`` in the candidate graph.

    Parameters
    ----------
    scores : dict
        ``{(i, j): score}`` over candidate pairs (``i < j``).
    K : int
        Number of pairs to return.
    weight : float, optional
        Weight of the closure bonus (default : 0.5).
    """
    nbr = {}
    for (i, j), v in scores.items():
        nbr.setdefault(i, {})[j] = v
        nbr.setdefault(j, {})[i] = v
    out = {}
    for (i, j), v in scores.items():
        ni, nj = nbr[i], nbr[j]
        if len(ni) > len(nj):
            ni, nj = nj, ni
        bonus = 0.0
        for k, vik in ni.items():
            vkj = nj.get(k)
            if vkj is not None:
                b = min(vik, vkj)
                if b > bonus:
                    bonus = b
        out[(i, j)] = v + weight * bonus
    return sorted(out, key=out.get, reverse=True)[:K]


def top_pairs(C, K):
    """Top-`K` unordered pairs of a full score matrix."""
    N = C.shape[0]
    iu = torch.triu_indices(N, N, 1, device=C.device)
    v = C[iu[0], iu[1]]
    t = torch.topk(v, min(K, v.numel())).indices
    return [(int(iu[0][i]), int(iu[1][i])) for i in t]


def detect(X, K, rank=None, lag=1, device=None, chunk=4096, parity=True, candidates=10, closure=0.5):
    """Top-`K` candidate colluding pairs after common-mode removal.

    Parameters
    ----------
    X : array_like
        Actions or prices, shape ``(N, T)``.
    K : int
        Number of pairs to return.
    rank : int, optional
        Common-mode rank to remove; estimated from the spectral edge when
        None (default : None).
    lag : int, optional
        Lag of the residual cross-correlation (default : 1).
    parity : bool, optional
        Add the `parity_features` channel for XOR-type coalitions. Each
        channel is scaled by its own `null_scale` and a pair's score is the
        best over channels (default : True).
    candidates : int, optional
        Candidate pairs kept per channel, as a multiple of `K`, for the
        closure step (default : 10).
    closure : float, optional
        Weight of the `closure_rerank` coalition bonus; 0 disables it
        (default : 0.5).

    Returns
    -------
    list of tuple
        Pairs ``(i, j)`` with ``i < j``, best first.
    """
    X = _as_tensor(X, device)
    if rank is None:
        rank, _ = estimate_rank(X)
    R, _ = common_mode_residual(X, rank)
    Rn = _normalise(R)
    T = Rn.shape[1]
    Kc = K * candidates if closure > 0 else K
    A1, B1 = (Rn[:, lag:], Rn[:, :T - lag]) if lag else (Rn, Rn)
    v1, i1 = pair_topk(A1, B1, Kc, chunk)
    channels = [(v1 / null_scale(A1, B1), i1)]
    if parity:
        f, g = parity_features(X)
        f, _ = common_mode_residual(f, rank)
        g, _ = common_mode_residual(g, rank)
        fn, gn = _normalise(f), _normalise(g)
        v2, i2 = pair_topk(fn, gn, Kc, chunk)
        channels.append((v2 / null_scale(fn, gn), i2))
    if closure <= 0:
        return merge_topk(channels, K)
    best = {}
    for vals, idx in channels:
        for v, (i, j) in zip(vals.tolist(), idx.tolist()):
            if best.get((i, j), -float('inf')) < v:
                best[(i, j)] = v
    return closure_rerank(best, K, closure)


# ---------------------------------------------------------------------------
# market-structured scoring: many markets, agents only interact inside a market
# ---------------------------------------------------------------------------

def global_basis(Xs, rank, extra=8, niter=4):
    """Right singular basis of a subsample of agents: the sketch of the global common mode.

    Returns ``V`` of shape ``(T, rank + extra)``; projecting any agent's centred series on
    ``V`` and subtracting removes the shared-model response without seeing every agent.
    """
    Xs = Xs - Xs.mean(1, keepdim=True)
    U, S, V = torch.svd_lowrank(Xs, q=rank + extra + 8, niter=niter)
    return V[:, :rank + extra]


def remove_global(X, V):
    X = X - X.mean(1, keepdim=True)
    return X - (X @ V) @ V.T


def market_pair_scores(X, m, lag=1, parity=True, market_mean=True):
    """Within-market pair scores for a chunk of ``M`` markets of ``m`` agents.

    `X` holds the agents market by market, shape ``(M * m, T)``, already cleaned of the global
    common mode. The market's own public state is removed as the market mean (rank one per
    market). Returns the lag-`lag` residual channel and, with `parity`, the parity channel, each
    of shape ``(M, m, m)`` with the best of one-way / mutual coupling (upper triangle meaningful).
    """
    M = X.shape[0] // m
    T = X.shape[1]
    R = X.reshape(M, m, T)
    if market_mean:
        R = R - R.mean(1, keepdim=True)
    R = R - R.mean(2, keepdim=True)
    R = R / (R.norm(dim=2, keepdim=True) + 1e-12)
    A = R[:, :, lag:] if lag else R
    B = R[:, :, :T - lag] if lag else R
    C = A @ B.transpose(1, 2)
    Ct = C.transpose(1, 2)
    out = {'lag': torch.maximum(torch.maximum(C, Ct), (C + Ct) * 0.7071067811865476)}
    if parity:
        s = torch.sign(R)
        s = torch.where(s == 0, torch.ones_like(s), s)
        f = s[:, :, 2:] * s[:, :, :-2]
        g = s[:, :, 1:-1] * s[:, :, :-2]
        f = f - f.mean(2, keepdim=True)
        g = g - g.mean(2, keepdim=True)
        f = f / (f.norm(dim=2, keepdim=True) + 1e-12)
        g = g / (g.norm(dim=2, keepdim=True) + 1e-12)
        C2 = f @ g.transpose(1, 2)
        C2t = C2.transpose(1, 2)
        out['parity'] = torch.maximum(torch.maximum(C2, C2t), (C2 + C2t) * 0.7071067811865476)
    return out


def robust_z(scores):
    """Scale pair scores by the median absolute deviation of all pairs (the vast majority are null)."""
    med = scores.median()
    mad = (scores - med).abs().median() * 1.4826
    return (scores - med) / (mad + 1e-12)


# ---------------------------------------------------------------------------
# multiscale activity coupling: the agent cut of the cross-replica tensor, level by level
# ---------------------------------------------------------------------------

def level_log_energy(X, levels=None, eps=1e-12):
    """Dyadic log-energy pyramid of every series.

    For level ``l`` the series is cut into ``2**l`` cells and ``e_l(k) = log sum_{t in k} x(t)^2``
    is the log of the cell's energy measure, i.e. the level-``l`` marginal of the quantized (QTT)
    measure of `replica.cell_measure`. Returns a list of tensors ``(N, 2**l)``, one per level.
    """
    N, T = X.shape
    nl = int(T).bit_length() - 1
    levels = list(range(2, nl - 1)) if levels is None else levels
    E2 = X[:, :2 ** nl] ** 2
    out = []
    for l in levels:
        e = E2.reshape(N, 2 ** l, -1).sum(2)
        out.append(torch.log(e + eps))
    return out, levels


def market_level_coupling(X, m, levels=None, rank=None, extra=4, market_mean=True):
    """Within-market activity coupling at each dyadic level.

    Under independence given the common modes, the q = 2 cross-replica tensor of two agents'
    energy measures factorizes, ``E[mu_i(w) mu_j(w')] = E mu_i(w) E mu_j(w')``, at every level;
    in log-energy coordinates the coupling at level ``l`` is the correlation of the two agents'
    level-``l`` log energies. Per level: centre each agent's log energies, remove the market mean
    (the market's own activity), then the global low-rank common mode across *all* agents
    (shared-model activity; `rank` estimated from the spectrum when None, plus `extra`), then
    correlate within market.

    Returns
    -------
    torch.Tensor
        ``(n_levels, M, m, m)`` correlations.
    list of int
        The levels.
    """
    M = X.shape[0] // m
    E, levels = level_log_energy(X, levels)
    out = []
    for e in E:
        e = e - e.mean(1, keepdim=True)
        R = e.reshape(M, m, -1)
        if market_mean:
            # market activity first: otherwise every market's own cascade counts as a global component
            R = R - R.mean(1, keepdim=True)
        e = R.reshape(M * m, -1)
        if rank != 0 and e.shape[1] > 8:
            r_ = estimate_rank(e, q=min(24, e.shape[1] - 1))[0] if rank is None else rank
            k = min(r_ + extra, e.shape[1] - 2)
            if k > 0:
                U, S, V = torch.svd_lowrank(e, q=min(k + 4, e.shape[1]), niter=3)
                e = e - (U[:, :k] * S[:k]) @ V[:, :k].T
        R = e.reshape(M, m, -1)
        R = R - R.mean(2, keepdim=True)
        R = R / (R.norm(dim=2, keepdim=True) + 1e-12)
        out.append(R @ R.transpose(1, 2))
    return torch.stack(out), levels


def combine_levels(C, iu, weights=None):
    """Combine per-level pair correlations into one score: Stouffer sum of Fisher-z times sqrt(cells).

    `C` is ``(n_levels, M, m, m)``; `weights` are the per-level effective sample sizes (cells).
    Returns ``(M, n_pairs)``.
    """
    L = C.shape[0]
    w = torch.ones(L, device=C.device) if weights is None else torch.as_tensor(weights, dtype=C.dtype, device=C.device)
    P = C[:, :, iu[0], iu[1]].clamp(-0.999, 0.999)
    Z = torch.atanh(P) * torch.sqrt(w)[:, None, None]
    return Z.sum(0) / torch.sqrt(torch.tensor(float(L), device=C.device))


# ---------------------------------------------------------------------------
# multi-lag, two-sided, reciprocal coupling (anti-coordination, delayed copying, benign reaction)
# ---------------------------------------------------------------------------

def market_lag_channels(X, m, lags=(0, 1, 2, 3), market_mean=True):
    """Signed within-market cross-correlations of residual actions at several lags.

    Returns ``{lag: C}`` with ``C`` of shape ``(M, m, m)``; for ``lag > 0`` ``C[k, i, j]`` is the
    correlation of agent i's moves with agent j's moves ``lag`` steps earlier (j leads), for
    ``lag == 0`` the symmetric contemporaneous correlation.
    """
    M = X.shape[0] // m
    T = X.shape[1]
    R = X.reshape(M, m, T)
    if market_mean:
        R = R - R.mean(1, keepdim=True)
    out = {}
    for lag in lags:
        A = R[:, :, lag:] if lag else R
        B = R[:, :, :T - lag] if lag else R
        A = A - A.mean(2, keepdim=True)
        B = B - B.mean(2, keepdim=True)
        A = A / (A.norm(dim=2, keepdim=True) + 1e-12)
        B = B / (B.norm(dim=2, keepdim=True) + 1e-12)
        out[lag] = A @ B.transpose(1, 2)
    return out


def coupling_scores(C, iu, mode='mutual'):
    """Pair scores from `market_lag_channels`, robust-z per lag, max over lags. Two-sided.

    ``mode='oneway'``: ``max(|C_ij|, |C_ji|)`` per lag, any directed coupling counts (also lawful
    one-way reaction to a rival). ``mode='mutual'``: ``|C_ij + C_ji| / sqrt 2`` per lag, so a
    reciprocal (both-ways) coupling scores twice the squared signal of a one-way one: the
    collusion-specific variant under the assumption that coordination is reciprocal while
    competitive reaction is typically one-way. Lag 0 is symmetric and counts in both modes.

    Returns ``(M, n_pairs)``.
    """
    best = None
    for lag, Cl in C.items():
        if lag == 0:
            S = Cl.abs()
        elif mode == 'oneway':
            S = torch.maximum(Cl.abs(), Cl.transpose(1, 2).abs())
        elif mode == 'mutual':
            S = (Cl + Cl.transpose(1, 2)).abs() * 0.7071067811865476
        else:
            raise ValueError(mode)
        S = S[:, iu[0], iu[1]]
        Z = robust_z(S.reshape(-1)).reshape(S.shape)
        best = Z if best is None else torch.maximum(best, Z)
    return best



# ---------------------------------------------------------------------------
# one decomposition per mode: the population common mode from a single streamed pass
# ---------------------------------------------------------------------------

def fit_common_mode(chunks, rank=None, extra=8, transform=None, q=48, device=None):
    """Right singular basis of the whole population's common mode, from streamed row chunks.

    `chunks` is anything `fathon.tensor.linalg.iter_chunks` accepts that can be iterated twice
    (a list, a tensor split into row blocks, or a zero-argument callable returning a fresh
    iterator). `transform` (optional) is applied to every chunk first, e.g. a market-mean removal.
    The rank is estimated with the parallel-analysis null over all rows (`estimate_rank_pa`)
    unless given; the basis comes from the streamed Gram of all rows (`tall_svd(method='gram')`),
    so the agent mode is never decomposed directly and memory is ``T x T`` whatever N.

    Returns
    -------
    torch.Tensor
        ``V`` of shape ``(T, rank + extra)``.
    int
        The estimated (or given) rank.
    """
    from . import linalg

    def source():
        it = chunks() if callable(chunks) else chunks
        for c in it:
            c = c if transform is None else transform(c)
            yield c.double()

    if rank is None:
        rank, _, _ = linalg.estimate_rank_pa(source, q=q)
        rank = int(rank)
    k = rank + extra
    S, V = linalg.tall_svd(source, k=k, method='gram', center=True)
    V = V[:, :k]
    if device is not None:
        V = V.to(device)
    return V, rank


def market_mean_removed(X, m):
    """Subtract each market's cross-sectional mean (rows arranged market by market)."""
    M = X.shape[0] // m
    R = X.reshape(M, m, -1)
    return (R - R.mean(1, keepdim=True)).reshape(X.shape[0], -1)


def base_log_energy(X, base_cell=4, eps=1e-12):
    """Log energy of every agent on cells of `base_cell` steps (the finest QTT level used)."""
    N, T = X.shape
    nl = int(T).bit_length() - 1
    T2 = 2 ** nl
    E2 = X[:, :T2] ** 2
    return torch.log(E2.reshape(N, T2 // base_cell, base_cell).sum(2) + eps)


def market_level_coupling_qtt(X, m, base_cell=4, V=None, rank=None, extra=4, min_cells=4, dof_factor=8):
    """Multiscale activity coupling with ONE decomposition for all levels.

    Log energies are computed once on cells of `base_cell` steps (2**L0 cells). Each row is
    centred and the market mean removed; the population common mode is removed once at that
    base level (basis `V` from `fit_common_mode` over all agents, or fitted here); every coarser
    level is then the dyadic average of the residual, i.e. the contraction of the trailing binary
    (QTT) modes with [1/2, 1/2]. Because the averaging is linear and the projection acts on the
    agent side of the same matrix, every level inherits the common-mode removal without its own
    SVD. Within-market correlations are returned per level.

    `V` holds the ``rank`` common components first and ``extra`` leakage components after them
    (as returned by `fit_common_mode`; pass `rank`). Removing a component costs every level one
    degree of freedom, which is negligible at fine levels and severe at coarse ones, so the extra
    components are removed only at levels with at least ``dof_factor * (rank + extra)`` cells;
    coarser levels use the rank-only residual. Both residuals come from the same basis.

    Returns
    -------
    torch.Tensor
        ``(n_levels, M, m, m)`` correlations, coarsest level first.
    list of int
        Number of cells per level.
    """
    M = X.shape[0] // m
    E = base_log_energy(X, base_cell)
    E = E - E.mean(1, keepdim=True)
    E = market_mean_removed(E, m)
    if V is None:
        rank = estimate_rank(E, q=min(48, E.shape[1] - 1))[0] if rank is None else rank
        from . import linalg
        _, Vf = linalg.tall_svd(E.double(), k=rank + extra, method='gram', center=False)
        V = Vf[:, :rank + extra].to(E.dtype)
    elif rank is None:
        rank = max(0, V.shape[1] - extra)
    V = V.to(E.dtype)
    Vr = V[:, :rank]
    R_rank = E - (E @ Vr) @ Vr.T
    R_full = E - (E @ V) @ V.T
    R = R_full
    n0 = R.shape[1]
    cells = []
    out = []
    c = n0
    while c >= min_cells:
        cells.append(c)
        c //= 2
    cells = cells[::-1]
    for c in cells:
        Rsrc = R_full if c >= dof_factor * V.shape[1] else R_rank
        Rl = Rsrc.reshape(Rsrc.shape[0], c, n0 // c).mean(2).reshape(M, m, c)
        Rl = Rl - Rl.mean(2, keepdim=True)
        Rl = Rl / (Rl.norm(dim=2, keepdim=True) + 1e-12)
        out.append(Rl @ Rl.transpose(1, 2))
    return torch.stack(out), cells
