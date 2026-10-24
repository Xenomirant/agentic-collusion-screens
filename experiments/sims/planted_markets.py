"""Many markets, many agents, a few base models: the threat model at scale.

``M`` markets of ``m`` agents each (N = M * m). Every agent is built on one of ``groups`` base
models: a dense loading on that model's global latent (rank ``r_global``), shared by all agents of
the group in *every* market. Every market also has its own public state (an AR(1) latent of rank
``r_market``) that all its agents respond to. Agents are sticky (``c``). Coalitions live inside
markets only, in the first agents of the block:

    market index mod 10 ==
      0  copy pair            agents 0, 1 follow each other's previous action (logit eps_copy)
      1  copy triad           agents 0, 1, 2 in a ring
      2  xor triad            agents 0, 1, 2; each follows the XOR of the other two (eps_xor):
                              every pair inside is uncorrelated
      3  copy pair, onset     as 0, but the coupling switches on at round ``onset``
      4-9 no coalition

Coalition members sit at random positions of their market and every agent's base model is drawn at
random (``layout='random'``, default), so planted pairs are same-model or cross-model at the population
rate; ``layout='fixed'`` reproduces the first version (model = agent parity, members at positions 0-2),
whose positives were almost always cross-model.

Simulation is chunked over markets (`chunks`), deterministic given ``seed`` and the chunk index,
so a 4-million-agent world never has to sit in memory at once. The global latents are generated
once from ``seed`` and shared by every chunk.
"""

import itertools
import torch

TYPES = {0: 'copy_pair', 1: 'copy_triad', 2: 'xor_triad', 3: 'copy_onset'}


def global_latents(T, r_global, groups, rho=0.9, seed=0, device='cuda', dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(int(seed))
    z = torch.zeros(T, groups * r_global, device=device, dtype=dtype)
    e = torch.randn(T, groups * r_global, device=device, generator=g, dtype=dtype)
    z[0] = e[0]
    for t in range(1, T):
        z[t] = rho * z[t - 1] + (1 - rho ** 2) ** 0.5 * e[t]
    return z



def _gen(seed, M0, salt, device):
    """Per-chunk generator; `salt` keeps the streams of different simulators and of the global
    latents (seeded with `seed` alone) apart."""
    return torch.Generator(device=device).manual_seed((int(seed) * 1_000_003 + int(M0)) * 16 + salt)


def _layout(Mc, m, groups, g, layout, device):
    """Base model of every agent and the local positions of the (up to 3) coalition slots.

    ``layout='random'``: each agent's base model is drawn uniformly, and the coalition occupies a
    random set of positions in its market, so planted pairs are same-model or cross-model at the
    population rate. ``layout='fixed'`` reproduces the first version (model = agent parity, slots
    0, 1, 2), in which pairs were almost always cross-model.
    """
    n = Mc * m
    agent = torch.arange(n, device=device)
    if layout == 'fixed':
        grp = agent % groups
        slots = torch.arange(3, device=device).expand(Mc, 3)
    elif layout == 'random':
        grp = torch.randint(0, groups, (n,), device=device, generator=g)
        slots = torch.argsort(torch.rand(Mc, m, device=device, generator=g), dim=1)[:, :3]
    else:
        raise ValueError(layout)
    return grp, slots


def chunk(M0, M1, m, T, zg, r_global=4, groups=2, r_market=1, eps_copy=0.5, eps_xor=2.0, c=0.5,
          w_global=0.8, w_market=0.6, rho=0.9, onset=None, seed=0, device='cuda', dtype=torch.float32,
          layout='random'):
    """Simulate markets ``M0 .. M1-1``. Returns actions ``(n, T)``, market index per agent,
    the set of true pairs (global agent indices) and a dict pair -> type."""
    Mc = M1 - M0
    n = Mc * m
    g = _gen(seed, M0, 1, device)
    rn = lambda *s: torch.randn(*s, device=device, generator=g, dtype=dtype)
    ru = lambda *s: torch.rand(*s, device=device, generator=g, dtype=dtype)
    onset = T // 2 if onset is None else onset
    grp, slots = _layout(Mc, m, groups, g, layout, device)
    # per-market public state
    zm = torch.zeros(Mc, T, r_market, device=device, dtype=dtype)
    e = rn(Mc, T, r_market)
    zm[:, 0] = e[:, 0]
    for t in range(1, T):
        zm[:, t] = rho * zm[:, t - 1] + (1 - rho ** 2) ** 0.5 * e[:, t]
    agent = torch.arange(n, device=device)
    market = agent // m                                  # 0 .. Mc-1 (local)
    Bg = torch.zeros(n, groups * r_global, device=device, dtype=dtype)
    for q in range(groups):
        sel = grp == q
        Bg[sel, q * r_global:(q + 1) * r_global] = w_global * rn(int(sel.sum()), r_global) / r_global ** 0.5
    Bm = w_market * rn(n, r_market) / r_market ** 0.5
    drive = Bg @ zg.T + torch.einsum('nr,ntr->nt', Bm, zm[market])          # (n, T)
    # coalition wiring
    mtype = (torch.arange(M0, M1, device=device) % 10)
    p1 = agent.clone()
    p2 = agent.clone()
    coupled = torch.zeros(n, device=device, dtype=dtype)
    kind_xor = torch.zeros(n, device=device, dtype=torch.bool)
    onset_mask = torch.zeros(n, device=device, dtype=torch.bool)
    base = torch.arange(Mc, device=device) * m
    S0, S1, S2 = base + slots[:, 0], base + slots[:, 1], base + slots[:, 2]   # global (chunk) indices
    pr = (mtype == 0) | (mtype == 3)                    # copy pair (3: switches on at `onset`)
    p1[S0[pr]] = S1[pr]
    p1[S1[pr]] = S0[pr]
    coupled[S0[pr]] = eps_copy
    coupled[S1[pr]] = eps_copy
    on = mtype == 3
    onset_mask[S0[on]] = True
    onset_mask[S1[on]] = True
    tr = (mtype == 1) | (mtype == 2)                    # triads: ring S0 -> S1 -> S2
    for a_, b_, c_ in ((S0, S1, S2), (S1, S2, S0), (S2, S0, S1)):
        p1[a_[tr]] = b_[tr]
        p2[a_[tr]] = c_[tr]
    for a_ in (S0, S1, S2):
        coupled[a_[mtype == 1]] = eps_copy
        coupled[a_[mtype == 2]] = eps_xor
        kind_xor[a_[mtype == 2]] = True
    A = torch.zeros(n, T, device=device, dtype=dtype)
    A[:, 0] = (ru(n) < 0.5).to(dtype)
    U = ru(n, T)
    for t in range(1, T):
        prev = A[:, t - 1]
        coup_copy = 2 * prev[p1] - 1
        coup_xor = 2 * (prev[p1] != prev[p2]).to(dtype) - 1
        coup = torch.where(kind_xor, coup_xor, coup_copy)
        active = coupled * torch.where(onset_mask, float(t >= onset), 1.0)
        logit = drive[:, t] + c * (2 * prev - 1) + active * coup
        A[:, t] = (U[:, t] < torch.sigmoid(logit)).to(dtype)
    truth, ttype = _truth(mtype, slots, m, M0, {0: 2, 1: 3, 2: 3, 3: 2}, TYPES)
    return A, market + M0, truth, ttype


def _truth(mtype, slots, m, M0, size, names):
    truth, ttype = set(), {}
    mt = mtype.tolist()
    sl = slots.tolist()
    for k, t_ in enumerate(mt):
        if t_ not in size:
            continue
        members = sorted(k * m + j + M0 * m for j in sl[k][:size[t_]])
        for pr in itertools.combinations(members, 2):
            truth.add(pr)
            ttype[pr] = names[t_]
    return truth, ttype


# ---------------------------------------------------------------------------
# Activity / timing coordination: coalitions share *when* (and how hard) they act, not which way
# ---------------------------------------------------------------------------

def log_cascade(n_series, T, levels, lam, g, device='cuda', dtype=torch.float32):
    """Log of a dyadic log-normal multiplicative cascade, one row per series, split by level.

    Returns a tensor ``(n_series, len(levels), T)``: component ``k`` is piecewise constant on
    blocks of ``T / 2**levels[k]`` steps with i.i.d. N(0, lam^2) values. The sum over components
    is the log-intensity of a multifractal (intermittent) activity process.
    """
    out = torch.empty(n_series, len(levels), T, device=device, dtype=dtype)
    for k, l in enumerate(levels):
        nb = 2 ** l
        v = lam * torch.randn(n_series, nb, device=device, generator=g, dtype=dtype)
        out[:, k] = v.repeat_interleave(T // nb, dim=1)
    return out


def chunk_activity(M0, M1, m, T, G, groups=2, lam=0.35, coarse=6, w_global_sd=0.8, w_market=0.5,
                   share=1.0, offset=0, seed=0, device='cuda', dtype=torch.float32, layout='random'):
    """Markets of agents whose *activity* is coupled; directions are independent noise.

    ``x_i(t) = sigma_i(t) * xi_i(t)`` with ``xi`` i.i.d. N(0, 1) for every agent, so every signed
    statistic (correlation at any lag, rho_DCCA, MFDCCA of x) has zero expectation for every pair.
    ``log sigma_i = b_i * G_{model(i)} + w_market * Mkt_{market} + Own_i`` where

    * ``G`` (argument, ``(groups, n_levels, T)``) is the shared-model activity cascade: every
      agent built on model ``q`` loads on ``G[q]`` with its own loading ``b_i ~ |N(0, w_global_sd)|``
      (heterogeneous exposure, the dense low-rank common mode);
    * ``Mkt`` is the market's own activity cascade;
    * ``Own_i`` is the agent's own cascade over all levels. In coalition markets (index mod 10 in
      0..3) the first ``k`` agents (k = 2 for 0, 1; k = 3 for 2, 3) replace a fraction ``share`` of
      the *coarse* levels (the first `coarse` levels, blocks of >= T / 2**coarse steps) of ``Own``
      by a cascade shared inside the coalition. The marginal law of every agent is unchanged,
      so single-series multifractal analysis cannot tell members from non-members.

    Types: 0 timing pair, 1 timing pair (half share), 2 timing triad, 3 timing triad (half share).

    ``offset > 0`` generates every cascade on ``2 T`` steps and keeps steps ``offset .. offset + T``
    (the same window for every agent and market), so the cascade blocks are *not* aligned with
    the dyadic cells a detector uses. `G` must then have length ``2 T`` (`shared_activity(2 T)`).
    """
    Tg = 2 * T if offset else T
    Mc = M1 - M0
    n = Mc * m
    g = _gen(seed, M0, 2, device)
    nl = int(round(torch.log2(torch.tensor(float(Tg))).item()))
    levels = list(range(1, nl + 1))
    # coalition levels: block sizes T/2 .. T/2**coarse in both modes (on the doubled grid these are
    # levels 2 .. coarse + 1; level 1 there has blocks of length T, coarser than in aligned mode)
    lc = slice(1, coarse + 1) if offset else slice(0, coarse)
    agent = torch.arange(n, device=device)
    market = agent // m
    grp, slots = _layout(Mc, m, groups, g, layout, device)
    b = (w_global_sd * torch.randn(n, device=device, generator=g, dtype=dtype)).abs()
    Gsum = G.sum(1)                                                     # (groups, T)
    logs = b[:, None] * Gsum[grp]
    Mk = log_cascade(Mc, Tg, levels, lam, g, device, dtype).sum(1)      # (Mc, Tg)
    logs = logs + w_market * Mk[market]
    own = log_cascade(n, Tg, levels, lam, g, device, dtype)             # (n, L, Tg)
    coal = log_cascade(Mc, Tg, levels[lc], lam, g, device, dtype)       # (Mc, coarse, Tg)
    mtype = torch.arange(M0, M1, device=device) % 10
    mt = mtype[market]
    k_of = torch.where((mt == 0) | (mt == 1), 2, torch.where((mt == 2) | (mt == 3), 3, 0))
    sh = torch.where((mt == 1) | (mt == 3), 0.5 * share, share).to(dtype)
    rank_in_market = torch.empty(n, dtype=torch.long, device=device)
    # slot order: the coalition occupies slots[:, :k]; mark members by their slot rank
    rank_in_market.fill_(m)
    base = torch.arange(Mc, device=device) * m
    for j in range(3):
        rank_in_market[base + slots[:, j]] = j
    member = rank_in_market < k_of
    w = torch.where(member, sh, torch.zeros_like(sh))[:, None, None]
    own[:, lc] = torch.sqrt(1 - w) * own[:, lc] + torch.sqrt(w) * coal[market]
    logs = logs + own.sum(1)
    if offset:
        logs = logs[:, offset:offset + T]
    x = torch.exp(logs) * torch.randn(n, T, device=device, generator=g, dtype=dtype)
    names = {0: 'timing_pair', 1: 'timing_pair_half', 2: 'timing_triad', 3: 'timing_triad_half'}
    truth, ttype = _truth(mtype, slots, m, M0, {0: 2, 1: 2, 2: 3, 3: 3}, names)
    return x, market + M0, truth, ttype


def shared_activity(T, groups=2, lam=0.35, seed=0, device='cuda', dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(int(seed) + 99)
    nl = int(round(torch.log2(torch.tensor(float(T))).item()))
    return log_cascade(groups, T, list(range(1, nl + 1)), lam, g, device, dtype)
