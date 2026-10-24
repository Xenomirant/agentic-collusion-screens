"""Planted coalitions among many agents that share a low-rank common mode.

Every agent emits a binary action each step from a logistic policy

    logit_i(t) = b_i . z_t  +  c * (2 a_i(t-1) - 1)  +  eps * coupling_i(t-1)

* ``z_t`` is an r-dimensional AR(1) latent, the shared "base model" response
  to public information; ``b_i`` are dense loadings, scaled so the drive has
  unit variance whatever ``r`` is. Independent agents are therefore
  correlated through ``z`` alone (a dense rank-``r`` spike).
* ``c`` is each agent's own stickiness (memory beyond any public state).
* ``groups > 1`` splits the agents into populations with their own latents
  (several base models in one market): the common mode is block-structured
  with rank ``groups * r``.
* coalitions are disjoint blocks of ``k`` agents. ``kind='copy'``: each
  member follows the previous action of the next member in the ring (a
  "follow / punish" response). ``kind='xor'`` (k = 3): each member follows
  the XOR of the other two members' previous actions, so every *pair* in the
  coalition is exactly uncorrelated and only a higher-order statistic sees it.

Ground truth is the set of unordered pairs inside a coalition.

The simulation runs on the GPU when available (``device``) and is
deterministic given ``seed`` and the device.
"""

import itertools
import torch


def simulate(N, T, r=4, kind='copy', k=2, n_coal=50, eps=0.5, c=0.3, bscale=1.0, rho=0.9,
             seed=0, device=None, dtype=torch.float32, groups=1):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    g = torch.Generator(device=device).manual_seed(int(seed))
    rn = lambda *s: torch.randn(*s, device=device, generator=g, dtype=dtype)
    ru = lambda *s: torch.rand(*s, device=device, generator=g, dtype=dtype)
    # shared latent(s): `groups` agent populations ("base models"), each with
    # its own r-dimensional latent; agents are assigned to groups round-robin
    # so coalitions (consecutive agents) straddle groups
    z = torch.zeros(T, groups * r, device=device, dtype=dtype)
    e = rn(T, groups * r)
    z[0] = e[0]
    for t in range(1, T):
        z[t] = rho * z[t - 1] + (1 - rho ** 2) ** 0.5 * e[t]
    B = torch.zeros(N, groups * r, device=device, dtype=dtype)
    grp = torch.arange(N, device=device) % groups
    for q in range(groups):
        m = grp == q
        B[m, q * r:(q + 1) * r] = bscale * rn(int(m.sum()), r) / r ** 0.5
    drive = B @ z.T                                     # (N, T)
    # coalition wiring
    M = n_coal * k
    if M > N:
        raise ValueError('n_coal * k exceeds N')
    idx = torch.arange(N, device=device)
    p1 = idx.clone()
    p2 = idx.clone()
    if M:
        blocks = idx[:M].reshape(n_coal, k)
        p1[:M] = torch.roll(blocks, -1, dims=1).reshape(-1)
        p2[:M] = torch.roll(blocks, -2, dims=1).reshape(-1)
    coupled = torch.zeros(N, device=device, dtype=dtype)
    coupled[:M] = 1.0
    if kind == 'xor' and k != 3:
        raise ValueError('xor coalitions need k = 3')
    A = torch.zeros(N, T, device=device, dtype=dtype)
    A[:, 0] = (ru(N) < 0.5).to(dtype)
    U = ru(N, T)
    for t in range(1, T):
        prev = A[:, t - 1]
        if kind == 'copy':
            coup = 2 * prev[p1] - 1
        elif kind == 'xor':
            x = (prev[p1] != prev[p2]).to(dtype)
            coup = 2 * x - 1
        else:
            raise ValueError(kind)
        logit = drive[:, t] + c * (2 * prev - 1) + eps * coupled * coup
        A[:, t] = (U[:, t] < torch.sigmoid(logit)).to(dtype)
    truth = set()
    for b in range(n_coal):
        members = range(b * k, (b + 1) * k)
        truth.update(itertools.combinations(members, 2))
    return A, truth
