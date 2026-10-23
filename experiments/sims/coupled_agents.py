"""Networks of binary agents with planted coalitions.

Every agent runs its own two-state Markov chain. With probability ``eps`` an
agent instead reacts to its coalition at the previous step:

* ``copy``: repeat the last action of a random coalition peer (a "follow /
  punish" response);
* ``xor``: play the XOR of two peers' last actions. Every *pair* of agents
  is then exactly independent, so only a cut of the agent set into groups
  can reveal the coalition (a genuinely higher-order, tensor effect).

At ``eps = 0`` the joint process is a tensor product of the agents' chains.
"""

import numpy as np


def simulate(coalitions, T=20_000, eps=0.1, kind='copy', stay=0.5, seed=0):
    """Simulate the joint symbol process.

    Parameters
    ----------
    coalitions : list of list of int
        Disjoint agent groups; singleton groups are uncoupled agents.
    T : int
        Number of steps.
    eps : float
        Coupling probability.
    kind : {'copy', 'xor'}
        Reaction rule.
    stay : float
        Own-chain probability of repeating the previous action (0.5 gives
        i.i.d. fair coins, the case in which copying is invisible to
        contemporaneous correlation).
    seed : int
        RNG seed.

    Returns
    -------
    numpy ndarray
        Symbols, shape ``(N, T)``, int64.
    """
    rng = np.random.default_rng(seed)
    N = sum(len(c) for c in coalitions)
    group = {a: c for c in coalitions for a in c}
    S = np.zeros((N, T), dtype=np.int64)
    S[:, 0] = rng.integers(0, 2, N)
    U = rng.random((T, N))
    V = rng.random((T, N))
    for t in range(1, T):
        prev = S[:, t - 1]
        own = np.where(U[t] < stay, prev, 1 - prev)
        nxt = own.copy()
        for a in range(N):
            peers = [b for b in group[a] if b != a]
            if not peers or V[t, a] >= eps:
                continue
            if kind == 'copy':
                nxt[a] = prev[peers[rng.integers(len(peers))]]
            elif kind == 'xor':
                if len(peers) >= 2:
                    i, j = rng.choice(peers, 2, replace=False)
                    nxt[a] = prev[i] ^ prev[j]
            else:
                raise ValueError(kind)
        S[:, t] = nxt
    return S


def xor_triad(T=20_000, eps=0.3, seed=0):
    """Agents 0, 1 are fair coins; agent 2 plays XOR of their last actions w.p. eps."""
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 2, size=(3, T))
    flip = rng.random(T) < eps
    S[2, 1:] = np.where(flip[1:], S[0, :-1] ^ S[1, :-1], S[2, 1:])
    return S
