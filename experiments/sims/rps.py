"""Iterated N-player Rock-Paper-Scissors with scripted (colluding) agents.

Actions: 0 = Rock, 1 = Paper, 2 = Scissors; ``a`` beats ``(a - 1) % 3``
(R beats S, P beats R, S beats P).

Payoffs follow Bonjour, Aggarwal & Bhargava (UAI 2022), Table 1, for any
number of players: all actions equal -> 0 for everyone; all three actions
present -> 1 for everyone; exactly two distinct actions -> 1 for each player
of the winning action, 0 otherwise.

Strategies
----------
random      uniform at random (Bonjour's players A and C).
assistant   with probability ``cp`` plays the move its partner beats, which
            guarantees the partner a point in the 3-player game (Bonjour's
            "manual assistant collusion", player B); random otherwise.
responder   with probability ``rho`` best-responds to the most frequent action
            of the previous round (public information), random otherwise.
            Two responders depend on the common state but not on each other:
            the confound that separates collusion from common responses.
"""

import numpy as np

BEATEN_BY = np.array([2, 0, 1])     # BEATEN_BY[a] is the move that a beats
BEST_RESPONSE = np.array([1, 2, 0])  # BEST_RESPONSE[a] beats a


def payoffs(actions):
    """Payoff vector of one round for any number of players."""
    actions = np.asarray(actions)
    present = np.unique(actions)
    if present.size == 1:
        return np.zeros(actions.size)
    if present.size == 3:
        return np.ones(actions.size)
    a, b = present
    winner = a if BEATEN_BY[a] == b else b
    return (actions == winner).astype(float)


def public_state(prev):
    """Most frequent action of the previous round (ties -> lowest action)."""
    return int(np.argmax(np.bincount(prev, minlength=3)))


def simulate(strategies, T, seed=0):
    """Play T rounds.

    Parameters
    ----------
    strategies : list of dict
        One dict per player: ``{'kind': 'random'}``,
        ``{'kind': 'assistant', 'partner': i, 'cp': 0.3}`` or
        ``{'kind': 'responder', 'rho': 0.5}``.
    T : int
        Number of rounds.

    Returns
    -------
    dict
        ``actions`` (N, T), ``rewards`` (N, T), ``state`` (T,) = public state
        each round was played in, ``joint_prev`` (T,) = code of the previous
        joint action (Bonjour's full fully-observable state).
    """
    rng = np.random.default_rng(seed)
    N = len(strategies)
    A = np.zeros((N, T), dtype=np.int64)
    R = np.zeros((N, T))
    S = np.zeros(T, dtype=np.int64)
    J = np.zeros(T, dtype=np.int64)
    prev = rng.integers(0, 3, N)
    order = [i for i, s in enumerate(strategies) if s['kind'] != 'assistant'] + \
            [i for i, s in enumerate(strategies) if s['kind'] == 'assistant']
    for t in range(T):
        s_t = public_state(prev)
        S[t] = s_t
        J[t] = int(np.dot(prev, 3 ** np.arange(N)))
        a = np.zeros(N, dtype=np.int64)
        for i in order:
            st = strategies[i]
            u = rng.random()
            if st['kind'] == 'random':
                a[i] = rng.integers(3)
            elif st['kind'] == 'assistant':
                a[i] = BEATEN_BY[a[st['partner']]] if u < st['cp'] else rng.integers(3)
            elif st['kind'] == 'responder':
                a[i] = BEST_RESPONSE[s_t] if u < st['rho'] else rng.integers(3)
            else:
                raise ValueError(st['kind'])
        A[:, t] = a
        R[:, t] = payoffs(a)
        prev = a
    return {'actions': A, 'rewards': R, 'state': S, 'joint_prev': J}
