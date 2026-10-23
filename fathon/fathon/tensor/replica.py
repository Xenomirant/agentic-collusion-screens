#    replica.py - empirical replica (moment) tensors of multiscale measures
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

"""Replica tensors: the law-level tensor network of a multiscale measure.

A measure on ``2**n`` dyadic cells is a tensor with ``n`` binary indices
(quantized / QTT format). A single sample of a random cascade is
incompressible, but its q-point moment tensor
``M_q(w^1, ..., w^q) = E[mu(w^1) ... mu(w^q)]``, written with the digits of the
q replicas interleaved level by level, is a uniform MPS of bond dimension at
most Bell(q) independent of depth (see ``multifractal_tensor_networks/REPORT.md``,
Theorem A). Statistical self-similarity is therefore bounded replica rank,
and these tensors are estimated here by averaging over samples (windows, runs).

For two agents the same construction with one replica per agent,
``M_ij(w, w') = E[mu_i(w) mu_j(w')]``, gives a cross-replica tensor. Under
independence it equals ``E mu_i kron E mu_j`` and its *agent cut* has rank
one; its level cuts show at which scales the agents' activity is coupled.
"""

import numpy as np
from . import decomp


def cell_measure(x, nLevels, power=2.0, increments=True):
    """Normalised activity measure on ``2**nLevels`` consecutive cells.

    Parameters
    ----------
    x : array_like
        Series, shape ``(..., T)`` with ``T >= 2**nLevels (+1)``.
    nLevels : int
        Number of dyadic levels.
    power : float, optional
        ``mu(cell) ~ |dx|^power`` (default : 2, the energy measure).
    increments : bool, optional
        Use increments of `x` (True) or `x` itself (default : True).

    Returns
    -------
    numpy ndarray
        Measures of shape ``(..., 2**nLevels)``, summing to 1 on the last axis.
    """
    x = np.asarray(x, dtype=float)
    L = 2 ** nLevels
    d = np.diff(x, axis=-1) if increments else x
    m = np.abs(d[..., :L]) ** power
    return m / m.sum(axis=-1, keepdims=True)


def sample_windows(x, nLevels, step=None, **kw):
    """Measures of consecutive (or strided) windows of one long series."""
    x = np.asarray(x, dtype=float)
    L = 2 ** nLevels + 1
    step = L - 1 if step is None else step
    starts = np.arange(0, x.shape[-1] - L + 1, step)
    return np.stack([cell_measure(x[..., s:s + L], nLevels, **kw) for s in starts], axis=-2)


def interleave(T, q, nLevels):
    """Reorder a ``(2,)*(q*nLevels)`` replica-major tensor to level-major.

    Input axes are ``(replica 0 levels, replica 1 levels, ...)``. The output
    has one axis per level of size ``2**q``, with replica ``a`` as bit ``a``
    of the letter, as in ``t1_replica_rank.py``.
    """
    T = np.asarray(T).reshape((2,) * (q * nLevels))
    perm = [a * nLevels + lv for lv in range(nLevels) for a in reversed(range(q))]
    return T.transpose(perm).reshape((2 ** q,) * nLevels)


def replica_tensor(mu, q):
    """Empirical q-point moment tensor in interleaved format.

    Parameters
    ----------
    mu : numpy ndarray
        Sample measures, shape ``(S, 2**n)``.
    q : int
        Number of replicas.

    Returns
    -------
    numpy ndarray
        Tensor of shape ``(2**q,) * n``.
    """
    mu = np.asarray(mu, dtype=float)
    return cross_replica_tensor([mu] * q)


def cross_replica_tensor(mus):
    """``E[mu_1(w^1) ... mu_q(w^q)]`` with one replica per agent, interleaved.

    Parameters
    ----------
    mus : list of numpy ndarray
        Per-agent sample measures, each ``(S, 2**n)``, samples aligned in time.
    """
    mus = [np.asarray(m, dtype=float) for m in mus]
    S, L = mus[0].shape
    n = int(round(np.log2(L)))
    if len(mus) == 1:
        return interleave(mus[0].mean(axis=0), 1, n)
    # sum_s (mu_1 x ... x mu_{q-1})[s] outer mu_q[s] as a matmul, in sample chunks
    chunk = max(1, int(2 ** 25 // L ** (len(mus) - 1)))
    acc = np.zeros((L ** (len(mus) - 1), L))
    for a in range(0, S, chunk):
        M = mus[0][a:a + chunk]
        for m in mus[1:-1]:
            M = (M[:, :, None] * m[a:a + chunk, None, :]).reshape(M.shape[0], -1)
        acc += M.T @ mus[-1][a:a + chunk]
    return interleave(acc / S, len(mus), n)


def agent_cut(mu_i, mu_j, centred=True):
    """Spectrum of the agent cut ``E[mu_i(w) mu_j(w')]`` (cells x cells).

    With ``centred=True`` the independent part ``E mu_i kron E mu_j`` is
    removed first, so the singular values measure coupling only.

    Returns
    -------
    numpy ndarray
        Singular values, decreasing.
    float
        Ratio of the coupling energy to the energy of the independent part.
    """
    C = mu_i.T @ mu_j / mu_i.shape[0]
    base = np.outer(mu_i.mean(axis=0), mu_j.mean(axis=0))
    D = C - base if centred else C
    s = np.linalg.svd(D, compute_uv=False)
    return s, float(np.linalg.norm(C - base) / np.linalg.norm(base))


def level_ranks(T, tol=1e-2):
    """TT ranks at level cuts of an interleaved replica tensor."""
    return decomp.tt_ranks(T, tol)


def level_entropies(T, alpha=2.0):
    """Renyi entropies of the level-cut spectra (scale-resolved complexity)."""
    return np.array([decomp.renyi_entropy(p, alpha) for p in decomp.cut_spectra(T)])


def partition_function(mu, qList, quenched=False):
    """Multifractal partition functions across dyadic levels.

    Parameters
    ----------
    mu : numpy ndarray
        Sample measures, shape ``(S, 2**n)``.
    qList : iterable of float
        Moment orders.
    quenched : bool, optional
        Average ``log Z`` (quenched) instead of ``log E Z`` (annealed).

    Returns
    -------
    numpy ndarray
        ``log2 Z_l(q)`` for levels ``l = 1 .. n``, shape ``(n_q, n)``.
    """
    mu = np.asarray(mu, dtype=float)
    S, L = mu.shape
    n = int(round(np.log2(L)))
    out = np.empty((len(qList), n))
    for lv in range(1, n + 1):
        m = mu.reshape(S, 2 ** lv, -1).sum(axis=-1)
        for a, q in enumerate(qList):
            Z = np.sum(np.where(m > 0, np.where(m > 0, m, 1.0) ** q, 0.0), axis=-1)
            out[a, lv - 1] = np.mean(np.log2(Z)) if quenched else np.log2(np.mean(Z))
    return out


def tau_spectrum(mu, qList, levels=None, quenched=False):
    """Mass exponents ``tau(q)`` from ``Z_l(q) ~ 2^(-l tau(q))``."""
    lz = partition_function(mu, qList, quenched)
    n = lz.shape[1]
    lv = np.arange(1, n + 1) if levels is None else np.asarray(levels)
    sel = lv - 1
    A = np.vstack([lv, np.ones_like(lv)]).T.astype(float)
    coef = np.linalg.lstsq(A, lz[:, sel].T, rcond=None)[0]
    return -coef[0]
