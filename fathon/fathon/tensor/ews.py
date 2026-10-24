#    ews.py - rolling early-warning indicators and their significance
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

"""Rolling-window early-warning signals (EWS) for multi-agent series."""

import numpy as np


def rolling(fn, X, window, step=None):
    """Apply ``fn`` to rolling windows of multivariate series.

    Parameters
    ----------
    fn : callable
        Maps an array ``(N, window)`` to a float or a dict of floats.
    X : array_like
        Series, shape ``(N, T)`` or ``(T,)``.
    window : int
        Window length.
    step : int, optional
        Stride between windows (default : ``window // 10``).

    Returns
    -------
    numpy ndarray
        Window end times.
    numpy ndarray or dict of numpy ndarray
        Indicator values.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    step = max(window // 10, 1) if step is None else step
    if X.shape[1] < window:
        raise ValueError('Error: series length {} is shorter than the window {}.'.format(X.shape[1], window))
    ends = np.arange(window, X.shape[1] + 1, step)
    vals = [fn(X[:, e - window:e]) for e in ends]
    if isinstance(vals[0], dict):
        return ends, {k: np.array([v[k] for v in vals]) for k in vals[0]}
    return ends, np.array(vals)


def variance(X):
    """Mean variance across agents."""
    return float(np.mean(np.var(X, axis=-1)))


def ar1(X):
    """Mean lag-1 autocorrelation across agents."""
    X = X - X.mean(axis=-1, keepdims=True)
    num = np.sum(X[:, 1:] * X[:, :-1], axis=-1)
    den = np.sum(X ** 2, axis=-1)
    return float(np.mean(num / np.where(den > 0, den, 1.0)))


def mean_corr(X):
    """Mean off-diagonal Pearson correlation."""
    C = np.corrcoef(np.atleast_2d(X))
    off = C[~np.eye(C.shape[0], dtype=bool)]
    return float(np.nanmean(off)) if np.any(np.isfinite(off)) else float('nan')


def kendall_trend(t, v):
    """Kendall tau of an indicator against time (and its p-value)."""
    from scipy.stats import kendalltau
    ok = np.isfinite(v)
    res = kendalltau(np.asarray(t)[ok], np.asarray(v)[ok])
    return float(res.statistic), float(res.pvalue)


def shuffle_surrogate(X, rng):
    """Independent random permutation of time per agent."""
    return np.stack([rng.permutation(x) for x in np.atleast_2d(X)])


def phase_surrogate(X, rng, common=True):
    """Fourier phase randomisation.

    With ``common=True`` the same random phases are applied to all agents,
    preserving auto- and cross-spectra (linear null). With ``common=False``
    each agent gets its own phases, destroying cross-dependence (factorized
    linear null).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    N, T = X.shape
    F = np.fft.rfft(X - X.mean(axis=1, keepdims=True), axis=1)
    nf = F.shape[1]
    if common:
        ph = np.exp(2j * np.pi * rng.random(nf))[None, :].repeat(N, axis=0)
    else:
        ph = np.exp(2j * np.pi * rng.random((N, nf)))
    ph[:, 0] = 1.0
    if T % 2 == 0:
        ph[:, -1] = 1.0
    return np.fft.irfft(F * ph, n=T, axis=1) + X.mean(axis=1, keepdims=True)


def iaaft_surrogate(X, rng, nIter=100, common=True):
    """Iterative amplitude-adjusted Fourier transform surrogate per agent."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    N, T = X.shape
    amp = np.abs(np.fft.rfft(X, axis=1))
    sorted_x = np.sort(X, axis=1)
    Y = np.stack([rng.permutation(x) for x in X])
    if common:
        # start from a common phase randomisation to keep cross-structure
        Y = phase_surrogate(X, rng, common=True)
    for _ in range(nIter):
        F = np.fft.rfft(Y, axis=1)
        Y = np.fft.irfft(amp * np.exp(1j * np.angle(F)), n=T, axis=1)
        ranks = np.argsort(np.argsort(Y, axis=1), axis=1)
        Y = np.take_along_axis(sorted_x, ranks, axis=1)
    return Y


def shift_surrogate(X, rng, minShift=None):
    """Circular shift of every agent but the first by an independent random lag."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    N, T = X.shape
    minShift = max(T // 20, 1) if minShift is None else minShift
    out = X.copy()
    for i in range(1, N):
        out[i] = np.roll(X[i], rng.integers(minShift, T - minShift))
    return out


def surrogate_test(stat, X, surrogate, nSurr=200, seed=0, **kw):
    """One-sided surrogate test of a scalar statistic.

    Returns
    -------
    float
        Observed statistic.
    numpy ndarray
        Surrogate values.
    float
        z-score.
    float
        p-value ``(1 + #{surr >= obs}) / (1 + nSurr)``.
    """
    rng = np.random.default_rng(seed)
    obs = stat(X)
    surr = np.array([stat(surrogate(X, rng, **kw)) for _ in range(nSurr)])
    sd = surr.std(ddof=1)
    z = (obs - surr.mean()) / sd if sd > 0 else np.inf
    return obs, surr, float(z), float((1 + np.sum(surr >= obs)) / (1 + nSurr))


def first_crossing(t, v, threshold, persist=1):
    """First time ``v`` stays above ``threshold`` for ``persist`` consecutive points."""
    above = np.asarray(v) > threshold
    run = 0
    for k, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= persist:
            return t[k - persist + 1]
    return None


def baseline_threshold(v, nBase, nSigma=3.0):
    """``mean + nSigma * std`` of the first ``nBase`` indicator values."""
    v = np.asarray(v, dtype=float)[:nBase]
    return float(np.nanmean(v) + nSigma * np.nanstd(v))


def auc(pos, neg):
    """Area under the ROC curve (probability a positive outranks a negative; ties count 1/2).

    Computed from ranks (Mann-Whitney U) in ``O((P + N) log(P + N))`` time and linear memory,
    so it scales to millions of positives and negatives.
    """
    from scipy.stats import rankdata
    pos = np.asarray(pos, dtype=float).ravel()
    neg = np.asarray(neg, dtype=float).ravel()
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    u = r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))
