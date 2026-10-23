#    mvdfa.py - multivariate (tensor) detrended fluctuation analysis
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

"""Multivariate DFA: the detrended covariance tensor of N series.

For profiles ``Y`` of shape ``(N, T)`` and window size ``s`` the residual
``eps[i, v, :]`` of series ``i`` in window ``v`` gives the per-window detrended
covariance ``C[v, i, j] = <eps_i eps_j>_v``. Stacking over scales gives the
tensor ``C[s][v, i, j]`` (agent x agent x window x scale). fathon's estimators
are slices of it:

* DFA / MFDFA: diagonal ``i == j`` with q-moments over ``v``;
* DCCA: one off-diagonal entry averaged over ``v``;
* rho_DCCA: the off-diagonal entry normalised by the diagonal.
"""

import numpy as np
from . import detrend


Q_ZERO = 3.0e-15   # same tolerance fathon uses to switch to the log-average at q = 0


def _norm(winSize, overlap):
    # per-window normalisation used by fathon (overlapping DCCA divides by s - 1)
    return float(winSize - 1) if overlap else float(winSize)


def window_covariances(Y, winSize, polOrd=1, revSeg=False, overlap=False):
    """Detrended covariance matrix of N profiles in every window.

    Parameters
    ----------
    Y : numpy ndarray
        Profiles, shape ``(N, T)``.
    winSize : int
        Window size.
    polOrd : int, optional
        Order of the detrending polynomial (default : 1).
    revSeg : bool, optional
        Repeat the segmentation from the end of the series (default : False).
    overlap : bool, optional
        Overlapping windows with fathon's DCCA convention (default : False).

    Returns
    -------
    numpy ndarray
        Array of shape ``(n_windows, N, N)``.
    """
    E = detrend.residuals(np.atleast_2d(Y), winSize, polOrd, revSeg, overlap)
    return np.einsum('ivk,jvk->vij', E, E) / _norm(winSize, overlap)


def covariance_tensor(Y, winSizes, polOrd=1, revSeg=False, overlap=False):
    """Scale-resolved detrended covariance, ``F2[s, i, j]``.

    ``F2[s, i, i]`` is the squared DFA fluctuation of series ``i`` and
    ``F2[s, i, j]`` the signed DCCA covariance of series ``i`` and ``j``.

    Parameters
    ----------
    Y : numpy ndarray
        Profiles, shape ``(N, T)``.
    winSizes : numpy ndarray
        Array of window sizes.
    polOrd, revSeg, overlap :
        See `window_covariances`.

    Returns
    -------
    numpy ndarray
        Array of shape ``(n_scales, N, N)``.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    winSizes = detrend.check_scales(winSizes, Y.shape[-1], polOrd)
    return np.stack([window_covariances(Y, s, polOrd, revSeg, overlap).mean(axis=0)
                     for s in winSizes])


def rho_matrix(Y, winSizes, polOrd=1, revSeg=False, overlap=False):
    """Matrix of DCCA cross-correlation coefficients at every scale.

    Entry ``[k, i, j]`` equals fathon's ``DCCA(Y[i], Y[j]).computeRho`` at
    ``winSizes[k]`` with the same options.

    Returns
    -------
    numpy ndarray
        Array of shape ``(n_scales, N, N)`` with unit diagonal.
    """
    F2 = covariance_tensor(Y, winSizes, polOrd, revSeg, overlap)
    d = np.sqrt(np.einsum('kii->ki', F2))
    return F2 / (d[:, :, None] * d[:, None, :])


def _q_average(x, qList, axis):
    # generalised mean (mean x^(q/2))^(1/q) of positive x, log-average at q = 0
    out = []
    for q in qList:
        if abs(q) <= Q_ZERO:
            out.append(np.exp(0.5 * np.mean(np.log(x), axis=axis)))
        else:
            out.append(np.mean(x ** (0.5 * q), axis=axis) ** (1.0 / q))
    return np.stack(out)


def mfdfa(Y, winSizes, qList, polOrd=1, revSeg=False):
    """MFDFA fluctuation functions of N series at once.

    Entry ``[i, a, k]`` equals fathon's ``MFDFA(Y[i]).computeFlucVec`` at
    ``qList[a]`` and ``winSizes[k]``.

    Parameters
    ----------
    Y : numpy ndarray
        Profiles, shape ``(N, T)`` or ``(T,)``.
    winSizes : numpy ndarray
        Array of window sizes.
    qList : iterable
        q-orders.
    polOrd : int, optional
        Order of the detrending polynomial (default : 1).
    revSeg : bool, optional
        Repeat the segmentation from the end (default : False).

    Returns
    -------
    numpy ndarray
        Array of shape ``(N, n_q, n_scales)``.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    winSizes = detrend.check_scales(winSizes, Y.shape[-1], polOrd)
    qList = np.atleast_1d(np.asarray(qList, dtype=float))
    F = np.empty((Y.shape[0], len(qList), len(winSizes)))
    for k, s in enumerate(winSizes):
        E = detrend.residuals(Y, s, polOrd, revSeg)
        F[:, :, k] = _q_average(np.mean(E ** 2, axis=-1), qList, axis=-1).T
    return F


def projected_mfdfa(Y, winSizes, qList, weights, polOrd=1, revSeg=False):
    """MFDFA of the linear combination ``weights @ Y`` (e.g. a common mode)."""
    w = np.asarray(weights, dtype=float)
    return mfdfa(w @ np.atleast_2d(Y), winSizes, qList, polOrd, revSeg)[0]


def fit_exponents(winSizes, F, nStart=None, nEnd=None):
    """Log-log slopes of fluctuation functions along the last axis.

    Parameters
    ----------
    winSizes : numpy ndarray
        Window sizes, matching the last axis of `F`.
    F : numpy ndarray
        Positive fluctuations, shape ``(..., n_scales)``.
    nStart, nEnd : int, optional
        Fit only scales in ``[nStart, nEnd]``.

    Returns
    -------
    numpy ndarray
        Slopes, shape ``F.shape[:-1]``.
    numpy ndarray
        Intercepts, shape ``F.shape[:-1]``.
    """
    n = np.asarray(winSizes, dtype=float)
    sel = np.ones(n.size, dtype=bool)
    if nStart is not None:
        sel &= n >= nStart
    if nEnd is not None:
        sel &= n <= nEnd
    x = np.log(n[sel])
    y = np.log(np.asarray(F)[..., sel])
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(A, y.reshape(-1, sel.sum()).T, rcond=None)[0]
    return coef[0].reshape(y.shape[:-1]), coef[1].reshape(y.shape[:-1])


def cross_hurst(winSizes, F2, nStart=None, nEnd=None):
    """DCCA scaling exponents ``lambda_ij`` with ``|F2_ij(s)|^(1/2) ~ s^lambda_ij``.

    The excess ``lambda_ij - (h_i + h_j) / 2`` is positive when the
    co-movement of ``i`` and ``j`` is more persistent than either series.

    Returns
    -------
    numpy ndarray
        Exponents, shape ``(N, N)``.
    numpy ndarray
        Excess over the mean of the individual exponents, shape ``(N, N)``.
    """
    lam, _ = fit_exponents(winSizes, np.moveaxis(np.sqrt(np.abs(F2)), 0, -1), nStart, nEnd)
    h = np.diag(lam)
    return lam, lam - 0.5 * (h[:, None] + h[None, :])


def common_mode(R):
    """Spectral summary of a stack of correlation matrices.

    Parameters
    ----------
    R : numpy ndarray
        Correlation matrices, shape ``(n_scales, N, N)``.

    Returns
    -------
    dict
        ``absorption``: leading eigenvalue over N per scale (1/N for
        independent series, 1 for a single common factor);
        ``effective_rank``: participation ratio of the eigenvalues;
        ``loading``: leading eigenvector per scale, sign fixed to positive sum;
        ``mean_offdiag``: mean off-diagonal correlation per scale.
    """
    R = np.asarray(R, dtype=float)
    N = R.shape[-1]
    lam, vec = np.linalg.eigh(R)
    lam = np.clip(lam, 0.0, None)
    lead = vec[..., -1]
    lead = lead * np.sign(lead.sum(axis=-1, keepdims=True) + 1e-300)
    offdiag = (R.sum(axis=(-1, -2)) - np.trace(R, axis1=-2, axis2=-1)) / (N * (N - 1)) if N > 1 \
        else np.zeros(R.shape[0])
    return {
        'absorption': lam[..., -1] / N,
        'effective_rank': lam.sum(axis=-1) ** 2 / np.sum(lam ** 2, axis=-1),
        'loading': lead,
        'mean_offdiag': offdiag,
    }


def window_tensor(Y, winSize, polOrd=1, overlap=False):
    """Per-window covariance tensor, ``(n_windows, N, N)``, as input to Tucker/CP."""
    return window_covariances(Y, winSize, polOrd, False, overlap)
