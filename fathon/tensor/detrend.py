#    detrend.py - vectorised segmentation and polynomial detrending
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

"""Batched detrending shared by the multivariate estimators.

The C loops of fathon fit a polynomial in every window separately. All windows
of the same size share the same design matrix, so the detrended residual is a
fixed orthogonal projection, ``r = (I - Q Q^T) y``, applied to all windows and
all series at once. The residuals are identical to a per-window least-squares
fit on ``t = 1..s``, because the span of polynomials does not depend on the
time origin.
"""

import functools
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def profile(x, axis=-1):
    """Profile (aggregated series) of one or more time series.

    Parameters
    ----------
    x : array_like
        Time series, time along `axis`.
    axis : int, optional
        Time axis (default : -1).

    Returns
    -------
    numpy ndarray
        Cumulative sum of `x` after subtraction of its mean along `axis`.
    """
    x = np.asarray(x, dtype=float)
    return np.cumsum(x - np.mean(x, axis=axis, keepdims=True), axis=axis)


@functools.lru_cache(maxsize=256)
def _trend_basis(size, polOrd):
    # orthonormal basis of polynomials of degree <= polOrd on `size` points;
    # centred and scaled abscissa keeps the Vandermonde matrix well conditioned
    t = np.linspace(-1.0, 1.0, size)
    V = np.vander(t, polOrd + 1, increasing=True)
    Q, _ = np.linalg.qr(V)
    Q.setflags(write=False)
    return Q


def detrend_windows(W, polOrd=1):
    """Remove the least-squares polynomial trend from every window.

    Parameters
    ----------
    W : numpy ndarray
        Array of windows, window samples along the last axis.
    polOrd : int, optional
        Order of the polynomial (default : 1).

    Returns
    -------
    numpy ndarray
        Residuals, same shape as `W`.
    """
    Q = _trend_basis(W.shape[-1], polOrd)
    return W - (W @ Q) @ Q.T


def segments(Y, winSize, revSeg=False, overlap=False):
    """Cut profiles into windows.

    Parameters
    ----------
    Y : numpy ndarray
        Profiles, time along the last axis, shape ``(..., T)``.
    winSize : int
        Window size `s`.
    revSeg : bool, optional
        If True, non-overlapping windows are also taken starting from the
        end of the series, as in fathon (default : False).
    overlap : bool, optional
        If True, use fathon's DCCA overlapping convention: windows of
        ``s + 1`` points starting at every ``v = 0 .. T - s - 1``
        (default : False).

    Returns
    -------
    numpy ndarray
        Windows of shape ``(..., n_windows, window_length)``.
    """
    T = Y.shape[-1]
    if overlap:
        return sliding_window_view(Y, winSize + 1, axis=-1)[..., :T - winSize, :]
    nSeg = T // winSize
    fw = Y[..., :nSeg * winSize].reshape(Y.shape[:-1] + (nSeg, winSize))
    if not revSeg:
        return fw
    bw = Y[..., T - nSeg * winSize:].reshape(Y.shape[:-1] + (nSeg, winSize))
    return np.concatenate([fw, bw], axis=-2)


def residuals(Y, winSize, polOrd=1, revSeg=False, overlap=False):
    """Detrended residuals of profiles in each window.

    Parameters
    ----------
    Y : numpy ndarray
        Profiles, shape ``(..., T)``.
    winSize : int
        Window size.
    polOrd : int, optional
        Order of the detrending polynomial (default : 1).
    revSeg : bool, optional
        See `segments` (default : False).
    overlap : bool, optional
        See `segments` (default : False).

    Returns
    -------
    numpy ndarray
        Residuals of shape ``(..., n_windows, window_length)``.
    """
    return detrend_windows(segments(np.asarray(Y, dtype=float), winSize, revSeg, overlap), polOrd)


def check_scales(winSizes, tsLen, polOrd):
    """Validate window sizes with the same rules as fathon."""
    winSizes = np.asarray(winSizes, dtype=np.int64)
    if polOrd < 1:
        raise ValueError('Error: Polynomial order must be greater than 0.')
    if len(winSizes) > 1 and winSizes[-1] <= winSizes[0]:
        raise ValueError('Error: `winSizes[-1]` must be greater than variable `winSizes[0]`.')
    if winSizes[-1] > tsLen:
        raise ValueError('Error: `winSizes[-1]` must be smaller than the input vector length.')
    if winSizes[0] < polOrd + 2:
        raise ValueError('Error: `winSizes[0]` must be at least equal to {}.'.format(polOrd + 2))
    return winSizes
