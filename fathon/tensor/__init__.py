#    __init__.py - init for fathon.tensor subpackage
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

"""Multivariate and tensor-network extensions of fathon (pure numpy).

* `mvdfa`: detrended covariance tensor of N series, batched MFDFA, rho_DCCA
  matrices, cross-Hurst exponents, common-mode spectra;
* `factorization`: whether a joint multi-agent process factorizes over a cut
  of the agents (Kronecker / canonical-correlation defect), coalition trees;
* `replica`: empirical replica (moment) tensors of multiscale measures and
  their tensor-train ranks;
* `decomp`: Tucker, tensor-train and CP decompositions;
* `ews`: rolling indicators, surrogates and early-warning scoring.
"""

from . import detrend, mvdfa, decomp, factorization, replica, ews
from .detrend import profile

__all__ = ['detrend', 'mvdfa', 'decomp', 'factorization', 'replica', 'ews', 'profile']
