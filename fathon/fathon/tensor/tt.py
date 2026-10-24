#    tt.py - replica (moment) tensors in quantized tensor-train format
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

"""Replica (moment) tensors of multiscale measures in QTT format.

Generic tensor-train machinery comes from established libraries; only the
project-specific numerics live here.

* `teneva` (Chertkov, Ryzhakov, Oseledets; MIT) does the CPU / numpy TT
  work: QR orthogonalisation, add / sub / mul, dense export, ranks, random
  TTs, batched entry evaluation (`teneva.get_many`), maxvol (Goreinov et al.
  2010) and the rank-adaptive one-site TT-cross (`teneva.cross`; Oseledets
  and Tyrtyshnikov 2010, Savostyanov and Oseledets 2011, rectangular maxvol
  of Mikhalev and Oseledets 2018; optional `tt_cross` backend for exactly
  low-rank black boxes). Call it directly for anything not wrapped
  below; its conventions are ours: a TT is a list of numpy cores of shape
  ``(r_{k-1}, n_k, r_k)`` with ``r_0 = r_d = 1`` and C-ordered multi-indices.
* `torchtt` (Ion, Pfeiffer, Schoeps; MIT) provides TT-SVD and TT-rounding
  (Oseledets 2011, Alg. 1 and 2; teneva's ``svd`` / ``truncate`` misplace the
  singular values in their sweeps, see `tt_from_dense` and `tt_round`), the
  two-site DMRG cross (`torchtt.interpolate.dmrg_cross`, default `tt_cross`
  backend) and the GPU path: after `to_torchtt`, rounding (``.round``),
  arithmetic and ``.full()`` run on the device of the cores; `from_torchtt`
  converts back.

Thin wrappers are kept only where they add a project convention or route
around a library defect: `tt_from_dense` / `tt_round` (relative accuracy,
torchtt instead of teneva), `tt_norm` (QR-based; ``teneva.norm`` squares the
tensor and so loses half the digits on differences), and `tt_cross` (our
batched evaluator interface, backend choice, evaluation count).

Project-specific numerics

* QTT of measures on ``2**n`` dyadic cells (`qtt_from_vector`), coarse-graining
  without decompression (`qtt_coarse_grain`) and the interleaved q-replica
  product (`qtt_interleave`), with the conventions of `fathon.tensor.replica`:
  coarsest bit first, one site per level, letter ``c = sum_a bit_a 2**a``
  (replica ``a`` is bit ``a``);
* empirical (cross-)replica tensors without ``2**(n q)`` arrays: `replica_tt`,
  `cross_replica_tt` (TT-cross on the moment-entry evaluator, or exact sample
  sums), and their level-cut spectra / ranks;
* the exact law-level MPS of a scalar random cascade (REPORT.md, Theorem A)
  `cascade_moment_tt` and the sampler `cascade_sample`.
"""

import contextlib
import io
import math
import sys
import time
import warnings
import numpy as np
from . import decomp

# Hard dependencies of this module only (the rest of fathon.tensor is pure numpy).
# Versions used and tested: teneva 0.14.11, torchtt 0.5.0, torch 2.x.
_DEPS_MSG = ('fathon.tensor.tt needs teneva, torchtt and torch '
             '(tested: pip install teneva==0.14.11 torchtt==0.5.0 torch).')
try:
    import teneva
    import torch
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')  # torchtt prints a notice when its C++ extension is absent
        import torchtt
        import torchtt.interpolate
except ImportError as _e:  # pragma: no cover - exercised only in environments without the deps
    raise ImportError(_DEPS_MSG + ' Missing: {}'.format(_e.name)) from _e


def _is_torch(x):
    return isinstance(x, torch.Tensor)


def _numpy(x):
    return x.detach().cpu().numpy() if _is_torch(x) else np.asarray(x)


# ---------------------------------------------------------------------------
# thin wrappers over teneva / torchtt (project conventions, library defects)
# ---------------------------------------------------------------------------

def _np_cores(cores):
    """Cores as float64 numpy arrays (torch cores, on any device, are copied to the CPU)."""
    return [np.asarray(_numpy(G), dtype=float) for G in cores]


def tt_from_dense(X, eps=1e-14, max_rank=None):
    """TT-SVD with relative accuracy ``||X - TT||_F <= eps ||X||_F`` (torchtt).

    Oseledets (2011), Alg. 1, as implemented by ``torchtt.TT(full, eps, rmax)``
    (per-cut threshold ``eps ||X||_F / sqrt(d - 1)``). `teneva.svd` is not
    used. Its ``e`` is an absolute per-cut tolerance (``matrix_skeleton`` with
    ``rel=False``; there is no relative mode), and it splits each SVD as
    ``U S^(1/2), S^(1/2) V``, so later cuts are truncated against a
    non-orthonormal left factor and the actual error depends on the scale of
    ``X``. teneva 0.14.11, exactly rank-6 ``teneva.rand([4] * 6, 6, seed=0)``
    tensor (``||X|| = 226``): ``e = 0.3 ||X|| / sqrt(5)``, the per-cut
    equivalent of a 30% relative bound, gives 93% error; on ``X / ||X||``,
    ``e = 0.3`` gives 28%, inside its own per-cut bound ``sqrt(5) 0.3``.
    """
    X = np.asarray(_numpy(X), dtype=float)
    if X.ndim == 1:
        return [X.reshape(1, -1, 1).copy()]
    t = torchtt.TT(torch.as_tensor(X), eps=eps, rmax=sys.maxsize if max_rank is None else int(max_rank))
    return from_torchtt(t)


def tt_to_dense(cores):
    """Dense tensor of shape ``(n_1, ..., n_d)`` (`teneva.full`)."""
    return teneva.full(_np_cores(cores))


def tt_ranks(cores):
    """Internal bond dimensions ``[r_1, ..., r_{d-1}]`` (`teneva.ranks` without the boundary 1s)."""
    return [int(r) for r in teneva.ranks(cores)[1:-1]]


def tt_round(cores, eps=1e-10, max_rank=None, device=None):
    """TT-rounding (Oseledets 2011, Alg. 2) by torchtt's ``TT.round``.

    ``||A - round(A)||_F <= eps ||A||_F`` unless `max_rank` binds. Runs on
    `device` (default: the device of torch cores, else the CPU) and returns
    numpy cores. `teneva.truncate` is not used: with ``is_eigh=False`` it keeps
    ``S`` in the core it just split (``give_to='r'``), so the next cut is
    truncated against a non-orthonormal environment and the bound fails (95%
    error at ``e = 0.1`` and ``0.3`` on the rank-6 tensor of `tt_from_dense`); its
    default Gram / eigen-decomposition path cannot resolve singular values
    below ``sqrt(machine eps)`` and keeps spurious ranks for ``eps < 1e-8``.
    """
    if len(cores) == 1:
        return _np_cores(cores)
    t = to_torchtt(cores, device=device).round(eps, sys.maxsize if max_rank is None else int(max_rank))
    return from_torchtt(t)


def tt_norm(cores):
    """Frobenius norm via QR orthogonalisation (`teneva.orthogonalize`).

    Unlike ``teneva.norm`` (square root of an inner product) it keeps full
    relative accuracy for differences of nearly equal tensors.
    """
    return float(np.linalg.norm(teneva.orthogonalize(_np_cores(cores))[-1]))


def tt_rel_err(a, b):
    """``||a - b||_F / ||b||_F`` in TT arithmetic (`teneva.sub`, `tt_norm`)."""
    a, b = _np_cores(a), _np_cores(b)
    return tt_norm(teneva.sub(a, b)) / tt_norm(b)


def tt_sum_samples(tts, eps=1e-12, max_rank=None, round_at=256):
    """Sum of many TTs (`teneva.add`) with periodic rounding (`tt_round`).

    Terms are added one at a time (block-diagonal cores, ranks add) and the
    partial sum is rounded whenever its summed bond dimension exceeds
    `round_at`. Each rounding adds at most ``eps`` times the norm of the
    partial sum, so for nonnegative terms (sample moment tensors) the relative
    error is at most ``eps`` times the number of roundings. Single-core
    terms (``d = 1``) are summed directly (``teneva.add`` would return a
    ``(1, n, 2)`` core). ``teneva.add_many`` is not used: it rounds every
    fixed number of terms regardless of rank, with teneva's truncation (see
    `tt_round`), and takes a list rather than a stream.

    Parameters
    ----------
    tts : iterable of lists of cores
        Terms (a generator is fine).
    eps : float, optional
        Relative accuracy of each rounding (default : 1e-12).
    max_rank : int, optional
        Rank cap of each rounding.
    round_at : int, optional
        Summed bond dimension that triggers a rounding (default : 256).

    Returns
    -------
    list
        Cores of the rounded sum.
    """
    acc, width = None, 0
    for t in tts:
        t = _np_cores(t)
        if acc is None:
            acc = t
        elif len(t) == 1:
            acc = [acc[0] + t[0]]
        else:
            acc = teneva.add(acc, t)
        width += max(tt_ranks(t) or [1])
        if width > round_at:
            acc = tt_round(acc, eps, max_rank)
            width = max(tt_ranks(acc) or [1])
    if acc is None:
        raise ValueError('Error: tt_sum_samples got no terms.')
    return tt_round(acc, eps, max_rank)


def to_torchtt(cores, device=None, dtype=None):
    """Convert numpy (teneva) or torch cores to a `torchtt.TT` (same core layout).

    Torch cores stay on their device unless `device` is given. GPU rounding /
    arithmetic then go through torchtt, e.g.
    ``from_torchtt(to_torchtt(Y, 'cuda').round(1e-8))``.
    """
    dtype = torch.float64 if dtype is None else dtype
    return torchtt.TT([torch.as_tensor(G, dtype=dtype, device=device) for G in cores])


def from_torchtt(t):
    """Numpy (teneva) cores of a `torchtt.TT` (moved to the CPU, float64)."""
    return [G.detach().cpu().double().numpy() for G in t.cores]


@contextlib.contextmanager
def _qr_accuracy():
    """Temporarily route ``teneva.accuracy`` (used by teneva.cross) to `tt_rel_err` (not thread-safe)."""
    orig = teneva.accuracy
    teneva.accuracy = tt_rel_err
    try:
        yield
    finally:
        teneva.accuracy = orig


# backend-specific `tt_cross` options and their defaults (anything else is a TypeError)
_CROSS_OPTS = {
    'teneva': {'init_rank': 2, 'dr_min': 1, 'dr_max': 2, 'cache': True, 'm_cache_scale': math.inf,
               'tau': 1.1, 'tau0': 1.05, 'k0': 100},
    'dmrg': {'kick': 2, 'core_device': None, 'stall': 0.5},
}
_CROSS_SWEEPS = {'teneva': 50, 'dmrg': 10}


class _BudgetExhausted(Exception):
    pass


def tt_cross(f, dims, *, eps=1e-10, max_rank=None, backend='dmrg', n_sweeps=None, seed=0, max_evals=None,
             log=False, **opts):
    """TT approximation from entry evaluations by a library TT-cross, then `tt_round`.

    Backends

    * ``'dmrg'`` (default): `torchtt.interpolate.dmrg_cross`, the two-site
      DMRG cross (Savostyanov and Oseledets 2011; Savostyanov 2014). Each
      superblock is evaluated and split by a truncated SVD at
      ``eps / sqrt(d - 1)``, so the rank adapts in one step with no ceiling,
      plus `kick` random enrichment directions. It has no entry cache, so on
      exactly low-rank targets it needs a few times more evaluations than
      teneva. Stops when the largest superblock change of a sweep is below
      `eps` (``stop = 'e'``), or when it fails to shrink by the factor `stall`
      from the previous sweep (``'stall'``; noisy targets such as empirical
      moment tensors plateau at the noise level after about two sweeps, and
      more sweeps only refit the noise: binary cascade, q = 3, n = 10, S =
      20000, error vs the law 0.115 after sweep 2, 0.11 to 0.27 over sweeps
      3 to 8 at 2.5 to 20x the evaluations), after `n_sweeps`, or at
      `max_evals`. Keep the cores on the CPU (`core_device`, default) even
      when `f` runs on a GPU: the many small QR / SVD / solve calls are ~15x
      slower as CUDA kernels (44 s vs 2.5 s for q = 3, n = 8 replica tensors).
    * ``'teneva'``: `teneva.cross`, the rank-adaptive one-site TT-cross
      (Oseledets and Tyrtyshnikov 2010; Savostyanov and Oseledets 2011) with
      rectangular maxvol (Mikhalev and Oseledets 2018) and a cache of
      evaluated entries (a Python dict; with cheap entries ``cache=False`` is
      faster: exact cascade law, q = 3, n = 12, 0.2 s / 82k evaluations vs
      0.8 s / 48k; dmrg 0.3 s / 120k). Ranks grow by `dr_min` .. `dr_max` per bond and
      half-sweep, so the reachable rank is about
      ``init_rank + 2 dr_max n_sweeps`` (202 with the defaults here). Use it
      for exactly low-rank black boxes with expensive entries; it is fragile
      otherwise. Two teneva defaults are changed because they stop it far
      from `eps` even on exact data: ``nswp`` (10 there, rank ceiling 42; 50
      here) and ``m_cache_scale`` (5 there: stop once cache hits exceed 5x
      the evaluations, which fired at 2e-2 to 1e-1 error for eps = 1e-4 on
      exact q = 2, 3 moment tensors of 25 samples; infinite here). A
      `RuntimeWarning` is issued whenever it stops before the sweep change
      reaches `eps`. During the run its sweep-change measure
      ``teneva.accuracy`` is swapped for `tt_rel_err` (module-level, not
      thread-safe): the original goes through ``teneva.mul_scalar``, which
      allocates ``(4 r^2, n, 4 r^2)`` arrays for the rank-``2r`` difference
      (gigabytes at ``r ~ 50``) and, as a squared norm, cannot resolve
      changes below ~1e-8.

    Parameters
    ----------
    f : callable
        Batched evaluator ``f(idx)``, ``idx`` an int64 numpy array ``(B, d)``,
        returning ``B`` values (numpy or torch).
    dims : sequence of int
        Mode sizes.
    eps : float, optional
        Accuracy: stopping tolerance, truncation threshold and final rounding
        (default : 1e-10). Keyword-only, like everything after `dims`.
    max_rank : int, optional
        Rank cap (dmrg: ``rmax``; teneva: stop once a bond exceeds it); the
        result is rounded to it.
    backend : {'dmrg', 'teneva'}, optional
        See above (default : 'dmrg').
    n_sweeps : int, optional
        Maximum number of full sweeps (default : 10 dmrg, 50 teneva).
    seed : int, optional
        Seed of the random start (and of the dmrg rank kicks). The dmrg run
        seeds and restores only the torch generator of `core_device`.
    max_evals : int, optional
        Evaluation budget. teneva never exceeds it. dmrg always completes its
        first sweep (which may exceed it); afterwards a batch that would
        exceed it stops the run, and the last complete sweep is returned.
    log : bool, optional
        Print the library's per-sweep log.
    **opts
        Backend-specific options; an option of the other backend is a
        `TypeError`.

        * dmrg: ``kick`` (rank enrichment per superblock, default 2),
          ``core_device`` (torch device of the cores, default CPU),
          ``stall`` (default 0.5; None disables the stagnation stop).
        * teneva: ``init_rank`` (2), ``dr_min`` (1; with 0 the ranks do not
          grow from a low-rank start and it stops early on a wrong answer),
          ``dr_max`` (2), ``cache`` (True), ``m_cache_scale`` (inf),
          ``tau`` (1.1), ``tau0`` (1.05), ``k0`` (100).

    Returns
    -------
    list of numpy ndarray
        TT cores.
    dict
        ``n_evals`` (entries requested from `f`), ``n_cache_hits``,
        ``sweeps``, ``stop``, ``converged`` (the sweep change reached `eps`;
        an a-posteriori indicator, not an error bound), ``ranks`` (after
        rounding), ``cross_ranks`` (before), ``time``, ``backend``; dmrg adds
        ``sweep_change``.
    """
    if backend not in _CROSS_OPTS:
        raise ValueError("Error: backend must be 'dmrg' or 'teneva'.")
    bad = sorted(set(opts) - set(_CROSS_OPTS[backend]))
    if bad:
        raise TypeError("tt_cross(backend='{}') got option(s) {} that it does not use; its options are {}.".format(
            backend, bad, sorted(_CROSS_OPTS[backend])))
    o = dict(_CROSS_OPTS[backend], **opts)
    n_sweeps = _CROSS_SWEEPS[backend] if n_sweeps is None else int(n_sweeps)
    dims = [int(n) for n in dims]
    t0 = time.perf_counter()
    if len(dims) == 1:  # nothing to cross: evaluate the whole vector
        v = np.asarray(_numpy(f(np.arange(dims[0], dtype=np.int64)[:, None])), dtype=float).reshape(1, -1, 1)
        return [v], {'n_evals': dims[0], 'n_cache_hits': 0, 'sweeps': 0, 'stop': 'full', 'converged': True,
                     'cross_ranks': [], 'ranks': [], 'time': time.perf_counter() - t0, 'backend': backend}
    if backend == 'teneva':
        def F(idx):
            return np.asarray(_numpy(f(np.asarray(idx, dtype=np.int64))), dtype=float).reshape(-1)

        def cb(Y, info, opts_):
            return max_rank is not None and int(max(teneva.ranks(Y))) > max_rank

        info = {}
        Y0 = teneva.rand(dims, o['init_rank'], seed=seed)
        with _qr_accuracy():
            Y = teneva.cross(F, Y0, m=max_evals, e=eps, nswp=n_sweeps, dr_min=o['dr_min'], dr_max=o['dr_max'],
                             tau=o['tau'], tau0=o['tau0'], k0=o['k0'], m_cache_scale=o['m_cache_scale'],
                             info=info, cache={} if o['cache'] else None, cb=cb, log=log)
        out = {'n_evals': int(info['m']), 'n_cache_hits': int(info['m_cache']), 'sweeps': int(info['nswp']),
               'stop': info['stop'], 'converged': info['stop'] == 'e'}
        if not out['converged']:
            warnings.warn("tt_cross(backend='teneva') stopped ('{}') before the sweep change reached eps = {:g} "
                          '(cross ranks {}); its ranks grow by at most dr_max per half-sweep (reachable rank about '
                          "init_rank + 2 dr_max n_sweeps = {}). Use backend='dmrg' or raise n_sweeps / dr_max."
                          .format(info['stop'], eps, tt_ranks(Y), o['init_rank'] + 2 * o['dr_max'] * n_sweeps),
                          RuntimeWarning, stacklevel=2)
    else:
        count, hist, last, stop = [0], [], [None], [None]

        def F(idx):
            B = int(idx.shape[0])
            if max_evals is not None and last[0] is not None and count[0] + B > max_evals:
                raise _BudgetExhausted
            count[0] += B
            v = f(idx.detach().cpu().numpy())
            return torch.as_tensor(v, dtype=torch.float64, device=idx.device).reshape(-1)

        def cb(Y, sweep, err):
            hist.append(err)
            last[0] = Y
            if max_evals is not None and count[0] >= max_evals:
                stop[0] = 'm'
            elif err >= eps and o['stall'] is not None and len(hist) >= 2 and err > o['stall'] * hist[-2]:
                stop[0] = 'stall'
            return False if stop[0] else None

        dev = torch.device(o['core_device']) if o['core_device'] is not None else torch.device('cpu')
        cuda = [dev] if dev.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda):  # seed (and restore) only the generator the cross uses
            torch.default_generator.manual_seed(seed)
            if cuda:
                with torch.cuda.device(dev):
                    torch.cuda.manual_seed(seed)
            try:
                Y = torchtt.interpolate.dmrg_cross(F, dims, eps=eps, nswp=n_sweeps, kick=o['kick'], device=dev,
                                                   rmax=max_rank if max_rank is not None else sys.maxsize,
                                                   verbose=log, callback=cb)
            except _BudgetExhausted:
                Y, stop[0] = last[0], 'm'
        Y = from_torchtt(Y)
        converged = bool(hist) and hist[-1] < eps
        out = {'n_evals': count[0], 'n_cache_hits': 0, 'sweeps': len(hist),
               'stop': 'e' if converged else (stop[0] or 'nswp'), 'converged': converged, 'sweep_change': hist}
    out['cross_ranks'] = tt_ranks(Y)
    Y = tt_round(Y, eps, max_rank)
    out.update(ranks=tt_ranks(Y), time=time.perf_counter() - t0, backend=backend)
    return Y, out


# ---------------------------------------------------------------------------
# QTT of dyadic measures
# ---------------------------------------------------------------------------

def qtt_from_vector(v, base=2, eps=1e-14, max_rank=None):
    """Quantized TT of a vector of length ``base**n`` (Oseledets 2010; Khoromskij 2011).

    The vector is reshaped to ``(base,) * n`` with the most significant digit
    (coarsest dyadic level) first, matching `replica.interleave`, and
    decomposed by `tt_from_dense`.
    """
    v = np.asarray(_numpy(v), dtype=float).reshape(-1)
    n = int(round(math.log(v.size, base))) if v.size > 1 else 0
    if base ** n != v.size:
        raise ValueError('Error: length {} is not a power of {}.'.format(v.size, base))
    if n == 0:
        return [v.reshape(1, 1, 1)]
    return tt_from_dense(v.reshape((base,) * n), eps=eps, max_rank=max_rank)


def qtt_coarse_grain(cores, n_keep, average=True):
    """Coarse-grain a (Q)TT measure to its first `n_keep` levels without decompressing.

    The trailing ``d - n_keep`` modes are contracted with ``[1/m, ..., 1/m]``
    (``average=True``; ``[1/2, 1/2]`` for binary modes) or ``[1, ..., 1]``
    (``average=False``: the projective coarse-grained measure
    ``mu_l(w) = sum_c mu_{l+1}(w c)``). For an interleaved replica TT (site
    dimension ``2**q``) this is the replica tensor of the coarse-grained
    measure. Returns a scalar for ``n_keep = 0``.
    """
    d = len(cores)
    if not 0 <= n_keep <= d:
        raise ValueError('Error: n_keep must lie in [0, {}].'.format(d))
    R = np.ones((1, 1))
    for G in reversed(cores[n_keep:]):
        R = (G.mean(axis=1) if average else G.sum(axis=1)) @ R
    if n_keep == 0:
        return float(R[0, 0])
    out = [G.copy() for G in cores[:n_keep]]
    out[-1] = np.tensordot(out[-1], R, 1)
    return out


def qtt_interleave(qtts):
    """Interleaved TT of the outer product of q QTTs (exact, ranks multiply).

    Site ``k`` has dimension ``prod_a n_k^(a)`` and letter
    ``c = sum_a c_a prod_{b < a} n_k^(b)``; for binary QTTs ``c = sum_a bit_a 2**a``
    with replica ``a`` as bit ``a``, level-major, the convention of
    `replica.interleave`. The site cores are slice-wise Kronecker products
    (teneva's `outer` concatenates trains instead, so it does not apply).
    """
    d = len(qtts[0])
    if any(len(t) != d for t in qtts):
        raise ValueError('Error: qtt_interleave needs QTTs with the same number of sites.')
    out = []
    for k in range(d):
        G = np.asarray(qtts[-1][k])  # highest replica = most significant digit of the letter
        for t in reversed(qtts[:-1]):
            H = np.asarray(t[k])
            G = np.einsum('ixj,kyl->ikxyjl', G, H).reshape(
                G.shape[0] * H.shape[0], G.shape[1] * H.shape[1], G.shape[2] * H.shape[2])
        out.append(G)
    return out


# ---------------------------------------------------------------------------
# replica (moment) tensors
# ---------------------------------------------------------------------------

class _MomentEvaluator:
    """Entries ``mean_s prod_a mu_a[s, w^a(c)]`` at interleaved letters ``c`` (batched).

    The measures are stored transposed, ``(2**n, S)``, so a batch of B letters
    gathers q blocks of ``(B, S)`` rows; evaluation runs in numpy, or in torch
    on the device of torch inputs or of `device`, in chunks of `chunk_elems`
    gathered elements (default 2**24: 0.2 GB of float32 / 0.4 GB of float64
    temporaries; the GPU speed-up comes from keeping the chunks on the device,
    and 2**26 was no faster but needed 4x the memory).
    """

    def __init__(self, mus, chunk_elems=None, device=None, dtype=None):
        self.q = len(mus)
        self.torch = _is_torch(mus[0]) or device is not None
        seen, self.cols = {}, []
        for m in mus:
            if id(m) not in seen:
                if self.torch:
                    t = m if _is_torch(m) else torch.as_tensor(np.asarray(m))
                    dt = dtype if dtype is not None else (t.dtype if t.is_floating_point() else torch.float64)
                    seen[id(m)] = t.to(device=device if device is not None else t.device, dtype=dt).T.contiguous()
                else:
                    a = np.asarray(m)
                    dt = dtype if dtype is not None else (a.dtype if a.dtype in (np.float32, np.float64) else np.float64)
                    seen[id(m)] = np.ascontiguousarray(a.astype(dt).T)
            self.cols.append(seen[id(m)])
        self.L, self.S = self.cols[0].shape
        self.n = int(round(math.log2(self.L)))
        if 2 ** self.n != self.L:
            raise ValueError('Error: measures need 2**n cells, got {}.'.format(self.L))
        if chunk_elems is None:
            chunk_elems = 2 ** 24
        self.rows = max(1, chunk_elems // self.S)

    def cells(self, idx):
        """Per-replica cell indices ``(q, B)``: bit ``a`` of letter ``k`` is bit ``n-1-k`` of ``w^a``."""
        idx = np.asarray(idx, dtype=np.int64)
        w = np.zeros((self.q, idx.shape[0]), dtype=np.int64)
        for k in range(self.n):
            for a in range(self.q):
                w[a] |= ((idx[:, k] >> a) & 1) << (self.n - 1 - k)
        return w

    def __call__(self, idx):
        w = self.cells(idx)
        B = w.shape[1]
        if self.torch:  # one transfer each way; chunks stay on the device
            w = torch.as_tensor(w, device=self.cols[0].device)
            out = torch.empty(B, dtype=torch.float64, device=w.device)
        else:
            out = np.empty(B)
        for a0 in range(0, B, self.rows):
            sl = slice(a0, a0 + self.rows)
            P = self.cols[0][w[0, sl]]
            for a in range(1, self.q):
                P = P * self.cols[a][w[a, sl]]
            out[sl] = P.mean(1)
        return out.cpu().numpy() if self.torch else out


def cross_replica_tt(mus, eps=1e-6, *, max_rank=None, method='cross', backend='dmrg', device=None, dtype=None,
                     round_at=512, max_sample_rank=4096, return_info=False, **cross_kw):
    """Cross-replica tensor ``E[mu_1(w^1) ... mu_q(w^q)]`` as an interleaved QTT, no dense arrays.

    Target: `replica.cross_replica_tensor` (shape ``(2**q,) * n``, letter
    ``c = sum_a bit_a 2**a``), whose dense construction needs ``2**(n q)``
    entries.

    Parameters
    ----------
    mus : list of array_like or torch.Tensor
        Per-agent sample measures, each ``(S, 2**n)``, samples aligned.
    eps : float, optional
        Relative accuracy (default : 1e-6); for empirical tensors choose it
        near the sampling noise level. Everything after `eps` is keyword-only.
    max_rank : int, optional
        Bond-dimension cap.
    method : {'cross', 'sum'}, optional
        'cross' (default): `tt_cross` on the entry evaluator
        ``M(c) = mean_s prod_a mu_a[s, w^a(c)]`` (``q S`` operations per
        entry; on the device of torch inputs, or of `device`).
        'sum': exact ``(1/S) sum_s qtt_interleave(QTT(mu_1[s]), ..., QTT(mu_q[s]))``
        with periodic rounding (`tt_sum_samples`). Cascade samples are
        QTT-incompressible, so terms have ranks up to ``2**(q n / 2)``; raises
        if a term exceeds `max_sample_rank`.
    backend : {'dmrg', 'teneva'}, optional
        'cross' only: TT-cross backend of `tt_cross` (default : 'dmrg').
        Exact (noise-free) targets: both reach `eps` (e.g. 25 cascade
        samples, q = 3, n = 5, eps = 1e-4, level-cut ranks up to 64); dmrg in
        ~0.2 s, teneva in ~4 s with 6x fewer evaluations. Empirical moment
        tensors are low-rank (law) plus sampling noise; choose `eps` above the
        noise. With moderate depth (experiments/bench_tt.py: binary cascade,
        S = 20000, q <= 3, n <= 10, eps = 2x the predicted noise) dmrg lands
        at 0.3-0.9 `eps` against the dense empirical tensor except q = 3,
        n = 10 (1.2-1.7 `eps` over runs), stopping after 2-3 sweeps
        (stagnation stop, see `tt_cross`), in about the time of the
        hand-written cross this module had before (which landed at up to 3.6
        `eps`); teneva lands at up to 5.7 `eps` for q = 3 and is not
        recommended for samples. Once the entrywise relative noise is large
        (it grows ~ (E W^(2q) / (E W^q)^2)^(n/2) / sqrt(S)) no cross works:
        q = 3, n = 12, S = 20000 (predicted noise 0.09, eps 0.18), errors
        against the law are 0.83 (dmrg, ``max_rank=256``; 1.16 uncapped, 0.85
        with ``max_rank=64``), 0.95 (teneva) and 0.65 (the previous
        hand-written cross, rank cap 64); at n = 14 dmrg reaches 4.1, teneva
        and the old cross ~1. All are unusable: maxvol pivots on outlier
        entries. A rank cap bounds the cost there (dmrg 8 s with
        ``max_rank=64`` vs 46 s without). Use ``method='sum'`` or more
        samples.
    device, dtype : optional
        Torch device / dtype of the entry evaluator (default : numpy
        float64). The cross itself stays on the CPU (see `tt_cross`).
    round_at, max_sample_rank : int, optional
        'sum' only.
    return_info : bool, optional
        Also return a dict (evaluation count, sweeps, stop reason, ranks).
    **cross_kw
        Passed to `tt_cross` (``n_sweeps``, ``seed``, ``max_evals`` and the
        backend's own options, e.g. ``stall`` / ``kick`` for dmrg,
        ``init_rank`` / ``dr_max`` for teneva).

    Returns
    -------
    list
        Interleaved TT cores ``(r_{k-1}, 2**q, r_k)`` (and the info dict).
    """
    q = len(mus)
    if method == 'cross':
        ev = _MomentEvaluator(mus, device=device, dtype=dtype)
        cores, info = tt_cross(ev, [2 ** q] * ev.n, eps=eps, max_rank=max_rank, backend=backend, **cross_kw)
    elif method == 'sum':
        S = int(mus[0].shape[0])

        def terms():
            for s in range(S):
                qs, seen = [], {}
                for m in mus:
                    if id(m) not in seen:
                        seen[id(m)] = qtt_from_vector(m[s], eps=1e-14)
                    qs.append(seen[id(m)])
                r = int(np.prod([max(tt_ranks(t) or [1]) for t in qs]))
                if r > max_sample_rank:
                    raise ValueError('Error: a sample term has interleaved rank {} > max_sample_rank = {}; '
                                     "samples are QTT-incompressible, use method='cross'.".format(r, max_sample_rank))
                yield qtt_interleave(qs)

        cores = teneva.mul(tt_sum_samples(terms(), eps=eps, max_rank=max_rank, round_at=round_at), 1.0 / S)
        info = {'n_evals': 0, 'ranks': tt_ranks(cores)}
    else:
        raise ValueError("Error: method must be 'cross' or 'sum'.")
    info['method'] = method
    return (cores, info) if return_info else cores


def replica_tt(mu, q, eps=1e-6, *, max_rank=None, method='cross', **kw):
    """Empirical q-point moment tensor ``E[mu(w^1) ... mu(w^q)]`` as an interleaved QTT.

    Target: `replica.replica_tensor` (shape ``(2**q,) * n``) without any
    ``2**(n q)`` array; `cross_replica_tt` with ``[mu] * q`` (methods and
    keywords as there; everything after `eps` is keyword-only).
    """
    return cross_replica_tt([mu] * q, eps=eps, max_rank=max_rank, method=method, **kw)


def level_cut_spectra(cores):
    """Singular values of every level-cut unfolding ``A[(i_1..i_k), (i_{k+1}..i_d)]``.

    `teneva.orthogonalize` brings the TT to left-canonical form (QR sweep,
    norm in the last core); a right-to-left sweep of small SVDs of the
    ``r_{k-1} x (n_k r_k)`` cores then yields exactly the singular values at
    each cut (canonical / Vidal form; Oseledets 2011, Sec. 3; Schollwoeck
    2011). Cost ``O(d n r^3)``; the TT counterpart of the dense
    `decomp.cut_spectra` (``p = s**2 / sum s**2``).

    Returns
    -------
    list of numpy ndarray
        ``d - 1`` vectors of decreasing singular values.
    """
    Z = teneva.orthogonalize(_np_cores(cores))
    out = []
    for k in range(len(Z) - 1, 0, -1):
        r0, n, r1 = Z[k].shape
        U, s, Vt = np.linalg.svd(Z[k].reshape(r0, n * r1), full_matrices=False)
        out.append(s)
        Z[k - 1] = np.tensordot(Z[k - 1], U * s, 1)
    return out[::-1]


def level_cut_ranks(cores, tol=1e-10):
    """Number of singular values above ``tol * s_max`` at each level cut (rule of `decomp.tt_ranks`)."""
    return [decomp._rank_from_tol(s, tol, None) for s in level_cut_spectra(cores)]


# ---------------------------------------------------------------------------
# exact replica MPS of a scalar random cascade (REPORT.md, Theorem A, r = 1)
# ---------------------------------------------------------------------------

def _set_partitions(q):
    """Set partitions of ``range(q)`` as restricted growth strings."""
    out = []

    def rec(prefix, m):
        if len(prefix) == q:
            out.append(tuple(prefix))
            return
        for b in range(m + 2):
            rec(prefix + [b], max(m, b))

    rec([0], 0) if q > 0 else out.append(())
    return out


def _canon(labels):
    seen = {}
    return tuple(seen.setdefault(x, len(seen)) for x in labels)


def cascade_moment_tt(atoms, probs, q, n):
    """Exact q-point moment tensor of a scalar random cascade as an interleaved uniform MPS.

    REPORT.md Theorem A with ``r = 1``: the bond space is the set partitions
    of the q replicas (dimension Bell(q)), the core is
    ``G[pi, c, pi'] = 1[pi' = pi ^ ker c] prod_{beta in pi} E[prod_{a in beta} W_{c_a}]``,
    the left boundary is the one-block partition and the right boundary sums
    over partitions. Equals ``E[prod_a mu_n(w^a)]`` for
    ``mu_n(w) = prod_k W^{(w_1..w_{k-1})}_{w_k}`` with iid node tuples ``W``.

    Parameters
    ----------
    atoms : array_like
        Atoms of the node law, shape ``(K, b)`` (``b`` children per node).
    probs : array_like
        Atom probabilities, shape ``(K,)``.
    q, n : int
        Number of replicas, depth.

    Returns
    -------
    list of numpy ndarray
        ``n`` cores of shape ``(B_q, b**q, B_q)`` (boundaries of size 1).
    """
    atoms = np.asarray(atoms, dtype=float)
    probs = np.asarray(probs, dtype=float)
    b = atoms.shape[1]
    parts = _set_partitions(q)
    pos = {p: i for i, p in enumerate(parts)}
    G = np.zeros((len(parts), b ** q, len(parts)))
    for i, p in enumerate(parts):
        for c in range(b ** q):
            digits = [(c // b ** a) % b for a in range(q)]
            val = 1.0
            for blk in set(p):
                m = np.zeros(b)
                for a in range(q):
                    if p[a] == blk:
                        m[digits[a]] += 1
                val *= float(np.sum(probs * np.prod(atoms ** m[None, :], axis=1)))
            G[i, c, pos[_canon(list(zip(p, digits)))]] = val
    top = pos[(0,) * q]
    if n == 1:
        return [G[top].sum(axis=1)[None, :, None]]
    return [G[top][None]] + [G.copy() for _ in range(n - 2)] + [G.sum(axis=2)[:, :, None]]


def cascade_sample(atoms, probs, n, S, seed=0):
    """S independent samples of a scalar random cascade on ``b**n`` cells, shape ``(S, b**n)`` (not renormalised)."""
    atoms = np.asarray(atoms, dtype=float)
    rng = np.random.default_rng(seed)
    K, b = atoms.shape
    mu = np.ones((S, 1))
    for _ in range(n):
        tup = atoms[rng.choice(K, size=mu.shape, p=probs)]  # (S, nodes, b)
        mu = (mu[:, :, None] * tup).reshape(S, -1)
    return mu
