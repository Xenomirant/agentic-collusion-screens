"""Batched GPU multifractal cross-correlation baselines on within-market pairs.

The "common multifractal methods" for pairs of series, computed for every within-market pair at
once. Data are arranged market by market: ``X`` has shape ``(M * m, T)``, rows ``k*m .. k*m+m-1``
are the ``m`` agents of market ``k``; every pair function returns ``(..., M, m, m)`` tensors.

Conventions (match fathon's MFDCCA, ``fathon/fathon/cLoops.c: flucMFDCCAForwCompute``):

* profile ``Y = cumsum(x - mean(x))`` (``fathonUtils.toAggregated``; fathon expects the profile
  as input), non-overlapping windows of size ``s`` starting at 0, the tail ``T mod s`` dropped
  (``rev_seg=True`` adds the windows aligned to the end and averages over the ``2 N_s`` windows),
  least-squares polynomial detrending of order ``pol_ord`` in every window;
* fathon's per-window cross fluctuation takes the absolute value *point by point*:
  ``F2_xy(v, s) = (1/s) sum_j |e_x(j) e_y(j)|`` (``rms += fabs(var_1 * var_2)``), so it is
  positive and the diagonal reduces to MFDFA. This is ``mode='fathon'`` (the default; parity
  with ``fathon.MFDCCA``). Zhou's MF-DXA (PRE 77, 066211, 2008) instead takes the absolute value
  of the *window covariance*, ``|(1/s) sum_j e_x(j) e_y(j)|``: ``mode='window'``. For an
  independent Gaussian pair the fathon version tends to ``2/pi * sigma_x sigma_y`` and not 0, so
  F_xy from it measures joint *volatility*, not co-movement;
* ``F_xy(q, s) = ( mean_v F2_xy(v, s)^{q/2} )^{1/q}``, ``q = 0`` by the log-average
  ``exp( mean_v log F2_xy(v, s) / 2 )`` (|q| <= 3e-15, fathon's tolerance).

Degenerate windows: a window whose detrended residual vanishes (a constant run of a binary
action series, e.g.) has ``F2 = 0`` and makes fathon's ``q <= 0`` moments infinite. Here windows
with ``F2 < rel_floor * mean_v F2`` are dropped from the ``q <= 0`` averages only (``rel_floor =
1e-10``; never triggered on continuous data, so parity is unaffected).

The q-dependent cross-correlation coefficient (`rho_q`). Kristoufek's form (Physica A 390, 4252
(2011), MF-HXA; Physica A 402, 291 (2014) for the DCCA coefficient) is
``rho(q, s) = F_xy(q, s)^2 / (F_xx(q, s) F_yy(q, s))``. With an absolute-valued F_xy it is
non-negative and does not reduce to rho_DCCA at q = 2. We use the *signed* F2 where it is
defined (q > 0), as in the q-dependent detrended cross-correlation coefficient of Kwapien,
Oswiecimka and Drozdz (PRE 92, 052815, 2015):

    S_xy(q, s)  = mean_v sign(F2_xy(v, s)) |F2_xy(v, s)|^{q/2}       (signed window covariance)
    rho(q, s)   = sign(S_xy) |S_xy|^{2/q} / (F_xx(q, s) F_yy(q, s))   (form='kristoufek', default)
    rho_K(q, s) = S_xy / (F_xx(q, s) F_yy(q, s))^{q/2}                 (form='kwapien')

Both equal rho_DCCA(s) exactly at q = 2, lie in [-1, 1] for q > 0 (Cauchy-Schwarz per window and
Hoelder over windows) and have the same sign; for fixed (q, s) they are monotone transforms of
each other. For q <= 0 the signed moment is not defined (windows with F2_xy ~ 0 dominate with
either sign), so there we fall back to Kristoufek's unsigned version with the window absolute
value (``mode='window'``).

Spectrum width (`hxy_width`): ``h_xy(q)`` is the least-squares slope of ``log F_xy(q, s)`` on
``log s`` (fathon's ``fitFlucVec``); the width is ``h_xy(min q) - h_xy(max q)``. The diagonal
``i == j`` is the per-agent MFDFA width.

Run as a script for the pilot comparison on the planted-market simulators:
    cd fathon && CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python ../experiments/baselines_mf.py
Output: experiments/results/baselines_mf_pilot.json
"""
import math

import numpy as np
import torch

SCALES = [8, 16, 32, 64, 128]
Q_ZERO = 3.0e-15
REL_FLOOR = 1e-10


# ---------------------------------------------------------------------------
# window residuals and per-window cross fluctuations
# ---------------------------------------------------------------------------

def _basis(s, pol_ord, dtype, device):
    """Orthonormal basis (s, pol_ord + 1) of polynomials of degree <= pol_ord on s points."""
    t = torch.linspace(-1.0, 1.0, s, dtype=torch.float64)
    V = torch.stack([t ** k for k in range(pol_ord + 1)], 1)
    Q, _ = torch.linalg.qr(V)
    return Q.to(dtype=dtype, device=device)


def _profiles(X, m, dtype):
    X = X.to(dtype)
    M, T = X.shape[0] // m, X.shape[1]
    if M * m != X.shape[0]:
        raise ValueError(f'{X.shape[0]} rows is not a multiple of m = {m}')
    return (X - X.mean(1, keepdim=True)).cumsum(1).reshape(M, m, T)


def _residuals(Y, s, pol_ord=1, rev_seg=False):
    """Detrended residuals of profiles Y (M, m, T) in windows of size s -> (M, nw, m, s)."""
    M, m, T = Y.shape
    nw = T // s
    if nw < 1:
        raise ValueError(f'scale {s} larger than the series length {T}')
    if s < pol_ord + 2:
        raise ValueError(f'scale {s} must be at least pol_ord + 2 = {pol_ord + 2}')
    W = Y[..., :nw * s].reshape(M, m, nw, s)
    if rev_seg:
        W = torch.cat([W, Y[..., T - nw * s:].reshape(M, m, nw, s)], 2)
    Q = _basis(s, pol_ord, Y.dtype, Y.device)
    E = W - (W @ Q) @ Q.T
    return E.permute(0, 2, 1, 3)                              # (M, nw, m, s)


def window_fluct(Y, s, pol_ord=1, rev_seg=False, kind='signed'):
    """Per-window cross fluctuation of every within-market pair, (M, nw, m, m).

    kind = 'signed'  (1/s) sum_j e_x e_y           (window covariance; diagonal = MFDFA F2)
           'fathon'  (1/s) sum_j |e_x e_y|         (fathon's MFDCCA, abs point by point)
           'window'  |(1/s) sum_j e_x e_y|         (Zhou's MF-DXA, abs of the window covariance)
    """
    E = _residuals(Y, s, pol_ord, rev_seg)
    if kind == 'fathon':
        Ea = E.abs()
        return Ea @ Ea.transpose(-1, -2) / s                  # |e_x e_y| = |e_x| |e_y|: one matmul
    C = E @ E.transpose(-1, -2) / s
    return C.abs() if kind == 'window' else C


# ---------------------------------------------------------------------------
# q-averages over windows (dim 1 of (M, nw, m, m))
# ---------------------------------------------------------------------------

def _qmean(A, q, rel_floor=REL_FLOOR):
    """(mean_v A^{q/2})^{1/q} over dim 1 for A >= 0, log-average at q = 0; returns (M, m, m)."""
    ref = A.mean(1, keepdim=True).clamp_min(torch.finfo(A.dtype).tiny)
    Z = A / ref                                              # O(1) values: no overflow at large |q|
    if q > Q_ZERO:
        out = (Z ** (0.5 * q)).mean(1) ** (1.0 / q)
    else:
        keep = Z > rel_floor
        n = keep.sum(1).to(A.dtype)
        Zs = torch.where(keep, Z, torch.ones_like(Z))
        if abs(q) <= Q_ZERO:
            out = torch.exp(0.5 * (torch.log(Zs) * keep).sum(1) / n)
        else:
            out = ((Zs ** (0.5 * q)) * keep).sum(1).div(n) ** (1.0 / q)
    return out * ref.squeeze(1).sqrt()


def _signed_moment(C, q):
    """S(q) = mean_v sign(C)|C|^{q/2} (q > 0), returned as (S / ref^{q/2}, ref) to avoid overflow."""
    ref = C.abs().mean(1, keepdim=True).clamp_min(torch.finfo(C.dtype).tiny)
    Z = C / ref
    S = (torch.sign(Z) * Z.abs() ** (0.5 * q)).mean(1)
    return S, ref.squeeze(1)


def _as_list(qs):
    return [float(q) for q in (qs.tolist() if torch.is_tensor(qs) or isinstance(qs, np.ndarray) else qs)]


def _chunks(M, chunk):
    chunk = M if not chunk else chunk
    return [(a, min(M, a + chunk)) for a in range(0, M, chunk)]


# ---------------------------------------------------------------------------
# 1. MFDCCA fluctuation functions
# ---------------------------------------------------------------------------

def mfdcca_fluct(X, m, scales=SCALES, qs=(-3, -1, 1, 2, 4), pol_ord=1, rev_seg=False, mode='fathon',
                 dtype=torch.float64, chunk=None):
    """MFDCCA fluctuation functions of every within-market pair.

    Parameters
    ----------
    X : torch.Tensor (M * m, T), market by market (raw increments; the profile is taken here).
    m : agents per market.
    scales, qs : window sizes and q-orders.
    mode : 'fathon' (abs point by point, parity with fathon.MFDCCA), 'window' (abs of the window
        covariance, Zhou 2008) or 'signed' (q > 0: ``sign(S)|S|^{1/q}`` with the signed moment
        ``S = mean_v sign(F2)|F2|^{q/2}``; q <= 0 falls back to 'window').
    chunk : markets per batch (None: all at once).

    Returns
    -------
    Fxy : (len(qs), len(scales), M, m, m) -- F_xy(q, s); its diagonal is the MFDFA F_xx(q, s).
    Fxx, Fyy : the same shape, expanded views of the diagonal (Fxx[..., i, j] = F_ii, Fyy = F_jj).
    """
    qs = _as_list(qs)
    M = X.shape[0] // m
    out = torch.empty(len(qs), len(scales), M, m, m, dtype=dtype, device=X.device)
    for a, b in _chunks(M, chunk):
        Y = _profiles(X[a * m:b * m], m, dtype)
        for k, s in enumerate(scales):
            C = window_fluct(Y, s, pol_ord, rev_seg, 'signed' if mode in ('signed', 'window') else 'fathon')
            A = C.abs() if mode in ('signed', 'window') else C
            for iq, q in enumerate(qs):
                if mode == 'signed' and q > Q_ZERO:
                    S, ref = _signed_moment(C, q)
                    out[iq, k, a:b] = torch.sign(S) * S.abs() ** (1.0 / q) * ref.sqrt()
                else:
                    out[iq, k, a:b] = _qmean(A, q)
    d = torch.diagonal(out, dim1=-2, dim2=-1)
    return out, d[..., :, None].expand_as(out), d[..., None, :].expand_as(out)


# ---------------------------------------------------------------------------
# 2. q-dependent detrended cross-correlation coefficient
# ---------------------------------------------------------------------------

def rho_q(X, m, scales=SCALES, qs=(1, 2, 4), pol_ord=1, rev_seg=False, form='kristoufek', agg='mean',
          abs_scales=False, dtype=torch.float64, chunk=None):
    """q-dependent detrended cross-correlation coefficient of every within-market pair.

    ``form='kristoufek'``: rho(q, s) = sign(S) |S|^{2/q} / (F_xx F_yy) (signed F_xy^2 / F_xx F_yy);
    ``form='kwapien'``: S / (F_xx F_yy)^{q/2}. q <= 0: Kristoufek's unsigned version with the
    window absolute value, F_xy(q, s)^2 / (F_xx F_yy) (resp. its q/2 power for 'kwapien').
    At q = 2 both are rho_DCCA(s).

    agg : 'mean' (over scales) -> (len(qs), M, m, m); 'none' -> (len(qs), len(scales), M, m, m).
    abs_scales : average |rho(q, s)| over scales (the convention of run_e11_scale.rho_dcca_markets).
    """
    qs = _as_list(qs)
    M = X.shape[0] // m
    R = torch.empty(len(qs), len(scales), M, m, m, dtype=dtype, device=X.device)
    for a, b in _chunks(M, chunk):
        Y = _profiles(X[a * m:b * m], m, dtype)
        for k, s in enumerate(scales):
            C = window_fluct(Y, s, pol_ord, rev_seg, 'signed')
            D = torch.diagonal(C, dim1=-2, dim2=-1)          # (Mc, nw, m) = MFDFA F2 per window
            for iq, q in enumerate(qs):
                Fd = _qmean(D[..., None], q).squeeze(-1)    # (Mc, m)
                den = Fd[:, :, None] * Fd[:, None, :]
                if q > Q_ZERO:
                    S, ref = _signed_moment(C, q)           # S_true = S * ref^{q/2}
                    if form == 'kristoufek':
                        r = torch.sign(S) * S.abs() ** (2.0 / q) * ref / den
                    else:
                        r = S * (ref / den) ** (0.5 * q)
                else:
                    r = _qmean(C.abs(), q) ** 2 / den
                    if form != 'kristoufek':
                        r = r ** (0.5 * q) if abs(q) > Q_ZERO else torch.ones_like(r)
                R[iq, k, a:b] = r
    if abs_scales:
        R = R.abs()
    return R.mean(1) if agg == 'mean' else R


# ---------------------------------------------------------------------------
# 3. generalized cross-Hurst exponents and the MFDCCA spectrum width
# ---------------------------------------------------------------------------

def hxy(X, m, scales=SCALES, qs=(-3, -2, -1, 1, 2, 3), mode='fathon', **kw):
    """h_xy(q): OLS slope of log F_xy(q, s) on log s, (len(qs), M, m, m) (fathon's fitFlucVec)."""
    if mode == 'signed':
        raise ValueError("h_xy needs a positive F_xy: mode 'fathon' or 'window'")
    F, _, _ = mfdcca_fluct(X, m, scales, qs, mode=mode, **kw)
    ls = torch.log(torch.tensor(scales, dtype=F.dtype, device=F.device))
    ls = ls - ls.mean()
    return torch.einsum('qsaij,s->qaij', torch.log(F), ls) / (ls ** 2).sum()


def hxy_width(X, m, scales=SCALES, qs=(-3, -2, -1, 1, 2, 3), mode='fathon', return_h=False, **kw):
    """MFDCCA spectrum width h_xy(min q) - h_xy(max q) of every within-market pair, (M, m, m).

    The diagonal is the per-agent MFDFA width h(min q) - h(max q).
    """
    ql = _as_list(qs)
    H = hxy(X, m, scales, ql, mode=mode, **kw)
    w = H[int(np.argmin(ql))] - H[int(np.argmax(ql))]
    return (w, H) if return_h else w


# ---------------------------------------------------------------------------
# 4. rolling rho_q for early warning
# ---------------------------------------------------------------------------

def rolling_rho_q(X, m, window, ends, scales=(8, 16, 32, 64), qs=(1, 2, 4), **kw):
    """rho_q on X[:, end - window : end] for every end in `ends` -> (len(ends), len(qs), M, m, m).

    ``window=None`` uses expanding windows X[:, :end]. The profile is rebuilt in every window
    (only past data enter the value at `end`); scales larger than the window are dropped.
    """
    out = []
    for e in ends:
        a = 0 if window is None else e - window
        if a < 0 or e > X.shape[1]:
            raise ValueError(f'window ({a}, {e}) outside the series')
        sc = [s for s in scales if s <= e - a]
        out.append(rho_q(X[:, a:e], m, sc, qs, **kw))
    return torch.stack(out)


# ---------------------------------------------------------------------------
# pilot comparison on the planted-market simulators
# ---------------------------------------------------------------------------

def _pilot():
    import argparse
    import json
    import os
    import sys
    import time
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, 'sims'))
    sys.path.insert(0, here)
    import planted_markets as pm
    import run_e12_activity as e12
    from fathon.tensor import ews

    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=int, default=1000)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(here, 'results', 'baselines_mf_pilot.json'))
    args = ap.parse_args()
    M, m, T, dev = args.markets, args.m, args.T, 'cuda'
    iu = torch.triu_indices(m, m, 1, device=dev)
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    QR = [1.0, 2.0, 4.0]
    QW = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    sync = torch.cuda.synchronize

    worlds = [('copy', dict())]
    for band in (None, (1, 4), (5, 8)):
        for off in (0, 37):
            worlds.append((f'timing_band{"default" if band is None else "%d-%d" % band}_off{off}', dict(band=band, offset=off)))

    def labels(ttype, codes):
        lab = torch.zeros(M, iu.shape[1], dtype=torch.int8)
        for (a, b), tp in ttype.items():
            if tp in codes:
                lab[a // m, pidx[(a % m, b % m)]] = codes[tp]
        return lab.reshape(-1).numpy()

    def aucs(score, lab, codes):
        s = score[:, iu[0], iu[1]].reshape(-1).double().cpu().numpy()
        pos, neg = lab > 0, lab == 0
        r = dict(auc=ews.auc(s[pos], s[neg]))
        for tp, c in codes.items():
            if (lab == c).any():
                r[f'auc_{tp}'] = ews.auc(s[lab == c], s[neg])
        r['nonfinite'] = int((~np.isfinite(s)).sum())
        return r

    res = dict(markets=M, m=m, T=T, seed=args.seed, scales=SCALES, qs_rho=QR, qs_width=QW,
               notes=('rho_q: signed Kristoufek form (q>0), mean over scales; "abs" = mean over scales of |rho(q,s)| '
                      '(the rho_dcca_markets convention). width: h_xy(-3) - h_xy(3) from fathon-convention F_xy '
                      '(abs point by point) and from window-abs F_xy (Zhou MF-DXA). AUC rank-based, within-market pairs, '
                      'all coalition pairs vs pairs outside any coalition; AUC < 0.5 means the score is lower for colluders. '
                      'chunk() has no offset parameter, so the copy world is run once.'),
               worlds={}, seconds={})
    tm = dict(rho_q=0.0, width_fathon=0.0, width_window=0.0, clean=0.0, n_calls=0)
    for wname, wkw in worlds:
        sync(); t0 = time.time()
        if wname == 'copy':
            X, _, truth, ttype = pm.chunk(0, M, m, T, pm.global_latents(T, 4, 2, seed=args.seed, device=dev),
                                          seed=args.seed, device=dev)
            codes = {'copy_pair': 1, 'copy_triad': 2, 'xor_triad': 3, 'copy_onset': 4}
        else:
            G = pm.shared_activity(2 * T if wkw['offset'] else T, seed=args.seed)
            X, _, truth, ttype = pm.chunk_activity(0, M, m, T, G, seed=args.seed, device=dev, **wkw)
            codes = e12.CODES
            codes = {k: v for k, v in codes.items() if v}
        sync(); tsim = time.time() - t0
        lab = labels(ttype, codes)
        W = dict(n_pairs=int(lab.size), n_true=int((lab > 0).sum()), simulate_s=tsim, results={})
        chans = dict(x=X, abs=X.abs(), logabs=torch.log(X.abs() + 1e-12))
        for cname, Xc in chans.items():
            for prep in ('raw', 'clean'):
                if prep == 'clean':
                    sync(); t0 = time.time()
                    Z = e12.clean(Xc.float(), m)
                    sync(); tm['clean'] += time.time() - t0
                else:
                    Z = Xc
                sync(); t0 = time.time()
                R = rho_q(Z, m, SCALES, QR, agg='none')
                sync(); tm['rho_q'] += time.time() - t0
                t0 = time.time()
                wf = hxy_width(Z, m, SCALES, QW, mode='fathon')
                sync(); tm['width_fathon'] += time.time() - t0
                t0 = time.time()
                ww = hxy_width(Z, m, SCALES, QW, mode='window')
                sync(); tm['width_window'] += time.time() - t0
                tm['n_calls'] += 1
                for iq, q in enumerate(QR):
                    W['results'][f'{cname}/{prep}/rho_q{q:g}'] = aucs(R[iq].mean(0), lab, codes)
                    W['results'][f'{cname}/{prep}/rho_q{q:g}_abs'] = aucs(R[iq].abs().mean(0), lab, codes)
                W['results'][f'{cname}/{prep}/hxy_width_fathon'] = aucs(wf, lab, codes)
                W['results'][f'{cname}/{prep}/hxy_width_window'] = aucs(ww, lab, codes)
                del R
        # early warning (copy world: onset pairs switch on at T/2): rolling rho_2 on cleaned x
        if wname == 'copy':
            Z = e12.clean(X.float(), m)
            ends = [256, 384, 512, 640, 768, 896, 1024]
            RR = rolling_rho_q(Z, m, 256, ends, qs=(2.0,))
            onset, null = lab == 4, lab == 0
            W['early_warning_rho2_clean_window256'] = {
                str(e): ews.auc(RR[k, 0][:, iu[0], iu[1]].reshape(-1).cpu().numpy()[onset],
                                RR[k, 0][:, iu[0], iu[1]].reshape(-1).cpu().numpy()[null]) for k, e in enumerate(ends)}
        best = max(W['results'].items(), key=lambda kv: kv[1]['auc'])
        W['best'] = dict(config=best[0], **best[1])
        bestdev = max(W['results'].items(), key=lambda kv: abs(kv[1]['auc'] - 0.5))
        W['best_either_direction'] = dict(config=bestdev[0], **bestdev[1])
        res['worlds'][wname] = W
        print(f'{wname:28s} best {best[0]:32s} AUC {best[1]["auc"]:.3f} | max|AUC-.5| {bestdev[0]} {bestdev[1]["auc"]:.3f}',
              file=sys.stderr, flush=True)
        del X, chans
    n = max(tm.pop('n_calls'), 1)
    res['seconds_per_1000_markets'] = {k: v / n * 1000.0 / M for k, v in tm.items()}
    res['clean_rank_estimates'] = e12.CLEAN_RANKS
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=2)
    print(json.dumps(res['seconds_per_1000_markets']), file=sys.stderr)


if __name__ == '__main__':
    _pilot()
