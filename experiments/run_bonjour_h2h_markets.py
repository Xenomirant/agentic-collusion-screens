"""Head-to-head on the planted-market world (E14 configuration): Bonjour et al. (UAI 2022) net
influence vs the fair classical set vs our detectors, on the same simulated data.

World: experiments/sims/planted_markets.py `chunk` with the E14 settings (7 coalition types, each in
a fraction `prev` of markets, benign one-way followers in half the markets with eps_benign 0.5,
w_global 2.0, eps_copy 0.75, eps_xor 2.0, eps_match 1.0, c 0.5, copy lag 2, m = 20, T = 1024).
A coalition-free twin (same parameters, mix empty, seed + 7, same global latents) sets thresholds.

Bonjour per market (binary actions, MI in bits, everything batched over markets on the GPU):
  gamma(i; j) = I(a_i(t - lag); a_j(t) | s_t), Gamma(i; j) = gamma(i; j) - max_{k != i, j} gamma(k; j)
  public states s_t (rounds t = 1 .. T-1; s_t is computed from round t - 1):
    none   no conditioning (reference)
    maj2   previous round's market majority action (mean >= 0.5 -> 1; ties to 1)
    q4     previous round's market mean action binned at that market's own quartiles (4 levels)
  target-specific coarsenings of the public previous joint action (2^20 values, too fine to use):
    te     (maj2_t, a_j(t-1)): previous majority x the TARGET's own previous action (4 levels).
           a_j(t-1) is part of the previous joint action, so this is a function of the public state
           (the coarsening that matches E5/E6). This is the main Bonjour configuration.
    teq4   (q4_t, a_j(t-1)) (8 levels)
    (the source's a_i(t-1) is NOT added: it would equal x_i(t - 1) and null the lag-1 channel.)
  extension, population information (NOT a within-market public state; the same information the
  classical cleaning uses): F_j(t) = agent j's fitted global common mode, the projection of its
  centred series on scale.global_basis (rank 8+8, the basis the classical pipeline removes), binned
  at j's own quartiles:
    cm       F_j quartile (4)        cmown   F_j quartile x a_j(t-1) (8)
    cmownmaj F_j quartile x a_j(t-1) x maj2 (16)
  lag specs: lag0 (simultaneous, the paper's setting), lag1 (sequential variant), lag123 (max of
  gamma over lags 1, 2, 3, then net influence).
  pair scores: mutual = min(Gamma_ij, Gamma_ji) (the paper's both-ways rule), oneway = max.
  calibrated: influence_z with null='markov' (order 1), re-implemented batched: B surrogates of
  every agent from its fitted P(a_j(t) | s_t, a_j(t-1)) (fallback P(a_j | s_t)), z of gamma for
  (actual i, surrogate j); lag123 uses max over lags with the same surrogate. Pair score min / max
  over the two directions (the library symmetrises lag 0 by resampling j only).
  Bonjour's rule (detect_pair): a market is flagged when exactly one pair has min(Gamma) >= alpha.

Classical (fair, from experiments/audit/e14_fair.py): rho_DCCA on cleaned data (global rank-8+8
common mode + market mean removed), lags 0-3, two-sided, signed rho at scales 4, 8, 16 averaged,
robust z per lag, max; MFDCCA rho_q at q = 4 with the same settings; volatility channel rho_DCCA of
|a(t) - a(t-h)|; their robust-z max (fair_classical_max). Also the original raw baselines and our
v1 / v2_oneway / v2_mutual (run_e14_threats.detectors).

Information-matched classical (classical_mm_max): the same fair set with the market mean removed but
no global basis (vol_dcca already uses only the market mean), to compare with Bonjour on within-market
states only.

Ablation (bjgauss_clean_*): Bonjour's net influence computed on the pipeline's cleaned residuals
with Gaussian MI -0.5 log2(1 - r^2) of the lagged residual correlations (lags 0, 1-3, 0-3), to
separate the effect of the statistic (net influence) from the effect of the input (common mode).

Output: experiments/results/bonjour_h2h/markets.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sims'))
sys.path.insert(0, HERE)
import planted_markets as pm  # noqa: E402
import baselines_mf as bmf  # noqa: E402
from run_e14_threats import detectors, TYPES, CODE  # noqa: E402
from run_e11_scale import triu  # noqa: E402
from fathon.tensor import scale, ews  # noqa: E402

DEV = 'cuda'
OUT = os.path.join(HERE, 'results', 'bonjour_h2h')
LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# fair classical set (copied from experiments/audit/e14_fair.py, parameterised)
# ---------------------------------------------------------------------------

def lag_rho(X, m, lag, scales, q=2):
    if lag == 0:
        return bmf.rho_q(X, m, scales=scales, qs=(q,))[0].float()
    Mk = X.shape[0] // m
    Xl = X[:, lag:].reshape(Mk, m, -1)
    Xe = X[:, :-lag].reshape(Mk, m, -1)
    Z = torch.cat([Xl, Xe], 1).reshape(Mk * 2 * m, -1)
    R = bmf.rho_q(Z, 2 * m, scales=scales, qs=(q,))[0]
    return R[:, :m, m:].float()


def classical_multilag(R, m, iu, scales, q=2, lags=(0, 1, 2, 3)):
    best = None
    for lag in lags:
        C = lag_rho(R, m, lag, scales, q)
        S = C.abs() if lag == 0 else torch.maximum(C.abs(), C.transpose(1, 2).abs())
        S = S[:, iu[0], iu[1]]
        Z = scale.robust_z(S.reshape(-1)).reshape(S.shape)
        best = Z if best is None else torch.maximum(best, Z)
    return best.reshape(-1)


def vol_channel(A, m, iu, lags=(0, 1, 2), horizons=(1, 2), scales=(4, 8, 16)):
    T = A.shape[1]
    V = {h: (A[:, h:] - A[:, :-h]).abs() for h in horizons}
    L = T - max(horizons)
    V = {h: v[:, v.shape[1] - L:] for h, v in V.items()}
    best = None
    for hi in horizons:
        for hj in horizons:
            for lag in lags:
                Vi = scale.market_mean_removed(V[hi], m)
                Vj = scale.market_mean_removed(V[hj], m)
                Mk = Vi.shape[0] // m
                if lag:
                    Xl, Xe = Vi[:, lag:].reshape(Mk, m, -1), Vj[:, :-lag].reshape(Mk, m, -1)
                else:
                    Xl, Xe = Vi.reshape(Mk, m, -1), Vj.reshape(Mk, m, -1)
                Z = torch.cat([Xl, Xe], 1).reshape(Mk * 2 * m, -1)
                C = bmf.rho_q(Z, 2 * m, scales=scales, qs=(2,))[0][:, :m, m:].float()
                S = torch.maximum(C.abs(), C.transpose(1, 2).abs())[:, iu[0], iu[1]]
                Zs = scale.robust_z(S.reshape(-1)).reshape(S.shape)
                best = Zs if best is None else torch.maximum(best, Zs)
    return best.reshape(-1)


# ---------------------------------------------------------------------------
# Bonjour et al., batched over markets
# ---------------------------------------------------------------------------

def public_states(A3):
    """A3 (M, m, T) in {0,1}. Returns dict name -> (S (M, T-1) long, ns) for rounds 1..T-1."""
    pm_ = A3[:, :, :-1].mean(1)                                   # previous-round market mean
    M = A3.shape[0]
    out = {'none': (torch.zeros(M, pm_.shape[1], dtype=torch.long, device=A3.device), 1),
           'maj2': ((pm_ >= 0.5).long(), 2)}
    qs = torch.quantile(pm_, torch.tensor([0.25, 0.5, 0.75], device=A3.device), dim=1).T   # (M, 3)
    out['q4'] = (((pm_[:, :, None] > qs[:, None, :]).sum(-1)).long(), 4)
    return out


def counts(X, Y, S, ns, lag):
    """C[k, i, j, s, a, b] = #{t: x_i(t - lag) = a, y_j(t) = b, s_t = s}; X, Y (M, m, T'), S (M, T')."""
    Tn = X.shape[2]
    Xl, Yl, Sl = X[:, :, :Tn - lag], Y[:, :, lag:], S[:, lag:]
    M, m = X.shape[0], X.shape[1]
    C = torch.empty(M, m, m, ns, 2, 2, device=X.device, dtype=torch.float32)
    Xs = (Xl, 1 - Xl)
    Ys = (Yl, 1 - Yl)
    for v in range(ns):
        w = (Sl == v).float()[:, None, :]
        for a in (0, 1):
            Xa = Xs[1 - a] * w                          # a = 0 -> 1 - x, a = 1 -> x
            for b in (0, 1):
                C[..., v, a, b] = Xa @ Ys[1 - b].transpose(1, 2)
    return C


def counts_tgt(X, Y, Sj, ns, lag):
    """As `counts` with a target-specific state Sj (M, m, T'): s_t of the pair (i, j) is Sj[:, j, t]."""
    Tn = X.shape[2]
    Xl, Yl, Sl = X[:, :, :Tn - lag], Y[:, :, lag:], Sj[:, :, lag:]
    M, m = X.shape[0], X.shape[1]
    C = torch.empty(M, m, m, ns, 2, 2, device=X.device, dtype=torch.float32)
    Xs = (Xl, 1 - Xl)
    Ys = (Yl, 1 - Yl)
    for v in range(ns):
        w = (Sl == v).float()
        for a in (0, 1):
            for b in (0, 1):
                C[..., v, a, b] = Xs[1 - a] @ (Ys[1 - b] * w).transpose(1, 2)
    return C


def target_states(A3, states, V=None):
    """Target-specific states (M, m, T-1) for rounds 1..T-1; see the module docstring."""
    M, m, T = A3.shape
    own = A3[:, :, :-1].long()                                    # a_j(t-1)
    maj = states['maj2'][0][:, None, :].expand(M, m, -1)
    q4 = states['q4'][0][:, None, :].expand(M, m, -1)
    out = {'te': (maj * 2 + own, 4), 'teq4': (q4 * 2 + own, 8)}
    if V is not None:
        Ac = A3.reshape(M * m, T)
        Ac = Ac - Ac.mean(1, keepdim=True)
        F = ((Ac @ V) @ V.T).reshape(M, m, T)[:, :, 1:]          # fitted common mode at round t
        qs = torch.quantile(F, torch.tensor([0.25, 0.5, 0.75], device=A3.device), dim=2).permute(1, 2, 0)
        Fq = (F[..., None] > qs[:, :, None, :]).sum(-1).long()
        out.update({'cm': (Fq, 4), 'cmown': (Fq * 2 + own, 8), 'cmownmaj': ((Fq * 2 + own) * 2 + maj, 16)})
    return out


def mi_bits(C):
    C = C.double()
    n = C.sum((-3, -2, -1))
    ps = C.sum((-2, -1), keepdim=True)
    px = C.sum(-1, keepdim=True)
    py = C.sum(-2, keepdim=True)
    with torch.no_grad():
        ratio = C * ps / (px * py)
        term = torch.where(C > 0, C * torch.log(torch.where(C > 0, ratio, torch.ones_like(ratio))), torch.zeros_like(C))
    return (term.sum((-3, -2, -1)) / n.clamp_min(1.0) / LN2).float()


def gamma_all(X, Y, S, ns, lags):
    """gamma[k, i, j] = I(x_i(t - lag); y_j(t) | s_t) in bits for each lag; dict lag -> (M, m, m)."""
    return {L: mi_bits(counts(X, Y, S, ns, L)) for L in lags}


def net_influence_b(G):
    """Gamma[k, i, j] = G[k, i, j] - max_{l != i, j} G[k, l, j]; diagonal NaN."""
    M, m, _ = G.shape
    eye = torch.eye(m, dtype=torch.bool, device=G.device)
    Gm = G.masked_fill(eye, float('-inf'))
    top2 = Gm.topk(2, dim=1)                             # over l (sources) for each column j
    v1, v2 = top2.values[:, 0, :], top2.values[:, 1, :]  # (M, m)
    i1 = top2.indices[:, 0, :]
    ar = torch.arange(m, device=G.device)
    sub = torch.where(i1[:, None, :] == ar[None, :, None], v2[:, None, :], v1[:, None, :])
    Gam = G - sub
    return Gam.masked_fill(eye, float('nan'))


def markov_surrogates(A3, S, ns, B, gen):
    """B surrogates of every agent from its fitted P(a(t) | s_t, a(t-1)); A3 (M, m, T'), S (M, T')."""
    M, m, Tn = A3.shape
    y = A3.long()
    s = S[:, None, :].expand(M, m, Tn)
    idx = (s[:, :, 1:] * 2 + y[:, :, :-1]) * 2 + y[:, :, 1:]
    Ccell = torch.zeros(M, m, ns * 4, device=A3.device)
    Ccell.scatter_add_(2, idx, torch.ones_like(idx, dtype=torch.float32))
    Ccell = Ccell.reshape(M, m, ns, 2, 2)
    Cs = torch.zeros(M, m, ns * 2, device=A3.device)
    Cs.scatter_add_(2, s * 2 + y, torch.ones_like(y, dtype=torch.float32))
    Cs = Cs.reshape(M, m, ns, 2)
    Ps = Cs[..., 1] / Cs.sum(-1).clamp_min(1.0)                      # P(1 | s)
    tot = Ccell.sum(-1)
    P = torch.where(tot > 0, Ccell[..., 1] / tot.clamp_min(1.0), Ps[..., None].expand_as(tot))   # (M, m, ns, 2)
    Yb = torch.empty(B, M, m, Tn, device=A3.device, dtype=torch.float32)
    Yb[..., 0] = A3[..., 0]
    Pflat = P.reshape(M, m, ns * 2)
    for t in range(1, Tn):
        cell = S[:, t][None, :, None] * 2 + Yb[..., t - 1].long()                 # (B, M, m)
        p = torch.gather(Pflat[None].expand(B, M, m, ns * 2), 3, cell[..., None]).squeeze(-1)
        Yb[..., t] = (torch.rand(B, M, m, device=A3.device, generator=gen) < p).float()
    return Yb


def bonjour_scores(A3, iu, nperm, seed, z_states=('maj2', 'q4'), z_markets=None, V=None):
    """All Bonjour variants for a world. Returns dict name -> flat pair scores (M * n_pairs,), and
    per-market Gamma-min matrices for the detection rule."""
    X = A3[:, :, 1:]
    states = public_states(A3)
    tstates = target_states(A3, states, V)
    out, rule = {}, {}
    lags = (0, 1, 2, 3)
    for sname, (S, ns) in list(states.items()) + list(tstates.items()):
        if S.dim() == 3:
            G = {L: mi_bits(counts_tgt(X, X, S, ns, L)) for L in lags}
        else:
            G = gamma_all(X, X, S, ns, lags)
        specs = {'lag0': G[0], 'lag1': G[1], 'lag123': torch.maximum(torch.maximum(G[1], G[2]), G[3])}
        for lname, g in specs.items():
            Gam = net_influence_b(g)
            Gt = Gam.transpose(1, 2)
            mn = torch.minimum(Gam, Gt)[:, iu[0], iu[1]]
            mx = torch.maximum(Gam, Gt)[:, iu[0], iu[1]]
            out[f'bj_{sname}_{lname}_mutual'] = mn.reshape(-1)
            out[f'bj_{sname}_{lname}_oneway'] = mx.reshape(-1)
            out[f'gamma_{sname}_{lname}_max'] = torch.maximum(g, g.transpose(1, 2))[:, iu[0], iu[1]].reshape(-1)
            rule[f'{sname}_{lname}'] = mn                                     # (M, n_pairs)
    # calibrated influence z, Markov null
    gen = torch.Generator(device=A3.device).manual_seed(int(seed) * 7919 + 13)
    Mz = A3.shape[0] if z_markets is None else min(z_markets, A3.shape[0])
    Xz = X[:Mz]
    for sname in z_states:
        S, ns = states[sname]
        S = S[:Mz]
        G = gamma_all(Xz, Xz, S, ns, lags)
        obs = {'lag0': G[0], 'lag1': G[1], 'lag123': torch.maximum(torch.maximum(G[1], G[2]), G[3])}
        s1 = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in obs.items()}
        s2 = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in obs.items()}
        bs = 11
        done = 0
        while done < nperm:
            b = min(bs, nperm - done)
            Yb = markov_surrogates(Xz, S, ns, b, gen)
            for k in range(b):
                Gb = gamma_all(Xz, Yb[k], S, ns, lags)                       # x actual, y surrogate
                nb = {'lag0': Gb[0], 'lag1': Gb[1], 'lag123': torch.maximum(torch.maximum(Gb[1], Gb[2]), Gb[3])}
                for kk, v in nb.items():
                    v = v.double()
                    s1[kk] += v
                    s2[kk] += v * v
            done += b
            del Yb
        for lname in obs:
            mu = s1[lname] / nperm
            sd = ((s2[lname] - nperm * mu * mu) / (nperm - 1)).clamp_min(0).sqrt()
            Z = ((obs[lname].double() - mu) / sd.clamp_min(1e-12)).float()      # Z[k, i, j]: j resampled
            Zt = Z.transpose(1, 2)
            out[f'z_{sname}_{lname}_mutual'] = torch.minimum(Z, Zt)[:, iu[0], iu[1]].reshape(-1)
            out[f'z_{sname}_{lname}_oneway'] = torch.maximum(Z, Zt)[:, iu[0], iu[1]].reshape(-1)
    return out, rule, Mz


# ---------------------------------------------------------------------------

def world(args, mix, zg, iu, pidx):
    m, T, M = args.m, args.T, args.markets
    kw = dict(eps_copy=0.75, eps_xor=2.0, w_global=2.0, c=0.5, mix=mix, copy_lag=2, eps_match=1.0, eps_benign=0.5,
              benign_frac=0.5, seed=args.seed + (0 if mix else 7), device=DEV)
    A, mk, truth, ttype = pm.chunk(0, M, m, T, zg, **kw)
    lab = torch.zeros(M, iu.shape[1], dtype=torch.int8)
    for (a, b), tp in ttype.items():
        lab[a // m, pidx[(a % m, b % m)]] = CODE[tp]
    t0 = time.time()
    V = scale.global_basis(A[:50000], rank=8)
    out = detectors(A, V, m, iu, raw=True)
    R = scale.market_mean_removed(scale.remove_global(A, V), m)
    out['dcca_clean_ml_s4_16'] = classical_multilag(R, m, iu, (4, 8, 16))
    out['rhoq4_clean_ml_s4_16'] = classical_multilag(R, m, iu, (4, 8, 16), q=4)
    out['vol_dcca'] = vol_channel(A, m, iu)
    out['fair_classical_max'] = torch.maximum(torch.maximum(out['dcca_clean_ml_s4_16'], out['rhoq4_clean_ml_s4_16']),
                                              out['vol_dcca'])
    # the same fair classical set restricted to within-market information (no global basis: market mean
    # removed only), the information-matched counterpart of Bonjour with within-market public states
    Rm = scale.market_mean_removed(A - A.mean(1, keepdim=True), m)
    out['dcca_mm_ml_s4_16'] = classical_multilag(Rm, m, iu, (4, 8, 16))
    out['rhoq4_mm_ml_s4_16'] = classical_multilag(Rm, m, iu, (4, 8, 16), q=4)
    out['classical_mm_max'] = torch.maximum(torch.maximum(out['dcca_mm_ml_s4_16'], out['rhoq4_mm_ml_s4_16']),
                                            out['vol_dcca'])
    # ablation: Bonjour's net influence on the pipeline's input (cleaned residuals), with Gaussian MI
    # -0.5 log2(1 - r^2) of the lagged residual correlations (C[l][k, i, j] = corr(i(t), j(t - l)),
    # so the influence of j on i is G[k, j, i])
    C = scale.market_lag_channels(R, m, lags=(0, 1, 2, 3))
    Gl = {l: (-0.5 * torch.log2((1 - c * c).clamp_min(1e-12))).transpose(1, 2) for l, c in C.items()}
    gspec = {'lag0': Gl[0], 'lag123': torch.maximum(torch.maximum(Gl[1], Gl[2]), Gl[3])}
    gspec['lag0123'] = torch.maximum(gspec['lag0'], gspec['lag123'])
    for lname, g in gspec.items():
        Gam = net_influence_b(g)
        Gt = Gam.transpose(1, 2)
        out[f'bjgauss_clean_{lname}_mutual'] = torch.minimum(Gam, Gt)[:, iu[0], iu[1]].reshape(-1)
        out[f'bjgauss_clean_{lname}_oneway'] = torch.maximum(Gam, Gt)[:, iu[0], iu[1]].reshape(-1)
        out[f'gammagauss_clean_{lname}_max'] = torch.maximum(g, g.transpose(1, 2))[:, iu[0], iu[1]].reshape(-1)
    torch.cuda.synchronize()
    t1 = time.time()
    bj, rule, Mz = bonjour_scores(A.reshape(M, m, T), iu, args.nperm, args.seed + (0 if mix else 7),
                                  z_markets=args.z_markets, V=V)
    torch.cuda.synchronize()
    t2 = time.time()
    out.update(bj)
    S = {k: v.float().cpu().numpy().astype(np.float64) for k, v in out.items()}
    rule = {k: v.cpu().numpy() for k, v in rule.items()}
    print(f'[world seed {kw["seed"]}] classical+ours {t1 - t0:.1f}s, bonjour {t2 - t1:.1f}s', file=sys.stderr, flush=True)
    return S, lab.numpy(), rule, Mz, dict(classical_ours=t1 - t0, bonjour=t2 - t1)


def evaluate(S, L, S0, L0, Mz, npair):
    res = {}
    Lf, L0f = L.reshape(-1), L0.reshape(-1)
    for k in S:
        s, s0 = S[k], S0[k]
        lab, lab0 = Lf, L0f
        if k.startswith('z_') and s.size < lab.size:          # Markov-z on a subset of markets
            lab, lab0 = Lf[:Mz * npair], L0f[:Mz * npair]
        clean, benign, pos = s[lab == 0], s[lab == -1], s[lab > 0]
        clean0 = s0[lab0 == 0]
        r = dict(auc_clean=ews.auc(pos, clean), auc_benign=ews.auc(pos, benign), n_pos=int(pos.size))
        for tp in TYPES:
            r[f'auc_clean_{tp}'] = ews.auc(s[lab == CODE[tp]], clean)
            r[f'auc_benign_{tp}'] = ews.auc(s[lab == CODE[tp]], benign)
        c0 = clean0[np.isfinite(clean0)]
        thr = float(np.quantile(c0, 1 - 1e-3))
        tp_ = int((pos > thr).sum())
        fpc, fpb = int((clean > thr).sum()), int((benign > thr).sum())
        r['fpr1e-3'] = dict(threshold=thr, recall=float((pos > thr).mean()), true_flags=tp_, false_clean=fpc,
                            false_benign=fpb, fpr_clean_observed=fpc / max(1, clean.size),
                            fpr_benign=fpb / max(1, benign.size), fdr=(fpc + fpb) / max(1, tp_ + fpc + fpb),
                            **{f'recall_{tp}': float((s[lab == CODE[tp]] > thr).mean()) for tp in TYPES})
        nm = s.size // npair
        ms = np.nan_to_num(s[:nm * npair].reshape(nm, npair), nan=-np.inf).max(1)
        ml = lab[:nm * npair].reshape(nm, npair)
        coal = (ml > 0).any(1)
        r['market_auc'] = ews.auc(ms[coal], ms[~coal])
        res[k] = r
    return res


def rule_eval(rule, L, rule0, npair, alphas=(0.005, 0.01, 0.02, 0.05)):
    """Bonjour's detect_pair per market: flagged iff exactly one pair has min(Gamma) >= alpha."""
    out = {}
    ml = L.reshape(-1, npair)
    coal = (ml > 0).any(1)
    for k, mn in rule.items():
        mn0 = rule0[k]
        r = {}
        for a in alphas:
            f = mn >= a
            one = f.sum(1) == 1
            hit = one & (ml[np.arange(ml.shape[0]), f.argmax(1)] > 0)
            f0 = (mn0 >= a).sum(1) == 1
            r[str(a)] = dict(true_pair_rate_coalition_markets=float(hit[coal].mean()),
                             false_pair_rate_coalition_markets=float((one & ~hit)[coal].mean()),
                             flag_rate_coalitionfree_markets_same_world=float(one[~coal].mean()),
                             flag_rate_null_world=float(f0.mean()),
                             **{f'true_pair_rate_{tp}': float(hit[(ml == CODE[tp]).any(1)].mean()) for tp in TYPES})
        out[k] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=int, default=1000)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--prev', type=float, default=0.08)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--nperm', type=int, default=99)
    ap.add_argument('--z_markets', type=int, default=None)
    ap.add_argument('--out', type=str, default=os.path.join(OUT, 'markets.json'))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    m = args.m
    iu = triu(m)
    npair = iu.shape[1]
    pidx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu[0].tolist(), iu[1].tolist()))}
    per_seed = {}
    for seed in args.seeds:
        args.seed = seed
        zg = pm.global_latents(args.T, 4, 2, seed=seed, device=DEV)
        S, L, rule, Mz, tm = world(args, {t: args.prev for t in TYPES}, zg, iu, pidx)
        S0, L0, rule0, _, tm0 = world(args, {}, zg, iu, pidx)
        ev = evaluate(S, L, S0, L0, Mz, npair)
        rv = rule_eval(rule, L, rule0, npair)
        per_seed[seed] = dict(eval=ev, rule=rv, seconds=dict(coalition_world=tm, null_world=tm0), z_markets=Mz,
                              n_pos={tp: int((L == CODE[tp]).sum()) for tp in TYPES},
                              n_benign=int((L == -1).sum()), n_clean=int((L == 0).sum()))
        for k in ('v2_oneway', 'fair_classical_max', 'classical_mm_max', 'bj_maj2_lag123_oneway', 'bj_te_lag123_oneway', 'bj_te_lag123_mutual',
                  'bj_cmown_lag123_oneway', 'gamma_cmownmaj_lag123_max', 'z_q4_lag123_oneway'):
            if k in ev:
                print(f"seed {seed} {k:24s} AUCc {ev[k]['auc_clean']:.3f} AUCb {ev[k]['auc_benign']:.3f} "
                      f"rec@1e-3 {ev[k]['fpr1e-3']['recall']:.3f}", file=sys.stderr, flush=True)
        json.dump(dict(args=vars(args), per_seed=per_seed), open(args.out, 'w'), indent=1, default=float)
    # aggregate over seeds
    agg = {}
    keys = per_seed[args.seeds[0]]['eval'].keys()
    for k in keys:
        a = {}
        fields = ['auc_clean', 'auc_benign', 'market_auc'] + [f'auc_clean_{tp}' for tp in TYPES] + [f'auc_benign_{tp}' for tp in TYPES]
        for f in fields:
            v = np.array([per_seed[s]['eval'][k][f] for s in args.seeds], dtype=float)
            a[f] = [float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0]
        for f in ['recall', 'fdr', 'fpr_benign', 'fpr_clean_observed'] + [f'recall_{tp}' for tp in TYPES]:
            v = np.array([per_seed[s]['eval'][k]['fpr1e-3'][f] for s in args.seeds], dtype=float)
            a[f'fpr1e-3_{f}'] = [float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0]
        agg[k] = a
    rule_agg = {}
    for k in per_seed[args.seeds[0]]['rule']:
        rule_agg[k] = {}
        for al, d in per_seed[args.seeds[0]]['rule'][k].items():
            rule_agg[k][al] = {f: float(np.mean([per_seed[s]['rule'][k][al][f] for s in args.seeds])) for f in d}
    json.dump(dict(args=vars(args), aggregate=agg, rule_aggregate=rule_agg, per_seed=per_seed), open(args.out, 'w'),
              indent=1, default=float)


if __name__ == '__main__':
    main()
