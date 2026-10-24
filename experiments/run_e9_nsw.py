"""E9: who prices together? NSW FuelCheck, thousands of real stations, weak ground truth.

Panel: daily carried-forward U91 prices of NSW petrol stations (data.nsw.gov.au FuelCheck,
CC BY-SA), loaded by experiments/data/nsw_fuelcheck.py. Every station follows the same wholesale
price and the Sydney price cycle: a strong dense common mode that makes all pairwise correlations
positive. Coordinated pricing is a *localised* dependence: stations under one owner / brand in one
area move together beyond the common mode. Brand x postcode is therefore a (weak, noisy) proxy label
for "same decision maker".

Pipeline: fathon.tensor.scale.detect on daily price changes (common-mode removal with the estimated
rank, lag-0 and lag-1 channels, closure); report the enrichment of top-K pairs in same-brand /
same-postcode / same-brand-and-suburb relations against the base rate over all pairs, with and
without common-mode removal, plus a circular-shift null for the number of pairs above threshold.

Output: experiments/results/e9/summary.json, top_pairs.csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'data'))
import nsw_fuelcheck as nsw  # noqa: E402
from fathon.tensor import scale  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e9' + os.environ.get('E9_TAG', ''))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FUEL = os.environ.get('E9_FUEL', 'U91')
START, END = os.environ.get('E9_START', '2022-01-01'), os.environ.get('E9_END', '2024-12-31')
MIN_COVER = float(os.environ.get('E9_COVER', 0.8))
MAX_GAP = int(os.environ.get('E9_GAP', 60))
KS = (200, 500, 1000, 2000, 5000)


def relations(st):
    """Boolean pair-relation matrices (N x N) for brand, postcode, brand & suburb."""
    b = np.asarray(st['brand'].astype(str).str.upper().tolist(), dtype=object)
    p = np.asarray(st['postcode'].astype(str).tolist(), dtype=object)
    s = np.asarray(st['suburb'].astype(str).str.upper().tolist(), dtype=object)
    same_brand = b[:, None] == b[None, :]
    same_post = p[:, None] == p[None, :]
    same_sub = s[:, None] == s[None, :]
    return dict(same_brand=same_brand, same_postcode=same_post, same_suburb=same_sub, same_brand_suburb=same_brand & same_sub)


def enrichment(pairs, rel, N):
    iu = np.triu_indices(N, 1)
    base = {k: float(v[iu].mean()) for k, v in rel.items()}
    out = {}
    for k, v in rel.items():
        hit = float(np.mean([v[i, j] for i, j in pairs])) if pairs else float('nan')
        out[k] = dict(rate=hit, base=base[k], lift=hit / base[k] if base[k] > 0 else None)
    # ownership given co-location: among same-suburb pairs, how often the same brand
    ss = rel['same_suburb']
    sb = rel['same_brand']
    base_cond = float(sb[iu][ss[iu]].mean())
    top_ss = [(i, j) for i, j in pairs if ss[i, j]]
    cond = float(np.mean([sb[i, j] for i, j in top_ss])) if top_ss else float('nan')
    out['same_brand_given_same_suburb'] = dict(rate=cond, base=base_cond, lift=cond / base_cond if base_cond > 0 else None,
                                               n_same_suburb_in_top=len(top_ss))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    panel, tindex, st = nsw.load_panel(fuel=FUEL, freq='D', ffill=True, max_gap_days=MAX_GAP)
    tindex = pd.DatetimeIndex(tindex)
    sel_t = (tindex >= START) & (tindex <= END)
    P = panel[:, sel_t]
    cover = np.isfinite(P).mean(1)
    keep = cover >= MIN_COVER
    P, st = P[keep], st.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    N, T = P.shape
    print(f'{FUEL} {START}..{END}: {N} stations with >= {MIN_COVER:.0%} coverage, {T} days')
    # daily log-price changes; a carried-forward day is a zero change
    L = np.log(np.where(np.isfinite(P), P, np.nan))
    D = np.diff(L, axis=1)
    D = np.nan_to_num(D, nan=0.0).astype(np.float32)
    rel = relations(st)
    Xt = torch.as_tensor(D, device=DEV)
    rank_est, S = scale.estimate_rank(Xt)
    svals = S.cpu().numpy()
    summ = dict(fuel=FUEL, start=START, end=END, N=int(N), T=int(T), rank_est=int(rank_est),
                top_sv=[float(v) for v in svals[:10]],
                var_share_top5=float((svals[:5] ** 2).sum() / float((Xt - Xt.mean(1, keepdim=True)).norm() ** 2)),
                enrichment={})
    summ['coverage_hist'] = np.histogram(cover, bins=[0, .1, .3, .5, .7, .8, .9, 1.01])[0].tolist()
    # local common mode: leave-one-out mean of the other stations in the same postcode
    pc = np.asarray(st['postcode'].astype(str).tolist(), dtype=object)
    codes = pd.factorize(pc)[0]
    G = torch.zeros(codes.max() + 1, N, device=DEV)
    G[torch.as_tensor(codes, device=DEV), torch.arange(N, device=DEV)] = 1.0
    cnt = G.sum(1)
    Xc = Xt - Xt.mean(1, keepdim=True)
    grp_sum = G @ Xc                                             # postcode sums
    others = (grp_sum[codes] - Xc)
    n_others = (cnt[codes] - 1).clamp(min=1)[:, None]
    local = Xc - others / n_others
    local[cnt[codes] < 3] = Xc[cnt[codes] < 3]                   # lone stations keep their series
    summ['postcodes_with_3plus'] = int((cnt >= 3).sum())
    rows = []
    for name, kw in (('common_mode_removed', dict(rank=None)), ('no_removal', dict(rank=0)),
                     ('local_postcode_removed', dict(rank=-1)), ('global_then_local', dict(rank=-2))):
        for lag in (0, 1):
            t0 = time.perf_counter()
            if kw['rank'] == 0:
                R = Xc
            elif kw['rank'] == -1:
                R = local
            elif kw['rank'] == -2:
                R, _ = scale.common_mode_residual(local, rank_est)
            else:
                R, _ = scale.common_mode_residual(Xt, rank_est)
            Rn = scale._normalise(R)
            A, B = (Rn[:, lag:], Rn[:, :T - 1 - lag]) if lag else (Rn, Rn)
            vals, idx = scale.pair_topk(A, B, max(KS))
            dt = time.perf_counter() - t0
            pairs = [tuple(p) for p in idx.tolist()]
            for K in KS:
                e = enrichment(pairs[:K], rel, N)
                summ['enrichment'][f'{name}_lag{lag}_K{K}'] = e
            summ[f'{name}_lag{lag}_seconds'] = dt
            for r_, (i, j) in enumerate(pairs[:500]):
                rows.append(dict(method=name, lag=lag, rank=r_, i=int(i), j=int(j), score=float(vals[r_]),
                                 brand_i=st.brand[i], brand_j=st.brand[j], suburb_i=st.suburb[i], suburb_j=st.suburb[j],
                                 postcode_i=st.postcode[i], postcode_j=st.postcode[j]))
    # circular-shift null (each station shifted independently) for the top score of the lag-0 residual channel
    R, _ = scale.common_mode_residual(Xt, rank_est)
    Rn = scale._normalise(R)
    obs_vals, _ = scale.pair_topk(Rn, Rn, 5000)
    g = torch.Generator(device=DEV).manual_seed(0)
    null_top = []
    for _ in range(10):
        sh = torch.randint(T // 10, T - T // 10, (N,), device=DEV, generator=g)
        ar = torch.arange(Rn.shape[1], device=DEV)
        Rs = torch.gather(Rn, 1, (ar[None, :] + sh[:, None]) % Rn.shape[1])
        Rs = scale._normalise(Rs)
        v, _ = scale.pair_topk(Rs, Rs, 5000)
        null_top.append(v.cpu().numpy())
    null_top = np.array(null_top)
    thr = float(null_top[:, 0].max())
    summ['shift_null'] = dict(null_max_score_mean=float(null_top[:, 0].mean()), null_max_score_max=thr,
                              observed_top=float(obs_vals[0]),
                              pairs_above_null_max=int((obs_vals.cpu().numpy() > thr).sum()),
                              null_5000th=float(null_top[:, -1].mean()), observed_5000th=float(obs_vals[-1]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'top_pairs.csv'), index=False)
    json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != 'enrichment'}, indent=1))
    for k, e in summ['enrichment'].items():
        if k.endswith('K1000') or k.endswith('K200'):
            print(k, {n: (round(x['rate'], 3), round(x['base'], 4), round(x['lift'], 1) if x['lift'] else None) for n, x in e.items() if n in ('same_brand', 'same_postcode', 'same_brand_given_same_suburb')})


if __name__ == '__main__':
    main()
