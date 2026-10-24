"""E10: documented retail-gasoline cartels in Brazil (ANP weekly station survey + CADE decisions).

Labels: external/data/anp_brazil_fuel/labels/cade_retail_fuel_cartels.csv (loaded by
experiments/data/anp_brazil_fuel.py: cartel_periods()). The four strong cases (Belo Horizonte,
Brasilia, Caxias do Sul, Sao Luis) have month-resolution cartel windows and later control windows
from Silveira et al.; the rest have year-resolution windows.

Design. Every (city, week) gets indicators computed from the stations surveyed that week:
  cv            price dispersion across stations (the classic screen: cartels compress dispersion)
  sync          share of stations surveyed in consecutive weeks whose price moved in the majority
                direction (coordination index; 0.5 = no coordination among movers)
  resid_ar1     lag-1 autocorrelation of the city mean price relative to the state mean, rolling 26 w
  resid_var     variance of weekly changes of that residual, rolling 26 w
  hurst         DFA Hurst exponent of the residual, rolling 52 w (fathon.tensor.mvdfa)
Two contrasts:
  A. cross-sectional: the labelled city-week against all other cities of the same state and
     similar size (station count within a factor 2) in the same week -> percentile of the cartel
     city; averaged over the cartel window. A screen that works gives percentiles far from 50.
  B. temporal (strong cases, once the post-2012 files are downloaded): AUC of each indicator for
     cartel weeks vs control weeks within the city.
Output: experiments/results/e10/{city_week_indicators.parquet, summary.json}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'data'))
import anp_brazil_fuel as anp  # noqa: E402
from fathon.tensor import detrend, mvdfa, ews  # noqa: E402

OUT = os.path.join(HERE, 'results', 'e10')
ROLL_SHORT, ROLL_LONG = 26, 52
MIN_N = 8                      # stations per city-week for cv / sync


def hurst(x):
    x = np.asarray(x, dtype=float)
    if np.isnan(x).mean() > 0.1 or x.size < 48:
        return np.nan
    x = pd.Series(x).interpolate(limit_direction='both').values
    Y = detrend.profile(x[None, :])
    sc = np.array([4, 6, 8, 12, 16])
    F2 = mvdfa.covariance_tensor(Y, sc)[:, 0, 0]
    return float(np.polyfit(np.log(sc), 0.5 * np.log(F2), 1)[0])


def main():
    os.makedirs(OUT, exist_ok=True)
    tidy, st = anp.build()
    cp = anp.cartel_periods()
    d = tidy[tidy['product'] == 'GASOLINA'].merge(st[['station_id', 'city', 'state']], on='station_id')
    d['week'] = d['date'].dt.to_period('W-SAT').dt.start_time
    # station-week price (mean if surveyed twice)
    sw = d.groupby(['state', 'city', 'station_id', 'week'])['price_sell'].mean().reset_index()
    # city-week moments
    cw = sw.groupby(['state', 'city', 'week'])['price_sell'].agg(mean='mean', sd='std', n='size').reset_index()
    cw['cv'] = cw['sd'] / cw['mean']
    # sync: among stations surveyed in consecutive weeks, share moving with the majority direction
    sw = sw.sort_values(['station_id', 'week'])
    sw['prev_week'] = sw.groupby('station_id')['week'].shift(1)
    sw['prev_price'] = sw.groupby('station_id')['price_sell'].shift(1)
    m = sw[(sw['week'] - sw['prev_week']) == pd.Timedelta(days=7)].copy()
    m['dir'] = np.sign(m['price_sell'] - m['prev_price'])
    mv = m[m['dir'] != 0]
    sy = mv.groupby(['state', 'city', 'week'])['dir'].agg(lambda v: max((v > 0).mean(), (v < 0).mean()) if len(v) >= 4 else np.nan).rename('sync')
    nmov = mv.groupby(['state', 'city', 'week']).size().rename('n_movers')
    cw = cw.merge(sy.reset_index(), how='left').merge(nmov.reset_index(), how='left')
    # residual of the city mean vs the state mean (weighted by station counts)
    stw = cw.groupby(['state', 'week']).apply(lambda g: np.average(g['mean'], weights=g['n']), include_groups=False).rename('state_mean').reset_index()
    cw = cw.merge(stw, on=['state', 'week'])
    cw['resid'] = np.log(cw['mean']) - np.log(cw['state_mean'])
    cw = cw.sort_values(['state', 'city', 'week'])
    out = []
    for (s_, c_), g in cw.groupby(['state', 'city']):
        g = g.set_index('week').asfreq('7D')
        r = g['resid'].interpolate(limit=3, limit_area='inside')     # bridge short survey gaps
        dr = r.diff()
        g['resid_var'] = dr.rolling(ROLL_SHORT, min_periods=ROLL_SHORT // 2).var()
        g['resid_ar1'] = dr.rolling(ROLL_SHORT, min_periods=ROLL_SHORT // 2).apply(lambda v: pd.Series(v).autocorr(1), raw=True)
        g['hurst'] = r.rolling(ROLL_LONG, min_periods=ROLL_LONG).apply(hurst, raw=True)
        g['state'], g['city'] = s_, c_
        out.append(g.reset_index())
    cw = pd.concat(out, ignore_index=True)
    cw.loc[cw['n'] < MIN_N, ['cv', 'sync']] = np.nan
    cw.to_parquet(os.path.join(OUT, 'city_week_indicators.parquet'), index=False)
    inds = ['cv', 'sync', 'resid_ar1', 'resid_var', 'hurst']
    # ---- contrast A: cross-sectional percentile within state, similar size, same week
    rows = []
    for _, lab in cp.iterrows():
        city = lab['city_key']
        tgt = cw[(cw['city'] == city) & (cw['week'] >= lab['start']) & (cw['week'] <= lab['end'])]
        if tgt.empty:
            rows.append(dict(entity=lab['entity'], label=lab['label'], start=str(lab['start'].date()), end=str(lab['end'].date()), weeks=0))
            continue
        state = tgt['state'].iloc[0]
        pct = {k: [] for k in inds}
        for _, row in tgt.iterrows():
            peers = cw[(cw['state'] == state) & (cw['week'] == row['week']) & (cw['city'] != city)
                       & (cw['n'] >= max(MIN_N, row['n'] / 2)) & (cw['n'] <= row['n'] * 2)]
            if len(peers) < 5:
                peers = cw[(cw['week'] == row['week']) & (cw['city'] != city) & (cw['n'] >= max(MIN_N, row['n'] / 2)) & (cw['n'] <= row['n'] * 2)]
            for k in inds:
                v = row[k]
                pv = peers[k].dropna()
                if np.isfinite(v) and len(pv) >= 5:
                    pct[k].append(float((pv < v).mean()))
        rec = dict(entity=lab['entity'], label=lab['label'], start=str(lab['start'].date()), end=str(lab['end'].date()),
                   weeks=int(len(tgt)), n_stations_mean=float(tgt['n'].mean()))
        for k in inds:
            rec[f'pct_{k}'] = float(np.mean(pct[k])) if pct[k] else None
            rec[f'nweeks_{k}'] = len(pct[k])
        rows.append(rec)
    A = pd.DataFrame(rows)
    # ---- contrast B: temporal AUC within city (cartel vs non_cartel windows), where both exist
    B = []
    for ent, g in cp.groupby('entity'):
        if set(g['label']) >= {'cartel', 'non_cartel'}:
            city = g['city_key'].iloc[0]
            cc = cw[cw['city'] == city]
            pos = pd.concat([cc[(cc['week'] >= r.start) & (cc['week'] <= r.end)] for r in g[g.label == 'cartel'].itertuples()])
            neg = pd.concat([cc[(cc['week'] >= r.start) & (cc['week'] <= r.end)] for r in g[g.label == 'non_cartel'].itertuples()])
            rec = dict(entity=ent, cartel_weeks=len(pos), control_weeks=len(neg))
            for k in inds:
                p, n = pos[k].dropna().values, neg[k].dropna().values
                rec[f'auc_{k}'] = float(ews.auc(p, n)) if len(p) >= 10 and len(n) >= 10 else None
            B.append(rec)
    B = pd.DataFrame(B)
    summ = dict(n_city_weeks=int(len(cw)), n_cities=int(cw['city'].nunique()), span=[str(cw['week'].min().date()), str(cw['week'].max().date())],
                cross_sectional=A.to_dict('records'), temporal=B.to_dict('records'))
    json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2, default=str)
    pd.set_option('display.width', 250)
    print(A[['entity', 'label', 'start', 'end', 'weeks', 'n_stations_mean'] + [f'pct_{k}' for k in inds]].round(2).to_string())
    print(B.round(3).to_string() if len(B) else 'no temporal contrasts yet')


if __name__ == '__main__':
    main()
