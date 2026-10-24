"""Loader for NSW FuelCheck monthly price-history files (data.nsw.gov.au dataset "fuel-check",
CC BY-SA). Raw: external/data/nsw_fuelcheck/raw/*.xlsx|*.csv, one file per month since Dec 2016.
Each row is a *price change event*: ServiceStationName, Address, Suburb, Postcode, Brand,
FuelCode, PriceUpdatedDate, Price (AUD cents per litre).

build(refresh=False)            -> tidy parquet external/data/nsw_fuelcheck/processed/prices.parquet
                                   + stations.parquet (station_id, name, address, suburb, postcode, brand)
load_panel(fuel="U91", freq="D", ffill=True, max_gap_days=14)
    -> (panel float32 [stations x days], time_index, stations DataFrame)
    panel[i,t] = last price posted by station i on day t; with ffill=True the last posted price
    is carried forward up to max_gap_days (a station's price stays valid until the next change).
"""
from __future__ import annotations

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "external/data/nsw_fuelcheck/raw"
PROC = ROOT / "external/data/nsw_fuelcheck/processed"
COLS = ["ServiceStationName", "Address", "Suburb", "Postcode", "Brand", "FuelCode", "PriceUpdatedDate", "Price"]


def _read_one(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        raw = pd.read_csv(path, header=None, dtype=str, encoding_errors="replace", on_bad_lines="skip")
    else:
        raw = pd.read_excel(path, header=None, dtype=str)
    # locate the header row (some files carry a 2-3 line report banner)
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("ServiceStationName")]
    if len(hdr) == 0:
        raise ValueError(f"no header row in {path}")
    df = raw.iloc[hdr[0] + 1:, :len(COLS)].copy()
    df.columns = COLS
    df = df.dropna(subset=["ServiceStationName", "FuelCode", "Price", "PriceUpdatedDate"])
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    raw_ts = df["PriceUpdatedDate"].astype(str).str.strip()
    # two formats occur: ISO "YYYY-MM-DD HH:MM:SS" (xlsx, most csv) and "D/MM/YYYY H:MM" (some csv, day first)
    ts = pd.to_datetime(raw_ts, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    m = ts.isna()
    if m.any():
        ts[m] = pd.to_datetime(raw_ts[m], errors="coerce", format="%d/%m/%Y %H:%M")
    m = ts.isna()
    if m.any():
        ts[m] = pd.to_datetime(raw_ts[m], errors="coerce", dayfirst=True)
    df["timestamp"] = ts
    df = df.dropna(subset=["Price", "timestamp"])
    df["source_file"] = os.path.basename(path)
    for c in ["ServiceStationName", "Address", "Suburb", "Brand", "FuelCode"]:
        df[c] = df[c].astype(str).str.strip()
    df["Postcode"] = df["Postcode"].astype(str).str.extract(r"(\d{4})")[0]
    return df.drop(columns=["PriceUpdatedDate"])


def build(refresh: bool = False):
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "prices.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out), pd.read_parquet(PROC / "stations.parquet")
    files = sorted(glob.glob(str(RAW / "*.xlsx")) + glob.glob(str(RAW / "*.csv")))
    with ProcessPoolExecutor(max_workers=min(32, len(files))) as ex:
        parts = list(ex.map(_read_one, files))
    df = pd.concat(parts, ignore_index=True)
    key = (df["ServiceStationName"].str.upper() + "|" + df["Address"].str.upper())
    stations = (df.assign(key=key).groupby("key")
                .agg(name=("ServiceStationName", "first"), address=("Address", "first"),
                     suburb=("Suburb", "first"), postcode=("Postcode", "first"),
                     brand=("Brand", lambda s: s.mode().iat[0] if len(s) else None),
                     first_seen=("timestamp", "min"), last_seen=("timestamp", "max"), n_events=("Price", "size"))
                .reset_index())
    stations.insert(0, "station_id", np.arange(len(stations), dtype=np.int32))
    df["station_id"] = key.map(dict(zip(stations["key"], stations["station_id"]))).astype(np.int32)
    tidy = df[["station_id", "timestamp", "FuelCode", "Price", "Brand", "source_file"]].rename(
        columns={"FuelCode": "fuel", "Price": "price", "Brand": "brand"})
    tidy = tidy.drop_duplicates(["station_id", "timestamp", "fuel"]).sort_values(["station_id", "fuel", "timestamp"])
    tidy["price"] = tidy["price"].astype(np.float32)
    tidy.to_parquet(out, index=False)
    stations.drop(columns=["key"]).to_parquet(PROC / "stations.parquet", index=False)
    return tidy, stations


def load_panel(fuel: str = "U91", freq: str = "D", ffill: bool = True, max_gap_days: int = 14):
    tidy, stations = build()
    d = tidy[tidy["fuel"] == fuel].copy()
    d["period"] = d["timestamp"].dt.floor(freq)
    last = d.sort_values("timestamp").groupby(["station_id", "period"])["price"].last()
    piv = last.unstack("period")
    full = pd.date_range(piv.columns.min(), piv.columns.max(), freq=freq)
    piv = piv.reindex(columns=full)
    if ffill:
        piv = piv.T.ffill(limit=max_gap_days).T
    st = stations.set_index("station_id").loc[piv.index].reset_index()
    return piv.to_numpy(dtype=np.float32), piv.columns, st


if __name__ == "__main__":
    tidy, st = build(refresh="--refresh" in sys.argv)
    print("events:", len(tidy), "stations:", len(st), "fuels:", tidy["fuel"].value_counts().to_dict())
    print("span:", tidy["timestamp"].min(), "->", tidy["timestamp"].max())
    print("months with data:", tidy["timestamp"].dt.to_period("M").nunique(), "files:", tidy["source_file"].nunique())
    for fuel in ["U91", "E10", "DL"]:
        P, t, s = load_panel(fuel)
        np.save(PROC / f"panel_daily_{fuel}.npy", P)
        print(fuel, "panel", P.shape, "filled frac", round(float(np.isfinite(P).mean()), 3), t[0].date(), "->", t[-1].date())
    pd.DataFrame({"date": t}).to_csv(PROC / "panel_time_index.csv", index=False)
    s.to_csv(PROC / "panel_stations.csv", index=False)
    print("sizes MB:", {p.name: round(p.stat().st_size / 2**20, 1) for p in PROC.iterdir()})
