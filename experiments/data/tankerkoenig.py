"""Loader for German retail fuel prices (Tankerkönig / MTS-K data) — 2022 daily station averages.

The Tankerkönig history repository itself (dev.azure.com/tankerkoenig/tankerkoenig-data, per-day CSVs of
every price change since 2014, and the PostgreSQL dumps at creativecommons.tankerkoenig.de/history/) is
no longer anonymously accessible (sign-in / onboarding required; see README.md). What we have is the
Mendeley Data release "2022 Daily German Retail Fuel Price Data" (10.17632/h8gb4rvb2x.1, file
2022-FY-combined.dta, Stata), which is derived from the Tankerkönig history and contains, for every
station and every day of 2022, the daily *average* price of E5, E10 and diesel and the number of price
entries that day, merged with the station master data.

Columns: date, uuid, daily_e10_avg, daily_e5_avg, daily_diesel_avg, numer_daily_entry, name, brand,
street, house_number, post_code, city, latitude, longitude, first_active, openingtimes_json, _merge

build(refresh=False) -> prices.parquet (station_uuid, date, diesel, e5, e10, n_entries)
                        + stations.parquet (uuid, name, brand, street, house_number, post_code, city, lat, lon, first_active)
load_panel(fuel="e5") -> (panel float32 [stations x days], time_index (DatetimeIndex), stations DataFrame)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "external/data/tankerkoenig/raw/2022-FY-combined.dta"
PROC = ROOT / "external/data/tankerkoenig/processed"
FUELS = {"diesel": "daily_diesel_avg", "e5": "daily_e5_avg", "e10": "daily_e10_avg"}


def build(refresh: bool = False):
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "prices.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out), pd.read_parquet(PROC / "stations.parquet")
    price_parts, station_parts = [], []
    reader = pd.read_stata(RAW, chunksize=2_000_000, convert_categoricals=False)
    for ch in reader:
        ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
        p = ch[["uuid", "date", "daily_diesel_avg", "daily_e5_avg", "daily_e10_avg", "numer_daily_entry"]].rename(
            columns={"uuid": "station_uuid", "daily_diesel_avg": "diesel", "daily_e5_avg": "e5",
                     "daily_e10_avg": "e10", "numer_daily_entry": "n_entries"})
        for c in ("diesel", "e5", "e10"):
            p[c] = pd.to_numeric(p[c], errors="coerce").astype(np.float32)
            p.loc[(p[c] <= 0.3) | (p[c] > 5), c] = np.nan  # 0 / implausible placeholders
        price_parts.append(p)
        station_parts.append(ch[["uuid", "name", "brand", "street", "house_number", "post_code", "city",
                                 "latitude", "longitude", "first_active"]].drop_duplicates("uuid"))
    prices = pd.concat(price_parts, ignore_index=True).dropna(subset=["date"])
    stations = (pd.concat(station_parts, ignore_index=True).drop_duplicates("uuid")
                .rename(columns={"latitude": "lat", "longitude": "lon"}).reset_index(drop=True))
    prices.to_parquet(out, index=False)
    stations.to_parquet(PROC / "stations.parquet", index=False)
    return prices, stations


def load_panel(fuel: str = "e5"):
    prices, stations = build()
    piv = prices.pivot_table(index="station_uuid", columns="date", values=fuel, aggfunc="mean")
    full = pd.date_range(piv.columns.min(), piv.columns.max(), freq="D")
    piv = piv.reindex(columns=full)
    st = stations.set_index("uuid").reindex(piv.index).reset_index().rename(columns={"index": "uuid"})
    return piv.to_numpy(dtype=np.float32), piv.columns, st


if __name__ == "__main__":
    prices, st = build(refresh="--refresh" in sys.argv)
    print("rows:", len(prices), "stations:", prices["station_uuid"].nunique(), "span:",
          prices["date"].min().date(), "->", prices["date"].max().date(), "days:", prices["date"].nunique())
    print("brands (top):", st["brand"].value_counts().head(8).to_dict())
    for f in FUELS:
        P, t, s = load_panel(f)
        np.save(PROC / f"panel_daily_{f}.npy", P)
        print(f, "panel", P.shape, "filled frac", round(float(np.isfinite(P).mean()), 3))
    pd.DataFrame({"date": t}).to_csv(PROC / "panel_time_index.csv", index=False)
    s.to_csv(PROC / "panel_stations.csv", index=False)
    print("sizes MB:", {p.name: round(p.stat().st_size / 2**20, 1) for p in PROC.iterdir()})
