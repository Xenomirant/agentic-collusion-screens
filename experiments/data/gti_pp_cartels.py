"""Loader for the GTI/CEU "Public procurement cartels" replication dataset
(Fazekas, Tóth, Wachs, Abdou; Zenodo record 17595875, CC BY-NC 3.0).

Raw file: external/data/gti_pp_cartels/raw/GTI_PP_cartels_contracts_data_NOV2023.Rdata
(one R data.frame `df_full`, 3,861,477 awarded contracts, 22 columns; see README.md there).

Functions
---------
load_contracts(refresh=False) -> pandas.DataFrame
    Contract-level table (cached as parquet in external/data/gti_pp_cartels/processed/).
cartel_periods() -> pandas.DataFrame
    One row per (cartel_id, bidder_id) with start/end of the labelled cartel window, plus one
    row per cartel_id. Columns: entity, entity_type, cartel_id, country, start, end, label,
    n_contracts, source.  The windows are *inferred* from the labelled contracts (award dates
    with is_cartel == 1), because the release does not ship the case-level dates.
load_panel(freq="QS", value="rel_price", min_contracts=1)
    Dense panel for the labelled bidders only (entities x time, float32, NaN where no contract
    in that period) together with the time index and an entity table carrying the cartel label
    per period (label matrix of the same shape: 1 during cartel, 0 after cartel, NaN unlabelled).

Run as a script to build the parquet cache and print summary numbers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "external/data/gti_pp_cartels/raw/GTI_PP_cartels_contracts_data_NOV2023.Rdata"
PROC = ROOT / "external/data/gti_pp_cartels/processed"
PARQ = PROC / "contracts.parquet"

DATE_COLS = ["tender_publications_firstcallfortenderdate",
             "tender_publications_firstdcontractawarddate"]


def load_contracts(refresh: bool = False) -> pd.DataFrame:
    if PARQ.exists() and not refresh:
        return pd.read_parquet(PARQ)
    import pyreadr  # only needed to build the cache
    df = pyreadr.read_r(str(RAW))["df_full"]
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    # logical columns arrive as object with NaN -> nullable boolean
    for c in ["is_cartel", "bid_iswinning", "bid_isconsortium", "bid_issubcontracted"]:
        df[c] = df[c].map({True: 1.0, False: 0.0}).astype("float32")
    df["tender_year"] = df["tender_year"].astype("float32")
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQ, index=False)
    return df


def _award_date(df: pd.DataFrame) -> pd.Series:
    d = df["tender_publications_firstdcontractawarddate"]
    d = d.fillna(df["tender_publications_firstcallfortenderdate"])
    # fall back to mid-year of tender_year when both dates are missing
    fb = pd.to_datetime(df["tender_year"].astype("Int64").astype(str) + "-07-01", errors="coerce")
    return d.fillna(fb)


def cartel_periods(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_contracts() if df is None else df
    lab = df[df["is_cartel"].notna()].copy()
    lab["date"] = _award_date(lab)
    rows = []
    src = "inferred from labelled contracts, Zenodo 17595875 (is_cartel==1 == during cartel period)"
    for (cid, country), g in lab.groupby(["cartel_id", "country"]):
        on = g[g["is_cartel"] == 1]
        off = g[g["is_cartel"] == 0]
        rows.append(dict(entity=cid, entity_type="cartel", cartel_id=cid, country=country,
                         start=on["date"].min(), end=on["date"].max(), label="cartel",
                         n_contracts=len(on), n_bidders=on["bidder_id"].nunique(), source=src))
        if len(off):
            rows.append(dict(entity=cid, entity_type="cartel", cartel_id=cid, country=country,
                             start=off["date"].min(), end=off["date"].max(), label="post_cartel",
                             n_contracts=len(off), n_bidders=off["bidder_id"].nunique(), source=src))
        for bid, gb in on.groupby("bidder_id"):
            rows.append(dict(entity=bid, entity_type="bidder", cartel_id=cid, country=country,
                             start=gb["date"].min(), end=gb["date"].max(), label="cartel",
                             n_contracts=len(gb), n_bidders=1, source=src))
    return pd.DataFrame(rows)


def load_panel(freq: str = "QS", value: str = "rel_price", min_contracts: int = 1):
    """Entities = labelled bidders (bidder_id, country). value in {rel_price, bid_price, n_bids,
    singleb, n_contracts}. Returns (panel[float32, E x T], time_index, entities, labels[E x T])."""
    df = load_contracts()
    lab = df[df["is_cartel"].notna()].copy()
    lab["date"] = _award_date(lab)
    lab["period"] = lab["date"].dt.to_period(freq[0] if freq[0] in "QMY" else "Q").dt.start_time
    est = lab["lot_estimatedprice"].fillna(lab["tender_estimatedprice"])
    lab["rel_price"] = (lab["bid_price"] / est).where(est > 0)
    lab.loc[(lab["rel_price"] > 5) | (lab["rel_price"] < 0.05), "rel_price"] = np.nan
    lab["n_bids"] = lab["lot_bidscount"].clip(upper=50)
    lab["n_contracts"] = 1.0
    lab["entity"] = lab["country"] + ":" + lab["bidder_id"]
    agg = "sum" if value == "n_contracts" else "mean"
    piv = lab.pivot_table(index="entity", columns="period", values=value, aggfunc=agg)
    cnt = lab.pivot_table(index="entity", columns="period", values="n_contracts", aggfunc="sum")
    piv = piv.where(cnt >= min_contracts)
    labp = lab.pivot_table(index="entity", columns="period", values="is_cartel", aggfunc="mean")
    full = pd.date_range(piv.columns.min(), piv.columns.max(), freq=freq)
    piv = piv.reindex(columns=full); labp = labp.reindex(index=piv.index, columns=full)
    ent = (lab.groupby("entity").agg(country=("country", "first"), bidder_id=("bidder_id", "first"),
                                      cartel_ids=("cartel_id", lambda s: ";".join(sorted(set(s)))),
                                      n_contracts=("n_contracts", "sum"))
           .reindex(piv.index).reset_index())
    return piv.to_numpy(dtype=np.float32), piv.columns, ent, labp.to_numpy(dtype=np.float32)


if __name__ == "__main__":
    df = load_contracts(refresh="--refresh" in sys.argv)
    print("contracts:", df.shape, "parquet:", PARQ, f"{PARQ.stat().st_size/2**20:.1f} MB")
    print("labelled:", int(df["is_cartel"].notna().sum()), "cartel:", int((df["is_cartel"] == 1).sum()),
          "post-cartel:", int((df["is_cartel"] == 0).sum()), "cartels:", df["cartel_id"].nunique())
    cp = cartel_periods(df)
    PROC.mkdir(parents=True, exist_ok=True)
    cp.to_csv(PROC / "cartel_periods.csv", index=False)
    cc = cp[cp.entity_type == "cartel"]
    print(cc[cc.label == "cartel"].groupby("country").agg(n_cartels=("cartel_id", "nunique"),
          first=("start", "min"), last=("end", "max"), contracts=("n_contracts", "sum")))
    print("labelled bidders:", cp[cp.entity_type == "bidder"]["entity"].nunique())
    P, t, ent, L = load_panel()
    print("panel:", P.shape, "time", t[0].date(), "->", t[-1].date(), "non-NaN frac", np.isfinite(P).mean().round(3))
    np.save(PROC / "panel_bidder_quarter_relprice.npy", P)
    np.save(PROC / "panel_bidder_quarter_label.npy", L)
    pd.DataFrame({"period": t}).to_csv(PROC / "panel_time_index.csv", index=False)
    ent.to_csv(PROC / "panel_entities.csv", index=False)
