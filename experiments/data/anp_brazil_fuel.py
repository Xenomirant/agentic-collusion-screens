"""Loader for the ANP (Brazil) weekly station-level fuel price survey
"Série Histórica de Preços de Combustíveis" (Levantamento de Preços de Combustíveis, LPC),
semi-annual files external/data/anp_brazil_fuel/raw/ca-YYYY-0S.csv (2004-2021) and .zip (2022+).

Raw schema (';'-separated, decimal ',', UTF-8-BOM or latin-1, dd/mm/yyyy):
Regiao - Sigla; Estado - Sigla; Municipio; Revenda; CNPJ da Revenda; Nome da Rua; Numero Rua;
Complemento; Bairro; Cep; Produto; Data da Coleta; Valor de Venda; Valor de Compra;
Unidade de Medida; Bandeira

build(refresh=False) -> prices.parquet (station_id, date, product, price_sell, price_buy)
                        + stations.parquet (station_id, cnpj, name, brand, city, state, region, cep)
load_panel(product="GASOLINA", freq="W-SAT", cities=None, states=None)
    -> (panel float32 [stations x weeks], time_index, stations) ; NaN where a station was not surveyed
cartel_periods() -> table (entity, state, start, end, label, decision_year, case_number, source, notes)
    from external/data/anp_brazil_fuel/labels/cade_retail_fuel_cartels.csv
city_panel(product, freq) -> mean/sd/n per (city,state) x week — the level the screen literature uses
"""
from __future__ import annotations

import glob
import io
import sys
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "external/data/anp_brazil_fuel/raw"
PROC = ROOT / "external/data/anp_brazil_fuel/processed"
LABELS = ROOT / "external/data/anp_brazil_fuel/labels/cade_retail_fuel_cartels.csv"

RENAME = {"Regiao - Sigla": "region", "Estado - Sigla": "state", "Municipio": "city", "Revenda": "name",
          "CNPJ da Revenda": "cnpj", "Nome da Rua": "street", "Numero Rua": "number", "Complemento": "compl",
          "Bairro": "district", "Cep": "cep", "Produto": "product", "Data da Coleta": "date",
          "Valor de Venda": "price_sell", "Valor de Compra": "price_buy", "Unidade de Medida": "unit",
          "Bandeira": "brand"}


def _parse_csv_bytes(b: bytes, src: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            txt = b.decode(enc); break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(txt), sep=";", dtype=str, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME)
    keep = [c for c in RENAME.values() if c in df.columns]
    df = df[keep].copy()
    for c in ("price_sell", "price_buy"):
        if c in df:
            df[c] = pd.to_numeric(df[c].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                                  errors="coerce").astype(np.float32)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date", "price_sell", "cnpj", "product"])
    for c in ("cnpj", "product", "city", "state", "region", "brand", "name", "cep"):
        if c in df:
            df[c] = df[c].astype(str).str.strip().str.upper()
    df["cnpj"] = df["cnpj"].str.replace(r"\D", "", regex=True)
    df["source_file"] = src
    return df


def _read_one(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".zip":
        parts = []
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    parts.append(_parse_csv_bytes(z.read(n), f"{p.name}:{n}"))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _parse_csv_bytes(p.read_bytes(), p.name)


def build(refresh: bool = False):
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "prices.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out), pd.read_parquet(PROC / "stations.parquet")
    files = sorted(glob.glob(str(RAW / "ca-*.csv")) + glob.glob(str(RAW / "ca-*.zip")))
    with ProcessPoolExecutor(max_workers=min(48, len(files))) as ex:
        parts = [d for d in ex.map(_read_one, files) if len(d)]
    df = pd.concat(parts, ignore_index=True)
    # station identity = CNPJ (14 digits); keep most frequent descriptive attributes
    st = (df.groupby("cnpj").agg(name=("name", "first"), brand=("brand", lambda s: s.mode().iat[0]),
                                 city=("city", "first"), state=("state", "first"), region=("region", "first"),
                                 cep=("cep", "first"), first_date=("date", "min"), last_date=("date", "max"),
                                 n_obs=("price_sell", "size")).reset_index())
    st.insert(0, "station_id", np.arange(len(st), dtype=np.int32))
    df["station_id"] = df["cnpj"].map(dict(zip(st["cnpj"], st["station_id"]))).astype(np.int32)
    tidy = (df[["station_id", "date", "product", "price_sell", "price_buy", "source_file"]]
            .drop_duplicates(["station_id", "date", "product"]).sort_values(["station_id", "product", "date"]))
    tidy["product"] = tidy["product"].astype("category")
    tidy.to_parquet(out, index=False)
    st.to_parquet(PROC / "stations.parquet", index=False)
    return tidy, st


def norm_city(s: str) -> str:
    """ANP spells municipalities in upper-case ASCII (BRASILIA, SAO LUIS); labels use accents."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.split(" (")[0].strip().upper()


def cartel_periods() -> pd.DataFrame:
    cp = pd.read_csv(LABELS, parse_dates=["start", "end"])
    cp["city_key"] = cp["entity"].map(norm_city)
    return cp


def load_panel(product: str = "GASOLINA", freq: str = "W-SAT", cities=None, states=None):
    tidy, st = build()
    d = tidy[tidy["product"] == product]
    if cities is not None or states is not None:
        m = pd.Series(True, index=st.index)
        if cities is not None:
            cities = [c.upper() for c in cities]; m &= st["city"].isin(cities)
        if states is not None:
            m &= st["state"].isin(states)
        d = d[d["station_id"].isin(st.loc[m, "station_id"])]
    d = d.assign(period=d["date"].dt.to_period(freq).dt.start_time)
    piv = d.groupby(["station_id", "period"])["price_sell"].mean().unstack("period")
    full = pd.date_range(piv.columns.min(), piv.columns.max(), freq=freq if freq.startswith("W") else freq)
    # period start_time for weekly periods is Sunday for W-SAT; align by re-deriving the grid
    full = pd.Series(full).dt.to_period(freq).dt.start_time.drop_duplicates()
    piv = piv.reindex(columns=full)
    stations = st.set_index("station_id").loc[piv.index].reset_index()
    return piv.to_numpy(dtype=np.float32), piv.columns, stations


def city_panel(product: str = "GASOLINA", freq: str = "W-SAT") -> pd.DataFrame:
    tidy, st = build()
    d = tidy[tidy["product"] == product].merge(st[["station_id", "city", "state"]], on="station_id")
    d["period"] = d["date"].dt.to_period(freq).dt.start_time
    g = d.groupby(["state", "city", "period"])["price_sell"].agg(mean="mean", sd="std", n="size").reset_index()
    g["cv"] = g["sd"] / g["mean"]
    return g


if __name__ == "__main__":
    tidy, st = build(refresh="--refresh" in sys.argv)
    print("rows:", len(tidy), "stations (CNPJ):", len(st), "span:", tidy["date"].min().date(), "->", tidy["date"].max().date())
    print("products:", tidy["product"].value_counts().to_dict())
    print("files:", tidy["source_file"].str.split(":").str[0].nunique())
    cp = cartel_periods(); print("cartel label rows:", len(cp), "cities:", cp["entity"].nunique())
    for prod in ["GASOLINA", "ETANOL", "DIESEL"]:
        P, t, s = load_panel(prod)
        np.save(PROC / f"panel_weekly_{prod}.npy", P)
        s.to_csv(PROC / f"panel_stations_{prod}.csv", index=False)
        pd.DataFrame({"week_start": t}).to_csv(PROC / f"panel_time_index_{prod}.csv", index=False)
        print(prod, "panel", P.shape, "filled frac", round(float(np.isfinite(P).mean()), 4), t[0].date(), "->", t[-1].date())
    cities = cp["city_key"].unique()
    cityp = city_panel("GASOLINA")
    cityp.to_parquet(PROC / "city_weekly_GASOLINA.parquet", index=False)
    sub = cityp[cityp["city"].isin(cities)]
    print("city-week rows (gasoline):", len(cityp), "; labelled cities present:", sorted(sub["city"].unique()))
    print("sizes MB:", {p.name: round(p.stat().st_size / 2**20, 1) for p in PROC.iterdir()})
