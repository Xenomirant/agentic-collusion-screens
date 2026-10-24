# Real-world datasets for collusion detection — status 2026-10-24

All raw downloads live in `external/data/<name>/raw/` (git-ignored), processed parquet/npy in
`external/data/<name>/processed/`, per-dataset documentation in `external/data/<name>/README.md`, loaders and
fetch scripts in `experiments/data/`. Run loaders with `.venv/bin/python -I experiments/data/<name>.py`.
Every loader exposes a dense panel (`load_panel` -> float32 ndarray entities x time, NaN where missing, plus the
time index and an entity table) and, for labelled sets, `cartel_periods()` -> (entity, start, end, label, source).

| # | dataset | status | entities x time | ground truth |
|---|---|---|---|---|
| 1 | `tankerkoenig` German retail fuel (target: 2022+2023 price changes) | **not obtained** — source gated; open substitute (Mendeley 2022 daily averages, 2.5 GB) only ~100 MB of 48 chunks fetched | expected ~15k stations x 365 days (2022 only) | none |
| 2 | `gti_pp_cartels` GTI/CEU procurement cartels (Zenodo 17595875) | **complete**, md5-verified | 3.86 M contracts; 611 labelled bidders x 69 quarters | **strong**: 73 proven cartels, firm x period labels |
| 3 | `anp_brazil_fuel` ANP weekly station survey + CADE cartel cases | **partial (23 % of 3.72 GB)**: 2004-05..2012-07, 26,440 stations; download resumable | 26,350 stations x 426 weeks (gasoline) | **medium-strong**: 4 cities with month-resolution cartel + control windows from the literature; 17 more CADE convictions with weaker dates |
| 4a | `nsw_fuelcheck` NSW (Australia) station price changes 2016-12..2026-09 | **complete** (107/107 files) | 4,581 stations x 3,591 days (U91) | none (null / scaling panel; known Sydney price cycle) |
| 4b | `qld_fuel` Queensland station price changes 2018-12.. | **failed** — AWS WAF JS challenge (HTTP 202) | — | none |
| 4c | WA FuelWatch (Perth, 2001-) | not downloaded — site is a JS app and the JS bundles could not be fetched through the sandbox proxy (503) | — | tacit-coordination episode documented by Byrne & de Roos (AER 2019), not a cartel finding |
| 4d | LIBOR individual panel submissions | not available without licence (ICE IBA; redistribution prohibited; 3-month embargo). No public mirror found. | — | would be strong (2005-2012 manipulation) |

## 1. Tankerkönig (Germany) — target NOT met
* https://creativecommons.tankerkoenig.de/ now says the history git repo is only for **registered and verified
  users** (onboarding.tankerkoenig.de) and that the history collection is licensed **CC BY-NC-SA 4.0** (the live
  API is CC BY 4.0). `git ls-remote` on the Azure DevOps repo prompts for credentials, the REST `items` endpoint
  redirects to Microsoft sign-in, and `history/history.dump.gz` returns 401. No open mirror of the raw history
  exists on Zenodo / HF / figshare / Dataverse / GitHub (searched). Kaggle copies require an account.
* Fallback: Mendeley Data 10.17632/h8gb4rvb2x.1 "2022 Daily German Retail Fuel Price Data" (Stata, 2.53 GB,
  derived from Tankerkönig, CC BY-NC-SA 4.0): per station-day **daily averages** of E5/E10/diesel + station
  master data, **2022 only**. Header read from the first chunk; columns documented in the README. The sandbox
  proxy throttled to ~0.1-0.4 MB/s and refused >10 concurrent connections, so the chunked download
  (`experiments/data/fetch_mendeley_h8gb4rvb2x.sh`, resumable) was stopped at ~100 MB to prioritise ANP.
* **Stations / span actually on disk: none usable (0 stations, 0 days).** Two-calendar-year, all-station,
  price-change-level target: not met and not achievable without a Tankerkönig account. Loader
  `experiments/data/tankerkoenig.py` is written for the Mendeley file and untested.

## 2. GTI public-procurement cartels — complete
* Zenodo 17595875 (Fazekas, Tóth, Wachs, Abdou; IJIO 2026), CC BY-NC 3.0, 51.6 MB Rdata + R code + README, md5 OK.
* One table `df_full`: 3,861,477 awarded contracts, 7 countries (BG, ES, FR, HU, LV, PT, SE), 2004-2021.
  Fields: anonymised `buyer_id`, `bidder_id`, anonymised CPV markets, first-call and award dates, `bid_price`,
  `lot/tender_estimatedprice`, `tender_finalprice`, `lot_bidscount`, `singleb`, consortium/subcontracting flags,
  `cartel_id` (73 ids) and `is_cartel` (**1 = won by a convicted cartel member inside the documented collusion
  window, 0 = won by the same firm after the window, NaN = firm never in a proven cartel**).
  Labelled rows: 18,807 (8,045 cartel / 10,762 post-cartel), 698 labelled bidder ids, label dates 2005-01..2018-11.
* Loader `experiments/data/gti_pp_cartels.py`: `load_contracts()` (parquet cache, 75 MB), `cartel_periods()`
  (inferred per cartel and per bidder from award dates — the release has no case-date table), `load_panel()`
  (labelled bidders x quarters, mean relative winning price bid/estimate; 611 x 69, 5 % populated).
* Strength: firm-period labels from competition-authority decisions (strong), but contract-level labels are noisy
  (not every contract of a cartel member in the window was rigged; windows are year-resolution in ES/PT/HU).
  No price *time series* per se — relative prices are sparse; the natural use is bidding-pattern screens.

## 3. ANP Brazil + CADE cartels — partial
* 45 semi-annual files (3.72 GB) listed on the ANP open-data page; 3 complete + 15 partial on disk (0.85 GB,
  2004-05..2012-07). gov.br was reachable (GET/range OK, HEAD 403) but the proxy was slow; `fetch_anp.sh`
  resumes with `curl -C -`. Rebuild with `anp_brazil_fuel.py --refresh` after it finishes.
* Partial build: 4,564,856 station-product-date rows, 26,440 stations (CNPJ), products GASOLINA/ETANOL/DIESEL/
  GNV; weekly panels stations x 426 weeks (13-14 % filled — the survey rotates stations in big cities);
  municipality-week moments (mean, sd, cv, n) for gasoline.
* Ground truth (`labels/cade_retail_fuel_cartels.csv`, 25 rows / 21 cities): Belo Horizonte 2004-01..2008-04,
  Brasília 2009-11..2015-11, Caxias do Sul 2004-01..2007-07, São Luís 2007-01..2011-03 with matching non-cartel
  control windows (Silveira et al. 2022 Energy Economics / 2023 CADE paper, Table 2) — the labels used by the
  published ML cartel screens on exactly this survey; plus Joinville 2013-06..11, Francisco Beltrão 2016-17,
  Londrina 2007 (month/year resolution from CADE press) and 14 further CADE convictions with only the
  proceeding year (weak). Cities are matched to the survey by accent-stripped name (`city_key`).
  In the partial build the four strong cities already have 159-167 weeks of gasoline data each
  (BH ~117 stations/week, Brasília ~106, São Luís ~57, Caxias ~30).

## 4. Other
* NSW FuelCheck (CC BY-SA): 5,662,446 price-change events, 5,415 station identities (~2.4-2.8k active per year),
  Dec 2016..Sep 2026, fuels E10/U91/P95/P98/DL/PDL/LPG; daily carried-forward panels U91 4,581 x 3,591
  (24 % filled), E10 3,550 x 3,591, DL 3,214 x 3,591. Good large null panel with a known price cycle; no cartel.
* Queensland: metadata and 93 URLs recorded; downloads blocked by AWS WAF challenge (needs a browser).
* WA FuelWatch: daily per-station prices since 2001 (CC BY 3.0 AU per data.gov.au); the download page is an
  Angular app whose JS the proxy refused (503 Forwarding failure); not obtained. Would pair with the Perth
  coordination literature (Byrne & de Roos 2019) if fetched from an unproxied machine.
* LIBOR: ICE IBA publishes historical individual submissions only under licence with a no-redistribution clause;
  pre-2014 BBA/Thomson Reuters submissions have no open mirror that I could find.

## What failed and why (summary)
1. Tankerkönig history: access policy changed (registration + CC BY-NC-SA); no anonymous route.
2. Sandbox network: Privoxy-based proxy, ~0.1-0.4 MB/s aggregate, 503 on >~10 concurrent connections, and the
   session restart killed the first round of downloads (truncated ANP files, lost scratch PDFs). All fetch
   scripts are idempotent/resumable (gov.br resume is best-effort: some transfers restart from zero); total
   remaining to fetch: ~2.9 GB ANP + ~2.4 GB Mendeley. Background fetchers stop when the agent session ends.
3. QLD (WAF challenge) and WA FuelWatch (JS app behind proxy) need a browser.

## Resume commands
```
bash experiments/data/fetch_anp.sh && .venv/bin/python -I experiments/data/anp_brazil_fuel.py --refresh
PAR=8 bash experiments/data/fetch_mendeley_h8gb4rvb2x.sh && .venv/bin/python -I experiments/data/tankerkoenig.py
```
