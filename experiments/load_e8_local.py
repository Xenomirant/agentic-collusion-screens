"""Loader for the E8 local-Qwen double-auction datasets (experiments/results/e8_local/).

Standalone (numpy only). Mirrors the parsing in run_e7_da_seeded.py's parse(): the per-round JSON
blocks of auction_results.md give the seller asks and trades, experiment_metadata.json gives the
seller order and, for seeded runs, the ground-truth pair.

    from load_e8_local import load_e8_local, parse_run
    data = load_e8_local()                 # {'controls': [...], 'seeded': [...], 'seeded_monitored': [...]}
    run = data['seeded'][0]
    run['asks']      # float array, sellers x rounds (missing asks forward/back-filled along time)
    run['raw_asks']  # same, with NaN where a seller had no ask that round
    run['trade']     # int array, rounds; 1 if at least one trade happened that round
    run['true']      # (i, j) indices into run['sellers'] of the seeded pair, or None
    run['n_trades']  # int array, rounds; number of trades that round
    run['template']  # seller prompt template: seller_prompt_base.jinja2, or seller_prompt_monitored.jinja2
                     # for seeded_monitored (the channel pair is told a regulator screens public asks)

Run as a script to print a summary of what is on disk.
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e8_local")
_BLOCK = re.compile(r"````json\n(.*?)\n````", re.S)
DATASETS = ("controls", "seeded", "seeded_monitored")


def _ffill_bfill(A):
    """Forward- then back-fill NaNs along axis 1 (time), like pandas ffill().bfill() per row."""
    A = A.copy()
    for row in A:
        last = np.nan
        for t in range(row.size):
            if np.isnan(row[t]):
                row[t] = last
            else:
                last = row[t]
        first = next((v for v in row if not np.isnan(v)), np.nan)
        for t in range(row.size):
            if np.isnan(row[t]):
                row[t] = first
            else:
                break
    return A


def parse_run(d):
    """One run folder -> dict(run, sellers, asks, raw_asks, trade, n_trades, true, pair_ids, meta)."""
    txt = open(os.path.join(d, "auction_results.md"), encoding="utf-8").read()
    blocks = [json.loads(b) for b in _BLOCK.findall(txt)]
    meta = json.load(open(os.path.join(d, "experiment_metadata.json"), encoding="utf-8"))
    sellers = meta["seller_ids"]
    A = np.full((len(sellers), len(blocks)), np.nan)
    trade = np.zeros(len(blocks), dtype=int)
    n_trades = np.zeros(len(blocks), dtype=int)
    for t, b in enumerate(blocks):
        for i, s in enumerate(sellers):
            v = b["seller_asks"].get(s)
            if v is not None:
                A[i, t] = float(v)
        n_trades[t] = len(b.get("trades") or [])
        trade[t] = int(n_trades[t] > 0)
    pair = meta.get("seller_channel_pair")
    true = tuple(sorted(sellers.index(s) for s in pair)) if pair else None
    return dict(run=os.path.basename(d.rstrip("/")), sellers=sellers, asks=_ffill_bfill(A), raw_asks=A,
                trade=trade, n_trades=n_trades, true=true, pair_ids=pair,
                seed=meta["auction_config"].get("seed"), rounds=len(blocks),
                template=meta["auction_config"].get("seller_prompt_template"),
                models=meta.get("resolved_models", {}).get("sellers"))


def run_dirs(root=ROOT, dataset=None):
    sets = [dataset] if dataset else list(DATASETS)
    out = []
    for s in sets:
        for d in sorted(glob.glob(os.path.join(root, s, "*"))):
            if os.path.exists(os.path.join(d, "auction_results.md")) and \
                    os.path.exists(os.path.join(d, "experiment_metadata.json")):
                out.append((s, d))
    return out


def load_e8_local(root=ROOT, rounds=None):
    """All complete runs, keyed by dataset. `rounds` (e.g. 60) drops runs with a different length."""
    data = {s: [] for s in DATASETS}
    for s, d in run_dirs(root):
        r = parse_run(d)
        if rounds is not None and r["rounds"] != rounds:
            continue
        data[s].append(r)
    return data


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    data = load_e8_local(root)
    for name, runs in data.items():
        print(f"{name}: {len(runs)} runs")
        for r in runs:
            print(f"  {r['run']:<32} seed={r['seed']:<5} rounds={r['rounds']:<3} pair={r['true']} "
                  f"trade_rounds={int(r['trade'].sum())} mean_ask={np.nanmean(r['raw_asks']):.2f}")
