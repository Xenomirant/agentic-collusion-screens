"""MVP4 on SALT-NLP/agent-collusion: restricted-disclosure audit of real two-agent episodes.

Splits (by run and by model): train = gemini-3.1-flash-lite runs; cal = gemini-3.7-flash
runs; test = the eight non-gemini model families of the cross_model experiment plus the
unconstrained communication runs (authorized controls), capped per condition.

    python experiments/run_salt.py --root <download dir> [--reader perfect|qwen3-8b] [--procs 16]
Writes results/salt/<reader>/{runs/*.jsonl.gz, metrics.csv, paired.csv, extraction.json}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import time

import numpy as np
import pandas as pd

from common import CACHE, RESULTS, cert_costs, run_many
from analysis import BUDGETS, paired, table
from certaudit.belief import train_qmodels
from certaudit.salt import episode_to_traj, extract_salt, load_episodes

SELS = ["random", "graph", "caes", "suspicion", "voi", "comp", "combined"]
CAP = {"coord": 400, "sim_relax": 400, "abandoned": 138, "authorized": 300, "none": 400}


def build_splits(root, seed=0):
    path = CACHE / "splits" / "salt.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    ep = load_episodes(root)
    rng = np.random.default_rng(seed)
    fam = ep.model_family
    pools = {
        "train": ep[(fam == "gemini") & ep.alice_model.str.contains("3.1-flash-lite") & (ep.experiment != "controlled_peer")],
        "cal": ep[(fam == "gemini") & ep.alice_model.str.contains("3.7-flash") & (ep.experiment != "controlled_peer")],
        "test": ep[((ep.experiment == "cross_model") & (fam != "gemini")) | (ep.experiment == "communication")],
    }
    out = {}
    for name, pool in pools.items():
        parts = []
        for cond, cap in CAP.items():
            sub = pool[pool.condition == cond]
            k = min(len(sub), cap if name == "test" else cap // 2)
            parts.append(sub.iloc[rng.permutation(len(sub))[:k]])
        df = pd.concat(parts)
        out[name] = [episode_to_traj(r, {"train": 0, "cal": 1, "test": 2}[name]) for r in df.itertuples()]
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    (RESULTS / "manifests").mkdir(parents=True, exist_ok=True)
    json.dump({n: [dict(tid=t.tid, condition=t.condition, coord=t.labels["coord"], model=t.knobs["model"],
                        n_events=len(t.events), total_cost=t.total_cost) for t in ts] for n, ts in out.items()},
              open(RESULTS / "manifests" / "salt.json", "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--reader", default="perfect")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    splits = build_splits(a.root)
    train, cal, test = splits["train"], splits["cal"], splits["test"]
    for n, ts in splits.items():
        agree = np.mean([t.labels["coord"] == t.labels["dataset_coord"] for t in ts])
        print(f"{n}: {len(ts)} episodes, verifier label == dataset label in {agree:.3f}", flush=True)
    qm_path = CACHE / "models" / "q_salt.pkl"
    if not qm_path.exists():
        train_qmodels(train, n_masks=4, seed=0).save(qm_path)
    out = RESULTS / "salt" / a.reader
    out.mkdir(parents=True, exist_ok=True)
    reader = "perfect"
    if a.reader != "perfect":
        ext = CACHE / "extract" / f"salt_{a.reader}.pkl"
        if not ext.exists():
            t0 = time.time()
            tab, usage = asyncio.run(extract_salt(cal + test, model=a.reader))
            pickle.dump(tab, open(ext, "wb"))
            acc = {}
            for t in cal + test:
                for e in t.events:
                    if e.pub.etype == "msg":
                        d = acc.setdefault(e.fact.kind, [0, 0])
                        d[0] += tab[(t.tid, e.eid)].kind == e.fact.kind
                        d[1] += 1
            rep = dict(model=a.reader, seconds=time.time() - t0, n=len(usage),
                       accuracy={k: v[0] / v[1] for k, v in acc.items()}, counts={k: v[1] for k, v in acc.items()},
                       mean_prompt_tokens=float(np.mean([p for p, _ in usage])),
                       mean_completion_tokens=float(np.mean([c for _, c in usage])))
            json.dump(rep, open(out / "extraction.json", "w"), indent=1)
            print(json.dumps(rep, indent=1), flush=True)
        reader = f"salt_{a.reader}"
    certs = cert_costs(test)
    cal_labels = {t.tid: t.labels["coord"] for t in cal}
    runs, cals = {}, {}
    for s in SELS:
        t0 = time.time()
        runs[s] = run_many(test, s, qm_path, reader, None, out / "runs" / f"test_{s}.jsonl.gz", a.procs)
        cals[s] = dict(runs=run_many(cal, s, qm_path, reader, None, out / "runs" / f"cal_{s}.jsonl.gz", a.procs),
                       labels=cal_labels)
        print(f"[salt/{a.reader}] {s}: {time.time() - t0:.0f}s", flush=True)
    df = table(runs, cals, test, certs)
    df.to_csv(out / "metrics.csv", index=False)
    pd.concat([paired(runs, test, "combined", b) for b in BUDGETS]).to_csv(out / "paired.csv", index=False)
    print(df[["selector", "budget", "auroc", "auroc_lo", "auroc_hi", "tpr@5", "fpr@5", "cert_found", "tokens",
              "records", "supported_on_neg", "overhead_median"]].round(3).to_string())


if __name__ == "__main__":
    main()
