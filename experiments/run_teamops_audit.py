"""Restricted-disclosure audit of TeamOps episodes (local LLM agents, 4 agents, actions on objects).

Held-out model: priors are trained on the judged episodes of --train-model and
tested on those of --test-model (a 25% seed-disjoint slice of the training model's
episodes serves as the calibration set).  Readers: the judge facts ("judge", the
offline ground truth) and a non-thinking extraction reader (--reader <model>).

    python experiments/run_teamops_audit.py --train-model qwen2.5-7b --test-model qwen3-8b [--reader qwen3-8b]
Writes results/teamops/audit_<train>_to_<test>/<reader>/{runs, metrics.csv, paired.csv, extraction.json}.
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
from certaudit import llm
from certaudit.belief import train_qmodels
from certaudit.verifier import labels

SELS = ["random", "random_groups", "graph", "caes", "suspicion", "voi", "comp", "combined"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-model", default="qwen2.5-7b")
    ap.add_argument("--test-model", default="qwen3-8b")
    ap.add_argument("--reader", default="judge")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--tag", default="L")
    ap.add_argument("--train-agents", type=int, nargs="*", default=None, help="restrict training/cal episodes to these team sizes")
    ap.add_argument("--test-agents", type=int, nargs="*", default=None, help="restrict test episodes to these team sizes")
    a = ap.parse_args()
    tr_all = list(pickle.load(open(CACHE / "splits" / f"teamops_{a.tag}_{a.train_model}_judged.pkl", "rb")).values())
    te_all = list(pickle.load(open(CACHE / "splits" / f"teamops_{a.tag}_{a.test_model}_judged.pkl", "rb")).values())
    for t in tr_all + te_all:          # labels always follow the current verifier semantics
        t.labels = labels(t)
    if a.train_agents:
        tr_all = [t for t in tr_all if t.knobs["n_agents"] in a.train_agents]
    if a.test_agents:
        te_all = [t for t in te_all if t.knobs["n_agents"] in a.test_agents]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(tr_all))
    if a.train_model == a.test_model and not (a.train_agents and a.test_agents and not set(a.train_agents) & set(a.test_agents)):
        # same model: disjoint episode slices (train 50% / cal 25% / test 25%)
        n = len(tr_all)
        test = [tr_all[i] for i in idx[: n // 4]]
        cal = [tr_all[i] for i in idx[n // 4: n // 2]]
        train = [tr_all[i] for i in idx[n // 2:]]
    else:
        test = te_all
        cal = [tr_all[i] for i in idx[: len(idx) // 4]]
        train = [tr_all[i] for i in idx[len(idx) // 4:]]
    sub = (f"_n{''.join(map(str, a.train_agents))}to{''.join(map(str, a.test_agents))}"
           if a.train_agents or a.test_agents else "")
    tag = f"{a.tag}_audit_{a.train_model}_to_{a.test_model}{sub}"
    print(f"{tag}: train {len(train)} cal {len(cal)} test {len(test)} "
          f"(test positives {sum(t.labels['coord'] for t in test)})", flush=True)
    out = RESULTS / "teamops" / tag / a.reader
    out.mkdir(parents=True, exist_ok=True)
    qm_path = CACHE / "models" / f"q_teamops_{a.tag}_{a.train_model}{sub}.pkl"
    if not qm_path.exists():
        train_qmodels(train, n_masks=6, seed=0).save(qm_path)
    reader = "perfect"
    if a.reader != "judge":
        ext = CACHE / "extract" / f"teamops_{a.tag}_{a.test_model}_{a.reader}.pkl"
        if not ext.exists():
            llm.EXTRACT_THINKING = False
            t0 = time.time()
            tab, usage = asyncio.run(llm.extract_all(cal + test, model=a.reader))
            pickle.dump(tab, open(ext, "wb"))
            acc = {}
            for t in cal + test:
                for e in t.events:
                    if e.pub.etype == "msg":
                        d = acc.setdefault(e.fact.kind, [0, 0])
                        g = tab[(t.tid, e.eid)]
                        d[0] += g.kind == e.fact.kind and (e.fact.aware is None or g.aware == e.fact.aware)
                        d[1] += 1
            json.dump(dict(reader=a.reader, seconds=time.time() - t0, n=len(usage),
                           agreement_with_judge={k: v[0] / v[1] for k, v in acc.items()},
                           counts={k: v[1] for k, v in acc.items()}), open(out / "extraction.json", "w"), indent=1)
        reader = f"teamops_{a.tag}_{a.test_model}_{a.reader}"
    certs = cert_costs(test)
    cal_labels = {t.tid: t.labels["coord"] for t in cal}
    runs, cals = {}, {}
    for s in SELS:
        t0 = time.time()
        runs[s] = run_many(test, s, qm_path, reader, None, out / "runs" / f"test_{s}.jsonl.gz", a.procs)
        cals[s] = dict(runs=run_many(cal, s, qm_path, reader, None, out / "runs" / f"cal_{s}.jsonl.gz", a.procs),
                       labels=cal_labels)
        print(f"[{tag}/{a.reader}] {s}: {time.time() - t0:.0f}s", flush=True)
    df = table(runs, cals, test, certs)
    df.to_csv(out / "metrics.csv", index=False)
    pd.concat([paired(runs, test, "combined", b) for b in BUDGETS]).to_csv(out / "paired.csv", index=False)
    print(df[["selector", "budget", "auroc", "auroc_lo", "auroc_hi", "tpr@5", "fpr@5", "cert_found", "tokens",
              "records", "supported_on_neg", "overhead_median"]].round(3).to_string())


if __name__ == "__main__":
    main()
