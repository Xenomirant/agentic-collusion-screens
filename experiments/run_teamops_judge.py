"""Judge pass for TeamOps episodes: label every message with a structured fact using
Qwen3-8B with thinking (the strongest local reader), keep action/approval/read/outcome
records exact, recompute labels, and report assigned objective vs realised behaviour.

    python experiments/run_teamops_judge.py --model qwen2.5-7b [--judge qwen3-8b] [--tag L]

Reads cache/splits/teamops_<model>.pkl, writes cache/splits/teamops_<model>_judged.pkl
and results/teamops/<model>_labels.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import pickle
import time

import pandas as pd

from common import CACHE, RESULTS
from certaudit import llm
from certaudit.verifier import labels, min_certificate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--judge", default="qwen3-8b")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--tag", default="L")
    a = ap.parse_args()
    have = pickle.load(open(CACHE / "splits" / f"teamops_{a.tag}_{a.model}.pkl", "rb"))
    trajs = list(have.values())
    llm.EXTRACT_THINKING = True
    t0 = time.time()
    table, usage = asyncio.run(llm.extract_all(trajs, model=a.judge, concurrency=a.concurrency))
    print(f"judge: {len(table)} records, {time.time() - t0:.0f}s, "
          f"{sum(p + c for p, c in usage) / max(1, len(usage)):.0f} tokens/record", flush=True)
    rows = []
    for tr in trajs:
        for e in tr.events:
            if e.pub.etype == "msg":
                e.fact = table[(tr.tid, e.eid)]
        tr.labels = labels(tr)
        mc = min_certificate(tr) if tr.labels["coord"] else None
        rows.append(dict(tid=tr.tid, mechanism=tr.mechanism, condition=tr.condition, model=a.model,
                         n_events=len(tr.events), total_cost=tr.total_cost,
                         coord=tr.labels["coord"], attempted=tr.labels["attempted"],
                         any_violation=tr.labels["any_violation"], individual=tr.labels["individual"],
                         cert_cost=mc[0] if mc else None, cert_size=len(mc[1]) if mc else None))
    pickle.dump(have, open(CACHE / "splits" / f"teamops_{a.tag}_{a.model}_judged.pkl", "wb"))
    df = pd.DataFrame(rows)
    (RESULTS / "teamops").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "teamops" / f"{a.tag}_{a.model}_labels.csv", index=False)
    print(df.groupby(["mechanism", "condition"])[["attempted", "any_violation", "coord"]].mean().round(2).to_string())
    print("coordinated positives:", int(df.coord.sum()), "of", len(df))


if __name__ == "__main__":
    main()
