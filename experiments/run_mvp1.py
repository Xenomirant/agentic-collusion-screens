"""MVP1: evidence complementarity across mechanisms A/B/C with matched information.

    python experiments/run_mvp1.py [--reader perfect|<extraction table>] [--ablations]

Writes results/mvp1/<reader>/{runs/*.jsonl.gz, metrics.csv, paired.csv}.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from common import RESULTS, cert_costs, get_qmodels, get_split, run_many
from analysis import BUDGETS, paired, table

SELS = ["random", "random_groups", "graph", "caes", "suspicion", "voi", "comp", "combined"]
ABL_SELS = ["random", "caes", "suspicion", "voi", "comp", "combined"]
ABLATIONS = ["shuffle_links", "no_provenance", "no_comms", "no_actions", "hide_outcomes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="perfect")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    out = RESULTS / "mvp1" / a.reader
    train, cal, test = get_split("mvp1_train"), get_split("mvp1_cal"), get_split("mvp1_test")
    certs = cert_costs(test)
    cal_labels = {t.tid: t.labels["coord"] for t in cal}
    settings = [(None, SELS)] + ([(ab, ABL_SELS) for ab in ABLATIONS] if a.ablations else [])
    tables, pairs = [], []
    for ab, sels in settings:
        qm = get_qmodels("mvp1_train", ab)
        runs, cals = {}, {}
        for s in sels:
            t0 = time.time()
            tag = f"{s}_{ab or 'none'}"
            runs[s] = run_many(test, s, qm, a.reader, ab, out / "runs" / f"test_{tag}.jsonl.gz", a.procs)
            c = run_many(cal, s, qm, a.reader, ab, out / "runs" / f"cal_{tag}.jsonl.gz", a.procs)
            cals[s] = dict(runs=c, labels=cal_labels)
            print(f"[{a.reader}] {tag}: {time.time() - t0:.0f}s", flush=True)
        df = table(runs, cals, test, certs)
        df["ablation"] = ab or "none"
        tables.append(df)
        for b in BUDGETS:
            p = paired(runs, test, "combined", b)
            p["ablation"] = ab or "none"
            pairs.append(p)
        pd.concat(tables).to_csv(out / "metrics.csv", index=False)
        pd.concat(pairs).to_csv(out / "paired.csv", index=False)
    print(pd.concat(tables)[["ablation", "selector", "budget", "auroc", "auroc_lo", "auroc_hi", "tpr@5", "fpr@5",
                            "cert_found", "tokens", "overhead_median"]].round(3).to_string())


if __name__ == "__main__":
    main()
