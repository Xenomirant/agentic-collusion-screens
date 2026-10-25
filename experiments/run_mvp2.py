"""MVP2: search complexity at a fixed minimum certificate.

One knob is varied at a time from the base setting; the core interaction (and
hence C_cert*) is fixed per core seed, the haystack changes.  Priors are trained
on controlled episodes from the whole grid (family 0, other core seeds).

    python experiments/run_mvp2.py [--procs 16]
Writes results/mvp2/{runs/*.jsonl.gz, metrics.csv}.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from common import CACHE, RESULTS, run_many
from certaudit.belief import train_qmodels
from certaudit.generator import Knobs, generate_controlled
from certaudit.metrics import auroc, disc_within, score_at
from certaudit.verifier import min_certificate

GRID = {"length": [60, 120, 240, 480], "n_decoys": [0, 2, 4, 8], "gap": [0.5, 1.0, 2.0, 4.0],
        "n_agents": [3, 4, 6, 8], "q_meta": [1.0, 0.85, 0.7]}
BASE = Knobs()
SELS = ["random", "graph", "caes", "suspicion", "voi", "comp", "combined"]
NEGS = ["benign_coop", "misled", "unexecuted", "authorized", "individual", "shared_error"]
BUDGETS = (0.05, 0.1, 0.2, 0.5, 1.0)
N_CORE = 12


def settings():
    out = [("base", None, BASE)]
    for k, vals in GRID.items():
        for v in vals:
            if v == getattr(BASE, k):
                continue
            out.append((f"{k}={v}", k, replace(BASE, **{k: v})))
    return out


def episodes(knobs, family, core0, ctx0, n_core=N_CORE):
    out = []
    for i in range(n_core):
        core = core0 + i
        for m in "ABC":
            out.append(generate_controlled(core, ctx0 + i, m, "pos", family, knobs, tid=f"c{core}-{m}-pos"))
            neg = NEGS[(i + "ABC".index(m)) % len(NEGS)]
            out.append(generate_controlled(core, ctx0 + i, m, neg, family, knobs, tid=f"c{core}-{m}-{neg}"))
    return out


def get_prior(procs):
    path = CACHE / "models" / "q_mvp2.pkl"
    if path.exists():
        return path
    trajs = []
    for j, (_, _, kn) in enumerate(settings()):
        trajs += episodes(kn, 0, 1000 + 20 * j, 5000 + 20 * j, n_core=3)
    t0 = time.time()
    qm = train_qmodels(trajs, n_masks=2, seed=1)      # kept small: the full grid never finished in 40 CPU-min
    qm.save(path)
    print(f"prior: {len(trajs)} episodes, {time.time() - t0:.0f}s", flush=True)
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    out = RESULTS / "mvp2"
    qm = get_prior(a.procs)
    rows = []
    for name, knob, kn in settings():
        cache = CACHE / "splits" / f"mvp2_{name}.pkl"
        if cache.exists():
            test = pickle.load(open(cache, "rb"))
        else:
            test = episodes(kn, 2, 0, 500)
            pickle.dump(test, open(cache, "wb"))
        certs = {t.tid: (min_certificate(t) or (None,))[0] for t in test}
        lab = {t.tid: t.labels["coord"] for t in test}
        for sel in SELS:
            t0 = time.time()
            runs = run_many(test, sel, qm, "perfect", None, out / "runs" / f"{name}_{sel}.jsonl.gz", a.procs)
            runs = sorted(runs, key=lambda r: r["tid"])
            y = np.array([lab[r["tid"]] for r in runs])
            pos = [r for r in runs if lab[r["tid"]]]
            disc = [r["disc_cost"] for r in pos if r["disc_cost"]]
            ov = [r["disc_cost"] / certs[r["tid"]] for r in pos if r["disc_cost"]]
            row = dict(setting=name, knob=knob or "base", value=getattr(kn, knob) if knob else None, selector=sel,
                       n=len(runs), n_events=np.mean([len(t.events) for t in test]),
                       total_tokens=np.mean([r["total_cost"] for r in runs]),
                       cert_tokens_median=np.median([certs[r["tid"]] for r in pos]),
                       disc_rate=len(disc) / len(pos), disc_tokens_median=np.median(disc) if disc else np.nan,
                       disc_tokens_mean=np.mean(disc) if disc else np.nan,
                       overhead_median=np.median(ov) if ov else np.nan, overhead_mean=np.mean(ov) if ov else np.nan,
                       seconds=time.time() - t0)
            for b in BUDGETS:
                s = np.array([score_at(r, b)[0] for r in runs])
                row[f"auroc@{b}"] = auroc(y, s)
                row[f"disc@{b}"] = np.mean([disc_within(r, b) for r in pos])
                row[f"supported_on_neg@{b}"] = np.mean([score_at(r, b)[1] == "supported" for r in runs if not lab[r["tid"]]])
            rows.append(row)
            print(f"{name:14s} {sel:10s} disc={row['disc_rate']:.2f} ov={row['overhead_median']:.2f} "
                  f"auroc@0.2={row['auroc@0.2']:.3f} {row['seconds']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
