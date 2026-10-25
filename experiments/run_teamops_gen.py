"""Generate TeamOps episodes with local LLM agents and store them as trajectories.

    python experiments/run_teamops_gen.py --model qwen2.5-7b [--n-pos 30 --n-neg 10 --n-null 20 --rounds 8]

Writes cache/splits/teamops_<model>.pkl (Trajectory objects; message facts are
placeholders until the judge pass in run_teamops_judge.py) and
results/teamops/<model>_episodes.jsonl.gz (full logs for inspection).
Episodes are resumable: existing episodes in the pickle are kept.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import json
import pickle
import time

from common import CACHE, RESULTS
from certaudit.teamops import generate_episodes
from certaudit.verifier import labels

CONDS = ["pos", "misled", "authorized", "unexecuted", "individual", "benign_coop", "shared_error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--mechs", default="ABC")
    ap.add_argument("--n-pos", type=int, default=30)
    ap.add_argument("--n-neg", type=int, default=10)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    path = CACHE / "splits" / f"teamops_{a.model}.pkl"
    have = pickle.load(open(path, "rb")) if path.exists() else {}
    specs = []
    k = 0
    for m in a.mechs:
        for c in CONDS:
            for r in range(a.n_pos if c == "pos" else a.n_neg):
                tid = f"to-{a.model}-{m}-{c}-{r}"
                if tid not in have:
                    specs.append((tid, a.seed0 + 10007 * k, c, m))
                k += 1
    for r in range(a.n_null):
        tid = f"to-{a.model}-null-{r}"
        if tid not in have:
            specs.append((tid, a.seed0 + 10007 * k, "null", "null"))
        k += 1
    print(f"{len(have)} cached, {len(specs)} to generate", flush=True)
    t0 = time.time()
    eps, usage = asyncio.run(generate_episodes([(s, c, m, 0) for _, s, c, m in specs], model=a.model,
                                               concurrency=a.concurrency))
    for e in eps:
        e.rounds = a.rounds
    out = RESULTS / "teamops"
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / f"{a.model}_episodes.jsonl.gz", "at") as fh:
        for (tid, _, _, _), ep in zip(specs, eps):
            tr = ep.to_trajectory(tid)
            tr.labels = labels(tr)          # execution/waiver part only; agreement needs the judge
            have[tid] = tr
            fh.write(json.dumps(dict(tid=tid, cond=ep.cond, mech=ep.mech, A=ep.A, B=ep.B, goal_ops=ep.goal_ops,
                                     log=[dict(t=r["t"], etype=r["etype"], actor=r["actor"], to=list(r["recipients"]),
                                               objects=list(r["objects"]), text=r["text"]) for r in ep.log])) + "\n")
    pickle.dump(have, open(path, "wb"))
    print(f"{time.time() - t0:.0f}s, {len(usage)} calls, {sum(p + c for p, c in usage)} tokens", flush=True)
    stat = collections.Counter()
    for tr in have.values():
        stat[(tr.mechanism, tr.condition, "exec" if tr.labels["any_violation"] else "no-exec")] += 1
    for key in sorted(stat):
        print(key, stat[key])


if __name__ == "__main__":
    main()
