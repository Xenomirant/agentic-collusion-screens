"""Generate TeamOps episodes with local LLM agents at scale and store them as trajectories.

    python experiments/run_teamops_gen.py --model qwen2.5-7b --tag L --n-episodes 600 --agents 8 12 --rounds 10

Episode design (seeded): 45% have one `pos` pair plus 2 other-condition pairs, 45% have
3 non-pos pairs, 10% have no private pair; pair mechanisms are drawn from A/A2/B/C/D
without repetition; the number of agents is drawn from --agents.
Writes cache/splits/teamops_<tag>_<model>.pkl (Trajectory objects; message facts are
placeholders until run_teamops_judge.py) and results/teamops/<tag>_<model>_episodes.jsonl.gz.
Generation is chunked and checkpointed: rerunning the same command resumes.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import json
import pickle
import time

import numpy as np

from common import CACHE, RESULTS
from certaudit.teamops import generate_episodes
from certaudit.verifier import labels

NEGS = ["misled", "authorized", "unexecuted", "individual", "benign_coop", "shared_error"]
MECHS = ["A", "A2", "B", "C", "D"]


def spec(rng, agents):
    n = int(rng.choice(agents))
    r = rng.random()
    mechs = list(rng.permutation(MECHS))
    if r < 0.10:
        return [], n
    k = 3 if n >= 8 else 2
    conds = list(rng.choice(NEGS, size=k, replace=False))
    if r < 0.55:
        conds[0] = "pos"
    return [(c, m) for c, m in zip(conds, mechs)], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--tag", default="L")
    ap.add_argument("--n-episodes", type=int, default=600)
    ap.add_argument("--agents", type=int, nargs="+", default=[8, 12])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=24)
    a = ap.parse_args()
    path = CACHE / "splits" / f"teamops_{a.tag}_{a.model}.pkl"
    have = pickle.load(open(path, "rb")) if path.exists() else {}
    rng = np.random.default_rng(a.seed0 + 17)
    specs = []
    for i in range(a.n_episodes):
        pairs, n = spec(rng, a.agents)          # drawn in order, so the design is reproducible
        tid = f"to{a.tag}-{a.model}-{i}"
        if tid not in have:
            specs.append((tid, a.seed0 + 10007 * i, pairs, n))
    print(f"{len(have)} cached, {len(specs)} to generate", flush=True)
    t0 = time.time()
    out = RESULTS / "teamops"
    out.mkdir(parents=True, exist_ok=True)
    n_calls = n_tok = 0
    for i in range(0, len(specs), a.chunk):
        chunk = specs[i:i + a.chunk]
        eps, usage = asyncio.run(generate_episodes([(s, p, n, a.rounds, 0) for _, s, p, n in chunk], model=a.model,
                                                   concurrency=a.concurrency))
        n_calls += len(usage)
        n_tok += sum(p + c for p, c in usage)
        with gzip.open(out / f"{a.tag}_{a.model}_episodes.jsonl.gz", "at") as fh:
            for (tid, _, _, _), ep in zip(chunk, eps):
                tr = ep.to_trajectory(tid)
                tr.labels = labels(tr)          # execution/waiver part only; agreement needs the judge
                have[tid] = tr
                fh.write(json.dumps(dict(tid=tid, cond=ep.cond, mech=ep.mech, n_agents=ep.n_agents,
                                         pairs=tr.knobs["pairs"],
                                         log=[dict(t=r["t"], etype=r["etype"], actor=r["actor"], to=list(r["recipients"]),
                                                   objects=list(r["objects"]), text=r["text"]) for r in ep.log])) + "\n")
        tmp = path.with_suffix(".tmp")
        pickle.dump(have, open(tmp, "wb"))
        tmp.replace(path)
        print(f"  {len(have)} episodes done ({time.time() - t0:.0f}s, {n_tok / 1e6:.1f}M tokens)", flush=True)
    print(f"{time.time() - t0:.0f}s, {n_calls} calls, {n_tok} tokens", flush=True)
    stat = collections.Counter()
    for tr in have.values():
        stat[(tr.condition, "exec" if tr.labels["any_violation"] else "no-exec")] += 1
    for key in sorted(stat):
        print(key, stat[key])
    ev = [len(t.events) for t in have.values()]
    print(f"events/episode mean {np.mean(ev):.0f}, tokens/episode mean {np.mean([t.total_cost for t in have.values()]):.0f}")


if __name__ == "__main__":
    main()
