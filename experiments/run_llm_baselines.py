"""Active LLM auditor (budgets 10/20/50%) and full-information LLM monitor on MVP1 splits.

    python experiments/run_llm_baselines.py --model qwen2.5-7b [--limit N]
"""
import argparse
import asyncio
import gzip
import json
import time

from common import RESULTS, get_split
from certaudit.llm import llm_audit, llm_monitor, run_llm_batch
from certaudit.oracle import make_view


def dump(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    ap.add_argument("--splits", nargs="+", default=["mvp1_cal", "mvp1_test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--pickle", default=None, help="audit a pickled {tid: Trajectory} dataset instead of named splits")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = RESULTS / (a.out or f"mvp1/llm/{a.model}")
    for s in a.splits:
        if a.pickle:
            import pickle
            import numpy as np
            d = list(pickle.load(open(a.pickle, "rb")).values())
            trajs = [d[i] for i in np.random.default_rng(0).permutation(len(d))][: a.limit]
        else:
            trajs = get_split(s)[: a.limit]
        p = out / f"{s}_monitor.jsonl.gz"
        if not p.exists():
            t0 = time.time()
            rows = asyncio.run(run_llm_batch(lambda t, sem, cli: llm_monitor(t, a.model, sem, cli), trajs, a.concurrency, a.model))
            dump(rows, p)
            print(s, "monitor", f"{time.time() - t0:.0f}s", flush=True)
        for b in a.budgets:
            p = out / f"{s}_auditor_b{int(b * 100)}.jsonl.gz"
            if p.exists():
                continue
            t0 = time.time()
            rows = asyncio.run(run_llm_batch(
                lambda t, sem, cli: llm_audit(t, make_view(t), b, a.model, sem, cli), trajs, a.concurrency, a.model))
            dump(rows, p)
            print(s, "auditor", b, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
