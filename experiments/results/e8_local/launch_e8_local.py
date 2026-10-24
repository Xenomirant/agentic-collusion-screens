#!/usr/bin/env python3
"""Launch many Agrawal double-auction markets against the local vLLM server (E8).

One harness process per seed, N at a time (vLLM batches the concurrent requests). Each run:
4 Qwen2.5-7B-Instruct sellers, 3 rule-based ZIP buyers, 60 rounds, temperature 0.7 (harness default).

    python launch_e8_local.py controls --seeds 0-39 --concurrency 16
    python launch_e8_local.py seeded   --seeds 1000-1039 --concurrency 16
    python launch_e8_local.py seeded_monitored --seeds 2000-2039 --concurrency 24

"seeded_monitored" is the seeded design with the seller prompt template swapped for
seller_prompt_monitored.jinja2 (one added sentence in the private-channel section telling the pair
that a regulator screens the public ask history for statistically coupled pairs and fines them).

The seeded dataset gives the private channel to PAIRS[seed % 6], so every unordered pair of the
four sellers is used equally often. A run counts as complete when experiment_metadata.json exists
and auction_results.md holds all 60 round blocks; anything else is moved to <dataset>/../failed/.
Per-run stdout/stderr goes to logs/<tag>.log, and a line per run is appended to runs.jsonl.
"""
import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[2] / "external" / "collusion_hackathon" / "llm-agent-collusion"
PY = HARNESS / ".venv" / "bin" / "python"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
PAIRS = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)]
PREFIX = {"controls": "ctrl", "seeded": "pair", "seeded_monitored": "mon"}
MONITORED_TEMPLATE = "seller_prompt_monitored.jinja2"
N_BLOCK = re.compile(r"````json\n(.*?)\n````", re.S)


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def tag_for(dataset, seed):
    return f"{PREFIX[dataset]}-s{seed}"


def existing_complete(out_dir, tag, rounds):
    for d in out_dir.glob(f"*_{tag}"):
        if complete(d, rounds):
            return d
    return None


def complete(d, rounds):
    ar = d / "auction_results.md"
    if not (d / "experiment_metadata.json").exists() or not ar.exists():
        return False
    return len(N_BLOCK.findall(ar.read_text())) == rounds


def run_one(dataset, seed, args, lock, results):
    tag = tag_for(dataset, seed)
    out_dir = HERE / dataset
    cmd = [str(PY), "src/continuous_double_auction/simulation.py", "--sellers", "4", "--buyer_models", "zip",
           "--seller_models", MODEL, "--rounds", str(args.rounds), "--seeds", str(seed), "-y",
           "--tag", tag, "--log_dir", str(out_dir)]
    pair = None
    if dataset in ("seeded", "seeded_monitored"):
        pair = PAIRS[seed % len(PAIRS)]
        cmd += ["--seller_channel_pair", str(pair[0]), str(pair[1])]
    if dataset == "seeded_monitored":
        cmd += ["--seller_prompt_template", MONITORED_TEMPLATE]
    before = set(out_dir.glob(f"*_{tag}"))
    t0 = time.time()
    with open(HERE / "logs" / f"{tag}.log", "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.call(cmd, cwd=HARNESS, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    new = [d for d in out_dir.glob(f"*_{tag}") if d not in before]
    folder = new[0] if new else None
    ok = folder is not None and rc == 0 and complete(folder, args.rounds)
    if folder is not None and not ok:
        failed = HERE / "failed"
        failed.mkdir(exist_ok=True)
        shutil.move(str(folder), str(failed / f"{dataset}_{folder.name}"))
    rec = dict(dataset=dataset, seed=seed, tag=tag, pair=pair, rc=rc, ok=ok, wall_s=round(wall, 1),
               folder=str(folder) if folder else None, cmd=" ".join(cmd))
    with lock:
        results.append(rec)
        with open(HERE / "runs.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {tag} pair={pair} rc={rc} ok={ok} {wall:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["controls", "seeded", "seeded_monitored"])
    ap.add_argument("--seeds", required=True, help="e.g. 0-39 or 0,1,5-9")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=60)
    args = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    out_dir = HERE / args.dataset
    out_dir.mkdir(exist_ok=True)
    todo = []
    for s in parse_seeds(args.seeds):
        done = existing_complete(out_dir, tag_for(args.dataset, s), args.rounds)
        if done:
            print(f"skip seed {s}: complete at {done}")
        else:
            todo.append(s)
    print(f"{len(todo)} runs to launch, {args.concurrency} at a time", flush=True)
    sem = threading.Semaphore(args.concurrency)
    lock = threading.Lock()
    results = []

    def worker(seed):
        with sem:
            run_one(args.dataset, seed, args, lock, results)

    threads = [threading.Thread(target=worker, args=(s,)) for s in todo]
    for t in threads:
        t.start()
        time.sleep(0.5)  # stagger so the timestamped folder names stay distinct
    for t in threads:
        t.join()
    n_ok = sum(r["ok"] for r in results)
    print(f"done: {n_ok}/{len(results)} complete")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
