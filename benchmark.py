"""Autoresearch harness: who-colludes recovery among many same-model agents.

Corpus: autoresearch/data/planted_cases.jsonl, one simulated market per line
(experiments/sims/planted.py; deterministic given the seed). Each case is run
through the real detector, fathon.tensor.scale.detect, and scored by
precision@K, K = number of planted pairs. Prints ``metric: <mean precision>``.

Optional env: AR_DEVICE (cuda:0), AR_CASES (comma-separated ids), AR_RANK.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'experiments' / 'sims'))
import planted  # noqa: E402
from fathon.tensor import scale  # noqa: E402

CORPUS = HERE / 'autoresearch' / 'data' / 'planted_cases.jsonl'
DEVICE = os.environ.get('AR_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')


def sync():
    if DEVICE.startswith('cuda'):
        torch.cuda.synchronize()


def run_case(case):
    groups = case.get('groups', 1)
    A, truth = planted.simulate(case['N'], case['T'], r=case['r'], kind=case['kind'], k=case['k'],
                                n_coal=case['n_coal'], eps=case['eps'], c=case['c'],
                                seed=case['seed'], device=DEVICE, groups=groups)
    K = len(truth)
    sync()
    t0 = time.perf_counter()
    pairs = scale.detect(A, K, rank=int(os.environ.get('AR_RANK', case['r'] * groups)), device=DEVICE)
    sync()
    dt = time.perf_counter() - t0
    hit = len(set(pairs) & truth) / K
    return hit, dt


def main():
    cases = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    sel = os.environ.get('AR_CASES')
    if sel:
        ids = {int(x) for x in sel.split(',')}
        cases = [c for c in cases if c['id'] in ids]
    print(f'corpus: {len(cases)} cases from {CORPUS}', file=sys.stderr)
    precs, times = [], []
    for c in cases:
        p, dt = run_case(c)
        precs.append(p)
        times.append(dt)
        print(f"case {c['id']:2d} N={c['N']:6d} r={c['r']:2d}x{c.get('groups', 1)} c={c['c']:.1f} {c['kind']:4s} k={c['k']} eps={c['eps']:.2f}  "
              f"precision@K={p:.3f}  detect={dt*1000:.0f}ms", file=sys.stderr)
    print(f'avg_latency_ms: {1000 * sum(times) / len(times):.1f}')
    print(f'metric: {sum(precs) / len(precs):.4f}')


if __name__ == '__main__':
    main()
