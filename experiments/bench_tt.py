"""Benchmark: replica tensors by TT-cross (fathon.tensor.tt) vs dense construction + TT-SVD.

For a binary random cascade (atoms 0.3 / 0.7, the BINARY law of the tests)
with S samples on 2**n cells:

* dense: `replica.replica_tensor` (2**(n q) entries) + `decomp.tt_svd`;
* cross: `tt.replica_tt(method='cross')` with the torchtt two-site DMRG
  backend (default; library defaults incl. the stagnation stop) and the
  teneva one-site backend (stock teneva sweep count 10), moment entries
  evaluated on the GPU, both with rank cap MAX_RANK;
* old: optionally (``--old PATH``) the hand-written cross this module had
  before the library refactor, with its own defaults (rank cap 64, 6
  sweeps), for an old-vs-new comparison. Get it with
  ``git show 400074c:fathon/fathon/tensor/tt.py > /tmp/tt_old.py``;
* law oracle: TT-cross (both backends, and old) on the entries of the exact
  law (`tt.cascade_moment_tt`, REPORT.md Theorem A), no sampling noise.

Reported: wall time, number of entry evaluations, level-cut ranks at
relative singular-value tolerance TOL, errors against the dense empirical
tensor (n <= 10) and against the exact law (TT arithmetic, any n), and the
predicted sampling noise ||T_S - T||_F / ||T||_F. The cross accuracy is set
to eps = max(0.05, 2 x predicted noise): below the sampling noise the
empirical tensor is full-rank and cross is the wrong tool.

Usage: CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 python experiments/bench_tt.py [--quick]
       [--part all|dense|cross] [--merge] [--old /tmp/tt_old.py]
Writes experiments/results/bench_tt.json (``--part`` writes bench_tt.<part>.json
after every configuration; ``--merge`` combines the two part files).
"""

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fathon'))
warnings.filterwarnings('ignore')

import torch  # noqa: E402
import teneva  # noqa: E402
from fathon.tensor import tt, replica, decomp  # noqa: E402

ATOMS = np.array([[0.3, 0.7], [0.7, 0.3]])
PROBS = np.array([0.5, 0.5])
TOL = 3e-2
S = 20000
MAX_EVALS = 3 * 10 ** 7  # per cross run (dmrg: never exceeded after its first sweep)
MAX_RANK = 256
OLD = None  # module of the pre-refactor tt.py (--old)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'bench_tt.json')


def ranks_from_spectra(spectra, tol):
    return [decomp._rank_from_tol(np.asarray(s), tol, None) for s in spectra]


def predicted_noise(mu, q, law):
    """||T_S - T||_F / ||T||_F in expectation: sqrt((E[Z_2^q] - ||T||^2) / S) / ||T||, Z_2 = sum_w mu(w)^2."""
    z2 = np.sum(mu ** 2, axis=1)
    nT2 = tt.tt_norm(law) ** 2
    return float(np.sqrt(max(np.mean(z2 ** q) - nT2, 0.0) / mu.shape[0]) / np.sqrt(nT2))


def load_old(path):
    spec = importlib.util.spec_from_file_location('fathon.tensor.tt_old', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cross(mu_dev, q, eps, backend, device, dtype=None):
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()
    if backend == 'old':
        C, info = OLD.replica_tt(mu_dev, q, eps=eps, return_info=True)
        C = [np.asarray(G.detach().cpu() if torch.is_tensor(G) else G, dtype=float) for G in C]
        info = dict(info, ranks=tt.tt_ranks(C), stop=None, converged=info.get('converged'))
    else:
        kw = {'n_sweeps': 10} if backend == 'teneva' else {}
        C, info = tt.replica_tt(mu_dev, q, eps=eps, max_rank=MAX_RANK, backend=backend, device=device,
                                dtype=dtype, max_evals=MAX_EVALS, return_info=True, **kw)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    return C, info, time.perf_counter() - t0


def bench_config(q, n, dense, device, backends):
    row = {'q': q, 'n': n, 'S': S, 'dense_entries': 2 ** (q * n)}
    mu = tt.cascade_sample(ATOMS, PROBS, n, S, seed=100 * q + n)
    law = tt.cascade_moment_tt(ATOMS, PROBS, q, n)
    noise = predicted_noise(mu, q, law)
    eps = max(0.05, 2 * noise)
    row.update(noise_pred=noise, eps=eps, tol=TOL,
               law_ranks_tol=tt.level_cut_ranks(law, TOL), law_ranks_exact=tt.level_cut_ranks(law, 1e-10))
    T = None
    if dense:
        t0 = time.perf_counter()
        T = replica.replica_tensor(mu, q)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, spectra = decomp.tt_svd(T, tol=TOL)
        t_svd = time.perf_counter() - t0
        nT = np.linalg.norm(T)
        row['dense'] = {'time_build': t_build, 'time_tt_svd': t_svd, 'time': t_build + t_svd,
                        'bytes': int(T.nbytes), 'ranks_tol': ranks_from_spectra(spectra, TOL),
                        'err_vs_law': float(np.linalg.norm(T - tt.tt_to_dense(law)) / nT)}
    # n >= 12: float32 evaluation (the GPU is shared; a float32 mean over S samples is ~1e-6 accurate)
    dtype = torch.float32 if n >= 12 else torch.float64
    row['eval_dtype'] = str(dtype)
    mu_dev = torch.as_tensor(mu, device=device, dtype=dtype)
    for be in backends:
        try:
            C, info, dt = run_cross(mu_dev, q, eps, be, device, dtype)
        except torch.OutOfMemoryError as e:  # the GPU is shared with other jobs
            row[be] = {'error': 'CUDA OOM: ' + str(e).split('.')[0]}
            print('  q={} n={} {}: CUDA OOM'.format(q, n, be), flush=True)
            torch.cuda.empty_cache()
            continue
        r = {'time': dt, 'n_evals': int(info['n_evals']), 'sweeps': info['sweeps'], 'stop': info['stop'],
             'converged': info['converged'], 'ranks': info['ranks'], 'ranks_tol': tt.level_cut_ranks(C, TOL),
             'err_vs_law': tt.tt_rel_err(C, law)}
        if T is not None:
            r['err_vs_dense'] = float(np.linalg.norm(tt.tt_to_dense(C) - T) / np.linalg.norm(T))
        row[be] = r
        print('  q={} n={} {}: {:.1f}s evals={} ranks_tol={} err_law={:.3f}{}'.format(
            q, n, be, dt, r['n_evals'], r['ranks_tol'], r['err_vs_law'],
            ' err_dense={:.3f}'.format(r['err_vs_dense']) if 'err_vs_dense' in r else ''), flush=True)
    del T
    return row


def bench_law_oracle(q, n, backends):
    law = tt.cascade_moment_tt(ATOMS, PROBS, q, n)
    row = {'q': q, 'n': n, 'law_ranks_exact': tt.level_cut_ranks(law, 1e-10), 'dense_entries': 2 ** (q * n)}
    for be in backends:
        t0 = time.perf_counter()
        if be == 'old':
            C, info = OLD.tt_cross(lambda idx: teneva.get_many(law, idx), [2 ** q] * n, max_rank=50, eps=1e-10)
            C = [np.asarray(G, dtype=float) for G in C]
        else:
            C, info = tt.tt_cross(lambda idx: teneva.get_many(law, idx), [2 ** q] * n, eps=1e-10, backend=be)
        dt = time.perf_counter() - t0
        row[be] = {'time': dt, 'n_evals': int(info['n_evals']), 'ranks': tt.level_cut_ranks(C, 1e-10),
                   'err_vs_law': tt.tt_rel_err(C, law)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='small sizes only (smoke test)')
    ap.add_argument('--part', choices=['all', 'dense', 'cross'], default='all',
                    help='dense: n <= 10 with dense reference; cross: cross-only sizes and law oracle')
    ap.add_argument('--merge', action='store_true', help='merge bench_tt.dense.json and bench_tt.cross.json')
    ap.add_argument('--old', help='path to the pre-refactor tt.py, adds an "old" column')
    args = ap.parse_args()
    global OLD
    backends = ['dmrg', 'teneva']
    if args.old:
        OLD = load_old(args.old)
        backends.append('old')
    if args.merge:
        parts = [json.load(open(OUT.replace('.json', '.{}.json'.format(p)))) for p in ('dense', 'cross')]
        merged = {'meta': {'dense_part': parts[0]['meta'], 'cross_part': parts[1]['meta']},
                  'samples': parts[0]['samples'] + parts[1]['samples'], 'law_oracle': parts[1]['law_oracle']}
        with open(OUT, 'w') as f:
            json.dump(merged, f, indent=1)
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with_dense = [(q, n) for q in (2, 3) for n in (6, 8, 10)]
    cross_only = [(q, n) for q in (3, 4) for n in (12, 14)]
    if args.quick:
        with_dense, cross_only = [(2, 6), (3, 6)], [(3, 12)]
    meta = {'max_evals': MAX_EVALS, 'max_rank': MAX_RANK, 'old': args.old, 'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
            'omp_threads': os.environ.get('OMP_NUM_THREADS'), 'torch_threads': torch.get_num_threads(),
            'load_avg_start': os.getloadavg(), 'python': platform.python_version(),
            'teneva': teneva.__version__ if hasattr(teneva, '__version__') else '0.14.11',
            'torch': torch.__version__, 'law': {'atoms': ATOMS.tolist(), 'probs': PROBS.tolist()},
            'S': S, 'tol': TOL, 'eps_rule': 'max(0.05, 2 * predicted sampling noise)'}
    if args.part == 'dense':
        cross_only = []
    elif args.part == 'cross':
        with_dense = []
    out = OUT.replace('.json', '_quick.json') if args.quick else OUT
    if args.part != 'all':
        out = out.replace('.json', '.{}.json'.format(args.part))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows, oracle = [], []

    def save():
        meta['load_avg_end'] = os.getloadavg()
        with open(out, 'w') as f:
            json.dump({'meta': meta, 'samples': rows, 'law_oracle': oracle}, f, indent=1)

    for q, n in cross_only:  # cheap and noise-free first
        oracle.append(bench_law_oracle(q, n, backends))
        print('  law oracle', oracle[-1], flush=True)
        save()
    for q, n in with_dense:
        print('dense + cross q={} n={}'.format(q, n), flush=True)
        rows.append(bench_config(q, n, True, device, backends))
        save()
    for q, n in cross_only:
        print('cross only q={} n={}'.format(q, n), flush=True)
        rows.append(bench_config(q, n, False, device, backends))
        save()


if __name__ == '__main__':
    main()
