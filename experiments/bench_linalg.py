"""Benchmark of fathon.tensor.linalg against torch built-ins on population-sized matrices.

Data: X = L (s F) + E, N agents x T rounds, a rank-4 common mode (strengths s = 8, 6, 4, 3 per
entry, i.e. S_j ~ s_j sqrt(N T)) plus unit noise, float32 on the GPU. Up to N = 1e6 the matrix is
materialised once and the streamed methods read it in chunks of `--chunk` rows; at N = 4e6 it is
never materialised: every pass regenerates the chunks on the fly from per-chunk seeds (the
generation time is reported separately as `generate`).

Methods (top right singular subspace and singular values of the whole matrix):
  tsqr / tsqr_f32   linalg.tall_svd(method='tsqr'), local QRs in float64 / float32
  gram / gram_f32   linalg.tall_svd(method='gram'), chunk GEMMs in float64 / float32 (float64 sum)
  cans_R            linalg.tall_svd(method='cans'): TSQR (float64) then CANS polar SVD of R
  cans_full         cans_svd on the whole matrix (where it fits)
  torch_svd         torch.linalg.svd(X, full_matrices=False) (where it fits)
  svd_lowrank       torch.svd_lowrank(X, q=16, niter=4) (where it fits)
Accuracy (N = 1e5): top-8 singular values and the largest principal angle of the top-4 (signal)
and top-8 right singular subspaces against float64 torch.linalg.svd.

Rank: linalg.estimate_rank_pa with its defaults (whole population, one pass, closed-form Toeplitz
null, 2 calibration surrogates, auto margin) against scale.estimate_rank(method='shift') (20k-row
subsample, 2 explicit surrogate SVDs), true rank 4, with white and AR(1) phi = 0.97 noise; and
(`rank_hard`, N = 1e5, 5 seeds) AR(1) phi = 0.97 noise alone (true rank 0), under a weak rank-4
mode (strengths x 0.12), and a common linear trend (amplitude 0.5 per entry) in white noise (true
rank 1; untestable under the Toeplitz null, see linalg.estimate_rank_pa), where null='circulant'
is reported too.

Precision and threads: by default the GEMMs of the baselines run with whatever TF32 setting the
environment gives (NGC: TF32 on), which is what a caller gets; linalg forces IEEE float32 for its
own float32 GEMMs. `--tf32 off` switches TF32 off globally instead. CPU threads default to the
cgroup CPU quota (torch would otherwise use one per host core). Both are recorded in the JSON.

Usage: CUDA_VISIBLE_DEVICES=1 python bench_linalg.py [--Ts 256 1024] [--Ns 1e5 1e6 4e6] [--hard-only]
       [--tf32 env|off] [--threads N]
Output: experiments/results/bench_linalg.json
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fathon'))
from fathon.tensor import linalg as la, scale  # noqa: E402

STRENGTHS = (8.0, 6.0, 4.0, 3.0)


def factors(T, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    F = torch.randn(len(STRENGTHS), T, generator=g, device=dev)
    return F * torch.tensor(STRENGTHS, device=dev)[:, None]


def ar1_(E, phi):
    """In-place AR(1) filter along time with unit marginal variance."""
    E[:, 0] /= math.sqrt(1 - phi ** 2)
    for t in range(1, E.shape[1]):
        E[:, t] += phi * E[:, t - 1]
    E *= math.sqrt(1 - phi ** 2)
    return E


def make_chunk(i, n, T, F, dev, noise='white', seed=0):
    g = torch.Generator(device=dev).manual_seed(1000003 * (seed + 1) + i)
    L = torch.randn(n, F.shape[0], generator=g, device=dev)
    E = torch.randn(n, T, generator=g, device=dev)
    if noise == 'ar97':
        E = ar1_(E, 0.97)
    return L @ F + E


def materialize(src, in_mem):
    """The whole matrix if it is wanted and fits on the (shared) device, else None (stream instead)."""
    if not in_mem:
        return None, False
    try:
        return torch.cat(list(src())), True
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print('  (matrix does not fit on the shared GPU: streaming instead)', flush=True)
        return None, False


def chunk_source(N, T, chunk, F, dev, noise='white'):
    sizes = [min(chunk, N - s) for s in range(0, N, chunk)]
    return lambda: (make_chunk(i, n, T, F, dev, noise) for i, n in enumerate(sizes))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, reps):
    """Median wall time over `reps` runs, peak extra device memory (GB), last result; or the error."""
    times, peak, out = [], 0.0, None
    for _ in range(reps):
        sync()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            out = fn()
            sync()
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            out = None
            torch.cuda.empty_cache()
            return {'error': type(exc).__name__ + ': ' + str(exc).split('\n')[0][:160]}, None
        times.append(time.perf_counter() - t)
        if torch.cuda.is_available():
            peak = max(peak, (torch.cuda.max_memory_allocated() - base) / 2 ** 30)
    times.sort()
    return {'time_s': times[len(times) // 2], 'times_s': times, 'peak_extra_gb': round(peak, 3)}, out


def accuracy(S, V, S0, V0):
    S, V = S.double().cpu(), V.double().cpu()
    return {'sv_relerr_top8': float(((S[:8] - S0[:8]) / S0[:8]).abs().max()),
            'angle_top4_rad': float(la.subspace_angles(V[:, :4], V0[:, :4]).max()),
            'angle_top8_rad': float(la.subspace_angles(V[:, :8], V0[:, :8]).max())}


def bench_svd(N, T, chunk, dev, reps, cans_svd):
    F = factors(T, dev)
    src = chunk_source(N, T, chunk, F, dev)
    in_mem = N <= 1_000_000
    res = {}
    r, _ = timed(lambda: [c.shape for c in src()], reps)
    res['generate'] = r
    X, in_mem = materialize(src, in_mem)
    stream = (lambda: X) if in_mem else src
    S0 = V0 = None
    if N <= 100_000 and in_mem:
        _, S0, V0h = torch.linalg.svd(X.double(), full_matrices=False)
        S0, V0 = S0.cpu(), V0h.T.cpu()
    methods = {
        'tsqr': lambda: la.tall_svd(stream(), k=8, method='tsqr', chunk_rows=chunk),
        'tsqr_f32': lambda: la.tall_svd(stream(), k=8, method='tsqr', dtype=torch.float32, chunk_rows=chunk),
        'gram': lambda: la.tall_svd(stream(), k=8, method='gram', chunk_rows=chunk),
        'gram_f32': lambda: la.tall_svd(stream(), k=8, method='gram', dtype=torch.float32, chunk_rows=chunk),
    }
    if cans_svd is not None:
        methods['cans_R'] = lambda: la.tall_svd(stream(), k=8, method='cans', chunk_rows=chunk)
    if in_mem:
        def full_cans():
            _, s, vt = cans_svd(X)
            return s[:8], vt.T[:, :8]
        if cans_svd is not None:
            methods['cans_full'] = full_cans
        methods['torch_svd'] = lambda: (lambda U, s, Vh: (s[:8], Vh.T[:, :8]))(*torch.linalg.svd(X, full_matrices=False))
        methods['svd_lowrank'] = lambda: (lambda U, s, V: (s[:8], V[:, :8]))(*torch.svd_lowrank(X, q=16, niter=4))
    for name, fn in methods.items():
        r, out = timed(fn, reps if N < 4_000_000 else 1)
        if out is not None and S0 is not None:
            r.update(accuracy(out[0], out[1], S0, V0))
        if out is not None:
            r['top8_sv'] = [round(float(v), 2) for v in out[0][:8]]
        res[name] = r
        print('  N={:.0e} T={} {:<12} {}'.format(N, T, name, {k: v for k, v in r.items() if k != 'times_s'}),
              flush=True)
    if cans_svd is not None:
        R = la.tsqr(stream(), chunk_rows=chunk)
        r, _ = timed(lambda: cans_svd(R), reps)
        res['cans_on_R_only'] = r
        r, _ = timed(lambda: torch.linalg.svd(R), reps)
        res['torch_svd_on_R_only'] = r
    del X
    torch.cuda.empty_cache()
    return res


def bench_rank(N, T, chunk, dev, noise):
    F = factors(T, dev)
    src = chunk_source(N, T, chunk, F, dev, noise)
    in_mem = N <= 1_000_000
    X, in_mem = materialize(src, in_mem)
    stream = (lambda: X) if in_mem else src
    out = {}
    r, res = timed(lambda: la.estimate_rank_pa(stream(), q=16, dtype=torch.float32, chunk_rows=chunk), 1)
    r['rank'] = res[0] if res else None
    out['estimate_rank_pa'] = r
    Xs = X if in_mem else next(iter(src()))          # scale.estimate_rank subsamples 20k rows anyway
    r, res = timed(lambda: scale.estimate_rank(Xs, q=16, method='shift'), 1)
    r['rank'] = res[0] if res else None
    r['rows_seen'] = min(Xs.shape[0], 20000)
    out['scale_estimate_rank_shift'] = r
    r, res = timed(lambda: la.estimate_rank_pa(stream(), q=16, null='mp', dtype=torch.float32, chunk_rows=chunk), 1)
    r['rank'] = res[0] if res else None
    out['mp_edge'] = r
    print('  rank N={:.0e} T={} noise={} {}'.format(N, T, noise, {k: (v['rank'], round(v.get('time_s', -1), 3))
                                                                  for k, v in out.items()}), flush=True)
    del X
    torch.cuda.empty_cache()
    return out


def hard_case(label, scale_s, N, T, dev, sd, ks):
    """One seed of a `bench_rank_hard` case, appended to the lists in `ks`."""
    if scale_s is None:                            # common linear trend (amplitude 0.5) in white noise
        t = torch.arange(T, device=dev, dtype=torch.float32)
        F = 0.5 * ((t - t.mean()) / (t - t.mean()).norm() * math.sqrt(T))[None, :]
        X = make_chunk(0, N, T, F, dev, 'white', seed=100 + sd)
    else:
        X = make_chunk(0, N, T, factors(T, dev, seed=100 + sd) * scale_s, dev, 'ar97', seed=100 + sd)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        k, _, _, info = la.estimate_rank_pa(X, q=16, dtype=torch.float32, seed=sd, return_info=True)
    ks['estimate_rank_pa'].append(k)
    ks['untestable'].append(info['untestable'])
    ks['circulant'].append(la.estimate_rank_pa(X, q=16, null='circulant', dtype=torch.float32, seed=sd)[0])
    ks['scale_estimate_rank_shift'].append(scale.estimate_rank(X, q=16, method='shift', seed=sd)[0])
    ks['mp_edge'].append(la.estimate_rank_pa(X, q=16, null='mp', dtype=torch.float32)[0])


def bench_rank_hard(N, T, dev, seeds=5):
    """Where the nulls differ: AR(1) phi = 0.97 noise alone (true rank 0), under a weak rank-4 mode,
    and a common linear trend in white noise (true rank 1). A seed that runs out of memory on the
    shared GPU is recorded in `oom_seeds`."""
    out = {}
    for label, scale_s in (('ar97_pure_true0', 0.0), ('ar97_weak_true4', 0.12), ('trend_white_true1', None)):
        ks = {'estimate_rank_pa': [], 'untestable': [], 'circulant': [], 'scale_estimate_rank_shift': [],
              'mp_edge': [], 'oom_seeds': []}
        for sd in range(seeds):
            try:
                hard_case(label, scale_s, N, T, dev, sd, ks)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                ks['oom_seeds'].append(sd)
        out[label] = ks
        print('  rank_hard N={:.0e} T={} {} {}'.format(N, T, label, ks), flush=True)
    return out


def cgroup_cpus():
    """CPU quota of the cgroup (v2 cpu.max or v1 cfs files), else the affinity mask size."""
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()
        if quota != 'max':
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
            quota = int(f.read())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ts', type=int, nargs='+', default=[256, 1024])
    ap.add_argument('--Ns', type=float, nargs='+', default=[1e5, 1e6, 4e6])
    ap.add_argument('--chunk', type=int, default=250_000)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'bench_linalg.json'))
    ap.add_argument('--hard-only', action='store_true', help='only (re)run the hard rank cases, into --out')
    ap.add_argument('--tf32', choices=['env', 'off'], default='env',
                    help="'env': keep the environment's TF32 setting for the baselines (default); 'off': IEEE")
    ap.add_argument('--threads', type=int, default=None, help='torch CPU threads (default: cgroup CPU quota)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.tf32 == 'off':
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.set_num_threads(a.threads or cgroup_cpus())
    try:
        cans_svd = la.load_cans_svd()
    except ImportError:
        cans_svd = None
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().split('\n')
    except OSError:
        smi = []
    out = {'device': torch.cuda.get_device_name() if dev == 'cuda' else platform.processor(),
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'nvidia_smi_at_start': smi,
           'torch': torch.__version__, 'chunk_rows': a.chunk, 'strengths': STRENGTHS, 'svd': {}, 'rank': {}}
    if a.hard_only and os.path.exists(a.out):
        with open(a.out) as f:
            out = json.load(f)
    out.update({'tf32_mode': a.tf32, 'fp32_precision': torch.backends.cuda.matmul.fp32_precision,
                'torch_threads': torch.get_num_threads(), 'host_cpus': os.cpu_count(),
                'estimate_rank_pa_defaults': 'calibrate=2, margin=auto, null=toeplitz'})
    out['rank_hard'] = {'N100000_T{}'.format(T): bench_rank_hard(100_000, T, dev) for T in a.Ts}
    if a.hard_only:
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)
        print('wrote', a.out)
        return
    for T in a.Ts:
        for N in (int(n) for n in a.Ns):
            print('svd N={:.0e} T={}'.format(N, T), flush=True)
            try:
                out['svd']['N{}_T{}'.format(N, T)] = bench_svd(N, T, a.chunk, dev, a.reps, cans_svd)
            except torch.OutOfMemoryError as exc:          # shared GPU: record and go on
                torch.cuda.empty_cache()
                out['svd']['N{}_T{}'.format(N, T)] = {'error': 'OutOfMemoryError: ' + str(exc).split('\n')[0][:160]}
            for noise in ('white', 'ar97'):
                try:
                    out['rank']['N{}_T{}_{}'.format(N, T, noise)] = bench_rank(N, T, a.chunk, dev, noise)
                except torch.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    out['rank']['N{}_T{}_{}'.format(N, T, noise)] = {
                        'error': 'OutOfMemoryError: ' + str(exc).split('\n')[0][:160]}
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, 'w') as f:
                json.dump(out, f, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
