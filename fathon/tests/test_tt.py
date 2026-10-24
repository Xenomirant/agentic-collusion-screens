"""Tests for fathon.tensor.tt (replica tensors in QTT on top of teneva / torchtt).

Generic TT arithmetic is teneva's and is not re-tested here; the tests cover
the project conventions of the thin wrappers, the converters, and the
project-specific numerics against dense references.
"""

import itertools
import os
import warnings
import numpy as np
import pytest

teneva = pytest.importorskip('teneva')  # optional dependencies of fathon.tensor.tt only
pytest.importorskip('torchtt')
torch = pytest.importorskip('torch')

from fathon.tensor import tt, replica, decomp  # noqa: E402


def rand_tt(dims, ranks, seed=0):
    rng = np.random.default_rng(seed)
    rs = [1] + list(ranks) + [1]
    return [rng.standard_normal((rs[k], n, rs[k + 1])) for k, n in enumerate(dims)]


def rel(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b))


def product_measures(n, S, seed=0):
    """S random binomial measures with per-level weights: QTT rank 1 each."""
    w = np.random.default_rng(seed).uniform(0.2, 0.8, size=(S, n))
    mu = np.ones((S, 1))
    for k in range(n):
        mu = (mu[:, :, None] * np.stack([w[:, k], 1 - w[:, k]], axis=1)[:, None, :]).reshape(S, -1)
    return mu


# t1_replica_rank.py draws atoms then probs from the same generator
_g = np.random.default_rng(0)
ATOMS4 = _g.uniform(0.2, 1.8, size=(4, 2))
PROBS4 = _g.dirichlet(np.ones(4))
BINARY = (np.array([[0.3, 0.7], [0.7, 0.3]]), np.array([0.5, 0.5]))


@pytest.fixture(scope='module', autouse=True)
def _warm_torch():
    # torch loads linalg / einsum backends lazily (tens of seconds on a cold, loaded filesystem);
    # pay it once here instead of inside the first test that rounds or runs the dmrg cross
    tt.replica_tt(product_measures(3, 2), 2, eps=1e-10, n_sweeps=1)


# ---------------------------------------------------------------- wrapper conventions

def test_from_dense_and_round_relative_accuracy():
    A = rand_tt([3, 4, 2, 5, 3], [3, 4, 5, 2])
    X = tt.tt_to_dense(A)
    B = tt.tt_from_dense(X)
    assert tt.tt_ranks(B) == [3, 4, 5, 2]
    assert rel(tt.tt_to_dense(B), X) < 1e-13
    # redundant A + A - A (ranks 12) rounds back to ranks <= 4
    A = rand_tt([3] * 8, [4] * 7, seed=4)
    R = tt.tt_round(teneva.add(teneva.add(A, A), teneva.mul(A, -1.0)), eps=1e-12)
    assert max(tt.tt_ranks(R)) <= 4 and tt.tt_rel_err(R, A) < 1e-11
    # relative eps bound, monotone ranks, and the same ranks as TT-SVD of the dense tensor
    X = rand_tt([4] * 6, [6] * 5, seed=5)
    D = tt.tt_to_dense(X)
    for eps in [3e-1, 1e-1, 1e-3]:
        Y = tt.tt_round(X, eps=eps)
        assert rel(tt.tt_to_dense(Y), D) <= eps
        assert tt.tt_ranks(Y) == tt.tt_ranks(tt.tt_from_dense(D, eps=eps))
    assert max(tt.tt_ranks(tt.tt_round(X, max_rank=2))) <= 2


def test_tt_norm_resolves_small_differences():
    A = rand_tt([3] * 8, [4] * 7, seed=1)
    B = teneva.copy(A)
    B[3] = B[3] * (1 + 1e-11)
    exact = 1e-11 * np.linalg.norm(tt.tt_to_dense(A))  # only core 3 changed, by a factor
    assert abs(tt.tt_norm(teneva.sub(B, A)) - exact) < 1e-3 * exact
    assert abs(tt.tt_norm(A) - np.linalg.norm(tt.tt_to_dense(A))) < 1e-12 * tt.tt_norm(A)


def test_sum_samples():
    terms = [rand_tt([2] * 6, [3] * 5, seed=20 + j) for j in range(30)]
    S = tt.tt_sum_samples(iter(terms), eps=1e-13, round_at=20)
    assert rel(tt.tt_to_dense(S), sum(tt.tt_to_dense(t) for t in terms)) < 1e-11
    assert max(tt.tt_ranks(S)) <= 8  # full rank of a (2^3 x 2^3) unfolding
    # d = 1 (teneva.add would give a (1, n, 2) core)
    one = tt.tt_sum_samples([[np.array([1., 2, 3]).reshape(1, 3, 1)], [np.array([10., 20, 30]).reshape(1, 3, 1)]])
    assert len(one) == 1 and one[0].shape == (1, 3, 1) and np.allclose(one[0].ravel(), [11, 22, 33])


def test_torchtt_converters():
    A = rand_tt([2, 3, 4, 3], [2, 3, 2], seed=8)
    X = tt.tt_to_dense(A)
    T = tt.to_torchtt(A)
    assert T.R == [1, 2, 3, 2, 1] and list(T.N) == [2, 3, 4, 3]
    assert rel(T.full().numpy(), X) < 1e-14
    back = tt.from_torchtt(T)
    assert all(np.array_equal(a, b) for a, b in zip(A, back))
    # rounding through torchtt agrees with teneva's
    R = tt.from_torchtt((T + T).round(1e-12))
    assert tt.tt_ranks(R) == [2, 3, 2] and rel(tt.tt_to_dense(R), 2 * X) < 1e-12
    # torch cores are accepted by the converter and the wrappers
    At = [torch.as_tensor(G) for G in A]
    assert rel(tt.to_torchtt(At).full().numpy(), X) < 1e-14
    assert rel(tt.tt_to_dense(At), X) < 1e-14 and tt.tt_ranks(At) == [2, 3, 2]
    assert abs(tt.tt_norm(At) - np.linalg.norm(X)) < 1e-12 * np.linalg.norm(X)
    assert rel(tt.tt_to_dense(tt.tt_round(At, 1e-12)), X) < 1e-12 and tt.tt_rel_err(At, A) < 1e-14
    if os.environ.get('FATHON_TEST_CUDA') and torch.cuda.is_available():  # opt-in: CUDA init is slow
        G = tt.to_torchtt(A, device='cuda')
        assert G.cores[0].is_cuda
        assert rel(tt.tt_to_dense(tt.from_torchtt(G.round(1e-12))), X) < 1e-12
        Ac = [G.cuda() for G in At]
        assert tt.to_torchtt(Ac).cores[0].is_cuda
        assert rel(tt.tt_to_dense(tt.tt_round(Ac, 1e-12)), X) < 1e-12 and rel(tt.tt_to_dense(Ac), X) < 1e-14


@pytest.mark.parametrize('backend', ['dmrg', 'teneva'])
def test_tt_cross_wraps_batched_evaluator(backend):
    dims = [5, 3, 6, 4, 7, 2]
    grids = [np.linspace(0, 1, m) for m in dims]
    calls = []

    def f(idx):  # sin of a sum: TT rank 2
        calls.append(idx.shape[0])
        assert idx.dtype == np.int64 and idx.shape[1] == len(dims)
        return np.sin(sum(g[idx[:, k]] for k, g in enumerate(grids)))
    C, info = tt.tt_cross(f, dims, eps=1e-12, backend=backend)
    X = np.sin(sum(np.meshgrid(*grids, indexing='ij')))
    assert rel(tt.tt_to_dense(C), X) < 1e-11
    assert info['ranks'] == [2] * 5 and info['converged'] and info['stop'] == 'e'
    assert info['n_evals'] == sum(calls)
    if backend == 'teneva':
        assert info['n_evals'] < X.size  # cache: every entry requested once


def test_tt_cross_d1_and_option_checks():
    f = lambda idx: (idx[:, 0] + 1.0) ** 2  # noqa: E731
    for be in ['dmrg', 'teneva']:  # d = 1: the vector itself (dmrg crashed here)
        C, info = tt.tt_cross(f, [5], backend=be)
        assert len(C) == 1 and C[0].shape == (1, 5, 1) and np.allclose(C[0].ravel(), [1, 4, 9, 16, 25])
        assert info['n_evals'] == 5
    g = lambda idx: np.ones(idx.shape[0])  # noqa: E731
    with pytest.raises(TypeError):  # old positional order (f, dims, max_rank, eps) must not be reinterpreted
        tt.tt_cross(g, [4] * 4, 20)
    with pytest.raises(TypeError):  # option of the other backend
        tt.tt_cross(g, [4] * 4, backend='dmrg', init_rank=7)
    with pytest.raises(TypeError):
        tt.tt_cross(g, [4] * 4, backend='teneva', kick=3)
    with pytest.raises(ValueError):
        tt.tt_cross(g, [4] * 4, backend='nope')
    mu = product_measures(3, 2)
    with pytest.raises(TypeError):  # cross_replica_tt(mus, eps, max_rank, ...) positional
        tt.cross_replica_tt([mu, mu], 1e-6, 10)
    with pytest.raises(TypeError):
        tt.replica_tt(mu, 2, 1e-6, 10)


def test_tt_cross_dmrg_seeds_only_its_own_generator(monkeypatch):
    # torch.manual_seed would also (lazily) reseed every CUDA generator of the caller
    called = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda s: called.append(s))
    monkeypatch.setattr(torch.cuda, 'manual_seed', lambda s: called.append(s))
    state = torch.get_rng_state()
    M = tt.cascade_moment_tt(ATOMS4, PROBS4, 2, 5)
    a, _ = tt.tt_cross(lambda idx: teneva.get_many(M, idx), [4] * 5, eps=1e-12, seed=3)
    b, _ = tt.tt_cross(lambda idx: teneva.get_many(M, idx), [4] * 5, eps=1e-12, seed=3)
    assert not called and torch.equal(torch.get_rng_state(), state)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))  # reproducible from the seed
    monkeypatch.undo()
    if os.environ.get('FATHON_TEST_CUDA') and torch.cuda.is_available():  # opt-in: CUDA init is slow
        torch.cuda.manual_seed(123)
        cstate = torch.cuda.get_rng_state()
        tt.tt_cross(lambda idx: teneva.get_many(M, idx), [4] * 5, eps=1e-12, seed=3)
        assert torch.equal(torch.cuda.get_rng_state(), cstate)
        tt.tt_cross(lambda idx: teneva.get_many(M, idx), [4] * 5, eps=1e-12, seed=3, core_device='cuda')
        assert torch.equal(torch.cuda.get_rng_state(), cstate)


def test_tt_cross_dmrg_max_evals():
    # after the first sweep the budget is never exceeded; the last complete sweep is returned
    M = tt.cascade_moment_tt(ATOMS4, PROBS4, 3, 8)
    f = lambda idx: teneva.get_many(M, idx)  # noqa: E731
    _, one = tt.tt_cross(f, [8] * 8, eps=1e-12, n_sweeps=1)
    _, full = tt.tt_cross(f, [8] * 8, eps=1e-12)
    assert full['n_evals'] > 2 * one['n_evals']
    budget = one['n_evals'] + (full['n_evals'] - one['n_evals']) // 3
    C, info = tt.tt_cross(f, [8] * 8, eps=1e-12, max_evals=budget)
    assert info['stop'] == 'm' and not info['converged'] and info['n_evals'] <= budget
    assert info['sweeps'] >= 1 and len(C) == 8
    _, small = tt.tt_cross(f, [8] * 8, eps=1e-12, max_evals=10)  # the first sweep always completes
    assert small['n_evals'] == one['n_evals'] and small['stop'] == 'm'


# ---------------------------------------------------------------- QTT

def test_qtt_vector_coarse_grain():
    rng = np.random.default_rng(9)
    n = 8
    v = rng.random(2 ** n)
    Q = tt.qtt_from_vector(v)
    assert rel(tt.tt_to_dense(Q).reshape(-1), v) < 1e-13
    assert tt.tt_ranks(tt.qtt_from_vector(product_measures(n, 1)[0], eps=1e-12)) == [1] * (n - 1)
    for keep in [0, 3, 8]:
        cg = tt.qtt_coarse_grain(Q, keep, average=False)
        ref = v.reshape(2 ** keep, -1).sum(axis=1)
        got = np.atleast_1d(tt.tt_to_dense(cg) if keep else cg).reshape(-1)
        assert np.allclose(got, ref)
    cg = tt.qtt_coarse_grain(Q, 5, average=True)
    assert np.allclose(tt.tt_to_dense(cg).reshape(-1), v.reshape(32, -1).mean(axis=1))
    v3 = rng.random(3 ** 4)
    assert rel(tt.tt_to_dense(tt.qtt_from_vector(v3, base=3)).reshape(-1), v3) < 1e-13


def test_coarse_grained_replica_tensor():
    # summing the fine levels of an interleaved replica TT = replica tensor of the coarse measure
    mu = tt.cascade_sample(ATOMS4, PROBS4, 5, 20, seed=2)
    C = tt.replica_tt(mu, 2, eps=1e-14, method='sum')
    coarse = mu.reshape(20, 2 ** 3, -1).sum(axis=-1)
    assert rel(tt.tt_to_dense(tt.qtt_coarse_grain(C, 3, average=False)), replica.replica_tensor(coarse, 2)) < 1e-12


@pytest.mark.parametrize('q', [1, 2, 3])
def test_qtt_interleave_matches_replica_interleave(q):
    rng = np.random.default_rng(10 + q)
    n = 4
    vs = [rng.random(2 ** n) for _ in range(q)]
    full = vs[0]
    for v in vs[1:]:
        full = np.multiply.outer(full, v)
    got = tt.tt_to_dense(tt.qtt_interleave([tt.qtt_from_vector(v) for v in vs]))
    assert got.shape == (2 ** q,) * n
    assert rel(got, replica.interleave(full, q, n)) < 1e-13


# ---------------------------------------------------------------- exact cascade law

def _brute_moment_tensor(atoms, probs, n, q):
    """Copy of t1_replica_rank.moment_tensor (exact enumeration)."""
    b = 2

    def mixed_moment(m):
        return float(np.sum(probs * np.prod(atoms ** np.array(m)[None, :], axis=1)))
    T = np.zeros((b ** q,) * n)
    for idx in itertools.product(range(b ** q), repeat=n):
        letters = np.array([[(s >> a) & 1 for a in range(q)] for s in idx])
        val = 1.0
        for k in range(n):
            groups = {}
            for a in range(q):
                groups.setdefault(tuple(letters[:k, a]), []).append(a)
            for g in groups.values():
                m = [0] * b
                for a in g:
                    m[letters[k, a]] += 1
                val *= mixed_moment(m)
        T[idx] = val
    return T


@pytest.mark.parametrize('q,n', [(1, 5), (2, 4), (3, 3), (4, 2)])
def test_cascade_moment_tt_matches_enumeration(q, n):
    M = tt.cascade_moment_tt(ATOMS4, PROBS4, q, n)
    assert rel(tt.tt_to_dense(M), _brute_moment_tensor(ATOMS4, PROBS4, n, q)) < 1e-12


def test_cascade_moment_tt_bell_ranks():
    exp = {2: [2] * 6, 3: [4, 5, 5, 5], 4: [8, 15, 13]}
    for q, n in [(2, 7), (3, 5), (4, 4)]:
        assert tt.level_cut_ranks(tt.cascade_moment_tt(ATOMS4, PROBS4, q, n), 1e-10) == exp[q]
    # depth independence (dense would need 2**(4*12) entries)
    r = tt.level_cut_ranks(tt.cascade_moment_tt(ATOMS4, PROBS4, 4, 12), 1e-10)
    assert max(r) == 15 and r[1:-2] == [15] * (len(r) - 3)


def test_cascade_moment_is_mean_of_samples():
    atoms, probs = BINARY
    mu = tt.cascade_sample(atoms, probs, 4, 40000, seed=3)
    M = tt.tt_to_dense(tt.cascade_moment_tt(atoms, probs, 2, 4))
    assert rel(replica.replica_tensor(mu, 2), M) < 0.02


def test_cross_recovers_bell_ranks_of_exact_law():
    # black-box teneva.cross on the exact-law entries, n = 10, q = 3 (2**30 dense entries)
    M = tt.cascade_moment_tt(ATOMS4, PROBS4, 3, 10)
    C, info = tt.tt_cross(lambda idx: teneva.get_many(M, idx), [8] * 10, eps=1e-12)
    assert tt.tt_rel_err(C, M) < 1e-10
    assert tt.level_cut_ranks(C, 1e-10) == [4] + [5] * 8
    assert info['n_evals'] < 2 ** 30 / 10000


# ---------------------------------------------------------------- empirical replica tensors

def test_level_cut_spectra_match_dense():
    mu = np.random.default_rng(12).random((30, 2 ** 5))
    T = replica.replica_tensor(mu, 2)
    C = tt.replica_tt(mu, 2, eps=1e-14, method='sum')
    for s, p in zip(tt.level_cut_spectra(C), decomp.cut_spectra(T)):
        pt = s ** 2 / np.sum(s ** 2)
        k = min(len(pt), len(p))
        assert np.allclose(pt[:k], p[:k], atol=1e-12)
    assert tt.level_cut_ranks(C, 1e-8) == decomp.tt_ranks(T, 1e-8)


@pytest.mark.parametrize('q,n', [(2, 6), (3, 4), (3, 5)])
def test_replica_tt_sum_vs_dense(q, n):
    mu = tt.cascade_sample(ATOMS4, PROBS4, n, 25, seed=q * 10 + n)
    C = tt.replica_tt(mu, q, eps=1e-13, method='sum')
    assert rel(tt.tt_to_dense(C), replica.replica_tensor(mu, q)) < 1e-10


@pytest.mark.parametrize('backend', ['dmrg', 'teneva'])
@pytest.mark.parametrize('q,n', [(2, 6), (3, 4), (3, 5)])
def test_replica_tt_cross_exact_high_rank_vs_dense(q, n, backend):
    # exact empirical tensor of 25 cascade samples: level-cut ranks up to the unfolding size (64),
    # beyond teneva's stock rank ceiling (~42) and cache-saturation stop, which ended at 2e-2..1e-1
    mu = tt.cascade_sample(ATOMS4, PROBS4, n, 25, seed=q * 10 + n)
    T = replica.replica_tensor(mu, q)
    C, info = tt.replica_tt(mu, q, eps=1e-4, backend=backend, return_info=True)
    assert rel(tt.tt_to_dense(C), T) < 10 * 1e-4
    assert info['converged'] and info['stop'] == 'e'


def test_tt_cross_teneva_warns_when_stopped_early():
    mu = tt.cascade_sample(ATOMS4, PROBS4, 5, 25, seed=35)
    with pytest.warns(RuntimeWarning, match='reachable rank'):  # stock teneva settings: rank ceiling ~42 < 64
        C, info = tt.replica_tt(mu, 3, eps=1e-4, backend='teneva', n_sweeps=10, m_cache_scale=5,
                                return_info=True)
    assert not info['converged'] and info['stop'] in ('conv', 'nswp')
    assert rel(tt.tt_to_dense(C), replica.replica_tensor(mu, 3)) > 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # converged runs do not warn
        tt.tt_cross(lambda idx: idx.sum(1) + 1.0, [3] * 4, eps=1e-12, backend='teneva')


@pytest.mark.parametrize('backend', ['teneva', 'dmrg'])
@pytest.mark.parametrize('q', [2, 3])
def test_replica_tt_cross_vs_dense_n6(q, backend):
    # S rank-1 samples: the empirical tensor has level-cut ranks <= S, so cross is exact
    mu = product_measures(6, 5, seed=q)
    T = replica.replica_tensor(mu, q)
    C, info = tt.replica_tt(mu, q, eps=1e-12, backend=backend, return_info=True)
    assert rel(tt.tt_to_dense(C), T) < 1e-10
    assert tt.level_cut_ranks(C, 1e-10) == decomp.tt_ranks(T, 1e-10)
    assert 0 < info['n_evals'] and info['backend'] == backend
    if backend == 'teneva':
        assert info['n_evals'] < T.size


def _predicted_noise(mu, q, law):
    z2 = np.sum(mu ** 2, axis=1)
    nT2 = tt.tt_norm(law) ** 2
    return np.sqrt((np.mean(z2 ** q) - nT2) / mu.shape[0] / nT2)


@pytest.mark.parametrize('q,n', [(2, 6), (3, 5)])
def test_replica_tt_cross_noisy_samples_meets_eps(q, n):
    # cascade samples: law (Bell ranks) + sampling noise; eps above the noise (default dmrg backend)
    mu = tt.cascade_sample(*BINARY, n, 4000, seed=q)
    T = replica.replica_tensor(mu, q)
    law = tt.cascade_moment_tt(*BINARY, q, n)
    noise = _predicted_noise(mu, q, law)
    assert abs(noise - rel(T, tt.tt_to_dense(law))) < 0.3 * noise  # the noise prediction is right
    eps = max(0.05, 2 * noise)
    C = tt.replica_tt(mu, q, eps=eps)
    # cross carries no a-priori bound; observed errors are 0.5-1.2 eps, so allow 1.5 eps
    assert rel(tt.tt_to_dense(C), T) < 1.5 * eps
    assert tt.tt_rel_err(C, law) < 1.5 * eps + noise
    assert max(tt.tt_ranks(C)) <= 2 ** q  # law-sized ranks, not the noise's full rank


def test_cross_replica_tt_vs_dense():
    rng = np.random.default_rng(13)
    n = 5
    mus = [rng.random((20, 2 ** n)) for _ in range(3)]
    T = replica.cross_replica_tensor(mus)
    assert rel(tt.tt_to_dense(tt.cross_replica_tt(mus, eps=1e-13, method='sum')), T) < 1e-10
    # full-rank random measures (ranks up to 64): both backends reach eps
    for be in ['dmrg', 'teneva']:
        assert rel(tt.tt_to_dense(tt.cross_replica_tt(mus, eps=1e-6, backend=be)), T) < 1e-5
    pm = [product_measures(n, 4, seed=s) for s in range(3)]
    pm = [m[[0, 1, 2, 3]] for m in pm]
    T = replica.cross_replica_tensor(pm)
    C = tt.cross_replica_tt(pm, eps=1e-12)
    assert rel(tt.tt_to_dense(C), T) < 1e-10


@pytest.mark.parametrize('backend', ['teneva', 'dmrg'])
def test_replica_tt_cross_large_n12_q3(backend):
    # 2**36 dense entries; S = 4 rank-1 samples give ranks <= 4; exact reference by the sum method
    mu = product_measures(12, 4, seed=7)
    ref = tt.replica_tt(mu, 3, eps=1e-13, method='sum')
    C, info = tt.replica_tt(mu, 3, eps=1e-11, backend=backend, return_info=True)
    assert max(info['ranks']) <= 4
    assert tt.level_cut_ranks(C, 1e-9) == tt.level_cut_ranks(ref, 1e-9) == [4] * 11
    assert tt.tt_rel_err(C, ref) < 1e-9
    assert info['n_evals'] < 2 ** 36 / 10 ** 5


def test_replica_tt_cross_torch_evaluator_matches_numpy():
    mu = product_measures(6, 5, seed=4)
    a = tt.replica_tt(mu, 2, eps=1e-12)
    b = tt.replica_tt(torch.as_tensor(mu), 2, eps=1e-12)
    assert rel(tt.tt_to_dense(a), tt.tt_to_dense(b)) < 1e-10
    assert rel(tt.tt_to_dense(a), replica.replica_tensor(mu, 2)) < 1e-10


def test_replica_tt_n1():
    mu = tt.cascade_sample(ATOMS4, PROBS4, 1, 100, seed=6)
    T = replica.replica_tensor(mu, 2)
    for kw in [dict(method='sum'), dict(backend='dmrg'), dict(backend='teneva')]:
        C = tt.replica_tt(mu, 2, eps=1e-12, **kw)
        assert len(C) == 1 and C[0].shape == (1, 4, 1)
        assert rel(tt.tt_to_dense(C), T) < 1e-12


def test_sum_method_refuses_incompressible_samples():
    mu = tt.cascade_sample(*BINARY, 10, 2, seed=5)
    with pytest.raises(ValueError):
        tt.replica_tt(mu, 4, method='sum', max_sample_rank=1000)
