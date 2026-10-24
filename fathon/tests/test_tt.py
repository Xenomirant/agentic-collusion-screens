"""Tests for fathon.tensor.tt (TT / QTT arithmetic, TT-cross, replica tensors in QTT)."""

import itertools
import numpy as np
import pytest

from fathon.tensor import tt, replica, decomp


def rand_tt(dims, ranks, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    rs = [1] + list(ranks) + [1]
    return [rng.standard_normal((rs[k], n, rs[k + 1])).astype(dtype) for k, n in enumerate(dims)]


def rel(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b))


# t1_replica_rank.py draws atoms then probs from the same generator
_g = np.random.default_rng(0)
ATOMS4 = _g.uniform(0.2, 1.8, size=(4, 2))
PROBS4 = _g.dirichlet(np.ones(4))
BINARY = (np.array([[0.3, 0.7], [0.7, 0.3]]), np.array([0.5, 0.5]))


# ---------------------------------------------------------------- arithmetic

def test_dense_roundtrip_and_ranks():
    A = rand_tt([3, 4, 2, 5, 3], [3, 4, 5, 2])
    X = tt.tt_to_dense(A)
    B = tt.tt_from_dense(X)
    assert tt.tt_ranks(B) == [3, 4, 5, 2]
    assert rel(tt.tt_to_dense(B), X) < 1e-13
    C = tt.tt_from_dense(X, tol=1e-10)  # decomp.tt_svd path
    assert rel(tt.tt_to_dense(C), X) < 1e-12


def test_orthogonalize_left_right():
    A = rand_tt([3, 4, 5, 3], [3, 6, 2], seed=1)
    X = tt.tt_to_dense(A)
    L = tt.tt_orthogonalize(A, 'left')
    R = tt.tt_orthogonalize(A, 'right')
    for G in L[:-1]:
        M = G.reshape(-1, G.shape[2])
        assert np.allclose(M.T @ M, np.eye(M.shape[1]), atol=1e-12)
    for G in R[1:]:
        M = G.reshape(G.shape[0], -1)
        assert np.allclose(M @ M.T, np.eye(M.shape[0]), atol=1e-12)
    assert rel(tt.tt_to_dense(L), X) < 1e-13 and rel(tt.tt_to_dense(R), X) < 1e-13
    assert abs(np.linalg.norm(L[-1]) - np.linalg.norm(X)) < 1e-10 * np.linalg.norm(X)


def test_add_scale_hadamard_dot_norm():
    A = rand_tt([2, 3, 4, 3], [2, 3, 2], seed=2)
    B = rand_tt([2, 3, 4, 3], [3, 2, 2], seed=3)
    XA, XB = tt.tt_to_dense(A), tt.tt_to_dense(B)
    assert rel(tt.tt_to_dense(tt.tt_add(A, B)), XA + XB) < 1e-13
    assert rel(tt.tt_to_dense(tt.tt_add(A, B, weights=[2.0, -0.5])), 2 * XA - 0.5 * XB) < 1e-13
    assert rel(tt.tt_to_dense(tt.tt_scale(A, -3.0)), -3 * XA) < 1e-14
    H = tt.tt_hadamard(A, B)
    assert tt.tt_ranks(H) == [6, 6, 4]
    assert rel(tt.tt_to_dense(H), XA * XB) < 1e-13
    assert abs(tt.tt_dot(A, B) - np.sum(XA * XB)) < 1e-11 * np.linalg.norm(XA) * np.linalg.norm(XB)
    assert abs(tt.tt_norm(A) - np.linalg.norm(XA)) < 1e-12 * np.linalg.norm(XA)
    idx = np.array(list(itertools.product(range(2), range(3), range(4), range(3))))
    assert np.allclose(tt.tt_eval(A, idx), XA.reshape(-1), atol=1e-12)


def test_round_exact_and_eps_bound():
    A = rand_tt([3] * 8, [4] * 7, seed=4)
    # redundant representation: A + A - A has ranks 12 but true ranks <= 4
    R = tt.tt_round(tt.tt_add(A, A, A, weights=[1, 1, -1]), eps=1e-12)
    assert max(tt.tt_ranks(R)) <= 4
    assert tt.tt_norm(tt.tt_add(R, A, weights=[1, -1])) / tt.tt_norm(A) < 1e-11
    # a tensor with decaying singular structure: error <= eps for every eps
    terms = [tt.tt_scale(rand_tt([3] * 8, [1] * 7, seed=10 + j), 0.5 ** j) for j in range(25)]
    X = tt.tt_add(*terms)
    nX = tt.tt_norm(X)
    prev_rank = 10 ** 9
    for eps in [3e-1, 1e-1, 1e-2, 1e-4, 1e-6]:
        Y = tt.tt_round(X, eps=eps)
        err = tt.tt_norm(tt.tt_add(X, Y, weights=[1, -1])) / nX
        assert err <= eps * (1 + 1e-8)
        assert sum(tt.tt_ranks(Y)) <= sum(tt.tt_ranks(X))
        prev_rank = sum(tt.tt_ranks(Y))
    assert prev_rank > 7  # nontrivial at the finest eps
    Y = tt.tt_round(X, max_rank=2)
    assert max(tt.tt_ranks(Y)) <= 2


def test_round_quasi_optimal_vs_dense_tt_svd():
    X = rand_tt([4] * 6, [6] * 5, seed=5)
    D = tt.tt_to_dense(X)
    for eps in [1e-1, 1e-3]:
        Y = tt.tt_round(X, eps=eps)
        Z = tt.tt_from_dense(D, eps=eps)
        assert rel(tt.tt_to_dense(Y), D) <= eps
        assert tt.tt_ranks(Y) == tt.tt_ranks(Z)


def test_contract_vectors_and_sum_samples():
    A = rand_tt([2, 3, 4, 3, 2], [2, 3, 3, 2], seed=6)
    X = tt.tt_to_dense(A)
    v1, v3 = np.arange(1.0, 4.0), np.array([0.2, -1.0, 0.5])
    C = tt.tt_contract_vectors(A, {1: v1, 3: v3})
    assert rel(tt.tt_to_dense(C), np.einsum('abcde,b,d->ace', X, v1, v3)) < 1e-13
    C = tt.tt_contract_vectors(A, {0: np.ones(2), 4: np.ones(2), 3: v3})
    assert rel(tt.tt_to_dense(C), np.einsum('abcde,d->bc', X, v3)) < 1e-13
    s = tt.tt_contract_vectors(A, {k: np.ones(n) for k, n in enumerate(X.shape)})
    assert abs(s - X.sum()) < 1e-11 * np.abs(X).sum()
    terms = [rand_tt([2] * 6, [3] * 5, seed=20 + j) for j in range(30)]
    S = tt.tt_sum_samples(terms, eps=1e-13, round_at=20)
    ref = sum(tt.tt_to_dense(t) for t in terms)
    assert rel(tt.tt_to_dense(S), ref) < 1e-11
    assert max(tt.tt_ranks(S)) <= 8  # full rank of a (2^3 x 2^3) unfolding


def test_float32_and_torch_backend():
    A32 = rand_tt([2, 3, 4, 3], [2, 3, 2], seed=7, dtype=np.float32)
    R = tt.tt_round(A32, eps=1e-5)
    assert R[0].dtype == np.float32
    assert rel(tt.tt_to_dense(R), tt.tt_to_dense(A32).astype(np.float64)) < 1e-4
    torch = pytest.importorskip('torch')
    A = rand_tt([2, 3, 4, 3], [2, 3, 2], seed=8)
    At = [torch.as_tensor(G) for G in A]
    X = tt.tt_to_dense(A)
    assert torch.is_tensor(tt.tt_to_dense(At))
    assert rel(tt.tt_to_dense(tt.tt_round(tt.tt_add(At, At), 1e-12)).numpy(), 2 * X) < 1e-12
    assert abs(tt.tt_norm(At) - np.linalg.norm(X)) < 1e-10
    assert abs(tt.tt_dot(At, At) - np.sum(X * X)) < 1e-8
    s_np = tt.level_cut_spectra(A)
    s_t = tt.level_cut_spectra(At)
    assert all(np.allclose(a, b) for a, b in zip(s_np, s_t))


# ---------------------------------------------------------------- QTT

def test_qtt_vector_coarse_grain():
    rng = np.random.default_rng(9)
    n = 8
    v = rng.random(2 ** n)
    Q = tt.qtt_from_vector(v)
    assert rel(tt.tt_to_dense(Q).reshape(-1), v) < 1e-13
    # binomial cascade is QTT-rank 1
    w = np.array([0.3, 0.7])
    mu = w
    for _ in range(n - 1):
        mu = np.kron(mu, w)
    assert tt.tt_ranks(tt.qtt_from_vector(mu, eps=1e-12)) == [1] * (n - 1)
    for keep in [0, 3, 8]:
        cg = tt.qtt_coarse_grain(Q, keep, average=False)
        ref = v.reshape(2 ** keep, -1).sum(axis=1)
        got = np.atleast_1d(tt.tt_to_dense(cg) if keep else cg).reshape(-1)
        assert np.allclose(got, ref)
    cg = tt.qtt_coarse_grain(Q, 5, average=True)
    assert np.allclose(tt.tt_to_dense(cg).reshape(-1), v.reshape(32, -1).mean(axis=1))
    # base 3
    v3 = rng.random(3 ** 4)
    assert rel(tt.tt_to_dense(tt.qtt_from_vector(v3, base=3)).reshape(-1), v3) < 1e-13


@pytest.mark.parametrize('q', [1, 2, 3])
def test_qtt_interleave_matches_replica_interleave(q):
    rng = np.random.default_rng(10 + q)
    n = 4
    vs = [rng.random(2 ** n) for _ in range(q)]
    full = vs[0]
    for v in vs[1:]:
        full = np.multiply.outer(full, v)
    ref = replica.interleave(full, q, n)
    got = tt.tt_to_dense(tt.qtt_interleave([tt.qtt_from_vector(v) for v in vs]))
    assert got.shape == (2 ** q,) * n
    assert rel(got, ref) < 1e-13


# ---------------------------------------------------------------- maxvol / cross

def test_maxvol_dominance():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((200, 7))
    piv = tt.maxvol(A, tol=1.01)
    assert len(set(piv.tolist())) == 7
    B = A @ np.linalg.inv(A[piv])
    assert np.max(np.abs(B)) <= 1.01 + 1e-10
    # volume is no smaller than that of the LU start
    assert abs(np.linalg.det(A[piv])) >= abs(np.linalg.det(A[tt._lu_pivots(A)])) * (1 - 1e-12)


@pytest.mark.parametrize('d,n,r', [(8, 2, 3), (10, 4, 5), (12, 8, 4), (9, 3, 5)])
def test_tt_cross_recovers_low_rank(d, n, r):
    A = rand_tt([n] * d, [r] * (d - 1), seed=d * 100 + n)
    C, info = tt.tt_cross(lambda idx: tt.tt_eval(A, idx), [n] * d, eps=1e-12, max_rank=20)
    err = tt.tt_norm(tt.tt_add(A, C, weights=[1, -1])) / tt.tt_norm(A)
    assert err < 1e-10
    assert info['converged'] and max(info['ranks']) <= r
    # evaluations per full sweep ~ 2 (d - 1) n^2 r^2 at most
    assert info['n_evals'] <= info['sweeps'] * 2 * (d - 1) * (n * r) ** 2 + 1
    assert info['sweeps'] <= 3


def test_tt_cross_mixed_dims_and_function():
    dims = [5, 3, 6, 4, 7, 2]
    grids = [np.linspace(0, 1, m) for m in dims]

    def f(idx):  # sin of a sum: TT rank 2
        return np.sin(sum(g[idx[:, k]] for k, g in enumerate(grids)))
    C, info = tt.tt_cross(f, dims, eps=1e-12)
    X = np.sin(sum(np.meshgrid(*grids, indexing='ij')))
    assert rel(tt.tt_to_dense(C), X) < 1e-11
    assert max(info['ranks']) == 2


# ---------------------------------------------------------------- replica tensors

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
        M = tt.cascade_moment_tt(ATOMS4, PROBS4, q, n)
        assert tt.level_cut_ranks(M, 1e-10) == exp[q]
    # depth independence (dense would need 2**(4*12) entries)
    M = tt.cascade_moment_tt(ATOMS4, PROBS4, 4, 12)
    r = tt.level_cut_ranks(M, 1e-10)
    assert max(r) == 15 and r[1:-2] == [15] * (len(r) - 3)


def test_cascade_moment_is_mean_of_samples():
    atoms, probs = BINARY
    n, q = 4, 2
    mu = tt.cascade_sample(atoms, probs, n, 40000, seed=3)
    emp = replica.replica_tensor(mu, q)
    M = tt.tt_to_dense(tt.cascade_moment_tt(atoms, probs, q, n))
    assert rel(emp, M) < 0.02


def test_level_cut_spectra_match_dense():
    rng = np.random.default_rng(12)
    mu = rng.random((30, 2 ** 5))
    T = replica.replica_tensor(mu, 2)
    C = tt.replica_tt(mu, 2, eps=1e-14, method='sum')
    sp_tt = tt.level_cut_spectra(C)
    sp_dense = decomp.cut_spectra(T)
    for s, p in zip(sp_tt, sp_dense):
        pt = s ** 2 / np.sum(s ** 2)
        k = min(len(pt), len(p))
        assert np.allclose(pt[:k], p[:k], atol=1e-12)
    assert tt.level_cut_ranks(C, 1e-8) == decomp.tt_ranks(T, 1e-8)


@pytest.mark.parametrize('q,n', [(2, 6), (3, 4), (3, 5)])
def test_replica_tt_sum_and_cross_vs_dense(q, n):
    mu = tt.cascade_sample(ATOMS4, PROBS4, n, 25, seed=q * 10 + n)
    T = replica.replica_tensor(mu, q)
    Cs = tt.replica_tt(mu, q, eps=1e-13, method='sum', max_rank=None)
    assert rel(tt.tt_to_dense(Cs), T) < 1e-10
    eps = 1e-4
    Cc, info = tt.replica_tt(mu, q, eps=eps, method='cross', max_rank=600, return_info=True)
    assert rel(tt.tt_to_dense(Cc), T) < 10 * eps
    assert info['n_evals'] > 0


def test_cross_replica_tt_vs_dense():
    rng = np.random.default_rng(13)
    n = 5
    mus = [rng.random((20, 2 ** n)) for _ in range(3)]
    T = replica.cross_replica_tensor(mus)
    Cs = tt.cross_replica_tt(mus, eps=1e-13, method='sum', max_rank=None)
    assert rel(tt.tt_to_dense(Cs), T) < 1e-10
    Cc = tt.cross_replica_tt(mus, eps=1e-6, method='cross', max_rank=600)
    assert rel(tt.tt_to_dense(Cc), T) < 1e-5
    # two agents: q = 2
    T2 = replica.cross_replica_tensor(mus[:2])
    assert rel(tt.tt_to_dense(tt.cross_replica_tt(mus[:2], eps=1e-12, max_rank=100)), T2) < 1e-10


def test_replica_tt_cross_recovers_exact_law_beyond_dense():
    # black-box cross on the exact-law entries at n = 10, q = 3 (2**30 dense entries)
    M = tt.cascade_moment_tt(ATOMS4, PROBS4, 3, 10)
    C, info = tt.tt_cross(lambda idx: tt.tt_eval(M, idx), [8] * 10, eps=1e-12, max_rank=30)
    assert tt.tt_norm(tt.tt_add(C, M, weights=[1, -1])) / tt.tt_norm(M) < 1e-10
    assert max(tt.level_cut_ranks(C, 1e-10)) == 5
    assert info['n_evals'] < 2 ** 30 / 1000


def test_replica_tt_cross_on_samples_torch_matches_numpy():
    torch = pytest.importorskip('torch')
    mu = tt.cascade_sample(*BINARY, 6, 50, seed=4)
    a = tt.replica_tt(mu, 2, eps=1e-10, max_rank=64)
    b = tt.replica_tt(torch.as_tensor(mu), 2, eps=1e-10, max_rank=64)
    assert rel(tt.tt_to_dense(a), tt.tt_to_dense(b)) < 1e-9
    assert rel(tt.tt_to_dense(a), replica.replica_tensor(mu, 2)) < 1e-8


def test_sum_method_refuses_incompressible_samples():
    mu = tt.cascade_sample(*BINARY, 10, 2, seed=5)
    with pytest.raises(ValueError):
        tt.replica_tt(mu, 4, method='sum', max_sample_rank=1000)
