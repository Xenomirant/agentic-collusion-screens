"""Regression tests for the review fixes of fathon.tensor (2026-10-23).

- influence_z: the permutation null is anti-conservative for agents with
  memory beyond the state; the Markov bootstrap null is calibrated.
- covariance_tensor matmul form equals the per-window mean.
- pairwise_matrix fast path equals the per-pair group_cut loop.
- robustness edges: cut_stats with empty rows, words overflow / alphabet
  check, cell_measure length check, mean_corr with a constant agent,
  cp_als with n_iter=0, rolling / shift_null short-input errors.
"""

import itertools
import warnings
import numpy as np
import pytest
from fathon.tensor import detrend, mvdfa, decomp, factorization as fz, replica, ews, influence as inf


def sticky(T, rng, stay=0.8, nA=3):
    a = np.zeros(T, dtype=int)
    a[0] = rng.integers(0, nA)
    for t in range(1, T):
        a[t] = a[t - 1] if rng.random() < stay else rng.integers(0, nA)
    return a


def test_markov_null_calibrated_for_sticky_agents():
    # independent sticky agents, exogenous public state: permutation null
    # rejects far too often, the Markov null stays near the nominal level
    rng = np.random.default_rng(3)
    R, T = 40, 500
    rej = {'permutation': 0, 'markov': 0}
    for rep in range(R):
        A = np.stack([sticky(T, rng) for _ in range(3)])
        S = rng.integers(0, 3, T)
        for null in rej:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _, P = inf.influence_z(A, S=S, nPerm=59, seed=rep, null=null)
            rej[null] += P[0, 1] < 0.05
    assert rej['permutation'] / R > 0.3
    assert rej['markov'] / R <= 0.15


def test_markov_null_keeps_power_and_warning_fires():
    rng = np.random.default_rng(5)
    T = 600
    a = sticky(T, rng)
    b = a.copy()
    flip = rng.random(T) < 0.5
    b[flip] = rng.integers(0, 3, flip.sum())        # b copies a half of the time
    c = sticky(T, rng)
    A = np.stack([a, b, c])
    with pytest.warns(RuntimeWarning):
        inf.influence_z(A, nPerm=59, seed=0, null='permutation')
    Z, P = inf.influence_z(A, nPerm=99, seed=0, null='markov')
    assert P[0, 1] < 0.05 and P[0, 2] > 0.05
    mem = inf.memory_beyond_state(A)
    assert np.all(mem > 0.02)
    assert np.all(inf.memory_beyond_state(rng.integers(0, 3, (3, T))) < 0.02)


def test_markov_bootstrap_preserves_policy():
    rng = np.random.default_rng(1)
    y = sticky(4000, rng, stay=0.9)
    Yb = inf.markov_bootstrap(y, None, rng, 20, order=1)
    stay_obs = np.mean(y[1:] == y[:-1])
    stay_boot = np.mean(Yb[:, 1:] == Yb[:, :-1])
    assert abs(stay_obs - stay_boot) < 0.03
    Yp = inf.conditional_permutation(y, None, rng, size=20)
    assert np.mean(Yp[:, 1:] == Yp[:, :-1]) < 0.45       # permutation destroys the memory


def test_conditional_mi_vectorised_matches_loop():
    rng = np.random.default_rng(2)
    x, y, s = rng.integers(0, 3, 500), rng.integers(0, 4, 500), rng.integers(0, 5, 500)
    ref = 0.0
    n = x.size
    for k in range(5):
        m = s == k
        if m.sum() == 0:
            continue
        C = np.zeros((3, 4))
        np.add.at(C, (x[m], y[m]), 1.0)
        P = C / C.sum()
        pa, pb = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        ref += m.sum() / n * np.sum(P[nz] * np.log(P[nz] / (pa @ pb)[nz]))
    assert abs(inf.conditional_mi(x, y, s) - ref) < 1e-12
    Cb = np.stack([inf._counts(x, y, s, 3, 4, 5)] * 3)
    assert np.allclose(inf._mi_from_counts(Cb), ref)


def test_covariance_tensor_matmul_equals_window_mean():
    rng = np.random.default_rng(4)
    Y = detrend.profile(rng.standard_normal((5, 3000)))
    sc = np.array([16, 32, 64, 128])
    for overlap in (False, True):
        for revSeg in (False, True):
            F2 = mvdfa.covariance_tensor(Y, sc, 2, revSeg, overlap)
            ref = np.stack([mvdfa.window_covariances(Y, s, 2, revSeg, overlap).mean(0) for s in sc])
            assert np.allclose(F2, ref, rtol=1e-10, atol=1e-14)


def test_pairwise_matrix_equals_group_cut_loop():
    rng = np.random.default_rng(6)
    S = rng.integers(0, 3, (6, 800))
    S[1, 1:] = np.where(rng.random(799) < 0.5, S[0, :-1], S[1, 1:])
    T = S.shape[1]
    for lag in (0, 1):
        for debias in (False, True):
            M = fz.pairwise_matrix(S, 'phi2', k=2, lag=lag, debias=debias)
            pairs = itertools.permutations(range(6), 2) if lag else itertools.combinations(range(6), 2)
            for i, j in pairs:
                v = fz.group_cut(S, [i], [j], 2, None, lag)['phi2']
                if debias:
                    Sr = S.copy()
                    Sr[j] = np.roll(S[j], T // 2)
                    v -= fz.group_cut(Sr, [i], [j], 2, None, lag)['phi2']
                assert abs(M[i, j] - v) < 1e-12
                if not lag:
                    assert M[j, i] == M[i, j]


def test_cut_stats_with_empty_rows_and_columns():
    P = np.array([[0.5, 0.0], [0.0, 0.0], [0.0, 0.5]])
    st = fz.cut_stats(P)
    assert abs(st['hgr'] - 1.0) < 1e-12 and abs(st['phi2'] - 1.0) < 1e-12
    assert fz.cut_spectrum(np.zeros((2, 2))).size == 0


def test_words_alphabet_and_overflow():
    S = np.array([[0, 1, 2, 1, 0, 2, 2, 1, 0, 1, 2, 0]] * 4)
    with pytest.raises(ValueError):
        fz.words(S, 2, alphabet=2)
    big = fz.words(S, 10, alphabet=3)              # 3 ** 40 would overflow int64
    assert big.min() >= 0 and big.size == 3
    small = fz.words(S[:1], 2, alphabet=3)
    assert small.tolist() == [1, 5, 7, 3, 2, 8, 7, 3, 1, 5, 6]
    with pytest.raises(ValueError):
        fz.shift_null(np.zeros((2, 30), dtype=int), [0], [1], k=2, nSurr=3)


def test_cell_measure_edges():
    with pytest.raises(ValueError):
        replica.cell_measure(np.arange(5.0), 3)
    m = replica.cell_measure(np.ones(20), 3)
    assert np.allclose(m, 1.0 / 8)


def test_ews_edges():
    X = np.random.default_rng(0).standard_normal((4, 200))
    X[2] = 1.0
    v = ews.mean_corr(X)
    assert np.isfinite(v)
    with pytest.raises(ValueError):
        ews.rolling(ews.variance, X, 500)


def test_cp_als_zero_iterations():
    X = np.random.default_rng(0).standard_normal((3, 4, 5))
    w, F, err = decomp.cp_als(X, 2, n_iter=0)
    assert np.isfinite(err) and len(F) == 3
