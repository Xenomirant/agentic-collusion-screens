import itertools
import os
import numpy as np
import pytest
import fathon
from fathon import fathonUtils as fu
from fathon.tensor import detrend, mvdfa, decomp, factorization, replica, ews

# TENSOR EXTENSION TESTS
# ----------------------
# - parity of the pure-numpy multivariate estimators with fathon's C loops
#   (single series = MFDFA, pairs = DCCA / rho)
# - exactness of the tensor decompositions
# - factorization defect is ~0 for independent agents and large for coupled ones
# - empirical replica tensors reproduce the Bell-number ranks of
#   multifractal_tensor_networks/REPORT.md (Theorem A)

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
mf = np.loadtxt(os.path.join(TESTS_PATH, 'mat/multifractal.txt'))
wn = np.loadtxt(os.path.join(TESTS_PATH, 'mat/whitenoise.txt'))

rng = np.random.default_rng(7)
X3 = rng.standard_normal((3, 2500))
X3[1] += 0.6 * X3[0]
X3[2] = np.cumsum(X3[2]) * 0.05 + X3[2]
Y3 = detrend.profile(X3)
scales = fu.linRangeByStep(10, 400, step=13)


#####
# Parity with the C implementation
#####
@pytest.mark.parametrize('polOrd', [1, 2, 3])
@pytest.mark.parametrize('revSeg', [False, True])
def test_mfdfa_parity(polOrd, revSeg):
    qs = [-5, -3, -1, 0, 1, 2, 3, 5]
    for x in (mf, wn):
        F = mvdfa.mfdfa(fu.toAggregated(x), scales, qs, polOrd, revSeg)[0]
        _, Fo = fathon.MFDFA(fu.toAggregated(x)).computeFlucVec(scales, qList=qs, polOrd=polOrd, revSeg=revSeg)
        # the C loops fit raw powers of t = 1..s, which loses ~1e-7 at order 3
        # (see test_detrend_exact); the orthonormal basis used here does not
        np.testing.assert_allclose(F, Fo, rtol=1e-9 if polOrd < 3 else 1e-6)


def _exact_ms_residual(w, deg):
    # mean squared least-squares residual in exact rational arithmetic
    from fractions import Fraction as Fr
    s = len(w)
    V = [[Fr(i + 1) ** k for k in range(deg + 1)] for i in range(s)]
    y = [Fr(float(v)) for v in w]
    n = deg + 1
    A = [[sum(V[r][i] * V[r][j] for r in range(s)) for j in range(n)] for i in range(n)]
    b = [sum(V[r][i] * y[r] for r in range(s)) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            f = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= f * A[i][k]
            b[j] -= f * b[i]
    c = [Fr(0)] * n
    for i in reversed(range(n)):
        c[i] = (b[i] - sum(A[i][k] * c[k] for k in range(i + 1, n))) / A[i][i]
    return float(sum((y[r] - sum(c[k] * V[r][k] for k in range(n))) ** 2 for r in range(s)) / s)


@pytest.mark.parametrize('deg', [1, 3])
def test_detrend_exact(deg):
    Y = fu.toAggregated(mf)[:200]
    for s in (10, 40):
        ex = [_exact_ms_residual(Y[v * s:(v + 1) * s], deg) for v in range(len(Y) // s)]
        ours = np.mean(detrend.residuals(Y, s, deg) ** 2, axis=-1)
        np.testing.assert_allclose(ours, ex, rtol=1e-10)


def test_mfdfa_batched_equals_single():
    qs = [-2, 0, 2]
    F = mvdfa.mfdfa(Y3, scales, qs)
    for i in range(3):
        _, Fo = fathon.MFDFA(Y3[i]).computeFlucVec(scales, qList=qs)
        np.testing.assert_allclose(F[i], Fo, rtol=1e-9)


@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.parametrize('revSeg', [False, True])
def test_rho_parity(overlap, revSeg):
    R = mvdfa.rho_matrix(Y3, scales, 2, revSeg, overlap)
    for i, j in itertools.combinations(range(3), 2):
        _, r = fathon.DCCA(Y3[i], Y3[j]).computeRho(scales, polOrd=2, overlap=overlap, revSeg=revSeg)
        np.testing.assert_allclose(R[:, i, j], r, atol=1e-12)
        np.testing.assert_allclose(R[:, j, i], r, atol=1e-12)
    np.testing.assert_allclose(np.einsum('kii->ki', R), 1.0)


@pytest.mark.parametrize('overlap', [False, True])
def test_signed_dcca_parity(overlap):
    F2 = mvdfa.covariance_tensor(Y3, scales, 1, False, overlap)
    _, F = fathon.DCCA(Y3[0], Y3[1]).computeFlucVec(scales, polOrd=1, absVals=False, overlap=overlap)
    np.testing.assert_allclose(F2[:, 0, 1], F, rtol=1e-9)


def test_scale_checks():
    with pytest.raises(ValueError):
        mvdfa.mfdfa(Y3, np.array([2, 10]), [2], polOrd=1)
    with pytest.raises(ValueError):
        mvdfa.covariance_tensor(Y3, np.array([10, 5000]))


#####
# Exponents on processes with known scaling
#####
def test_hurst_white_noise_and_walk():
    x = np.random.default_rng(0).standard_normal((2, 2 ** 14))
    x[1] = np.cumsum(x[1])
    n = fu.powRangeByStep(4, 10)
    H, _ = mvdfa.fit_exponents(n, mvdfa.mfdfa(detrend.profile(x), n, [2])[:, 0])
    assert abs(H[0] - 0.5) < 0.05
    assert abs(H[1] - 1.5) < 0.05


def test_common_mode():
    rng = np.random.default_rng(1)
    common = rng.standard_normal(4000)
    X = 0.9 * common + 0.3 * rng.standard_normal((5, 4000))
    cm = mvdfa.common_mode(mvdfa.rho_matrix(detrend.profile(X), scales))
    assert np.all(cm['absorption'] > 0.8)
    Xi = rng.standard_normal((5, 4000))
    cmi = mvdfa.common_mode(mvdfa.rho_matrix(detrend.profile(Xi), scales[:5]))
    assert np.all(cmi['absorption'] < 0.4)


#####
# Decompositions
#####
def test_tt_svd_roundtrip_and_ranks():
    rng = np.random.default_rng(2)
    cores = [rng.standard_normal((1, 3, 2)), rng.standard_normal((2, 4, 3)),
             rng.standard_normal((3, 2, 2)), rng.standard_normal((2, 3, 1))]
    X = decomp.tt_to_full(cores)
    G, _ = decomp.tt_svd(X)
    np.testing.assert_allclose(decomp.tt_to_full(G), X, atol=1e-10)
    assert decomp.tt_ranks(X) == [2, 3, 2]


def test_tucker_and_cp():
    rng = np.random.default_rng(3)
    A, B, C = rng.standard_normal((6, 2)), rng.standard_normal((5, 2)), rng.standard_normal((4, 2))
    X = decomp.cp_to_full(np.array([3.0, 1.0]), [A, B, C])
    core, U, _ = decomp.tucker(X, tol=1e-10)
    assert core.shape == (2, 2, 2)
    np.testing.assert_allclose(decomp.tucker_to_full(core, U), X, atol=1e-10)
    w, F, err = decomp.cp_als(X, 2, n_iter=500)
    assert err < 1e-6


def test_nearest_kron():
    rng = np.random.default_rng(4)
    A, B = rng.random((3, 3)), rng.random((2, 2))
    _, _, s = decomp.nearest_kron(np.kron(A, B), (3, 3), (2, 2))
    assert s[1] / s[0] < 1e-12


#####
# Factorization defect
#####
def _markov(T, P, rng):
    s = np.zeros(T, dtype=np.int64)
    for t in range(1, T):
        s[t] = rng.choice(P.shape[0], p=P[s[t - 1]])
    return s


def test_kron_rearrangement_is_pair_cut():
    # rearranged joint transition matrix == cut matrix of length-2 windows
    rng = np.random.default_rng(5)
    S = rng.integers(0, 2, size=(2, 500))
    z = S[0] * 2 + S[1]
    Pt = np.zeros((4, 4))
    np.add.at(Pt, (z[:-1], z[1:]), 1.0)
    R = decomp.kron_rearrange(Pt / Pt.sum(), (2, 2), (2, 2))
    a = factorization.words(S[[0]], 2, 2)
    b = factorization.words(S[[1]], 2, 2)
    C = np.zeros((4, 4))
    np.add.at(C, (a, b), 1.0)
    np.testing.assert_allclose(R, C / C.sum())


def test_defect_independent_vs_coupled():
    rng = np.random.default_rng(6)
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    T = 4000
    a, b = _markov(T, P, rng), _markov(T, P, rng)
    S_ind = np.stack([a, b])
    # b copies a's previous symbol with probability 0.5
    c = np.where(rng.random(T) < 0.5, np.roll(a, 1), b)
    S_cpl = np.stack([a, c])
    _, _, z_ind, p_ind = factorization.shift_null(S_ind, [0], [1], nSurr=100)
    _, _, z_cpl, p_cpl = factorization.shift_null(S_cpl, [0], [1], nSurr=100)
    assert p_ind > 0.01
    assert z_cpl > 10 and p_cpl < 0.02
    # directed: a's past predicts c's future, not the reverse
    fwd = factorization.group_cut(S_cpl, [0], [1], k=1, lag=1)['phi2']
    bwd = factorization.group_cut(S_cpl, [1], [0], k=1, lag=1)['phi2']
    assert fwd > 10 * bwd


def test_coalition_tree_recovers_groups():
    rng = np.random.default_rng(8)
    T = 3000
    srcs = rng.standard_normal((2, T))
    X = np.stack([srcs[g] + 0.7 * rng.standard_normal(T) for g in (0, 0, 0, 1, 1, 1)])
    S = factorization.discretize(X, 3)
    M = factorization.pairwise_matrix(S, k=1)
    from scipy.cluster.hierarchy import fcluster
    lab = fcluster(factorization.coalition_tree(M), 2, 'maxclust')
    assert len(set(lab[:3])) == 1 and len(set(lab[3:])) == 1 and lab[0] != lab[3]
    best = factorization.best_bipartition(S, k=1)[0]
    assert sorted(best[1]) in ([0, 1, 2], [3, 4, 5])


def test_gaussian_cut():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 5000))
    ind = factorization.gaussian_cut(x[[0]], x[[1]], k=2)['gauss_mi']
    y = 0.8 * x[0] + 0.6 * x[1]
    cpl = factorization.gaussian_cut(x[[0]], y[None], k=2)['gauss_mi']
    assert ind < 0.01 and cpl > 0.4


#####
# Replica tensors (Theorem A of multifractal_tensor_networks/REPORT.md)
#####
def _cascade_samples(S, n, atoms, probs, rng):
    # unnormalised Mandelbrot cascade (normalising by the total mass couples
    # all cells and breaks the exact MPS structure of Theorem A)
    mu = np.ones((S, 1))
    for _ in range(n):
        idx = rng.choice(len(probs), size=mu.shape, p=probs)
        mu = (mu[:, :, None] * atoms[idx]).reshape(S, -1)
    return mu


def _exact_moment_tensor(n, q, atoms, probs):
    # exact enumeration, as in multifractal_tensor_networks/code/t1_replica_rank.py
    def mixed(m):
        return float(np.sum(probs * np.prod(atoms ** np.array(m)[None, :], axis=1)))
    T = np.zeros((2 ** q,) * n)
    for idx in itertools.product(range(2 ** q), repeat=n):
        letters = np.array([[(s >> a) & 1 for a in range(q)] for s in idx])
        val = 1.0
        for k in range(n):
            groups = {}
            for a in range(q):
                groups.setdefault(tuple(letters[:k, a]), []).append(a)
            for g in groups.values():
                m = [0, 0]
                for a in g:
                    m[letters[k, a]] += 1
                val *= mixed(m)
        T[idx] = val
    return T


def test_binomial_cascade_rank_one():
    w = np.array([0.3, 0.7])
    mu = w
    for _ in range(9):
        mu = np.kron(mu, w)
    assert decomp.tt_ranks(mu.reshape((2,) * 10)) == [1] * 9


@pytest.mark.parametrize('q,n,bell', [(2, 5, 2), (3, 4, 5)])
def test_replica_tensor_matches_exact(q, n, bell):
    rng = np.random.default_rng(10)
    atoms = rng.uniform(0.2, 1.8, size=(4, 2))
    probs = rng.dirichlet(np.ones(4))
    exact = _exact_moment_tensor(n, q, atoms, probs)
    assert max(decomp.tt_ranks(exact)) == bell
    mu = _cascade_samples(60000, n, atoms, probs, rng)
    emp = replica.replica_tensor(mu, q)
    err = np.linalg.norm(emp - exact) / np.linalg.norm(exact)
    assert err < 0.05, err
    # truncating the estimate to the Bell rank loses only sampling noise
    G, _ = decomp.tt_svd(emp, max_rank=bell)
    trunc = np.linalg.norm(decomp.tt_to_full(G) - exact) / np.linalg.norm(exact)
    assert trunc < err * 1.05


def test_replica_rank_two_resolved():
    # q = 2: the second (non-diagonal) Bell direction stands far above noise
    rng = np.random.default_rng(14)
    atoms = rng.uniform(0.2, 1.8, size=(4, 2))
    probs = rng.dirichlet(np.ones(4))
    sp = decomp.cut_spectra(replica.replica_tensor(_cascade_samples(20000, 5, atoms, probs, rng), 2))
    for p in sp:
        assert p[1] > 10 * p[2:].sum()


def test_replica_interleave_ordering():
    # for q = 1 interleaving is the identity reshape
    mu = np.random.default_rng(11).random((3, 16))
    T = replica.replica_tensor(mu, 1)
    np.testing.assert_allclose(T.reshape(-1), mu.mean(0))


def test_agent_cut_independent():
    # agents driven by a shared slowly varying volatility vs by their own
    rng = np.random.default_rng(12)
    T = 2 ** 16

    def vol():
        return np.exp(np.convolve(rng.standard_normal(T), np.ones(64) / 8, mode='same'))

    v = vol()
    shared = [v * rng.standard_normal(T) for _ in range(2)]
    own = [vol() * rng.standard_normal(T) for _ in range(2)]
    ms = [replica.sample_windows(np.cumsum(x), 5) for x in shared]
    mo = [replica.sample_windows(np.cumsum(x), 5) for x in own]
    _, r_shared = replica.agent_cut(*ms)
    _, r_own = replica.agent_cut(*mo)
    assert r_shared > 3 * r_own


def test_tau_binomial():
    p = 0.3
    w = np.array([p, 1 - p])
    mu = w
    for _ in range(11):
        mu = np.kron(mu, w)
    qs = [-2.0, 0.0, 1.0, 2.0, 3.0]
    tau = replica.tau_spectrum(mu[None], qs)
    np.testing.assert_allclose(tau, [-np.log2(p ** q + (1 - p) ** q) for q in qs], atol=1e-9)


#####
# EWS helpers
#####
def test_ews_helpers():
    rng = np.random.default_rng(13)
    t, v = ews.rolling(ews.variance, rng.standard_normal((2, 1000)) * np.linspace(1, 3, 1000), 200, 50)
    tau, p = ews.kendall_trend(t, v)
    assert tau > 0.8 and p < 0.01
    X = rng.standard_normal((3, 512))
    for sur in (ews.phase_surrogate, ews.iaaft_surrogate):
        Y = sur(X, rng)
        assert Y.shape == X.shape
    np.testing.assert_allclose(np.sort(ews.iaaft_surrogate(X, rng), axis=1), np.sort(X, axis=1))
    assert ews.auc([2, 3], [0, 1]) == 1.0
    assert ews.first_crossing(np.arange(5), [0, 2, 0, 2, 2], 1, persist=2) == 3


#####
# Influence (Bonjour, Aggarwal & Bhargava 2022)
#####
from fathon.tensor import influence as inf  # noqa: E402


def _assistant_mi(cp):
    # I(A; B) when B plays f(A) w.p. cp, uniform otherwise (3 actions), nats
    p1 = cp + (1 - cp) / 3
    p2 = (1 - cp) / 3
    return np.log(3) + p1 * np.log(p1) + 2 * p2 * np.log(p2)


def test_conditional_mi_analytic_and_units():
    rng = np.random.default_rng(15)
    T, cp = 200000, 0.3
    a = rng.integers(0, 3, T)
    b = np.where(rng.random(T) < cp, (a + 2) % 3, rng.integers(0, 3, T))
    np.testing.assert_allclose(inf.conditional_mi(a, b), _assistant_mi(cp), atol=2e-3)
    np.testing.assert_allclose(inf.conditional_mi(a, b, base=2), _assistant_mi(cp) / np.log(2), atol=3e-3)


def test_conditioning_removes_common_state():
    # x and y both copy a common state s; independent given s
    rng = np.random.default_rng(16)
    T = 50000
    s = rng.integers(0, 3, T)
    x = np.where(rng.random(T) < 0.6, s, rng.integers(0, 3, T))
    y = np.where(rng.random(T) < 0.6, s, rng.integers(0, 3, T))
    assert inf.conditional_mi(x, y) > 0.05
    assert inf.conditional_mi(x, y, s) < 1e-3


def test_net_influence_and_detection():
    rng = np.random.default_rng(17)
    T = 3000
    A = rng.integers(0, 3, size=(3, T))
    A[1] = np.where(rng.random(T) < 0.6, (A[0] + 2) % 3, A[1])
    G = inf.influence_matrix(A)
    Gam = inf.net_influence(G)
    assert inf.detect_pair(Gam, 0.05) == (0, 1)
    assert inf.pair_scores(Gam)[0][1] == (0, 1)
    assert inf.rank_of_pair(Gam, (1, 0)) == 1
    with pytest.raises(ValueError):
        inf.net_influence(G[:2, :2])


def test_conditional_permutation_keeps_strata():
    rng = np.random.default_rng(18)
    y = rng.integers(0, 3, 500)
    s = rng.integers(0, 4, 500)
    yp = inf.conditional_permutation(y, s, rng)
    for v in range(4):
        np.testing.assert_array_equal(np.sort(y[s == v]), np.sort(yp[s == v]))


def test_influence_z_null_and_alternative():
    rng = np.random.default_rng(19)
    T = 600
    s = rng.integers(0, 3, T)
    A = np.stack([np.where(rng.random(T) < 0.7, s, rng.integers(0, 3, T)) for _ in range(3)])
    A[2] = np.where(rng.random(T) < 0.5, (A[0] + 1) % 3, A[2])
    Z, P = inf.influence_z(A, s, nPerm=99, seed=1)
    assert P[0, 1] > 0.01          # common-state dependence only
    assert P[0, 2] <= 0.02 and Z[0, 2] > 5


def test_directed_influence():
    rng = np.random.default_rng(20)
    T = 5000
    A = rng.integers(0, 3, size=(3, T))
    A[1, 1:] = np.where(rng.random(T - 1) < 0.5, A[0, :-1], A[1, 1:])
    G = inf.influence_matrix(A, lag=1)
    assert G[0, 1] > 10 * G[1, 0]
