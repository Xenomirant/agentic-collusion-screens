"""Tests for fathon.tensor.linalg (streamed tall-skinny numerics; CPU is enough)."""

import math
import os
import types
import warnings

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from fathon.tensor import linalg as la  # noqa: E402

F64 = torch.float64


@pytest.fixture(autouse=True, scope='module')
def _few_threads():
    """The closed-form null is many small tensor ops; dozens of threads only add overhead."""
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


def tall(N, T, svals, seed):
    """``N x T`` matrix with prescribed singular values (random orthonormal factors)."""
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(N, T, generator=g, dtype=F64)).Q
    V = torch.linalg.qr(torch.randn(T, T, generator=g, dtype=F64)).Q
    return (U * torch.as_tensor(svals, dtype=F64)) @ V.T


def ar1(N, T, phi, g):
    """Stationary AR(1) rows with unit marginal variance (vectorised by a cumulative-product trick)."""
    e = torch.randn(N, T, generator=g, dtype=F64)
    e[:, 0] /= math.sqrt(1 - phi ** 2)
    if phi == 0:
        return e
    # x_t = sum_{s<=t} phi^{t-s} e_s, in blocks of 32 steps to keep phi^{-s} bounded
    x = torch.empty_like(e)
    prev = torch.zeros(N, dtype=F64)
    for s in range(0, T, 32):
        blk = e[:, s:s + 32]
        p = phi ** torch.arange(blk.shape[1], dtype=F64)
        x[:, s:s + 32] = p * (torch.cumsum(blk / p, 1) + phi * prev[:, None])
        prev = x[:, s + blk.shape[1] - 1]
    return x * math.sqrt(1 - phi ** 2)


def uneven(X, sizes=(1, 37, 400, 3, 900)):
    out, s, i = [], 0, 0
    while s < X.shape[0]:
        out.append(X[s:s + sizes[i % len(sizes)]])
        s += sizes[i % len(sizes)]
        i += 1
    return out


def shift_surrogate(X, g):
    N, T = X.shape
    sh = torch.randint(0, T, (N, 1), generator=g)
    return torch.gather(X, 1, (torch.arange(T)[None, :] + sh) % T)


def test_ar1_helper_is_ar1():
    x = ar1(4000, 200, 0.9, torch.Generator().manual_seed(0))
    assert abs(float(x.var()) - 1) < 0.05
    assert abs(float((x[:, 1:] * x[:, :-1]).mean()) - 0.9) < 0.02


# ---------------------------------------------------------------------------
# streamed factorizations and tall SVD
# ---------------------------------------------------------------------------

def test_tsqr_and_gram_streamed_equal_in_memory():
    X = tall(3000, 40, torch.logspace(0, -3, 40), 0) + 0.01
    R0 = torch.linalg.qr(X, mode='r').R
    R0 = R0 * torch.sign(torch.diagonal(R0))[:, None]
    for src in (X, uneven(X), lambda: iter(uneven(X))):
        R = la.tsqr(src, chunk_rows=777)
        assert torch.allclose(torch.triu(R), R)
        assert torch.allclose(R, R0, rtol=1e-9, atol=1e-12)
        assert torch.allclose(la.gram(src, chunk_rows=777), X.T @ X, rtol=1e-12, atol=1e-12)
    # fan-in reduction with many chunks, and centring
    R = la.tsqr(X, chunk_rows=50, fan_in=4, center=True)
    Xc = X - X.mean(1, keepdim=True)
    assert torch.allclose(R.T @ R, Xc.T @ Xc, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('method', ['tsqr', 'gram', 'torch'])
def test_tall_svd_well_conditioned(method):
    s = torch.linspace(10, 1, 30)
    X = tall(5000, 30, s, 1)
    S, V = la.tall_svd(uneven(X), k=8, method=method)
    _, S0, V0h = torch.linalg.svd(X, full_matrices=False)
    assert S.shape == (8,) and V.shape == (30, 8)
    assert torch.allclose(S, S0[:8], rtol=1e-10)
    assert float(la.subspace_angles(V, V0h[:8].T).max()) < 1e-8


def test_tall_svd_ill_conditioned_tsqr_beats_gram():
    s = torch.logspace(0, -12, 24)
    X = tall(4000, 24, s, 2)
    S_t, V_t = la.tall_svd(X, method='tsqr', chunk_rows=999)
    S_g, _ = la.tall_svd(X, method='gram', chunk_rows=999)
    rel_t = ((S_t - s) / s).abs()
    rel_g = ((S_g - s) / s).abs()
    assert float(rel_t.max()) < 1e-3                       # TSQR resolves all 12 decades
    assert float(rel_t[:12].max()) < 1e-9
    assert float(rel_g[:6].max()) < 1e-6                   # Gram: fine down to ~1e-6 S[0]
    assert float(rel_g[-6:].min()) > 1e-2                  # and lost below sqrt(eps) S[0]
    # leading subspace of the tsqr path vs the float64 reference
    V0 = torch.linalg.svd(X, full_matrices=False).Vh[:6].T
    assert float(la.subspace_angles(V_t[:, :6], V0).max()) < 1e-9


def test_tall_svd_float32_inputs():
    s = torch.logspace(0, -2, 32)
    X = tall(6000, 32, s, 3)
    X32 = X.float()
    ref = torch.linalg.svdvals(X32.double())
    for method, dtype, tol in [('tsqr', F64, 1e-10), ('gram', F64, 1e-9), ('tsqr', torch.float32, 1e-4),
                               ('gram', torch.float32, 1e-3)]:
        S, V = la.tall_svd(uneven(X32), method=method, dtype=dtype)
        assert float(((S.double() - ref) / ref).abs().max()) < tol, (method, dtype)
    S, V = la.tall_svd(X32, method='tsqr', dtype=torch.float32, chunk_rows=1000)
    assert S.dtype == torch.float32 and V.dtype == torch.float32


def test_left_factors_reassemble_u():
    X = tall(2500, 20, torch.linspace(5, 1, 20), 4) + 0.1
    S, V = la.tall_svd(X, method='tsqr', center=True)           # centred rows: rank T - 1
    assert float(S[-1] / S[0]) < 1e-12
    U = torch.cat([la.left_factors(c, S, V, center=True) for c in uneven(X)])
    assert torch.allclose(U.T @ U, torch.diag(torch.tensor([1.0] * 19 + [0.0], dtype=F64)), atol=1e-10)
    Xc = X - X.mean(1, keepdim=True)
    assert torch.allclose((U * S) @ V.T, Xc, atol=1e-10)


def test_cans_backend_or_skip():
    try:
        la.load_cans_svd()
    except ImportError:
        pytest.skip('external CANS repository not present')
    s = torch.linspace(4, 1, 24)
    X = tall(3000, 24, s, 5)
    S, V = la.tall_svd(uneven(X), k=6, method='cans')
    # CANS' own singular values carry a ~1e-9 relative error; tall_svd recomputes them as ||R v_j||
    assert torch.allclose(S, s[:6].to(S.dtype), rtol=1e-12)
    V0 = torch.linalg.svd(X, full_matrices=False).Vh[:6].T
    assert float(la.subspace_angles(V, V0).max()) < 1e-10
    X = tall(6000, 96, torch.logspace(0, -8, 96), 17)                  # regression: was 2.6e-6 at cond 1e8
    S, V = la.tall_svd(X, method='cans')
    s0 = torch.linalg.svdvals(X)
    assert float(((S - s0) / s0).abs().max()) < 1e-8
    U = torch.cat([la.left_factors(c, S, V, method='cans') for c in X.split(1000)])
    keep = U.norm(dim=0) > 0
    assert float((U[:, keep].T @ U[:, keep] - torch.eye(int(keep.sum()), dtype=F64)).abs().max()) < 1e-7


def test_errors():
    with pytest.raises(ValueError):
        la.tall_svd(torch.randn(10, 4), method='nope')
    with pytest.raises(ValueError):
        la.gram([])
    with pytest.raises(TypeError):                      # multi-pass functions need a re-iterable source
        list(la.causal_projector(iter([torch.randn(5, 8)]), 4, 1))


# ---------------------------------------------------------------------------
# closed-form parallel-analysis null
# ---------------------------------------------------------------------------

def test_periodogram_eigenvalues_are_circulant_projection():
    g = torch.Generator().manual_seed(6)
    X = ar1(700, 96, 0.8, g) + 0.3
    lam, n_eff, n = la.periodogram_eigenvalues(uneven(X), center=True)
    Xc = X - X.mean(1, keepdim=True)
    C = la._null_gram(Xc.T @ Xc, 'circulant', center=True)
    assert n == 700 and 0.5 * n < n_eff < 1.5 * n
    assert torch.allclose(lam.sort().values, torch.linalg.eigvalsh(C), rtol=1e-9, atol=1e-8)
    assert torch.allclose(lam[1:], lam[1:].flip(0))                    # lam_k = lam_{T-k}
    assert math.isclose(float(lam.sum()), float((Xc ** 2).sum()), rel_tol=1e-12)   # Parseval
    # the circulant is the exact expectation of circular-shift surrogate Grams
    Gm = torch.zeros_like(C)
    for _ in range(400):
        Y = shift_surrogate(Xc, g)
        Gm = Gm + Y.T @ Y / 400
    assert float((Gm - C).norm() / C.norm()) < 0.02


def test_toeplitz_centring_debias_is_exact():
    T = 64
    ar = torch.arange(T)
    Gam = (0.95 ** ar.double())[(ar[None] - ar[:, None]).abs()]
    P = torch.eye(T, dtype=F64) - 1.0 / T
    M = P @ Gam @ P
    assert torch.allclose(la._null_gram(M, 'toeplitz', center=True), M, atol=1e-10)
    assert not torch.allclose(la._null_gram(M, 'toeplitz', center=False), M, atol=1e-3)
    # linear in its argument (the deflation relies on it)
    A = torch.randn(T, T, generator=torch.Generator().manual_seed(0), dtype=F64)
    A = A + A.T
    lhs = la._null_gram(2 * M - 3 * A, 'toeplitz', True)
    rhs = 2 * la._null_gram(M, 'toeplitz', True) - 3 * la._null_gram(A, 'toeplitz', True)
    assert torch.allclose(lhs, rhs, atol=1e-9)


@pytest.mark.parametrize('phi', [0.0, 0.9, 0.97])
def test_parallel_analysis_null_matches_monte_carlo(phi):
    g = torch.Generator().manual_seed(7)
    N, T = 2000, 128
    X = ar1(N, T, phi, g)
    X = X - X.mean(1, keepdim=True)
    null = la.parallel_analysis_null(X, q=6)
    mc = torch.stack([torch.linalg.svdvals(shift_surrogate(X, g))[:6] for _ in range(12)]).mean(0)
    assert float(((null - mc) / mc).abs().max()) < 0.02
    moment = la.parallel_analysis_null(X, q=6, n_eff='moment')     # homogeneous rows: N_eff ~ N
    assert float(((moment - mc) / mc).abs().max()) < 0.02
    if phi == 0:      # the bare expected-Gram spectrum misses the random-matrix spread by far
        bare = la.parallel_analysis_null(X, q=6, spread='none')
        assert float((bare / mc).max()) < 0.85


def test_deflation_keeps_a_weak_component_next_to_strong_ones():
    # two strong components and a weak one barely above the noise edge, white and AR noise
    g = torch.Generator().manual_seed(9)
    N, T = 2000, 256
    for phi, weak in ((0.0, 0.12), (0.97, 0.75)):
        L = torch.randn(N, 3, generator=g, dtype=F64)
        F = torch.randn(3, T, generator=g, dtype=F64) * torch.tensor([[2.0], [1.5], [weak]], dtype=F64)
        noise = ar1(N, T, phi, g)
        k, S, thr = la.estimate_rank_pa(L @ F + noise, q=8)
        assert k == 3, (phi, S[:5], thr)
        # after deflating the two strong components the null is that of the noise alone: neither
        # inflated by their smear over all frequencies nor stripped of the noise's share along them
        # (closed form only: the calibration surrogates also carry the strong components)
        _, _, thr = la.estimate_rank_pa(L @ F + noise, q=8, calibrate=0)
        k0, S0, thr0 = la.estimate_rank_pa(noise, q=2, calibrate=0)
        assert abs(float(thr[2] / thr0[0]) - 1) < 0.02, (phi, thr, thr0)


# ---------------------------------------------------------------------------
# rank estimation
# ---------------------------------------------------------------------------

def test_rank2_in_ar_noise_while_mp_overcounts():
    N, T = 2000, 256
    for seed in range(2):
        g = torch.Generator().manual_seed(seed)
        X = torch.randn(N, 2, generator=g, dtype=F64) @ torch.randn(2, T, generator=g, dtype=F64) * 0.7 \
            + ar1(N, T, 0.97, g)
        k, S, thr = la.estimate_rank_pa(X if seed else uneven(X), q=12, factor='tsqr' if seed else 'gram')
        assert k == 2 and thr.numel() == 3
        assert la.estimate_rank_pa(X, q=12, null='mp')[0] > 4


def test_pure_ar_false_positive_rate():
    N, T = 1000, 128
    ks = []
    for rep in range(50):
        g = torch.Generator().manual_seed(100 + rep)
        ks.append(la.estimate_rank_pa(ar1(N, T, 0.97 if rep % 2 else 0.6, g), q=4)[0])
    assert sum(k == 0 for k in ks) >= 45, ks


def test_circulant_null_overcounts_long_memory_toeplitz_does_not():
    g = torch.Generator().manual_seed(8)
    X = ar1(2000, 128, 0.97, g)
    assert la.estimate_rank_pa(X, q=4, null='toeplitz')[0] == 0
    assert la.estimate_rank_pa(X, q=4, null='circulant')[0] >= 1


@pytest.mark.parametrize('k_true', [1, 3, 5])
def test_heterogeneous_loadings_and_smooth_factors(k_true):
    g = torch.Generator().manual_seed(10 + k_true)
    N, T = 3000, 256
    L = torch.randn(N, k_true, generator=g, dtype=F64) * (0.8 ** torch.arange(k_true, dtype=F64))
    L = L * torch.exp(0.5 * torch.randn(N, 1, generator=g, dtype=F64))         # unequal agents
    F = ar1(k_true, T, 0.8, g)                                                 # autocorrelated factors
    X = L @ F + ar1(N, T, 0.9, g)
    assert la.estimate_rank_pa(X, q=12)[0] == k_true


def test_calibration_handles_heavy_tailed_row_energies():
    g = torch.Generator().manual_seed(11)
    N, T = 2000, 128
    X = ar1(N, T, 0.5, g) * torch.exp(1.0 * torch.randn(N, 1, generator=g, dtype=F64))
    S = torch.linalg.svdvals(X - X.mean(1, keepdim=True))[0]
    closed = la.parallel_analysis_null(X, q=1)[0]
    assert float(S / closed) > 1.06                        # closed form alone is too low here
    assert la.estimate_rank_pa(X, q=4, calibrate=2)[0] == 0


# ---------------------------------------------------------------------------
# causal projection
# ---------------------------------------------------------------------------

def test_causal_projector_removes_later_common_mode():
    g = torch.Generator().manual_seed(12)
    N, t0, T, k = 4000, 200, 320, 3
    L = torch.randn(N, k, generator=g, dtype=F64)
    F = ar1(k, T, 0.7, g) * 3.0
    X = L @ F + 0.1 * torch.randn(N, T, generator=g, dtype=F64)
    chunks = uneven(X, (999, 1234, 500))
    R = torch.cat(list(la.causal_projector(chunks, t0, k)))
    assert R.shape == (N, T - t0)
    later = X[:, t0:] - X[:, t0:].mean(1, keepdim=True)
    common = L @ (F[:, t0:] - F[:, t0:].mean(1, keepdim=True))
    Qc = torch.linalg.qr(common).Q[:, :k]
    assert float((Qc.T @ R).norm() ** 2 / common.norm() ** 2) < 1e-3
    assert float((Qc.T @ later).norm() ** 2 / common.norm() ** 2) > 0.9
    # the residual keeps the idiosyncratic part
    noise = later - common
    assert 0.9 < float(R.norm() / noise.norm()) < 1.1
    # class interface, streamed in two passes with an explicit pair factory
    cp = la.CausalProjector(k, method='tsqr').fit([c[:, :t0] for c in chunks])
    R2 = torch.cat(list(cp.transform(lambda: ((c[:, :t0], c[:, t0:]) for c in chunks))))
    assert torch.allclose(R, R2, atol=1e-8)



# ---------------------------------------------------------------------------
# regressions from the verification round
# ---------------------------------------------------------------------------

def test_ieee_fp32_guard_switches_tf32_off_and_restores():
    """NGC containers enable TF32 (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1); float32 CUDA GEMMs must run in IEEE."""
    m = torch.backends.cuda.matmul
    before = m.fp32_precision
    with la._ieee_fp32(types.SimpleNamespace(is_cuda=True, dtype=torch.float32)):
        assert m.fp32_precision == 'ieee'
    assert m.fp32_precision == before
    with la._ieee_fp32(torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=F64)):    # CPU: no-op
        assert m.fp32_precision == before


def test_subspace_angles_unequal_column_counts():
    g = torch.Generator().manual_seed(18)
    A = torch.randn(50, 2, generator=g, dtype=F64)
    B = torch.randn(50, 4, generator=g, dtype=F64)
    ab, ba = la.subspace_angles(A, B), la.subspace_angles(B, A)          # used to raise a size mismatch
    assert ab.shape == (2,) and torch.allclose(ab, ba, atol=1e-14)
    assert bool((ab[1:] >= ab[:-1]).all())                               # increasing
    # tiny angles keep full precision (sine formula)
    Q = torch.linalg.qr(torch.randn(50, 3, generator=g, dtype=F64)).Q
    W = Q[:, :2].clone()
    W[:, 0] = math.cos(1e-9) * Q[:, 0] + math.sin(1e-9) * Q[:, 2]
    ang = la.subspace_angles(Q[:, :2], W)
    assert abs(float(ang[-1]) - 1e-9) < 1e-15 and float(ang[0]) < 1e-15


def test_tall_svd_fewer_rows_than_columns():
    X = torch.randn(50, 96, generator=torch.Generator().manual_seed(19), dtype=F64)
    s0 = torch.linalg.svdvals(X)
    for method in ('tsqr', 'gram', 'torch'):
        S, V = la.tall_svd(uneven(X, (7, 20)), method=method)
        assert S.shape == (50,) and V.shape == (96, 50), method           # gram used to return (96,), (96, 96)
        assert torch.allclose(S, s0, rtol=1e-6), method


def test_left_factors_default_rtol_follows_the_method():
    X = tall(6000, 96, torch.logspace(0, -8, 96), 14)
    S, V = la.tall_svd(X, method='tsqr', chunk_rows=1000)
    U = torch.cat([la.left_factors(c, S, V) for c in X.split(1000)])
    keep = U.norm(dim=0) > 0
    assert int(keep.sum()) >= 90                                         # tsqr resolves ~eps^(1/2) S0
    assert float((U.T @ U - torch.diag(keep.double())).abs().max()) < 1e-7
    S, V = la.tall_svd(X, method='gram', chunk_rows=1000)
    U = torch.cat([la.left_factors(c, S, V, method='gram') for c in X.split(1000)])
    keep = U.norm(dim=0) > 0
    assert 40 <= int(keep.sum()) <= 56                                   # gram: ~eps^(1/4) S0
    assert float((U.T @ U - torch.diag(keep.double())).abs().max()) < 1e-7    # was 0.27 with rtol = T eps
    # exactly rank-deficient: the null components get zero columns
    Xr = tall(3000, 40, [1.0] * 10 + [0.0] * 30, 15)
    for method in ('tsqr', 'gram'):
        S, V = la.tall_svd(Xr, method=method)
        U = la.left_factors(Xr, S, V, method=method)
        assert int((U.norm(dim=0) > 0).sum()) == 10, method
    # centring: the annihilated constant direction is zeroed whatever rtol
    Xo = tall(3000, 48, torch.logspace(0, -3, 48), 16) + 5.0
    S, V = la.tall_svd(Xo, method='tsqr', center=True)
    U = la.left_factors(Xo, S, V, center=True, rtol=0.0)
    assert float(U[:, -1].abs().max()) == 0.0
    assert float((U[:, :-1].T @ U[:, :-1] - torch.eye(47, dtype=F64)).abs().max()) < 1e-10
    with pytest.raises(ValueError):
        la.left_factors(Xo, S, V, method='nope')


def smooth_factor(kind, T, g):
    t = torch.arange(T, dtype=F64)
    f = {'trend': t,
         'half_cos': torch.cos(math.pi * t / T),
         'rw_drift': torch.cumsum(torch.randn(T, generator=g, dtype=F64) + 0.3, 0)}[kind]
    f = f - f.mean()
    return f / f.norm() * math.sqrt(T)


@pytest.mark.parametrize('kind', ['trend', 'half_cos', 'rw_drift'])
def test_smooth_common_factor_is_flagged_untestable(kind):
    """The centred Toeplitz null absorbs smooth common shapes: report them; null='circulant' counts them."""
    g = torch.Generator().manual_seed(20)
    N, T = 1000, 256
    X = torch.randn(N, T, generator=g, dtype=F64) + 3.0 * torch.outer(torch.randn(N, generator=g, dtype=F64),
                                                                       smooth_factor(kind, T, g))
    with pytest.warns(UserWarning, match='untestable'):
        k, S, thr, info = la.estimate_rank_pa(X, q=4, return_info=True)
    assert k == 0 and info['untestable'] == [0]
    assert info['self_null'][0] >= 1 / info['margin'] ** 2
    assert float(S[0]) > 20 * float(S[1])                               # far above the noise, yet untestable
    assert la.estimate_rank_pa(X, q=4, null='circulant')[0] == 1


def test_component_behind_an_untestable_one_is_still_found():
    """Regression: the leading-run rule stopped at the trend and hid the strong i.i.d. factor behind it."""
    g = torch.Generator().manual_seed(21)
    N, T = 1000, 256
    L = torch.randn(N, 2, generator=g, dtype=F64)
    F = torch.stack([3.0 * smooth_factor('trend', T, g), 2.0 * torch.randn(T, generator=g, dtype=F64)])
    X = torch.randn(N, T, generator=g, dtype=F64) + L @ F
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        k, S, thr, info = la.estimate_rank_pa(X, q=6, return_info=True)
    assert k == 1 and info['untestable'] == [0]
    assert la.estimate_rank_pa(X, q=6, null='circulant')[0] == 2


def test_persistent_noise_with_untestable_lead_gives_rank_zero():
    # the leading direction of AR(0.99) noise at T = 128 sits in the blind spot; skipping it must not let
    # the next noise component through
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for rep in range(10):
            g = torch.Generator().manual_seed(300 + rep)
            k, S, thr, info = la.estimate_rank_pa(ar1(1000, 128, 0.99, g), q=4, return_info=True)
            out.append((k, tuple(info['untestable'])))
    assert all(k == 0 for k, _ in out), out
    assert sum(len(u) > 0 for _, u in out) >= 5, out                    # the lead is indeed flagged


def test_unequal_row_variances_give_no_false_positives_by_default():
    N, T = 1000, 128
    fp = {'default': 0, 'calibrate0': 0, 'standardize': 0}
    for rep in range(8):
        g = torch.Generator().manual_seed(400 + rep)
        if rep % 2:
            scale = torch.exp(0.5 * torch.randn(N, 1, generator=g, dtype=F64))             # lognormal sd
        else:
            scale = torch.where(torch.arange(N)[:, None] < N // 20, 10.0, 1.0).double()  # 5% of rows at 10x
        X = ar1(N, T, 0.5, g) * scale
        fp['default'] += la.estimate_rank_pa(X, q=4)[0] > 0
        fp['calibrate0'] += la.estimate_rank_pa(X, q=4, calibrate=0)[0] > 0
        fp['standardize'] += la.estimate_rank_pa(X, q=4, calibrate=0, standardize=True)[0] > 0
    assert fp['default'] == 0 and fp['standardize'] == 0, fp
    assert fp['calibrate0'] >= 6, fp                                     # the homogeneous spread model alone


def test_auto_margin_covers_small_populations():
    assert math.isclose(la.auto_margin(1000), 1 + 3 / math.sqrt(2000))
    assert la.auto_margin(1e5) == 1.05
    N, T = 300, 256
    ks = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for rep in range(20):
            g = torch.Generator().manual_seed(500 + rep)
            k, S, thr, info = la.estimate_rank_pa(ar1(N, T, 0.97, g), q=3, calibrate=0, return_info=True)
            assert info['margin'] == la.auto_margin(N)
            ks.append(k)
    assert sum(k > 0 for k in ks) == 0, ks                              # margin 1.05: 4 of 50 false positives


@pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get('FATHON_TEST_CUDA')),
                    reason='set FATHON_TEST_CUDA=1 to run on a GPU')
def test_cuda_smoke():
    X = tall(20000, 64, torch.linspace(3, 1, 64), 13).float().cuda()
    S, V = la.tall_svd(X, k=4, chunk_rows=3000)
    assert S.is_cuda and torch.allclose(S.cpu(), torch.linspace(3, 1, 64)[:4].double(), rtol=1e-5)


@pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get('FATHON_TEST_CUDA')),
                    reason='set FATHON_TEST_CUDA=1 to run on a GPU')
def test_cuda_float32_gram_is_ieee_even_with_tf32_on():
    m = torch.backends.cuda.matmul
    old = m.fp32_precision
    m.fp32_precision = 'tf32'
    try:
        g = torch.Generator(device='cuda').manual_seed(22)
        N, T = 60000, 256
        F = torch.randn(4, T, device='cuda', generator=g) * torch.tensor([[8.0], [6.0], [4.0], [3.0]], device='cuda')
        X = torch.randn(N, 4, device='cuda', generator=g) @ F + torch.randn(N, T, device='cuda', generator=g)
        S64 = la.tall_svd(X, method='gram', chunk_rows=20000)[0]
        S32 = la.tall_svd(X, method='gram', dtype=torch.float32, chunk_rows=20000)[0]
        assert float(((S32 - S64) / S64).abs().max()) < 5e-5          # 1.1e-5; TF32 GEMMs: 1.9e-4
        assert m.fp32_precision == 'tf32'                               # restored
    finally:
        m.fp32_precision = old
