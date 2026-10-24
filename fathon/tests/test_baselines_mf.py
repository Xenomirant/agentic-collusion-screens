"""Tests for experiments/baselines_mf.py (batched GPU MFDCCA / rho_q / h_xy baselines)."""

import os
import sys

# fathon's C loops use OpenMP over every core; on a shared 192-core box that oversubscription makes
# one 1000-step MFDCCA call ~50x slower (0.47 s vs 0.01 s). Only effective if libgomp is not loaded yet.
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np  # noqa: E402
import pytest

torch = pytest.importorskip('torch')
import fathon  # noqa: E402
from fathon import fathonUtils as fu  # noqa: E402

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments')
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, 'sims'))
bm = pytest.importorskip('baselines_mf')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
QS = [-3, -1, 0, 1, 2, 4]


def _pairs_world(seed=0, M=2, m=4, T=1000):
    """Gaussian, coupled, heavy-tailed and volatility-coupled rows, market by market."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((M * m, T))
    X[1] += 0.7 * X[0]
    X[2] *= np.exp(np.cumsum(rng.standard_normal(T)) * 0.05)
    X[m + 1] = np.abs(X[m + 1]) * X[m]
    X[m + 2] = rng.standard_t(3, T)
    return X


@pytest.mark.skipif(not hasattr(fathon, 'MFDCCA'), reason='compiled fathon (GSL) not available')
@pytest.mark.parametrize('revSeg', [False, True])
@pytest.mark.parametrize('seed', [0, 1])
def test_mfdcca_parity_with_fathon(revSeg, seed):
    m = 4
    X = _pairs_world(seed, m=m)
    sc = np.array([8, 16, 32, 64, 128, 250])
    F, Fxx, Fyy = bm.mfdcca_fluct(torch.tensor(X, device=DEV), m, list(sc), QS, rev_seg=revSeg)
    F, Fxx, Fyy = F.cpu().numpy(), Fxx.cpu().numpy(), Fyy.cpu().numpy()
    for a in range(2):
        for i in range(m):
            for j in range(i, m):
                _, Fo = fathon.MFDCCA(fu.toAggregated(X[a * m + i]), fu.toAggregated(X[a * m + j])).computeFlucVec(
                    sc, qList=QS, revSeg=revSeg)
                np.testing.assert_allclose(F[:, :, a, i, j], Fo, rtol=1e-10)
                np.testing.assert_allclose(F[:, :, a, j, i], Fo, rtol=1e-10)
            # diagonal = MFDFA, and Fxx / Fyy are views of it
            _, Fd = fathon.MFDFA(fu.toAggregated(X[a * m + i])).computeFlucVec(sc, qList=QS, revSeg=revSeg)
            np.testing.assert_allclose(Fxx[:, :, a, i, (i + 1) % m], Fd, rtol=1e-10)
            np.testing.assert_allclose(Fyy[:, :, a, (i + 1) % m, i], Fd, rtol=1e-10)


@pytest.mark.skipif(not hasattr(fathon, 'MFDCCA'), reason='compiled fathon (GSL) not available')
def test_hxy_matches_fathon_fit():
    m = 4
    X = _pairs_world(3, m=m)
    sc = [8, 16, 32, 64, 128]
    qs = [-3, -1, 1, 2, 4]
    w, H = bm.hxy_width(torch.tensor(X, device=DEV), m, sc, qs, return_h=True)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        o = fathon.MFDCCA(fu.toAggregated(X[i]), fu.toAggregated(X[j]))
        o.computeFlucVec(np.array(sc), qList=qs)
        h, _ = o.fitFlucVec()
        np.testing.assert_allclose(H[:, 0, i, j].cpu().numpy(), h, atol=1e-10)
        assert abs(w[0, i, j].item() - (h[0] - h[-1])) < 1e-10


@pytest.mark.skipif(not hasattr(fathon, 'DCCA'), reason='compiled fathon (GSL) not available')
def test_rho_q2_is_fathon_rho_dcca():
    m = 4
    X = _pairs_world(4, m=m)
    sc = np.array([8, 16, 32, 64, 128])
    R = bm.rho_q(torch.tensor(X, device=DEV), m, list(sc), [2.0], agg='none')[0].cpu().numpy()
    R_kw = bm.rho_q(torch.tensor(X, device=DEV), m, list(sc), [2.0], agg='none', form='kwapien')[0].cpu().numpy()
    for i, j in [(0, 1), (0, 2), (1, 3)]:
        _, rho = fathon.DCCA(fu.toAggregated(X[i]), fu.toAggregated(X[j])).computeRho(sc, overlap=False)
        np.testing.assert_allclose(R[:, 0, i, j], rho, atol=1e-12)
        np.testing.assert_allclose(R_kw[:, 0, i, j], rho, atol=1e-12)


@pytest.mark.skipif(DEV != 'cuda', reason='run_e11_scale.rho_dcca_markets runs on CUDA')
def test_rho_q2_equals_existing_rho_dcca_markets():
    e11 = pytest.importorskip('run_e11_scale')
    g = torch.Generator(device=DEV).manual_seed(0)
    M, m, T = 30, 6, 1024
    X = torch.randn(M * m, T, device=DEV, generator=g)
    X[::m] += 0.8 * X[1::m]
    old = e11.rho_dcca_markets(X, m)
    new = bm.rho_q(X, m, e11.SCALES, [2.0], abs_scales=True)[0]
    # rho_dcca_markets runs in float32 (cumsum over 1024 steps, float32 projector): ~1e-4 error
    d = (new.float() - old).abs()
    assert d.max() < 1e-3 and d.median() < 1e-4


def test_shapes():
    M, m, T = 3, 5, 512
    X = torch.randn(M * m, T, device=DEV)
    sc, qs = [8, 16, 32], [-2.0, 0.0, 2.0, 3.0]
    F, Fxx, Fyy = bm.mfdcca_fluct(X, m, sc, qs)
    assert F.shape == Fxx.shape == Fyy.shape == (4, 3, M, m, m)
    assert bm.mfdcca_fluct(X, m, sc, qs, mode='signed')[0].shape == (4, 3, M, m, m)
    assert bm.rho_q(X, m, sc, qs).shape == (4, M, m, m)
    assert bm.rho_q(X, m, sc, qs, agg='none').shape == (4, 3, M, m, m)
    assert bm.hxy_width(X, m, sc, qs).shape == (M, m, m)
    assert bm.rolling_rho_q(X, m, 128, [128, 256, 512], qs=(1, 2)).shape == (3, 2, M, m, m)
    assert bm.rolling_rho_q(X, m, None, [256, 512]).shape == (2, 3, M, m, m)
    # chunking over markets does not change anything
    assert torch.allclose(bm.rho_q(X, m, sc, qs, chunk=2), bm.rho_q(X, m, sc, qs))
    # symmetric, unit diagonal, bounded for q > 0
    R = bm.rho_q(X, m, sc, [1.0, 2.0, 4.0])
    assert torch.allclose(R, R.transpose(-1, -2))
    assert torch.allclose(torch.diagonal(R, dim1=-2, dim2=-1), torch.ones(3, M, m, device=DEV, dtype=R.dtype))
    assert R.abs().max() <= 1 + 1e-12


def test_independent_pairs_rho_near_zero():
    g = torch.Generator(device=DEV).manual_seed(1)
    M, m, T = 50, 10, 1024
    X = torch.randn(M * m, T, device=DEV, generator=g)
    X = X * torch.exp(0.5 * torch.randn(M * m, 1, device=DEV, generator=g))   # heterogeneous scales
    iu = torch.triu_indices(m, m, 1, device=DEV)
    R = bm.rho_q(X, m, bm.SCALES, [1.0, 2.0, 4.0])[:, :, iu[0], iu[1]]
    assert R.mean(dim=(1, 2)).abs().max() < 0.01
    # Kristoufek form sign(S)|S|^{2/q}: the null spread grows with q (|S|^{1/2} at q = 4)
    assert R[1].abs().mean() < 0.1 and R[0].abs().mean() < R[1].abs().mean() < R[2].abs().mean() < 0.2
    # Kwapien form S / (Fxx Fyy)^{q/2} has a tight null at every q
    K = bm.rho_q(X, m, bm.SCALES, [1.0, 2.0, 4.0], form='kwapien')[:, :, iu[0], iu[1]]
    assert K.mean(dim=(1, 2)).abs().max() < 0.01 and K.abs().mean(dim=(1, 2)).max() < 0.1


def test_coupled_pair_large_rho():
    g = torch.Generator(device=DEV).manual_seed(2)
    M, m, T = 20, 4, 1024
    X = torch.randn(M * m, T, device=DEV, generator=g)
    X[1::m] = X[0::m] + 0.5 * X[1::m]
    X[3::m] = -X[2::m] - 0.5 * X[3::m]
    R = bm.rho_q(X, m, bm.SCALES, [1.0, 2.0, 4.0])
    assert R[:, :, 0, 1].min() > 0.75
    assert R[:, :, 2, 3].max() < -0.75
    assert R[:, :, 0, 2].abs().mean() < 0.1
    # the unsigned (fathon-convention) cross fluctuation does not vanish for independent pairs
    F, Fxx, Fyy = bm.mfdcca_fluct(X, m, bm.SCALES, [2.0])
    ratio = (F ** 2 / (Fxx * Fyy))[0, :, :, 0, 2]
    assert abs(ratio.mean().item() - 2 / np.pi) < 0.03


def test_zero_windows_are_dropped_for_negative_q():
    # a binary action series with long constant runs has windows with zero residual
    T, m = 1024, 2
    x = torch.zeros(m, T, dtype=torch.float64, device=DEV)
    x[:, ::64] = 1.0
    x[1, 1::3] = 1.0
    w = bm.hxy_width(x, m, [8, 16, 32, 64], [-3.0, 3.0])
    assert torch.isfinite(w).all()
