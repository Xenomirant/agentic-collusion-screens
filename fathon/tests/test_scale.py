"""Tests for fathon.tensor.scale (torch backend; CPU is enough)."""

import itertools
import os
import sys
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from fathon.tensor import scale  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments', 'sims'))
planted = pytest.importorskip('planted')
DEV = 'cpu'


def test_estimate_rank_recovers_planted_rank():
    g = torch.Generator().manual_seed(0)
    N, T, r = 400, 1500, 3
    X = torch.randn(N, r, generator=g) @ torch.randn(r, T, generator=g) * 3 + torch.randn(N, T, generator=g)
    k, S = scale.estimate_rank(X)
    assert k == r
    R, _ = scale.common_mode_residual(X, r, extra=0)
    assert torch.linalg.svdvals(R)[0] < S[0] / 5


def test_pair_topk_matches_full_matrix():
    g = torch.Generator().manual_seed(1)
    A = scale._normalise(torch.randn(50, 300, generator=g))
    B = scale._normalise(torch.randn(50, 300, generator=g))
    C = A @ B.T
    full = torch.maximum(torch.maximum(C, C.T), (C + C.T) * 0.7071067811865476)
    iu = torch.triu_indices(50, 50, 1)
    v = full[iu[0], iu[1]]
    top = torch.topk(v, 10)
    ref = [(int(iu[0][i]), int(iu[1][i])) for i in top.indices]
    vals, idx = scale.pair_topk(A, B, 10, chunk=7)
    assert [tuple(p) for p in idx.tolist()] == ref
    assert torch.allclose(vals, top.values, atol=1e-6)


def test_detect_recovers_copy_and_xor_coalitions():
    A, truth = planted.simulate(300, 3000, r=2, kind='copy', k=2, n_coal=10, eps=1.0, seed=3, device=DEV)
    pairs = scale.detect(A, len(truth), device=DEV)
    assert len(set(pairs) & truth) / len(truth) >= 0.9
    A, truth = planted.simulate(300, 3000, r=2, kind='xor', k=3, n_coal=6, eps=2.5, seed=4, device=DEV)
    pairs = scale.detect(A, len(truth), device=DEV)
    assert len(set(pairs) & truth) / len(truth) >= 0.9
    # XOR coalitions are invisible to plain pairwise correlation
    pairs0 = scale.detect(A, len(truth), device=DEV, parity=False)
    assert len(set(pairs0) & truth) / len(truth) < 0.3


def test_closure_rerank_prefers_triangles():
    sc = {(0, 1): 5.0, (1, 2): 5.0, (0, 2): 4.0, (7, 8): 4.5, (3, 4): 6.0}
    top = scale.closure_rerank(sc, 3, weight=0.5)
    assert (0, 2) in top and (7, 8) not in top


def test_simulator_truth_and_determinism():
    A, truth = planted.simulate(60, 200, r=2, kind='copy', k=3, n_coal=4, eps=0.5, seed=9, device=DEV)
    assert A.shape == (60, 200) and set(A.unique().tolist()) <= {0.0, 1.0}
    assert truth == {p for b in range(4) for p in itertools.combinations(range(3 * b, 3 * b + 3), 2)}
    B, _ = planted.simulate(60, 200, r=2, kind='copy', k=3, n_coal=4, eps=0.5, seed=9, device=DEV)
    assert torch.equal(A, B)


def test_coincidence_pairs_finds_price_matchers():
    rng = np.random.default_rng(0)
    N, T = 30, 80
    X = np.round(80 + rng.standard_normal((N, T)).cumsum(1) * 0.3, 2)
    X[1] = X[0]                                   # a pair that matches prices to the cent
    X[1, ::7] += 0.5                              # most of the time
    top, info = scale.coincidence_pairs(X, 3)
    assert top[0] == (0, 1)
    assert info[(0, 1)][0] > 60 and info[(0, 1)][1] < 10
