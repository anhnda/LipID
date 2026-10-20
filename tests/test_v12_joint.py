"""
tests/test_v12_joint.py — Proposition J (exact joint identified set) and the LP machinery (reviewer Q3).
"""
import numpy as np
import pytest
import torch

from bounds import shapley_from_vertex_values
from envelope import anchor_lipschitz
from lipid import pairwise as PW
from lipid.joint import JointPolytope, shapley_matrix

torch.set_default_dtype(torch.float64)


def _inst(d, n, seed, fixed, kappa):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(n, d, generator=g)
    x, r = torch.randn(d, generator=g), torch.randn(d, generator=g)
    sigma = 0.5 + torch.rand(d, generator=g)
    f = lambda Z: torch.sin(Z[:, 0]) + Z[:, 1:].sum(1) * 0.5        # noqa: E731
    A = torch.cat([C, x[None]] + ([r[None]] if fixed else []))
    fA = f(A)
    return x, r, A, fA, kappa * anchor_lipschitz(A, fA, sigma), sigma


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_shapley_matrix(m):
    g = torch.Generator().manual_seed(m)
    t = torch.randn(1 << m, generator=g)
    M = shapley_matrix(m)
    np.testing.assert_allclose(M @ t.numpy(), shapley_from_vertex_values(t, m).numpy(), atol=1e-12)
    # efficiency: sum of the Shapley values = v(all) - v(empty)
    np.testing.assert_allclose(M.sum(0)[[0, (1 << m) - 1]], [-1.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("fixed", [False, True])
def test_marginal_lp_equals_theorem1(d, fixed):
    x, r, A, fA, L, sigma = _inst(d, 8, 3 + d, fixed, 1.3)
    jp = JointPolytope(x, r, A, fA, L, sigma)
    lo, hi = jp.marginal()
    res = PW.pairwise_bounds(x, r, A, fA, L, sigma)
    Ld = res["Ld"].numpy()
    np.testing.assert_allclose(lo / Ld, res["lo"].numpy() / Ld, atol=1e-8)
    np.testing.assert_allclose(hi / Ld, res["hi"].numpy() / Ld, atol=1e-8)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("fixed", [False, True])
def test_edge_reduction_equals_all_pairs_on_random_directions(d, fixed):
    """Proposition J: under weighted l1 the edge constraints define the same polytope as all vertex-pair constraints,
    so their support functions agree in every direction (here: random contrasts of the attribution vector)."""
    x, r, A, fA, L, sigma = _inst(d, 6, 11 + d, fixed, 1.7)
    jE = JointPolytope(x, r, A, fA, L, sigma, constraints="edges")
    jA = JointPolytope(x, r, A, fA, L, sigma, constraints="all_pairs")
    g = torch.Generator().manual_seed(d)
    scale = float(L * jE.dl.sum())
    for _ in range(6):
        coef = torch.randn(jE.m, generator=g).numpy()
        lE, hE, tlo, thi = jE.interval(jE.contrast(coef))
        lA, hA, _, _ = jA.interval(jA.contrast(coef))
        assert abs(lE - lA) / scale < 1e-8 and abs(hE - hA) / scale < 1e-8
        # the edge-LP optimum is feasible for ALL pairs and for the anchors directly
        assert max(jE.violation(tlo).values()) / scale < 1e-8
        assert max(jE.violation(thi).values()) / scale < 1e-8


@pytest.mark.parametrize("d", [3, 4])
def test_pair_objective_identity(d):
    x, r, A, fA, L, sigma = _inst(d, 6, 21, False, 1.2)
    jp = JointPolytope(x, r, A, fA, L, sigma)
    for a in range(jp.m):
        for b in range(jp.m):
            if a != b:
                np.testing.assert_allclose(jp.M[a] - jp.M[b], jp.pair_diff_obj_closed(a, b), atol=1e-14)


def test_infeasible_class_is_rejected():
    from lipid.joint import Infeasible
    x, r, A, fA, L, sigma = _inst(3, 8, 1, False, 1.0)
    with pytest.raises(Infeasible):
        JointPolytope(x, r, A, fA, 0.5 * L, sigma)
