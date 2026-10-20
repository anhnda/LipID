"""
tests/test_metric_lp.py — MET-01: the l2 counterexample (appendix B) and l1 sharpness of the edgewise aggregate.
"""
import math

import numpy as np
import pytest
import torch

from bounds import pair_bounds_exact
from envelope import anchor_lipschitz
from lipid import metric_lp as ML

torch.set_default_dtype(torch.float64)


def _fa(Z, a=2.0):
    t = Z[:, :1] - Z[:, 1:]
    return Z[:, 0] + a * torch.tanh(Z[:, 0]) * (t * t / (1 + t * t)).sum(1)


def _inst(d, n, seed, fixed, p, kappa, f=_fa):
    g = torch.Generator().manual_seed(seed)
    ch = torch.linalg.cholesky(0.4 * torch.eye(d) + 0.6 * torch.ones(d, d))
    C = torch.randn(n, d, generator=g) @ ch.T
    x = (torch.randn(1, d, generator=g) @ ch.T)[0]
    r = (torch.randn(1, d, generator=g) @ ch.T)[0]
    sigma = 0.5 + torch.rand(d, generator=g)
    A = torch.cat([C, x[None]] + ([r[None]] if fixed else []))
    fA = f(A)
    LA = ML.anchor_L_p(A, fA, sigma, p)
    return x, r, A, fA, kappa * LA, sigma


def test_euclidean_counterexample_exact():
    out = ML.euclidean_counterexample()
    assert abs(out["l2"]["edgewise_upper_phi1"] - 1.0) < 1e-8
    assert abs(out["l2"]["lp_upper_phi1"] - 1.0 / math.sqrt(2.0)) < 1e-8
    assert abs(out["l2"]["lp_lower_phi1"] + 1.0 / math.sqrt(2.0)) < 1e-8
    assert abs(out["l1"]["edgewise_upper_phi1"] - 1.0) < 1e-8
    assert abs(out["l1"]["lp_upper_phi1"] - 1.0) < 1e-8


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("kappa", [1.0, 1.5, 3.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_l1_lp_equals_theorem1(d, fixed, kappa, seed):
    x, r, A, fA, L, sigma = _inst(d, 12, seed, fixed, 1, kappa)
    assert abs(L / kappa - anchor_lipschitz(A, fA, sigma)) < 1e-10
    lpv = ML.lp_interval(x, r, A, fA, L, sigma, 1)
    e = ML.edgewise_interval(x, r, A, fA, L, sigma, 1)
    lo, hi, _ = pair_bounds_exact(x, r, A, fA, L, sigma)
    act = lpv["active"]
    np.testing.assert_allclose(lpv["lo"], lo[act].numpy(), atol=1e-7)
    np.testing.assert_allclose(lpv["hi"], hi[act].numpy(), atol=1e-7)
    np.testing.assert_allclose(e["lo"], lo[act].numpy(), atol=1e-12)
    np.testing.assert_allclose(e["hi"], hi[act].numpy(), atol=1e-12)
    assert lpv["max_constraint_violation"] < 1e-7


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("kappa", [1.0, 2.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_l2_edgewise_is_a_valid_outer_interval(d, fixed, kappa, seed):
    x, r, A, fA, L, sigma = _inst(d, 10, seed, fixed, 2, kappa)
    lpv = ML.lp_interval(x, r, A, fA, L, sigma, 2)
    e = ML.edgewise_interval(x, r, A, fA, L, sigma, 2)
    assert (e["lo"] <= lpv["lo"] + 1e-7).all() and (e["hi"] >= lpv["hi"] - 1e-7).all()
    assert (lpv["lo"] <= lpv["hi"] + 1e-9).all()
    assert lpv["max_constraint_violation"] < 1e-7


def test_l2_lp_contains_true_attribution_of_l2_lipschitz_model():
    d = 3
    w = torch.tensor([1.0, -0.5, 0.3])
    f = lambda Z: Z @ w
    x, r, A, fA, _, sigma = _inst(d, 15, 4, False, 2, 1.0, f=f)
    L = float(torch.linalg.norm(w * sigma))        # exact weighted-l2 Lipschitz constant of a linear map
    lpv = ML.lp_interval(x, r, A, fA, L, sigma, 2)
    phi = (w * (x - r)).numpy()[lpv["active"]]
    assert (lpv["lo"] <= phi + 1e-7).all() and (phi <= lpv["hi"] + 1e-7).all()
