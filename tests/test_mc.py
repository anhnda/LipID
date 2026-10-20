"""
tests/test_mc.py — SCALE-01: appendix-F stratified Hoeffding bounds.
"""
import numpy as np
import pytest
import torch

from envelope import anchor_lipschitz
from lipid.core import marginal_intervals
from lipid.mc import sample_coalitions, stratified_bounds

torch.set_default_dtype(torch.float64)


def _inst(d, n, seed, kappa=1.5):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(n, d, generator=g)
    x, r = torch.randn(d, generator=g), torch.randn(d, generator=g)
    sigma = 0.5 + torch.rand(d, generator=g)
    w = torch.linspace(1.0, 0.2, d)
    A = torch.cat([C, x[None]])
    fA = A @ w + 0.5 * torch.tanh(A.sum(1))
    L = kappa * anchor_lipschitz(A, fA, sigma)
    return x, r, A, fA, L, sigma


def test_coalition_sampler_sizes_and_exclusion():
    g = torch.Generator().manual_seed(0)
    for k in range(6):
        M = sample_coalitions(2, 6, k, 500, g)
        assert (M.sum(1) == k).all() and not M[:, 2].any()


@pytest.mark.parametrize("exact_singletons", [False, True])
def test_m2_is_deterministically_conservative(exact_singletons):
    """For m = 2 every stratum has one coalition, so the stratum mean is exact and the bound must hold surely."""
    x, r, A, fA, L, sigma = _inst(2, 20, 1)
    ex = marginal_intervals(x, r, A, fA, L, sigma)
    mc = stratified_bounds(x, r, A, fA, L, sigma, n_per_stratum=8, alpha=0.05,
                           gen=torch.Generator().manual_seed(3), exact_singletons=exact_singletons)
    assert (mc["lo"] <= ex["lo"] + 1e-12).all() and (mc["hi"] >= ex["hi"] - 1e-12).all()
    if exact_singletons:
        np.testing.assert_allclose(mc["lo"].numpy(), ex["lo"].numpy(), atol=1e-12)
        np.testing.assert_allclose(mc["hi"].numpy(), ex["hi"].numpy(), atol=1e-12)


def test_simultaneous_coverage_m5():
    x, r, A, fA, L, sigma = _inst(5, 64, 2)
    ex = marginal_intervals(x, r, A, fA, L, sigma)
    fails = 0
    for rep in range(40):
        mc = stratified_bounds(x, r, A, fA, L, sigma, n_per_stratum=16, alpha=0.05,
                               gen=torch.Generator().manual_seed(100 + rep))
        ok = bool((mc["lo"] <= ex["lo"] + 1e-12).all() and (mc["hi"] >= ex["hi"] - 1e-12).all())
        fails += not ok
        assert (mc["lo"] <= mc["hi"] + 1e-12).all()
    assert fails <= 2


def test_estimates_are_unbiased_in_the_limit():
    x, r, A, fA, L, sigma = _inst(4, 32, 5)
    ex = marginal_intervals(x, r, A, fA, L, sigma)
    mc = stratified_bounds(x, r, A, fA, L, sigma, n_per_stratum=4000, alpha=0.05,
                           gen=torch.Generator().manual_seed(9))
    Ld = (L * (x - r).abs() / sigma)
    assert (np.abs((mc["a_hat"] - ex["lo"]).numpy()) < 0.03 * Ld.numpy()).all()
    assert (np.abs((mc["b_hat"] - ex["hi"]).numpy()) < 0.03 * Ld.numpy()).all()
