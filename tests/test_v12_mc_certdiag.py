"""
tests/test_v12_mc_certdiag.py — SCL-02 Monte Carlo variants and CERT-02 relaxation chain.
"""
import math

import numpy as np
import pytest
import torch

from envelope import anchor_lipschitz, envelopes, weighted_l1
from lipid.certdiag import classify, geo_pair_min, relaxation_chain
from lipid.core import edge_tensors, marginal_intervals
from lipid.mc import stratified_bounds
from lipid.mc_v2 import all_coalitions, sample_coalitions_wor, stratified_bounds_v2

torch.set_default_dtype(torch.float64)


def _inst(d, n, seed, kappa=1.5, fixed=False):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(n, d, generator=g)
    x, r = torch.randn(d, generator=g), torch.randn(d, generator=g)
    sigma = 0.5 + torch.rand(d, generator=g)
    w = torch.linspace(1.0, 0.2, d)
    A = torch.cat([C, x[None]] + ([r[None]] if fixed else []))
    fA = A @ w + 0.5 * torch.tanh(A.sum(1))
    return x, r, A, fA, kappa * anchor_lipschitz(A, fA, sigma), sigma


# ---------------------------------------------------------------------------------------------------- mc_v2
def test_hoeffding_paper_reproduces_lipid_mc():
    x, r, A, fA, L, sigma = _inst(6, 40, 1)
    a = stratified_bounds(x, r, A, fA, L, sigma, n_per_stratum=16, alpha=0.05, gen=torch.Generator().manual_seed(9))
    b = stratified_bounds_v2(x, r, A, fA, L, sigma, n_per_stratum=16, alpha=0.05, gen=torch.Generator().manual_seed(9),
                             method="hoeffding_paper")
    torch.testing.assert_close(a["lo"], b["lo"], rtol=0, atol=1e-14)
    torch.testing.assert_close(a["hi"], b["hi"], rtol=0, atol=1e-14)


@pytest.mark.parametrize("method", ["hoeffding_tight", "bernstein", "serfling_wor"])
def test_full_enumeration_is_exact(method):
    """m = 4, n_k = 8 >= C(3, k) for every k: every stratum is enumerated, so the bound equals Theorem 1."""
    x, r, A, fA, L, sigma = _inst(4, 30, 2)
    ex = marginal_intervals(x, r, A, fA, L, sigma)
    mc = stratified_bounds_v2(x, r, A, fA, L, sigma, n_per_stratum=8, alpha=0.05, gen=torch.Generator().manual_seed(1),
                              method=method)
    assert mc["exhaustive_strata"] == [0, 1, 2, 3] and mc["n_events"] == 0
    torch.testing.assert_close(mc["lo"], ex["lo"], rtol=0, atol=1e-12)
    torch.testing.assert_close(mc["hi"], ex["hi"], rtol=0, atol=1e-12)


def test_wor_sampler_and_enumeration():
    g = torch.Generator().manual_seed(0)
    M = sample_coalitions_wor(2, 8, 3, 20, g)
    keys = {tuple(row.tolist()) for row in M}
    assert len(keys) == 20 and (M.sum(1) == 3).all() and not M[:, 2].any()
    E = all_coalitions(1, 6, 2)
    assert E.shape[0] == math.comb(5, 2) and (E.sum(1) == 2).all() and not E[:, 1].any()


@pytest.mark.parametrize("method", ["hoeffding_tight", "bernstein", "serfling_wor"])
def test_variants_contain_exact_interval(method):
    """Validity (probabilistic, alpha = .05 simultaneous): fixed seeds, failures must stay rare."""
    x, r, A, fA, L, sigma = _inst(7, 60, 3)
    ex = marginal_intervals(x, r, A, fA, L, sigma)
    fails = 0
    for rep in range(30):
        mc = stratified_bounds_v2(x, r, A, fA, L, sigma, n_per_stratum=16, alpha=0.05,
                                  gen=torch.Generator().manual_seed(100 + rep), method=method)
        fails += not bool((mc["lo"] <= ex["lo"] + 1e-12).all() and (mc["hi"] >= ex["hi"] - 1e-12).all())
    assert fails <= 4


# ---------------------------------------------------------------------------------------------------- certdiag
def _chain_inputs(d, n, seed, fixed):
    from bounds import all_masks, vertex_points
    x, r, A, fA, LA0, sigma = _inst(d, n, seed, kappa=1.0, fixed=fixed)
    LA = anchor_lipschitz(A, fA, sigma)
    delta = (x - r).abs() / sigma
    Z = vertex_points(x, r, list(range(d)), all_masks(d))
    Dk = weighted_l1(Z, A, sigma)
    Sx, Px, W = edge_tensors(d)
    return x, r, A, fA, LA, sigma, delta, Z, Dk, Sx, Px, W


@pytest.mark.parametrize("fixed", [False, True])
def test_abs_relaxation_identity_and_geo_brute_force(fixed):
    x, r, A, fA, LA, sigma, delta, Z, Dk, Sx, Px, W = _chain_inputs(3, 7, 4, fixed)
    L = 1.8 * LA
    ell, h = envelopes(Z, A, fA, L, sigma)
    M = weighted_l1(A, A, sigma)
    k = LA / L
    for jj in range(3):
        for s_, p_ in zip(Sx[jj].tolist(), Px[jj].tolist()):
            dl = float(delta[jj])
            cp = float(h[p_] - ell[s_]) / (L * dl)
            cm = float(h[s_] - ell[p_]) / (L * dl)
            a, b = Dk[p_], Dk[s_]
            brute_abs = float((a[:, None] + b[None, :] - (fA[:, None] - fA[None, :]).abs() / L).min()) / dl
            assert abs(brute_abs - min(cp, cm)) < 1e-12
            brute_geo = float((a[:, None] + b[None, :] - k * M).min()) / dl
            fast = float(geo_pair_min(a[None], b[None], M, [k])[0, 0]) / dl
            assert abs(brute_geo - fast) < 1e-12
            assert brute_geo <= min(cp, cm) + 1e-12


@pytest.mark.parametrize("fixed", [False, True])
def test_relaxation_chain_order(fixed):
    x, r, A, fA, LA, sigma, delta, Z, Dk, Sx, Px, W = _chain_inputs(4, 25, 6, fixed)
    Lvec = torch.tensor([LA, 1.3 * LA, 3.0 * LA])
    Omega = float(fA.max() - fA.min())
    R = relaxation_chain(Dk, fA, Lvec, LA, Omega, Sx, Px, W, delta, D_AA=weighted_l1(A, A, sigma),
                         n_background=25, geo="all")
    t = 1e-12
    assert (R["rho_abs"] <= R["rho_exact"] + t).all()
    assert (R["rho_pairbest"] <= R["rho_abs"] + t).all()
    assert (R["rho_uni"] <= R["rho_pairbest"] + t).all()
    assert (R["rho_t3"] <= R["rho_geo"] + t).all()
    # rho_exact equals the Theorem-1 normalized width
    for kk in range(3):
        ex = marginal_intervals(x, r, A, fA, float(Lvec[kk]), sigma)
        torch.testing.assert_close(R["rho_exact"][kk], ex["rho"], rtol=0, atol=1e-12)


def test_classify():
    assert classify(0.5, 0.0, 0.0, 0.0) == "not_saturated"
    assert classify(0.9, 0.85, 0.9, 0.9) == "certified"
    assert classify(0.9, 0.1, 0.7, 0.1) == "sign_asymmetry"
    assert classify(0.9, 0.1, 0.85, 0.5) == "slope_magnitude"
    assert classify(0.9, 0.1, 0.85, 0.82) == "decoupling"
    assert classify(0.9, 0.1, 0.85, float("nan")) == "abs_ok_geo_not_computed"
