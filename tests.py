"""
tests.py — unit tests for Phase 1 (run: python -m pytest -q tests.py).

Deterministic tests must pass at numerical tolerance (a failure is a bug, a solver issue,
or a violated assumption). Monte Carlo coverage is tested statistically in phase1.py, not here.
"""
import math

import numpy as np
import pytest
import torch

from bounds import (coalition_masks, mc_pair_bounds, pair_bounds_exact, shapley_weights)
from envelope import (Member, RandomLipschitz, anchor_lipschitz, envelopes, lipschitz_violation)
from hypercube_lp import HypercubeLP

torch.set_default_dtype(torch.float64)
TOL = 1e-6


def _fa(Z, a=2.0):
    t = Z[:, :1] - Z[:, 1:]
    return Z[:, 0] + a * torch.tanh(Z[:, 0]) * (t * t / (1 + t * t)).sum(1)


def _setup(d=3, n=24, seed=0, rho=0.8, kappa=1.5, fixed=False):
    g = torch.Generator().manual_seed(seed)
    cov = (1 - rho) * torch.eye(d) + rho * torch.ones(d, d)
    ch = torch.linalg.cholesky(cov)
    C = torch.randn(n, d, generator=g) @ ch.T
    x = (torch.randn(1, d, generator=g) @ ch.T)[0]
    r = (torch.randn(1, d, generator=g) @ ch.T)[0]
    sigma = 0.5 + torch.rand(d, generator=g)
    pts = [C, x[None]] + ([r[None]] if fixed else [])
    A = torch.cat(pts)
    fA = _fa(A)
    L = kappa * anchor_lipschitz(A, fA, sigma)
    return dict(C=C, x=x, r=r, A=A, fA=fA, L=L, sigma=sigma, g=g, d=d)


def test_shapley_weights_sum_to_one():
    for m in range(1, 9):
        w = shapley_weights(m)
        counts = torch.tensor([float(math.comb(m - 1, s)) for s in range(m)])
        assert abs(float((w * counts).sum()) - 1.0) < 1e-12


def test_envelope_tight_at_anchors():
    s = _setup()
    lo, hi = envelopes(s["A"], s["A"], s["fA"], s["L"], s["sigma"])
    assert torch.allclose(lo, s["fA"], atol=1e-10)
    assert torch.allclose(hi, s["fA"], atol=1e-10)


def test_random_lipschitz_is_lipschitz():
    s = _setup(d=4)
    for k in range(10):
        psi = RandomLipschitz(s["x"], 1.0, s["L"], s["sigma"], s["g"], offset=0.3)
        P = torch.randn(200, 4, generator=s["g"]) * 2
        assert lipschitz_violation(P, psi(P), s["L"], s["sigma"]) <= TOL


def test_member_is_in_class():
    s = _setup(d=3)
    for k in range(10):
        psi = RandomLipschitz(s["x"], 1.0, s["L"], s["sigma"], s["g"])
        g = Member(s["A"], s["fA"], s["L"], s["sigma"], psi)
        assert torch.allclose(g(s["A"]), s["fA"], atol=1e-10)
        P = torch.cat([torch.randn(150, 3, generator=s["g"]) * 2, s["A"]])
        assert lipschitz_violation(P, g(P), s["L"], s["sigma"]) <= TOL


def test_shapley_of_linear_function():
    s = _setup(d=4)
    beta = torch.tensor([1.0, -0.5, 0.25, 2.0])
    lp = HypercubeLP(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    phi = lp.shapley(lp.Z @ beta)
    assert np.allclose(phi, (beta * (s["x"] - s["r"])).numpy(), atol=1e-10)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chain_and_containment(fixed, seed):
    s = _setup(d=4, seed=seed, fixed=fixed)
    lo_p, hi_p, _ = (v.numpy() for v in pair_bounds_exact(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"]))
    lp = HypercubeLP(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    lo_l, hi_l = lp.interval()
    two_ld = 2 * s["L"] * lp.delta.numpy()
    assert np.all(lo_p <= lo_l + TOL) and np.all(hi_l <= hi_p + TOL)
    assert np.all((hi_p - lo_p) <= two_ld + TOL)


@pytest.mark.parametrize("fixed", [False, True])
def test_members_inside_intervals_and_certificates(fixed):
    s = _setup(d=4, seed=3, fixed=fixed, kappa=1.2)
    lp = HypercubeLP(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    lo, hi = lp.interval()
    sg = lp.signs(1e-7)
    T = lp.candidate_topk(2)
    cert = lp.certify_topk_abs(T)
    for k in range(60):
        psi = RandomLipschitz(s["x"], 1.0, s["L"], s["sigma"], s["g"], offset=float(s["fA"].mean()))
        phi = lp.shapley(Member(s["A"], s["fA"], s["L"], s["sigma"], psi)(lp.Z))
        assert np.all(phi >= lo - TOL) and np.all(phi <= hi + TOL)
        assert not np.any((sg == 1) & (phi < -TOL)) and not np.any((sg == -1) & (phi > TOL))
        if cert["certified"]:
            ab = np.abs(phi)
            out = [j for j in range(4) if j not in T]
            assert ab[T].min() >= ab[out].max() - TOL


def test_lp_solutions_lipschitz_on_all_pairs():
    """Theorem 2 sanity check: edge constraints imply all-pairs Lipschitz under weighted l1."""
    s = _setup(d=5, seed=4)
    lp = HypercubeLP(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    lp.interval()
    assert lp.verify_solutions(all_pairs=True) <= TOL


def test_abs_margin_is_a_lower_bound_over_members():
    s = _setup(d=4, seed=5)
    lp = HypercubeLP(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    phis = []
    for k in range(80):
        psi = RandomLipschitz(s["x"], 1.0, s["L"], s["sigma"], s["g"], offset=float(s["fA"].mean()))
        phis.append(lp.shapley(Member(s["A"], s["fA"], s["L"], s["sigma"], psi)(lp.Z)))
    phis = np.abs(np.array(phis))
    for i in range(4):
        for j in range(4):
            if i != j:
                pc = lp.abs_margin(i, j)
                if pc.kind in ("exact", "lower"):
                    assert pc.margin <= (phis[:, i] - phis[:, j]).min() + TOL


@pytest.mark.parametrize("kappa", [1.0, 1.1, 1.5, 2.0])
def test_saturated_anchor_closed_form(kappa):
    sig = torch.ones(2)
    x, a = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])
    A, fA = torch.stack([x, a]), torch.tensor([1.0, 0.0])
    L = kappa * anchor_lipschitz(A, fA, sig)
    for eps in (0.05, 0.2, 0.5, 0.8):
        lo, _ = HypercubeLP(x, torch.tensor([1.0 - eps, 1.0]), A, fA, L, sig).interval()
        assert abs(lo[0] - (1.0 - min(1.0 + L * eps, L * (1.0 - eps)))) <= TOL


def test_coalition_masks():
    d, n0 = 6, 50
    g = torch.Generator().manual_seed(0)
    for i in range(d):
        M = coalition_masks(i, d, n0, g)
        assert not M[..., i].any()
        assert torch.equal(M.sum(-1), torch.arange(d).view(d, 1).expand(d, n0))


@pytest.mark.parametrize("mode", ["domain", "fixed"])
def test_mc_is_exact_when_d_is_2(mode):
    """d = 2: both strata contain a single coalition, so the MC bounds equal the exact pair bounds."""
    s = _setup(d=2, seed=6, fixed=(mode == "fixed"))
    fx = _fa(s["x"][None])[0]
    fr = _fa(s["r"][None])
    res = mc_pair_bounds(s["x"], s["r"][None], s["C"], _fa(s["C"]), s["L"], s["sigma"], fx=fx, fr=fr,
                         mode=mode, n_per_stratum=8, eta=0.05, method="best",
                         gen=torch.Generator().manual_seed(1))
    lo, hi, _ = pair_bounds_exact(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    assert torch.allclose(res.lo[0], lo, atol=1e-10) and torch.allclose(res.hi[0], hi, atol=1e-10)
