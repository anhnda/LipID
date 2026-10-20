"""
tests/test_calibration.py — CAL-01. Gradient sampling here is a TEST of the analytic certificates, not a proof.
"""
import inspect
import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

import lipid.calibration as CAL
import phase1 as P

torch.set_default_dtype(torch.float64)
REL = 1e-9
MODELS = CAL.MODELS


def _sigma(kind, d, seed):
    if kind == "unit":
        return torch.ones(d)
    g = torch.Generator().manual_seed(1000 + seed)
    return 0.2 + 3.0 * torch.rand(d, generator=g)


def _weighted_grad(f, Z, sigma):
    Z = Z.clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(f(Z).sum(), Z)
    return (gr.abs() * sigma).max(1).values          # max_j sigma_j |d_j f(z)|  (dual norm of weighted l1)


# ------------------------------------------------------------------------------------------------- constants
def test_cq_closed_form():
    qp = lambda t: 2 * t / (1 + t * t) ** 2
    res = minimize_scalar(lambda t: -abs(qp(t)), bounds=(0.0, 5.0), method="bounded", options=dict(xatol=1e-12))
    assert abs(-res.fun - CAL.CQ) < 1e-10
    t = np.linspace(-50, 50, 2_000_001)
    assert np.max(np.abs(qp(t))) <= CAL.CQ * (1 + 1e-15)
    assert abs(abs(qp(1 / math.sqrt(3))) - CAL.CQ) < 1e-15


def test_ct_certified_upper_bound():
    ub, lo, hi = CAL.certify_ct()
    g = lambda u: math.sqrt(u) / math.cosh(u) ** 2
    res = minimize_scalar(lambda u: -g(u), bounds=(0.0, 5.0), method="bounded", options=dict(xatol=1e-12))
    assert lo < res.x < hi or abs(res.x - 0.5 * (lo + hi)) < 1e-6
    assert ub >= -res.fun
    assert ub - (-res.fun) < 1e-8                    # certificate is valid AND essentially sharp
    u = np.linspace(0.0, 20.0, 2_000_001)
    assert np.max(np.sqrt(u) / np.cosh(u) ** 2) <= ub
    assert 4 * lo * math.tanh(lo) < 1 < 4 * hi * math.tanh(hi)


# ------------------------------------------------------------------------- certificate >= observed gradients
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("d", [2, 3, 4, 6])
@pytest.mark.parametrize("a", [4.0, -4.0, 0.5])
@pytest.mark.parametrize("skind", ["unit", "nonuniform"])
def test_gradient_never_exceeds_certificate(name, d, a, skind):
    sigma = _sigma(skind, d, d)
    Lc = CAL.L_cert(name, d, sigma, a)
    f = P.make_model(name, a)
    g = torch.Generator().manual_seed(7 + d)
    worst = 0.0
    for scale in (0.05, 0.5, 1.0, 3.0, 10.0, 1e3):           # includes tanh saturation (large |z|)
        Z = scale * torch.randn(4000, d, generator=g)
        worst = max(worst, float(_weighted_grad(f, Z, sigma).max()))
    # structured points: diagonal (unstable_id peak), z_1 - z_j = 1/sqrt(3) (fa), sum (z_1 - z_j)^2 = u* (stable)
    t = torch.linspace(-3, 3, 601)
    Zs = [t[:, None].repeat(1, d)]
    Zs.append(torch.cat([t[:, None], (t - 1 / math.sqrt(3))[:, None].repeat(1, d - 1)], 1))
    us = math.sqrt(0.5216 / max(d - 1, 1))
    Zs.append(torch.cat([t[:, None], (t - us)[:, None].repeat(1, d - 1)], 1))
    for Zt in Zs:
        worst = max(worst, float(_weighted_grad(f, Zt, sigma).max()))
    # adversarial: gradient ascent on the weighted gradient norm from random starts
    Z0 = torch.randn(64, d, generator=g)
    Z = Z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        (gr,) = torch.autograd.grad(f(Z).sum(), Z, create_graph=True)
        obj = (gr.abs() * sigma).max(1).values
        worst = max(worst, float(obj.max()))
        if not obj.requires_grad:        # constant gradient (linear generator): nothing to ascend
            break
        (-obj.sum()).backward()
        opt.step()
    assert worst <= Lc * (1 + REL), (name, d, a, skind, worst, Lc)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("a", [4.0, -4.0])
def test_finite_difference_slopes_below_certificate(name, a):
    d = 5
    sigma = _sigma("nonuniform", d, 3)
    Lc = CAL.L_cert(name, d, sigma, a)
    f = P.make_model(name, a)
    g = torch.Generator().manual_seed(11)
    for scale, eps in ((1.0, 1.0), (1.0, 1e-3), (10.0, 1e-2), (0.3, 1e-4)):
        U = scale * torch.randn(20000, d, generator=g)
        V = U + eps * torch.randn(20000, d, generator=g)
        D = ((U - V).abs() / sigma).sum(1)
        sl = (f(U) - f(V)).abs() / D
        assert float(sl.max()) <= Lc * (1 + 1e-7)


@pytest.mark.parametrize("d", [2, 3, 5, 6])
def test_exact_for_linear_and_unstable_positive_a(d):
    """For a > 0 the certificate equals the analytic L_true of the archived code (no change of the oracle L*)."""
    sigma = _sigma("nonuniform", d, d)
    for name in ("linear", "unstable_id"):
        Lt = P.lipschitz_true(name, d, sigma, 4.0)
        assert abs(CAL.L_cert(name, d, sigma, 4.0) - Lt) <= 1e-13 * Lt


def test_legacy_unstable_ltrue_is_invalid_for_negative_a():
    """Documents the legacy bug (signed a). Not used in archived runs (a = 4)."""
    d, a = 3, -4.0
    sigma = torch.ones(d)
    legacy = P.lipschitz_true("unstable_id", d, sigma, a)
    f = P.make_model("unstable_id", a)
    obs = float(_weighted_grad(f, torch.zeros(1, d), sigma).max())
    assert legacy < obs <= CAL.L_cert("unstable_id", d, sigma, a) * (1 + REL)


# -------------------------------------------------------------------------------- independence from responses
def test_certificate_never_calls_a_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("calibration must not construct or evaluate a response function")
    monkeypatch.setattr(P, "make_model", boom)
    for name in MODELS:
        for d in (2, 3, 6):
            CAL.certify(name, d, torch.ones(d), 4.0)
    src = inspect.getsource(CAL)
    code_lines = [ln for ln in src.splitlines() if not ln.strip().startswith(("#", '"', "'")) and "make_model(" in ln]
    assert code_lines == []


def test_certificate_depends_only_on_sigma_and_parameters():
    s = torch.tensor([0.7, 1.3, 2.0])
    a = CAL.certify("fa", 3, s, 4.0)
    b = CAL.certify("fa", 3, s.numpy(), 4.0)
    assert a.L_cert == b.L_cert and not a.uses_hybrid_responses and not a.uses_anchor_labels
    assert a.scope == "global"


def test_rejects_bad_sigma():
    with pytest.raises(ValueError):
        CAL.certify("fa", 3, torch.tensor([1.0, 0.0, 1.0]), 4.0)
    with pytest.raises(ValueError):
        CAL.certify("fa", 3, torch.ones(2), 4.0)
