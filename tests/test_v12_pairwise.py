"""
tests/test_v12_pairwise.py — Revision A2/A3: sharp pairwise bounds, attainment, certificates (reviewer Q3).
"""
import math

import numpy as np
import pytest
import torch

from bounds import all_masks, shapley_from_vertex_values, vertex_points
from envelope import anchor_lipschitz
from lipid import pairwise as PW
from lipid.joint import JointPolytope, labeling_violation, mcshane

torch.set_default_dtype(torch.float64)


def _inst(d, n, seed, fixed, kappa, scale="nonuniform"):
    g = torch.Generator().manual_seed(seed)
    ch = torch.linalg.cholesky(0.4 * torch.eye(d) + 0.6 * torch.ones(d, d))
    C = torch.randn(n, d, generator=g) @ ch.T
    x = (torch.randn(1, d, generator=g) @ ch.T)[0]
    r = (torch.randn(1, d, generator=g) @ ch.T)[0]
    sigma = 0.5 + torch.rand(d, generator=g) if scale == "nonuniform" else torch.ones(d)
    w = torch.randn(d, generator=g)
    f = lambda Z: Z @ w + 0.7 * torch.tanh(Z[:, 0] - Z[:, -1])          # noqa: E731
    A = torch.cat([C, x[None]] + ([r[None]] if fixed else []))
    fA = f(A)
    L = kappa * anchor_lipschitz(A, fA, sigma)
    return x, r, A, fA, L, sigma, f


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6])
def test_pairwise_identity_on_random_games(m):
    g = torch.Generator().manual_seed(m)
    vals = torch.randn(1 << m, generator=g)
    phi = shapley_from_vertex_values(vals, m)
    pt = PW.pair_tensors(m)
    diff = PW.pair_diff_from_vertices(vals, pt)
    ref = phi[pt["ia"]] - phi[pt["ib"]]
    torch.testing.assert_close(diff, ref, rtol=0, atol=1e-12)
    # beta weights of one pair sum to 1 (they are the Shapley weights of the (m-1)-player game)
    np.testing.assert_allclose(pt["B"].sum(1).numpy(), np.ones(len(pt["pairs"])), atol=1e-14)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("kappa", [1.0, 2.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_closed_form_equals_lp(d, fixed, kappa, seed):
    x, r, A, fA, L, sigma, _ = _inst(d, 10, seed, fixed, kappa)
    res = PW.pairwise_bounds(x, r, A, fA, L, sigma)
    for constraints in ("all_pairs", "edges"):
        jp = JointPolytope(x, r, A, fA, L, sigma, constraints=constraints)
        for p, (a, b) in enumerate(res["pt"]["pairs"]):
            lo, hi, tlo, thi = jp.pair_diff(a, b)
            c = float(res["c"][p])
            assert abs(lo - float(res["pair_lo"][p])) / c < 1e-8, (constraints, a, b)
            assert abs(hi - float(res["pair_hi"][p])) / c < 1e-8, (constraints, a, b)
            assert max(jp.violation(tlo).values()) / c < 1e-8
            assert max(jp.violation(thi).values()) / c < 1e-8


@pytest.mark.parametrize("d", [2, 3, 4, 5])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("kappa", [1.0, 1.5])
def test_extremal_labeling_attains_closed_form(d, fixed, kappa):
    x, r, A, fA, L, sigma, _ = _inst(d, 12, 7 + d, fixed, kappa)
    res = PW.pairwise_bounds(x, r, A, fA, L, sigma)
    D = float(res["delta"].sum())
    m = res["m"]
    for p, (a, b) in enumerate(res["pt"]["pairs"]):
        c = float(res["c"][p])
        P_, Q_ = res["pt"]["P"][p], res["pt"]["Q"][p]
        for side, target in (("lower", float(res["pair_lo"][p])), ("upper", float(res["pair_hi"][p]))):
            vP, vQ = PW.extremal_pair_labels(res["ell"], res["h"], P_, Q_, c, side)
            pts = torch.cat([res["Z"][P_], res["Z"][Q_]])
            vals = torch.cat([vP, vQ])
            lv = labeling_violation(pts, vals, A, fA, L, sigma)
            assert max(lv["points"], lv["points_anchors"]) <= 1e-12 * L * D
            G = mcshane(torch.cat([pts, A]), torch.cat([vals, fA]), L, sigma)
            phiG = shapley_from_vertex_values(G(res["Z"]), m)
            assert abs(float(phiG[a] - phiG[b]) - target) <= 1e-10 * c
            assert float((G(A) - fA).abs().max()) <= 1e-12 * L * D


@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("fixed", [False, True])
def test_dominance_and_containment(d, fixed):
    g = torch.Generator().manual_seed(100 + d)
    w = torch.randn(d, generator=g)
    sigma = 0.5 + torch.rand(d, generator=g)
    C = torch.randn(30, d, generator=g)
    x, r = torch.randn(d, generator=g), torch.randn(d, generator=g)
    A = torch.cat([C, x[None]] + ([r[None]] if fixed else []))
    L = float((w.abs() * sigma).max())                      # exact Lipschitz constant of the linear model
    res = PW.pairwise_bounds(x, r, A, A @ w, L, sigma)
    mlo, mhi = PW.marginal_pair_diff(res["lo"], res["hi"], res["pt"])
    c = res["c"]
    assert (res["pair_lo"] >= mlo - 1e-12 * c).all() and (res["pair_hi"] <= mhi + 1e-12 * c).all()
    phi = w * (x - r)
    td = phi[res["active"]][res["pt"]["ia"]] - phi[res["active"]][res["pt"]["ib"]]
    assert (td >= res["pair_lo"] - 1e-9 * c).all() and (td <= res["pair_hi"] + 1e-9 * c).all()
    assert (res["pair_lo"] >= -c - 1e-12).all() and (res["pair_hi"] <= c + 1e-12).all()
    # certificate monotonicity: pairwise certification contains marginal certification
    cp = PW.signed_cert(res["pair_lo"], res["pair_hi"], c)
    cm = PW.signed_cert(mlo, mhi, c)
    assert ((cm == 0) | (cp == cm)).all()


def test_null_players_consistency():
    g = torch.Generator().manual_seed(5)
    d = 5
    x, r = torch.randn(d, generator=g), torch.randn(d, generator=g)
    r[1], r[4] = x[1], x[4]
    w = torch.randn(d, generator=g)
    A = torch.cat([torch.randn(10, d, generator=g), x[None]])
    L = 1.5 * anchor_lipschitz(A, A @ w, torch.ones(d))
    act = PW.pairwise_bounds(x, r, A, A @ w, L, torch.ones(d))
    full = PW.pairwise_bounds(x, r, A, A @ w, L, torch.ones(d), include_inactive=True)
    assert act["active"] == [0, 2, 3]
    pos = {j: k for k, j in enumerate(act["active"])}
    for j in range(d):
        if j in pos:
            assert abs(float(full["lo"][j] - act["lo"][pos[j]])) < 1e-12
            assert abs(float(full["hi"][j] - act["hi"][pos[j]])) < 1e-12
        else:
            assert float(full["lo"][j]) == 0.0 and float(full["hi"][j]) == 0.0
    for p, (a, b) in enumerate(full["pairs"]):
        if a in pos and b not in pos:
            assert abs(float(full["pair_lo"][p] - act["lo"][pos[a]])) < 1e-12
            assert abs(float(full["pair_hi"][p] - act["hi"][pos[a]])) < 1e-12


def test_point_identification_when_all_hybrids_are_observed():
    d = 4
    w = torch.tensor([2.0, -1.5, 0.5, -0.2])
    x, r = torch.tensor([0.3, -1.0, 0.8, 1.2]), torch.tensor([-0.5, 0.4, 0.1, 0.0])
    Z = vertex_points(x, r, list(range(d)), all_masks(d))
    res = PW.pairwise_bounds(x, r, Z, Z @ w, float(w.abs().max()), torch.ones(d))
    phi = w * (x - r)
    td = phi[res["pt"]["ia"]] - phi[res["pt"]["ib"]]
    torch.testing.assert_close(res["pair_lo"], td, rtol=0, atol=1e-12)
    torch.testing.assert_close(res["pair_hi"], td, rtol=0, atol=1e-12)


def test_certified_topk_relation():
    # strict order 2 > 0 > 3 > 1 fully certified
    m = 4
    order = [2, 0, 3, 1]
    cert = []
    for a, b in PW.pair_list(m):
        cert.append(1 if order.index(a) < order.index(b) else -1)
    G = PW.dominance_matrix(cert, m)
    assert PW.top1(G) == 2
    assert sorted(PW.certified_topk(G, 2)) == [0, 2]
    # remove 2 > 0: top-1 is lost, top-2 {0, 2} still certified
    cert2 = [0 if (a, b) == (0, 2) else v for (a, b), v in zip(PW.pair_list(m), cert)]
    G2 = PW.dominance_matrix(cert2, m)
    assert PW.top1(G2) == -1
    assert sorted(PW.certified_topk(G2, 2)) == [0, 2]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_abs_exact_certificate_matches_four_lp_margin(seed):
    import hypercube_lp as HL
    x, r, A, fA, L, sigma, _ = _inst(4, 40, 50 + seed, True, 1.0)
    res = PW.pairwise_bounds(x, r, A, fA, L, sigma)
    jp = JointPolytope(x, r, A, fA, L, sigma)
    hl = HL.HypercubeLP(x, r, A, fA, L, sigma)
    am = PW.abs_marginal_cert(res["lo"], res["hi"], res["pt"], res["c"])
    for p, (a, b) in enumerate(res["pt"]["pairs"]):
        c = float(res["c"][p])
        dlo, dhi = float(res["pair_lo"][p]), float(res["pair_hi"][p])
        slo, shi, _, _ = jp.pair_sum(a, b)
        ex = int(PW.abs_exact_cert(np.array(dlo), np.array(dhi), np.array(slo), np.array(shi), c))
        mab = hl.abs_margin(res["active"][a], res["active"][b], tol=0.0)
        mba = hl.abs_margin(res["active"][b], res["active"][a], tol=0.0)
        if mab.margin > 1e-5 * c:
            assert ex == 1
        if mba.margin > 1e-5 * c:
            assert ex == -1
        if ex == 1:
            assert mab.margin > -1e-9 * c
        if ex == -1:
            assert mba.margin > -1e-9 * c
        if int(am[p]) != 0:               # Proposition 2 is sufficient, never stronger than the exact certificate
            assert ex == int(am[p])
