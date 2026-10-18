"""
tests.py — unit tests for Phase 1 (run: python -m pytest -q tests.py).

Deterministic tests must pass at numerical tolerance (a failure is a bug, a solver issue,
or a violated assumption). Monte Carlo coverage is tested statistically in phase1.py, not here.
"""
import math

import numpy as np
import pytest
import torch

from scipy.optimize import linprog

from bounds import (coalition_masks, geometric_rho_inf, hybrid_vertices, mc_pair_bounds,
                    pair_bounds_exact, shapley_full, shapley_weights)
from envelope import (Member, RandomLipschitz, anchor_lipschitz, envelopes, lipschitz_after_adding,
                      lipschitz_violation)
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


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("kappa", [1.0, 1.3, 2.0, 5.0])
def test_lp_equals_pair_d2_endpoint_fixed(seed, kappa):
    """Structural identity (regression test for envelopes + hypercube LP):
    d = 2, endpoint-fixed anchors, weighted l1  =>  LP interval == pairwise interval (so Gamma_LP == B_pair),
    with extra calibration anchors inside and far outside the hybrid box, for any feasible L."""
    s = _setup(d=2, n=30, seed=seed, fixed=True)
    far = 5.0 * torch.randn(10, 2, generator=torch.Generator().manual_seed(100 + seed))
    A = torch.cat([s["A"], far])
    fA = _fa(A)
    L = kappa * anchor_lipschitz(A, fA, s["sigma"])
    lo_p, hi_p, _ = (v.numpy() for v in pair_bounds_exact(s["x"], s["r"], A, fA, L, s["sigma"]))
    lp = HypercubeLP(s["x"], s["r"], A, fA, L, s["sigma"])
    lo_l, hi_l = lp.interval()
    assert np.allclose(lo_l, lo_p, atol=TOL) and np.allclose(hi_l, hi_p, atol=TOL)
    assert np.allclose(hi_l - lo_l, hi_p - lo_p, atol=TOL)
    assert np.all(lp.cross_free_edges() == 0)


@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("kappa", [1.0, 1.5, 3.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_pairwise_interval_is_exact(d, fixed, kappa, seed):
    """Proposition (bounds.py): under weighted l1 the pairwise interval equals the exact LP interval,
    for any d, anchor mode, feasible L, with extra anchors far outside the hybrid box."""
    s = _setup(d=d, n=40, seed=10 + seed, fixed=fixed)
    far = 4.0 * torch.randn(8, d, generator=torch.Generator().manual_seed(200 + seed))
    A = torch.cat([s["A"], far])
    fA = _fa(A)
    L = kappa * anchor_lipschitz(A, fA, s["sigma"])
    lo_p, hi_p, _ = (v.numpy() for v in pair_bounds_exact(s["x"], s["r"], A, fA, L, s["sigma"]))
    lp = HypercubeLP(s["x"], s["r"], A, fA, L, s["sigma"])
    lo_l, hi_l = lp.interval()
    assert np.allclose(lo_l, lo_p, atol=TOL) and np.allclose(hi_l, hi_p, atol=TOL)


def test_abs_margin_never_below_interval_separation():
    """Exact joint LP margin >= interval-separation lower bound min|phi_i| - max|phi_j|."""
    s = _setup(d=4, seed=11)
    lp = HypercubeLP(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    lo, hi = lp.interval()
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            pc = lp.abs_margin(i, j)
            if pc.kind == "exact":
                min_i = 0.0 if lo[i] <= 0 <= hi[i] else min(abs(lo[i]), abs(hi[i]))
                max_j = max(abs(lo[j]), abs(hi[j]))
                assert pc.margin >= min_i - max_j - TOL


def test_cross_free_edges_d2_domain():
    """d = 2, domain mode (r not an anchor): exactly one useful cross edge per feature (r -- other hybrid)."""
    s = _setup(d=2, seed=7)
    lp = HypercubeLP(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    assert lp.cross_free_edges().tolist() == [1, 1]


# ============================================================ amendment v3 (P0)
# Independent references: plain Python loops / an LP written from scratch, sharing no helper with the
# code under test (envelopes, vertex_points, all_masks, shapley_weights, HypercubeLP).

def _envelopes_loop(Z, A, fA, L, sigma):
    sig = sigma.tolist()
    lo, hi = [], []
    for z in Z.tolist():
        best_lo, best_hi = -math.inf, math.inf
        for u, fu in zip(A.tolist(), fA.tolist()):
            dist = sum(abs(p - q) / s for p, q, s in zip(z, u, sig))
            best_lo = max(best_lo, fu - L * dist)
            best_hi = min(best_hi, fu + L * dist)
        lo.append(best_lo)
        hi.append(best_hi)
    return torch.tensor(lo), torch.tensor(hi)


def _independent_interval(x, r, A, fA, L, sigma):
    """Exact class interval by an LP over ALL vertex pairs + explicit anchor constraints
    (no hypercube edges, no envelope box). Valid by McShane: such vertex values extend to a member."""
    d = x.numel()
    xs, rs, ss = x.tolist(), r.tolist(), sigma.tolist()
    active = [j for j in range(d) if xs[j] != rs[j]]
    m = len(active)
    verts = []
    for v in range(2 ** m):
        z = list(rs)
        for jj, j in enumerate(active):
            if (v >> jj) & 1:
                z[j] = xs[j]
        verts.append(z)

    def dist(a, b):
        return sum(abs(p - q) / s for p, q, s in zip(a, b, ss))

    n = len(verts)
    rows, rhs = [], []
    for a in range(n):
        for b in range(a + 1, n):
            dd = L * dist(verts[a], verts[b])
            row = [0.0] * n
            row[a], row[b] = 1.0, -1.0
            rows.append(row)
            rhs.append(dd)
            row = [0.0] * n
            row[a], row[b] = -1.0, 1.0
            rows.append(row)
            rhs.append(dd)
        for u, fu in zip(A.tolist(), fA.tolist()):
            dd = L * dist(verts[a], u)
            row = [0.0] * n
            row[a] = 1.0
            rows.append(row)
            rhs.append(fu + dd)        # t_a - f(u) <= L d
            row = [0.0] * n
            row[a] = -1.0
            rows.append(row)
            rhs.append(dd - fu)        # f(u) - t_a <= L d
    lo, hi = np.zeros(d), np.zeros(d)
    for jj, j in enumerate(active):
        c = [0.0] * n
        for v in range(n):
            if (v >> jj) & 1:
                continue
            k = bin(v).count("1")
            w = math.factorial(k) * math.factorial(m - 1 - k) / math.factorial(m)
            c[v | (1 << jj)] += w
            c[v] -= w
        rmin = linprog(c, A_ub=rows, b_ub=rhs, bounds=[(None, None)] * n, method="highs")
        rmax = linprog([-ci for ci in c], A_ub=rows, b_ub=rhs, bounds=[(None, None)] * n, method="highs")
        assert rmin.status == 0 and rmax.status == 0
        lo[j], hi[j] = rmin.fun, -rmax.fun
    return lo, hi


def test_envelopes_match_independent_loop():
    s = _setup(d=4, seed=8)
    Z = torch.randn(30, 4, generator=s["g"]) * 2
    lo, hi = envelopes(Z, s["A"], s["fA"], s["L"], s["sigma"])
    lo2, hi2 = _envelopes_loop(Z, s["A"], s["fA"], s["L"], s["sigma"])
    assert torch.allclose(lo, lo2, atol=1e-10) and torch.allclose(hi, hi2, atol=1e-10)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("fixed", [False, True])
def test_pairwise_matches_independent_lp(d, fixed):
    """Proposition 3 checked against an LP that shares no code with pair_bounds_exact / HypercubeLP."""
    s = _setup(d=d, n=20, seed=20 + d, fixed=fixed)
    lo_p, hi_p, _ = pair_bounds_exact(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    lo_i, hi_i = _independent_interval(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    assert np.allclose(lo_p.numpy(), lo_i, atol=TOL) and np.allclose(hi_p.numpy(), hi_i, atol=TOL)


@pytest.mark.parametrize("fixed", [False, True])
def test_rho_monotone_nested_and_bounded_by_rho_inf(fixed):
    """rho(L) non-decreasing in L, intervals nested, rho(L) <= rho_inf, rho(L) -> rho_inf."""
    s = _setup(d=4, seed=9, fixed=fixed)
    LA = anchor_lipschitz(s["A"], s["fA"], s["sigma"])
    delta = (s["x"] - s["r"]).abs() / s["sigma"]
    rinf = geometric_rho_inf(s["x"], s["r"], s["A"], s["sigma"])
    prev = None
    for k in (1.0, 1.1, 1.5, 2.0, 4.0, 16.0, 1e3):
        L = k * LA
        lo, hi, _ = pair_bounds_exact(s["x"], s["r"], s["A"], s["fA"], L, s["sigma"])
        rho = (hi - lo) / (2 * L * delta)
        assert torch.all(rho <= rinf + TOL)
        if prev is not None:
            p_rho, p_lo, p_hi = prev
            assert torch.all(rho >= p_rho - TOL)
            assert torch.all(lo <= p_lo + TOL) and torch.all(hi >= p_hi - TOL)
        prev = (rho, lo, hi)
    L = 1e9 * LA
    lo, hi, _ = pair_bounds_exact(s["x"], s["r"], s["A"], s["fA"], L, s["sigma"])
    assert torch.allclose((hi - lo) / (2 * L * delta), rinf, atol=1e-5)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_frozen_phi_inside_interval_at_LV(fixed, seed):
    """At L_V = Lip(f on hybrids ∪ A) the certified interval contains phi(f) (McShane), and L_V >= L_A."""
    s = _setup(d=4, seed=30 + seed, fixed=fixed)
    LA = anchor_lipschitz(s["A"], s["fA"], s["sigma"])
    Z, active = hybrid_vertices(s["x"], s["r"])
    fZ = _fa(Z)
    LV = lipschitz_after_adding(LA, s["A"], s["fA"], Z, fZ, s["sigma"])
    assert LV >= LA
    phi = shapley_full(fZ, active, 4)
    lo, hi, _ = pair_bounds_exact(s["x"], s["r"], s["A"], s["fA"], LV, s["sigma"])
    assert torch.all(phi >= lo - TOL) and torch.all(phi <= hi + TOL)


def test_linear_LA_le_LV_le_Ltrue():
    s = _setup(d=4, seed=12)
    w = torch.tensor([1.0, -0.5, 0.25, 2.0])
    A = s["A"]
    fA = A @ w
    LA = anchor_lipschitz(A, fA, s["sigma"])
    Z, _ = hybrid_vertices(s["x"], s["r"])
    LV = lipschitz_after_adding(LA, A, fA, Z, Z @ w, s["sigma"])
    Ltrue = float((w.abs() * s["sigma"]).max())
    assert LA <= LV <= Ltrue * (1 + 1e-9)


@pytest.mark.parametrize("L", [1.0, 1.1, 1.5, 2.0, 5.0, 100.0])
def test_uv_counterexample_rho_closed_form(L):
    """f = u + v, anchors {(0,0), (1,1)}: rho(L) = 1 - 1/L while rho_inf = 1 (upper bound only)."""
    sig = torch.ones(2)
    x, r = torch.tensor([1.0, 1.0]), torch.zeros(2)
    A = torch.stack([r, x])
    fA = A.sum(1)
    assert abs(anchor_lipschitz(A, fA, sig) - 1.0) < 1e-12
    lo, hi, _ = pair_bounds_exact(x, r, A, fA, L, sig)
    assert torch.allclose((hi - lo) / (2 * L), torch.full((2,), 1 - 1 / L), atol=1e-12)
    assert torch.allclose(geometric_rho_inf(x, r, A, sig), torch.ones(2), atol=1e-12)


def test_mc_radius_scale_zero_gives_clamped_point_estimates():
    """Negative-control mechanics: radius_scale = 0 -> bounds are the clamped stratified point estimates."""
    s = _setup(d=4, seed=13)
    fx = _fa(s["x"][None])[0]
    res = mc_pair_bounds(s["x"], s["r"][None], s["C"], _fa(s["C"]), s["L"], s["sigma"], fx=fx,
                         mode="domain", n_per_stratum=8, method="hoeffding", radius_scale=0.0,
                         gen=torch.Generator().manual_seed(3))
    Ld = s["L"] * (s["x"] - s["r"]).abs() / s["sigma"]
    assert torch.allclose(res.lo[0], torch.maximum(res.a_hat[0], -Ld), atol=1e-12)
    assert torch.allclose(res.hi[0], torch.minimum(res.b_hat[0], Ld), atol=1e-12)


def test_shapley_full_matches_lp_shapley():
    s = _setup(d=4, seed=14)
    Z, active = hybrid_vertices(s["x"], s["r"])
    lp = HypercubeLP(s["x"], s["r"], s["A"], s["fA"], s["L"], s["sigma"])
    assert np.allclose(shapley_full(_fa(Z), active, 4).numpy(), lp.shapley(_fa(lp.Z)), atol=1e-12)