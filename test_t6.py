"""
test_t6.py — tests for the directional decomposition (run: python -m pytest -q test_t6.py).
Independent references: pair_bounds_exact (bounds.py) for rho, and the O(|A|^2) formulas for box / det.
"""
import itertools
import math

import pytest
import torch

from bounds import all_masks, pair_bounds_exact, vertex_points
from envelope import anchor_lipschitz, weighted_l1
from t6_directional import (box_oracle_T, det_T, directional, edge_sets, efficiency_cap, endpoint_band_bound,
                            envelopes_from_D, feature_stats, lb_from_T, margin_diagnostic, pair_tensor,
                            psi_certificate, quantized_T, separation_decomposition, slope_ratio,
                            straddler_exact)

torch.set_default_dtype(torch.float64)
TOL = 1e-9


def _fa(Z, a=2.0):
    t = Z[:, :1] - Z[:, 1:]
    return Z[:, 0] + a * torch.tanh(Z[:, 0]) * (t * t / (1 + t * t)).sum(1)


def _lin(Z):
    return Z @ torch.linspace(1.0, 0.2, Z.shape[1])


def _setup(d, n, seed, rho, kappa, fixed, f):
    g = torch.Generator().manual_seed(seed)
    ch = torch.linalg.cholesky((1 - rho) * torch.eye(d) + rho * torch.ones(d, d))
    C = torch.randn(n, d, generator=g) @ ch.T
    x = (torch.randn(1, d, generator=g) @ ch.T)[0]
    r = (torch.randn(1, d, generator=g) @ ch.T)[0]
    sigma = 0.5 + torch.rand(d, generator=g)
    A = torch.cat([C, x[None]] + ([r[None]] if fixed else []))
    fA = f(A)
    L_min = anchor_lipschitz(A, fA, sigma)
    return x, r, A, fA, kappa * L_min, L_min, sigma


def _instance(d, seed, kappa, fixed, f, rho=0.6, n=20):
    x, r, A, fA, L, L_min, sigma = _setup(d, n, seed, rho, kappa, fixed, f)
    delta = (x - r).abs() / sigma
    active = torch.nonzero(delta > 0).flatten().tolist()
    m = len(active)
    Z = vertex_points(x, r, active, all_masks(m))
    D = weighted_l1(Z, A, sigma)
    DA = weighted_l1(A, A, sigma)
    ell, h = envelopes_from_D(D, fA, L)
    return dict(x=x, r=r, A=A, fA=fA, L=L, L_min=L_min, sigma=sigma, delta=delta, active=active,
                Z=Z, D=D, DA=DA, ell=ell, h=h, edges=edge_sets(m))


GRID = list(itertools.product((2, 3, 4, 5), (0, 1, 2), (1.0, 1.5), (False, True), (_fa, _lin)))


@pytest.mark.parametrize("d,seed,kappa,fixed,f", GRID)
def test_directional_identities(d, seed, kappa, fixed, f):
    I = _instance(d, seed, kappa, fixed, f)
    lo, hi, _ = pair_bounds_exact(I["x"], I["r"], I["A"], I["fA"], I["L"], I["sigma"])
    dF = (I["fA"][:, None] - I["fA"][None, :]).abs()
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        Ld = float(I["L"] * I["delta"][j])
        cp, cm = directional(I["ell"], I["h"], S, P, Ld)
        st = feature_stats(w, cp, cm)
        # exact rho and interval from (c+, c-) match pair_bounds_exact
        assert abs(st.rho - float((hi[j] - lo[j]) / (2 * Ld))) < TOL
        assert abs(Ld * st.lo_n - float(lo[j])) < TOL and abs(Ld * st.hi_n - float(hi[j])) < TOL
        # identity T_box = min(B+, B-): envelope version == O(|A|^2) oracle
        R = pair_tensor(I["D"], S, P)
        assert abs(lb_from_T(box_oracle_T(R, dF, I["L"]), Ld, w) - st.dmin) < TOL
        # (I1)  2 rho - 1 <= dmin <= rho
        assert st.dmin <= st.rho + TOL
        assert st.dmin >= 2 * st.rho - 1 - TOL
        # (I3)  certified sign => rho < 1/2
        if st.sign(Ld, 1e-9) != 0:
            assert st.rho < 0.5 + TOL
        # (I2) when every edge has c± >= 0
        if bool((torch.minimum(cp, cm) >= 0).all()):
            a = float((w * (cp.clamp(max=1) - cm.clamp(max=1)).abs()).sum()) / 2
            assert abs(st.asym - a) < TOL


@pytest.mark.parametrize("d,seed,kappa,fixed,f", GRID)
def test_slope_ladder(d, seed, kappa, fixed, f):
    I = _instance(d, seed, kappa, fixed, f)
    dF = (I["fA"][:, None] - I["fA"][None, :]).abs()
    ratio = slope_ratio(I["DA"], dF, I["L_min"])
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        Ld = float(I["L"] * I["delta"][j])
        R = pair_tensor(I["D"], S, P)
        det = lb_from_T((I["L"] * R - I["L_min"] * I["DA"][None]).amin(dim=(1, 2)), Ld, w)
        box = lb_from_T(box_oracle_T(R, dF, I["L"]), Ld, w)
        vals = [lb_from_T(quantized_T(R, I["DA"], ratio, I["L"], I["L_min"], K), Ld, w) for K in (1, 2, 4, 8, 64)]
        assert abs(vals[0] - det) < TOL                       # K = 1 is det
        assert all(b >= a - TOL for a, b in zip(vals, vals[1:]))  # non-decreasing
        assert vals[-1] <= box + TOL                          # never above box


@pytest.mark.parametrize("d,seed,fixed", list(itertools.product((3, 4, 5), (0, 1, 2), (False, True))))
def test_endpoint_band(d, seed, fixed):
    """(I4): s_S <= 1 + 2 D_S / delta_i (x in A); fixed mode 1 + 2 min(B_S, D_S) / delta_i."""
    I = _instance(d, seed, 1.0, fixed, _fa)
    q = I["D"].min(dim=1).values
    da = I["delta"][I["active"]]
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        s = (q[P] + q[S]) / I["delta"][j]
        assert bool((s <= endpoint_band_bound(da, S, jj, fixed=fixed) + TOL).all())


@pytest.mark.parametrize("d,seed,f", list(itertools.product((2, 3, 4, 5), (0, 1, 2), (_fa, _lin))))
def test_fixed_mode_det_is_zero_at_Lmin(d, seed, f):
    """(I5): with x, r in A and L = L_min, the pair (x, r) gives R - d = -delta_i on every edge -> det = 0."""
    I = _instance(d, seed, 1.0, True, f)
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        Ld = float(I["L"] * I["delta"][j])
        R = pair_tensor(I["D"], S, P)
        assert lb_from_T(det_T(R, I["DA"], I["L"], I["L_min"]), Ld, w) <= TOL
        # B_S is never empty in fixed mode: (x, r) has g = -1 on every edge -> psi_empty_share = 0
        assert bool((((R - I["DA"][None]) / float(I["delta"][j])) < 1.0 - 1e-12).flatten(1).any(1).all())


@pytest.mark.parametrize("d,seed,n", list(itertools.product((2, 3, 4, 5), range(6), (2, 4, 8))))
def test_domain_top_edge_G_equals_one_iff_x_extreme(d, seed, n):
    """ChatGPT round 6: domain mode, top edge (S = N \\ {i}, z+ = x): G_S = 1 iff every background anchor lies on
    the far side of x from r in coordinate i (x_i > r_i: min_C v_i >= x_i; x_i < r_i: max_C v_i <= x_i);
    otherwise G_S < 1. Rank argument: P(G_S = 1) = 2 / ((n+1)(n+2)) for exchangeable coordinates."""
    I = _instance(d, seed, 1.0, False, _fa, n=n)
    C = I["A"][:-1]
    m = len(I["active"])
    top = 2 ** m - 1
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        e = int((P == top).nonzero()[0])
        R = pair_tensor(I["D"], S, P)
        G = float(((R[e] - I["DA"]) / float(I["delta"][j])).min())
        xi, ri = float(I["x"][j]), float(I["r"][j])
        extreme = bool((C[:, j] >= xi).all()) if xi > ri else bool((C[:, j] <= xi).all())
        assert G <= 1.0 + 1e-12
        assert (abs(G - 1.0) < 1e-9) == extreme


@pytest.mark.parametrize("d,seed,kappa,f", list(itertools.product((2, 3, 4, 5), (0, 1, 2), (1.0, 1.5), (_fa, _lin))))
def test_fixed_mode_efficiency_cap(d, seed, kappa, f):
    """(I6): rho <= (1 + min(1, X))/2, dmin <= clamp(X), X < 0 forces sign(phi_i) = sign(f(x) - f(r))."""
    I = _instance(d, seed, kappa, True, f)
    D_xr = float(I["delta"].sum())
    Delta = float(f(I["x"][None]) - f(I["r"][None]))
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        dj = float(I["delta"][j])
        Ld = I["L"] * dj
        cp, cm = directional(I["ell"], I["h"], S, P, Ld)
        st = feature_stats(w, cp, cm)
        cap = efficiency_cap(D_xr, Delta, I["L"], dj, 1e-9)
        assert st.rho <= cap["rho_cap"] + TOL
        assert st.dmin <= cap["box_cap"] + TOL
        if cap["forced_sign"] != 0:
            assert st.sign(Ld, 1e-9) == cap["forced_sign"]
        assert abs(st.asym - st.asym_pos - st.neg_part) < TOL


@pytest.mark.parametrize("d,seed,kappa,fixed", list(itertools.product((3, 4, 5), (0, 1), (1.0, 1.5), (False, True))))
def test_margin_diagnostic_identity(d, seed, kappa, fixed):
    """min over pairs of the box margin V equals min(c+, c-) on every edge (no argmin involved)."""
    I = _instance(d, seed, kappa, fixed, _fa)
    dF = (I["fA"][:, None] - I["fA"][None, :]).abs()
    ratio = slope_ratio(I["DA"], dF, I["L_min"])
    endpoints = [I["A"].shape[0] - (2 if fixed else 1)] + ([I["A"].shape[0] - 1] if fixed else [])
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        dj = float(I["delta"][j])
        cp, cm = directional(I["ell"], I["h"], S, P, I["L"] * dj)
        md = margin_diagnostic(pair_tensor(I["D"], S, P), I["DA"], ratio, cp, cm, w, I["L"], I["L_min"], dj,
                               endpoints)
        assert md["m_err"] < 1e-9
        assert md["cert_viol"] == 0                    # decoupled certificates imply full saturation
        assert abs(md["G_strad"] + md["G_neg"] + md["G_01"] + md["G_ge1"] - 1.0) < 1e-12
        # exact coordinate-order straddler test agrees with the round-off-tolerance test
        Sx = straddler_exact(I["A"], I["x"], I["r"], I["Z"][P], j)
        md2 = margin_diagnostic(pair_tensor(I["D"], S, P), I["DA"], ratio, cp, cm, w, I["L"], I["L_min"], dj,
                                endpoints, strad_exact=Sx)
        assert md2["strad_mismatch"] == 0


@pytest.mark.parametrize("d,seed,kappa,fixed", list(itertools.product((2, 3), (0, 1, 2), (1.0, 1.5), (False, True))))
def test_psi_certificate_bruteforce(d, seed, kappa, fixed):
    """psi certificate recomputed by explicit loops over pairs (independent of the vectorized code)."""
    I = _instance(d, seed, kappa, fixed, _fa, n=8)
    dF = (I["fA"][:, None] - I["fA"][None, :]).abs()
    ratio = slope_ratio(I["DA"], dF, I["L_min"])
    k = I["L_min"] / I["L"]
    N = I["A"].shape[0]
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        dj = float(I["delta"][j])
        R = pair_tensor(I["D"], S, P)
        g = (R - I["DA"][None]) / dj
        h = (I["DA"] / dj)[None]
        cert, empty, _ = psi_certificate(g, h, ratio, k)
        sd = separation_decomposition(g, h, ratio, k)
        for e in range(R.shape[0]):
            psi, Q, any_b, Vmin, max_ab, nB = -math.inf, 0.0, False, math.inf, -math.inf, 0
            for u in range(N):
                for v in range(N):
                    gg, hv, t = float(g[e, u, v]), float(h[0, u, v]), float(ratio[u, v])
                    Vmin = min(Vmin, gg + (1 - k * t) * hv)
                    if gg < 1.0 - 1e-12:
                        any_b, nB = True, nB + 1
                        a = (1 - gg) / hv
                        psi, Q, max_ab = max(psi, a), max(Q, t), max(max_ab, a + k * t)
            ref = (not any_b) or (k * Q <= 1 - psi)
            assert bool(cert[e]) == ref
            if ref:
                assert Vmin >= 1 - 1e-9                    # soundness against the pairwise minimum
            assert bool(sd["empty"][e]) == (not any_b) and int(sd["nB"][e]) == nB
            if any_b:
                C, mu = psi + k * Q - max_ab, 1 - max_ab
                assert abs(float(sd["C"][e]) - C) < 1e-9 and abs(float(sd["mu"][e]) - mu) < 1e-9
                assert C >= -1e-12                         # separation loss is non-negative
                if abs(psi + k * Q - 1) > 1e-9:
                    assert ref == (C <= mu)                # C1: psi certifies <=> C <= mu
                if abs(mu) > 1e-6:                         # exact saturation <=> mu >= 0
                    assert (mu > 0) == (Vmin >= 1 - 1e-9)


def test_G_equal_one_is_not_a_null_event():
    """ChatGPT round 5: domain mode, x=(1,1), r=(0,0), A={(2,2),(1,1)}, f=(1,0), L_min=1/2. For feature 1 and the
    edge z+=(1,1), z-=(0,1): G_S = 1 (pair (x,x)), B_S empty, min V = 1 -> trivially certified, yet 'tight'."""
    x, r = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    A = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
    fA = torch.tensor([1.0, 0.0])
    sigma = torch.ones(2)
    L_min = anchor_lipschitz(A, fA, sigma)
    assert abs(L_min - 0.5) < 1e-12
    Z = vertex_points(x, r, [0, 1], all_masks(2))
    S, P, w = edge_sets(2)[0]                          # feature 0; edges S = {} and S = {feature 1}
    R = pair_tensor(weighted_l1(Z, A, sigma), S, P)
    DA = weighted_l1(A, A, sigma)
    g, h = (R - DA[None]) / 1.0, DA[None] / 1.0
    e = int((S == 2).nonzero()[0])                     # the edge with z+ = (1,1) = x, z- = (0,1)
    assert abs(float(g[e].min()) - 1.0) < 1e-12
    ratio = slope_ratio(DA, (fA[:, None] - fA[None, :]).abs(), L_min)
    sd = separation_decomposition(g, h, ratio, 1.0)
    assert bool(sd["empty"][e])
    assert abs(float((g[e] + (1 - ratio) * h[0]).min()) - 1.0) < 1e-12


def test_p7a_counterexample_fixed_gain_without_I6():
    """ChatGPT round 4: x=(1,1), r=(0,0), C={(1,0),(2,1)}, f=z_1, sigma=1, L=1. Fixed certifies sign(phi_1) > 0
    (interval [0.5, 1]); domain does not ([-1, 1]); I6 does not fire (X_1 = 0)."""
    x, r = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    C = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    sigma = torch.ones(2)
    f = lambda Z: Z[:, 0]
    A_dom, A_fix = torch.cat([C, x[None]]), torch.cat([C, x[None], r[None]])
    assert abs(anchor_lipschitz(A_dom, f(A_dom), sigma) - 1.0) < 1e-12
    assert abs(anchor_lipschitz(A_fix, f(A_fix), sigma) - 1.0) < 1e-12
    lo_d, hi_d, _ = pair_bounds_exact(x, r, A_dom, f(A_dom), 1.0, sigma)
    lo_f, hi_f, _ = pair_bounds_exact(x, r, A_fix, f(A_fix), 1.0, sigma)
    assert abs(float(lo_d[0]) + 1) < 1e-12 and abs(float(hi_d[0]) - 1) < 1e-12
    assert abs(float(lo_f[0]) - 0.5) < 1e-12 and abs(float(hi_f[0]) - 1) < 1e-12
    cap = efficiency_cap(2.0, 1.0, 1.0, 1.0, 1e-9)
    assert abs(cap["X"]) < 1e-12 and cap["forced_sign"] == 0


def test_quantized_straddler_counterexample():
    """Finite K needs q_K(t), not t: g = -1, h = 3, t = 0.3, K = 4 -> V_box = 1.1 >= 1 but V_4 = 0.5 < 1."""
    g, h, t, K = -1.0, 3.0, 0.3, 4
    q = max(math.ceil(K * t), 1) / K
    assert abs((g + (1 - t) * h) - 1.1) < 1e-12 and abs((g + (1 - q) * h) - 0.5) < 1e-12
