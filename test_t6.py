"""
test_t6.py — tests for the directional decomposition (run: python -m pytest -q test_t6.py).
Independent references: pair_bounds_exact (bounds.py) for rho, and the O(|A|^2) formulas for box / det.
"""
import itertools

import pytest
import torch

from bounds import all_masks, pair_bounds_exact, vertex_points
from envelope import anchor_lipschitz, weighted_l1
from t6_directional import (box_oracle_T, directional, edge_sets, endpoint_band_bound, envelopes_from_D,
                            feature_stats, lb_from_T, pair_tensor, quantized_T, slope_ratio)

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
                D=D, DA=DA, ell=ell, h=h, edges=edge_sets(m))


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
    """(I4): with x in A, s_S <= 1 + 2 D_S / delta_i (the near-x band can never be certified by rho_low)."""
    I = _instance(d, seed, 1.0, fixed, _fa)
    q = I["D"].min(dim=1).values
    da = I["delta"][I["active"]]
    for jj, j in enumerate(I["active"]):
        S, P, w = I["edges"][jj]
        s = (q[P] + q[S]) / I["delta"][j]
        assert bool((s <= endpoint_band_bound(da, S, jj) + TOL).all())
