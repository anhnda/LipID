"""
lipid/certdiag.py — CERT-02: why the geometry/range saturation certificates miss saturated instances (reviewer Q6).

Notation per i-edge (z-, z+) = (z_S, z_{S+i}), delta = delta_i, anchors u, v in A, k = min(L_A / L, 1) (anchor slopes
are at most k L), q(z) = d(z, A), Omega = max_A f - min_A f. Directional gaps (paper eq. 10):
    c+ = (h(z+) - ell(z-)) / (L delta) = min_{u,v} [ d(z+,u) + d(z-,v) + (f(u) - f(v)) / L ] / delta
    c- = (h(z-) - ell(z+)) / (L delta)  (same with z+ and z- exchanged),
and the exact edge contribution is rho_S = (min(1, c+) + min(1, c-)) / 2 (paper eq. 11).

Lower bounds on rho_S, each obtained by discarding information (all clipped to [0, 1]); per edge
    exact  >=  ABS  >=  PAIRBEST = max(GEO, RANGE)  >=  UNI = max(T3, RANGE)        and   GEO >= T3:
  ABS    pairwise |slope| relaxation: replace (f(u) - f(v)) by -|f(u) - f(v)|. Since -|t| = min(t, -t) the double
         minimum separates and the relaxation equals min(c+, c-) EXACTLY (identity proved in paper/v12_theory.tex,
         checked by brute force in tests/test_v12_certdiag.py): it forgets only the SIGN/direction of the slopes.
  GEO    geometry-only pairwise relaxation: |f(u) - f(v)| <= k L d(u, v):  min_{u,v} [d(z+,u) + d(z-,v) - k d(u,v)] / delta
         (symmetric in the direction; O(|A|^2) per edge — a diagnostic, not a cheaper certificate).
  RANGE  |f(u) - f(v)| <= Omega and d(z+,u) + d(z-,v) >= q(z+) + q(z-):   s - Omega / (L delta),  s = (q(z+) + q(z-)) / delta
  T3     triangle inequality d(u,v) <= d(u,z+) + delta + d(z-,v) in GEO:  (1 - k) s - k
UNI = max(T3, RANGE) is the paper's Proposition-7 certificate (nested_anchors_v2 `uni_com` at L*).
Summing w_S times each term gives feature-level rho's with the same ordering, so for a saturated feature
    rho_exact - rho_uni = [rho_exact - rho_abs] + [rho_abs - rho_pairbest] + [rho_pairbest - rho_uni]
                           sign/direction asymmetry   slope magnitude             decoupling (T3 vs pairwise geometry)
with three nonnegative terms. Classification at the saturation threshold 0.8 (protocol CERT-02):
    not_saturated (exact <= .8) | certified (uni > .8) | sign_asymmetry (abs <= .8) | slope_magnitude (pairbest <= .8)
    | decoupling (pairbest > .8 >= uni).
Binding-pair diagnostics: for each edge and direction, the anchors attaining h and ell (argmin / argmax in the
envelopes): share of edges where a binding anchor is an endpoint (x or r), mean d(u*, v*) / delta, mean signed slope
(f(u*) - f(v*)) / (L d(u*, v*)).
"""
from __future__ import annotations

import torch

SAT = 0.8
CLASSES = ("not_saturated", "certified", "sign_asymmetry", "slope_magnitude", "decoupling")


def geo_pair_min(DP: torch.Tensor, DS: torch.Tensor, M: torch.Tensor, kvec, max_elems: int = 2 ** 26) -> torch.Tensor:
    """min_{u,v} [DP[e,u] + DS[e,v] - k M[u,v]] for every edge e and every k: (K, E). DP, DS: (E, nA); M: (nA, nA)."""
    E, nA = DP.shape
    step = max(1, max_elems // max(1, nA * nA))
    out = []
    for s0 in range(0, E, step):
        T = DP[s0:s0 + step, :, None] + DS[s0:s0 + step, None, :]
        out.append(torch.stack([(T - float(kk) * M).amin(dim=(1, 2)) for kk in kvec]))
    return torch.cat(out, 1) if out else DP.new_zeros(len(kvec), 0)


def relaxation_chain(Dk, fA, Lvec, LA: float, Omega: float, Sx, Px, W, dl, *, D_AA=None, n_background: int,
                     geo: str = "target", max_elems: int = 2 ** 26) -> dict:
    """All feature-level rho's for K Lipschitz levels at once (Lvec, shape (K,)); returns tensors of shape (K, m)
    plus binding-pair diagnostics. `geo`: 'none' | 'target' (only features saturated-but-uncertified at some level)
    | 'all'. Anchors are ordered (background C[:n], x, (r)); index >= n_background is an endpoint."""
    K = Lvec.shape[0]
    LD = Lvec[:, None, None] * Dk[None]
    hv = fA[None, None, :] + LD
    lv = fA[None, None, :] - LD
    h, hu = hv.min(2)
    ell, lu = lv.max(2)
    Ld = Lvec[:, None] * dl[None, :]                                               # (K, m)
    cp = (h[:, Px] - ell[:, Sx]) / Ld[..., None]                                   # (K, m, R)
    cm = (h[:, Sx] - ell[:, Px]) / Ld[..., None]
    wsum = lambda t: (W * t).sum(-1)                                               # noqa: E731
    rho_exact = wsum((cp.clamp(max=1.0) + cm.clamp(max=1.0)) / 2.0)
    rho_abs = wsum(torch.minimum(cp, cm).clamp(0.0, 1.0))
    asym = wsum((cp.clamp(max=1.0) - cm.clamp(max=1.0)).abs() / 2.0)
    q = Dk.min(1).values
    s = (q[Px] + q[Sx]) / dl[:, None]                                              # (m, R)
    k = (LA / Lvec).clamp(max=1.0)                                                 # (K,)
    t3 = ((1.0 - k)[:, None, None] * s[None] - k[:, None, None]).clamp(0.0, 1.0)
    rg = (s[None] - Omega / Ld[..., None]).clamp(0.0, 1.0)
    rho_t3, rho_range = wsum(t3), wsum(rg)
    rho_uni = wsum(torch.maximum(t3, rg))
    m = dl.shape[0]
    rho_geo = torch.full_like(rho_exact, float("nan"))
    rho_pairbest = torch.full_like(rho_exact, float("nan"))
    if geo != "none" and D_AA is not None:
        need = ((rho_exact > SAT) & (rho_uni <= SAT)).any(0) if geo == "target" else torch.ones(m, dtype=torch.bool)
        for jj in torch.nonzero(need).flatten().tolist():
            G = geo_pair_min(Dk[Px[jj]], Dk[Sx[jj]], D_AA, k.tolist(), max_elems) / dl[jj]   # (K, R)
            g = G.clamp(0.0, 1.0)
            rho_geo[:, jj] = (W[jj] * g).sum(-1)
            rho_pairbest[:, jj] = (W[jj] * torch.maximum(g, rg[:, jj])).sum(-1)
    # binding pairs: c+ uses (u = argmin h at z+, v = argmax ell at z-); c- uses (u at z-, v at z+)
    diag = {}
    if D_AA is not None:
        up, vp = hu[:, Px], lu[:, Sx]
        um, vm = hu[:, Sx], lu[:, Px]
        endp = ((up >= n_background) | (vp >= n_background)).to(Dk.dtype) + \
               ((um >= n_background) | (vm >= n_background)).to(Dk.dtype)
        dpp, dmm = D_AA[up, vp], D_AA[um, vm]
        hpp = (dpp + dmm) / (2.0 * dl[None, :, None])
        sl = lambda u, v, dd: torch.where(dd > 1e-12, (fA[u] - fA[v]) / (Lvec[:, None, None] * dd.clamp_min(1e-12)),  # noqa: E731
                                          torch.zeros_like(dd))
        slope = (sl(up, vp, dpp) + sl(um, vm, dmm)) / 2.0
        diag = dict(share_endpoint_binding=wsum(endp / 2.0), mean_h_binding=wsum(hpp), mean_slope_binding=wsum(slope))
    return dict(rho_exact=rho_exact, rho_abs=rho_abs, rho_geo=rho_geo, rho_pairbest=rho_pairbest, rho_range=rho_range,
                rho_t3=rho_t3, rho_uni=rho_uni, asym=asym, kappa=Lvec / LA, **diag)


def classify(rho_exact: float, rho_uni: float, rho_abs: float, rho_pairbest: float) -> str:
    if not rho_exact > SAT:
        return "not_saturated"
    if rho_uni > SAT:
        return "certified"
    if not rho_abs > SAT:
        return "sign_asymmetry"
    if rho_pairbest != rho_pairbest:          # NaN: GEO not computed (geo='none' or n above --geo-max-n)
        return "abs_ok_geo_not_computed"
    if not rho_pairbest > SAT:
        return "slope_magnitude"
    return "decoupling"


__all__ = ["SAT", "CLASSES", "geo_pair_min", "relaxation_chain", "classify"]
