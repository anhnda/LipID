"""
t6_directional.py — directional decomposition of the exact normalized AIW (torch, any device).

Feature i, coalition S not containing i, z+ = z_{S+i}, z- = z_S, Ld = L * delta_i,
ell / h = McShane–Whitney envelopes of (A, f|_A, L):

    B+ = h(z+) - ell(z-) = min_{u,v} [ f(u) - f(v) + L (d(z+,u) + d(z-,v)) ]
    B- = h(z-) - ell(z+) = min_{u,v} [ f(v) - f(u) + L (d(z+,u) + d(z-,v)) ]
    c+ = B+ / Ld,  c- = B- / Ld          (non-empty class:  c± >= -1,  c+ + c- >= 0)

Exact (P3):          b_S = Ld min(c+, 1),  a_S = -Ld min(c-, 1)
                     rho_S = (min(c+,1) + min(c-,1)) / 2
Former 'box' (T6):   T_box = min_{u,v} [L R_uv - |f(u) - f(v)|] = min(B+, B-)        (exact identity)
                     dmin_S = clamp(min(c+, c-), 0, 1)

So rho and rho_box are the SAME two numbers combined by mean vs min. Deterministic consequences,
per feature-instance (Shapley weights sum to 1):
    (I1)  2 rho - 1  <=  rho_dmin  <=  rho           =>  rho - rho_dmin <= 1 - rho
    (I2)  rho - rho_dmin = sum_S w_S |ĉ+ - ĉ-| / 2   when c± >= 0 on every edge (ĉ = min(c, 1))
    (I3)  certified sign (lo > 0 or hi < 0)  =>  rho < 1/2
    (I4)  domain mode (x in A):  s_S = (q+ + q-) / delta_i <= 1 + 2 D_S / delta_i,
          D_S = sum_{j not in S+i} delta_j            (endpoint band; fixed mode: also with r)

Slope-quantized ladder between det and box (K slope levels, K = 1 is det, K -> inf is box):
    Lam_K(u,v) = L_min * max(ceil(K * lambda_uv / L_min), 1) / K  >=  lambda_uv
    T_K = min_{u,v} [L R_uv - Lam_K(u,v) d_uv],   rho_K = sum_S w_S clamp(T_K / Ld, 0, 1)
    rho_1 = rho_det <= rho_2 <= rho_4 <= ... <= rho_box   (non-decreasing along powers of two)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from bounds import shapley_weights

Tensor = torch.Tensor


# ----------------------------------------------------------------------------- edges
def edge_sets(m: int, dtype=torch.float64, device="cpu") -> list[tuple[Tensor, Tensor, Tensor]]:
    """For each active position jj: (S, P, w) with S = masks without bit jj, P = S | bit, w = Shapley weights.
    Same bit order as bounds.vertex_points / pair_bounds_exact."""
    idx = torch.arange(2 ** m, device=device)
    pop = ((idx.unsqueeze(1) >> torch.arange(m, device=device)) & 1).sum(1)
    w_all = shapley_weights(m, dtype, device)
    out = []
    for jj in range(m):
        bit = 1 << jj
        S = idx[(idx & bit) == 0]
        out.append((S, S | bit, w_all[pop[S]]))
    return out


# ------------------------------------------------------------------------- envelopes
def envelopes_from_D(D: Tensor, fA: Tensor, L: float) -> tuple[Tensor, Tensor]:
    """ell, h at every hybrid from precomputed hybrid-to-anchor distances D (2^m, |A|)."""
    LD = L * D
    return (fA[None, :] - LD).max(dim=1).values, (fA[None, :] + LD).min(dim=1).values


def directional(ell: Tensor, h: Tensor, S: Tensor, P: Tensor, Ld: float) -> tuple[Tensor, Tensor]:
    """Normalized directional slacks (c+, c-) on the edges (S, P) of one feature."""
    return (h[P] - ell[S]) / Ld, (h[S] - ell[P]) / Ld


@dataclass
class FeatureStats:
    rho: float      # exact normalized AIW
    dmin: float     # former rho_box  (= sum w clamp(min(c+,c-),0,1))
    asym: float     # rho - dmin
    negmass: float  # Shapley weight of edges with min(c+, c-) < 0  (one direction forced)
    lo_n: float     # lo / Ld
    hi_n: float     # hi / Ld

    def sign(self, Ld: float, tol: float) -> int:
        if Ld * self.lo_n > tol:
            return 1
        if Ld * self.hi_n < -tol:
            return -1
        return 0


def feature_stats(w: Tensor, cp: Tensor, cm: Tensor) -> FeatureStats:
    hp, hm = cp.clamp(max=1.0), cm.clamp(max=1.0)
    rho = float((w * (hp + hm)).sum()) / 2.0
    mn = torch.minimum(cp, cm)
    dmin = float((w * mn.clamp(0.0, 1.0)).sum())
    return FeatureStats(rho=rho, dmin=dmin, asym=rho - dmin, negmass=float(w[mn < 0].sum()),
                        lo_n=-float((w * hm).sum()), hi_n=float((w * hp).sum()))


# ------------------------------------------------------------------- pair-based bounds
def pair_tensor(D: Tensor, S: Tensor, P: Tensor) -> Tensor:
    """R[e, u, v] = d(z+_e, u) + d(z-_e, v): (2^(m-1), |A|, |A|). Model- and L-independent."""
    return D[P][:, :, None] + D[S][:, None, :]


def lb_from_T(T: Tensor, Ld: float, w: Tensor) -> float:
    return float((w * (T / Ld).clamp(0.0, 1.0)).sum())


def box_oracle_T(R: Tensor, dF: Tensor, L: float) -> Tensor:
    """O(|A|^2)-per-edge reference for T_box (kept as a test oracle for the envelope identity)."""
    return (L * R - dF[None]).amin(dim=(1, 2))


def slope_ratio(DA: Tensor, dF: Tensor, L_min: float) -> Tensor:
    """lambda_uv / L_min in [0, 1]; 0 on the diagonal."""
    lam = torch.where(DA > 1e-12, dF / DA.clamp_min(1e-12), torch.zeros_like(DA))
    return (lam / L_min).clamp(0.0, 1.0)


def quantized_T(R: Tensor, DA: Tensor, ratio: Tensor, L: float, L_min: float, K: int) -> Tensor:
    """T_K with K slope levels. K = 1 reproduces det exactly (Lam = L_min everywhere)."""
    q = torch.clamp(torch.ceil(K * ratio), min=1.0) / K
    return (L * R - (L_min * q * DA)[None]).amin(dim=(1, 2))


def det_argmin(R: Tensor, DA: Tensor, L: float, L_min: float) -> tuple[Tensor, Tensor, Tensor]:
    """T_det per edge and the anchor pair (u*, v*) attaining it."""
    T = (L * R - L_min * DA[None]).flatten(1)
    val, k = T.min(dim=1)
    nA = DA.shape[0]
    return val, k // nA, k % nA


def det_diagnostic(R: Tensor, DA: Tensor, ratio: Tensor, cp: Tensor, cm: Tensor, w: Tensor,
                   L: float, L_min: float, Ld: float, endpoints: list[int], sat: float = 0.8) -> dict:
    """On edges that det fails to saturate but dmin (= box) does, which anchor pairs make det fail?
      share_endpoint : Shapley-weight share whose det-argmin pair involves x (or r), u != v
      share_self     : share whose det-argmin is a self pair u = v (pure distance-to-anchor limit)
      ratio_at_argmin: Shapley-weighted mean of lambda_{u*v*} / L_min (how far below worst case)
    Returns None-valued fields when no such edge exists."""
    Tdet, u, v = det_argmin(R, DA, L, L_min)
    e_det = (Tdet / Ld).clamp(0.0, 1.0)
    e_box = torch.minimum(cp, cm).clamp(0.0, 1.0)
    mask = (e_det < sat) & (e_box >= sat)
    wm = float(w[mask].sum())
    out = dict(mass=wm, share_endpoint=None, share_self=None, ratio_at_argmin=None)
    if wm <= 0.0:
        return out
    ep = torch.tensor(endpoints, device=u.device)
    is_self = u == v
    is_ep = (torch.isin(u, ep) | torch.isin(v, ep)) & ~is_self
    out["share_endpoint"] = float(w[mask & is_ep].sum()) / wm
    out["share_self"] = float(w[mask & is_self].sum()) / wm
    out["ratio_at_argmin"] = float((w * ratio[u, v])[mask].sum()) / wm
    return out


# --------------------------------------------------------------------- endpoint band
def endpoint_band_bound(delta_active: Tensor, S: Tensor, jj: int) -> Tensor:
    """(I4) upper bound on s_S from the anchor x alone: 1 + 2 D_S / delta_i,
    D_S = sum of delta_j over active j not in S and j != i (bit order of edge_sets)."""
    m = delta_active.numel()
    bits = ((S.unsqueeze(1) >> torch.arange(m, device=S.device)) & 1).bool()
    comp = ~bits
    comp[:, jj] = False
    D_S = (comp.to(delta_active.dtype) * delta_active[None, :]).sum(1)
    return 1.0 + 2.0 * D_S / delta_active[jj]
