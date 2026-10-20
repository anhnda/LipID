"""
lipid/core.py — small, general helpers built on the historical modules (Theorem 1, eqs. 8-11).

Used by the real audit, the scaling benchmark and the figure-2 script. The nested P0'' pipeline uses
nested_anchors_v2.edge_terms directly so that its arithmetic is identical to the archived run.
"""
from __future__ import annotations

import torch

import lipid  # noqa: F401  (puts the repository root on sys.path)
from bounds import all_masks, vertex_points
from envelope import envelopes, weighted_l1
from t6_directional import edge_sets

Tensor = torch.Tensor


def active_features(x: Tensor, r: Tensor, sigma: Tensor):
    delta = (x - r).abs() / sigma
    return delta, torch.nonzero(delta > 0).flatten().tolist()


def edge_tensors(m: int, dtype=torch.float64, device="cpu"):
    """Stacked (S, P, W), each (m, 2^{m-1}): lower/upper vertex index of every i-edge and its Shapley weight."""
    ed = edge_sets(m, dtype, device)
    return (torch.stack([e[0] for e in ed]), torch.stack([e[1] for e in ed]), torch.stack([e[2] for e in ed]))


def edge_from_envelopes(ell: Tensor, h: Tensor, Sx: Tensor, Px: Tensor, W: Tensor, dl: Tensor, L: float):
    """Same formulas as nested_anchors_v2.edge_terms, starting from envelopes:
    rho_i = sum_S w (min(1,c+) + min(1,c-)) / 2,  normalized lower = -sum w min(1,c-), upper = sum w min(1,c+)."""
    Ld = L * dl
    cp = ((h[Px] - ell[Sx]) / Ld).clamp(max=1.0)
    cm = ((h[Sx] - ell[Px]) / Ld).clamp(max=1.0)
    return (W * (cp + cm)).sum(1) / 2.0, -(W * cm).sum(1), (W * cp).sum(1)


def marginal_intervals(x: Tensor, r: Tensor, A: Tensor, fA: Tensor, L: float, sigma: Tensor, *,
                       max_players: int = 20, max_elems: int = 2 ** 24) -> dict:
    """Sharp marginal identification intervals (Theorem 1) for every active feature, by exact enumeration.
    A must already contain x (domain) or x and r (fixed). No response-function evaluation happens here."""
    delta, active = active_features(x, r, sigma)
    m = len(active)
    out = dict(delta=delta, active=active, m=m)
    if m == 0:
        return out
    if m > max_players:
        raise ValueError(f"{m} active players > max_players={max_players}")
    Z = vertex_points(x, r, active, all_masks(m, x.device))
    ell, h = envelopes(Z, A, fA, L, sigma, max_elems=max_elems)
    Sx, Px, W = edge_tensors(m, x.dtype, x.device)
    dl = delta[active][:, None]
    rho, lo_n, hi_n = edge_from_envelopes(ell, h, Sx, Px, W, dl, L)
    Ld = L * delta[active]
    out.update(Z=Z, Sx=Sx, Px=Px, W=W, rho=rho, lo_norm=lo_n, hi_norm=hi_n, lo=lo_n * Ld, hi=hi_n * Ld,
               Gamma=2.0 * rho * Ld, L=float(L))
    return out


def shapley_on_vertices(fZ: Tensor, Sx: Tensor, Px: Tensor, W: Tensor) -> Tensor:
    """Exact reference-conditioned Shapley values of the active players from vertex values (evaluation only)."""
    return (W * (fZ[Px] - fZ[Sx])).sum(1)


def sign_from_norm(lo_n: Tensor, hi_n: Tensor, tol: float) -> Tensor:
    """Strict sign certificate on the normalized endpoints (same rule and tol as nested_anchors_v2)."""
    return torch.where(lo_n > tol, torch.ones_like(lo_n), torch.where(hi_n < -tol, -torch.ones_like(lo_n),
                                                                    torch.zeros_like(lo_n)))


def contained(phi: Tensor, lo_n: Tensor, hi_n: Tensor, Ld: Tensor, tol: float = 1e-9) -> Tensor:
    """Scale-aware containment, identical to the cover check of nested_anchors_v2."""
    return (phi >= (lo_n - tol) * Ld) & (phi <= (hi_n + tol) * Ld)


def anchor_L(A: Tensor, fA: Tensor, sigma: Tensor) -> float:
    from envelope import anchor_lipschitz
    return anchor_lipschitz(A, fA, sigma)


__all__ = ["active_features", "edge_tensors", "edge_from_envelopes", "marginal_intervals", "shapley_on_vertices",
           "sign_from_norm", "contained", "anchor_L", "weighted_l1"]
