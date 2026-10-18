"""
envelope.py — exact McShane–Whitney envelopes, weighted-l1 metric, members of F_L(A).

Conventions (fixed for the whole study, sigma chosen BEFORE experiments)
    metric    d(z, u) = sum_j |z_j - u_j| / sigma_j
    class     F_L(A) = { g : g(u) = f(u) for all u in A,  Lip_d(g) <= L }
    envelopes ell(z) = max_{u in A} [f(u) - L d(z,u)],   h(z) = min_{u in A} [f(u) + L d(z,u)]

Facts used everywhere
    F_L(A) non-empty  <=>  L >= L_A := max_{u != v} |f(u) - f(v)| / d(u, v)
    if non-empty: ell, h are members, and ell <= g <= h for every member g
    for any L-Lipschitz psi:  min(h, max(ell, psi)) is a member  (min/max keep Lip <= L,
    and ell = h = f on A)

Envelopes are computed by exact brute force over all anchors (chunked), never by kNN.
All code is dtype-agnostic; use float64.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

Tensor = torch.Tensor
DEFAULT_MAX_ELEMS = 2 ** 24  # max entries of one distance block kept in memory


def _step(n_cols: int, max_elems: int) -> int:
    return max(1, max_elems // max(1, n_cols))


# ----------------------------------------------------------------------------- metric
def weighted_l1(Z: Tensor, U: Tensor, sigma: Tensor) -> Tensor:
    """Pairwise weighted-l1 distances: (m,d), (n,d) -> (m,n)."""
    return torch.cdist(Z / sigma, U / sigma, p=1.0)


def dist_to_set(Z: Tensor, U: Tensor, sigma: Tensor, *, max_elems: int = DEFAULT_MAX_ELEMS) -> Tensor:
    """d(z, U) = min_u d(z, u) for every row z of Z."""
    if Z.shape[0] == 0:
        return Z.new_zeros(0)
    step = _step(U.shape[0], max_elems)
    return torch.cat([weighted_l1(Z[s:s + step], U, sigma).min(dim=1).values
                      for s in range(0, Z.shape[0], step)])


# ------------------------------------------------------------------ Lipschitz constants
def max_ratio(P: Tensor, fP: Tensor, Q: Tensor, fQ: Tensor, sigma: Tensor, *,
              max_elems: int = DEFAULT_MAX_ELEMS, dtol: float = 1e-12, ftol: float = 1e-10) -> float:
    """max over p in P, q in Q with d(p,q) > dtol of |f(p) - f(q)| / d(p, q).
    Returns inf if two (numerically) identical points carry different values."""
    best = 0.0
    step = _step(Q.shape[0], max_elems)
    for s in range(0, P.shape[0], step):
        D = weighted_l1(P[s:s + step], Q, sigma)
        dF = (fP[s:s + step, None] - fQ[None, :]).abs()
        tiny = D <= dtol
        if bool((tiny & (dF > ftol)).any()):
            return math.inf
        ratio = torch.where(tiny, torch.zeros_like(D), dF / D.clamp_min(dtol))
        best = max(best, float(ratio.max()))
    return best


def anchor_lipschitz(A: Tensor, fA: Tensor, sigma: Tensor, **kw) -> float:
    """L_A: the smallest L for which F_L(A) is non-empty."""
    return max_ratio(A, fA, A, fA, sigma, **kw)


def lipschitz_after_adding(LA_base: float, A: Tensor, fA: Tensor, P: Tensor, fP: Tensor,
                           sigma: Tensor) -> float:
    """L_A of the anchor set A ∪ P, given LA_base = L_A(A) (avoids recomputing A x A)."""
    out = max(LA_base, max_ratio(P, fP, A, fA, sigma))
    if P.shape[0] > 1:
        out = max(out, max_ratio(P, fP, P, fP, sigma))
    return out


# ------------------------------------------------------------------------- envelopes
def envelopes(Z: Tensor, A: Tensor, fA: Tensor, L: float, sigma: Tensor, *,
              max_elems: int = DEFAULT_MAX_ELEMS) -> tuple[Tensor, Tensor]:
    """Exact (ell(z), h(z)) for every row of Z, by brute force over all anchors."""
    if Z.shape[0] == 0:
        return Z.new_zeros(0), Z.new_zeros(0)
    lo, hi = [], []
    step = _step(A.shape[0], max_elems)
    for s in range(0, Z.shape[0], step):
        LD = L * weighted_l1(Z[s:s + step], A, sigma)
        lo.append((fA[None, :] - LD).max(dim=1).values)
        hi.append((fA[None, :] + LD).min(dim=1).values)
    return torch.cat(lo), torch.cat(hi)


def clamp_to_envelope(vals: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    return torch.minimum(hi, torch.maximum(lo, vals))


def envelope_violation(vals: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """V_L(z) = dist(f(z), [ell(z), h(z)]).
    Assumption-violation diagnostic only: V_L = 0 does NOT prove membership."""
    return torch.clamp(lo - vals, min=0.0) + torch.clamp(vals - hi, min=0.0)


def lipschitz_violation(P: Tensor, vals: Tensor, L: float, sigma: Tensor, *,
                        max_elems: int = DEFAULT_MAX_ELEMS) -> float:
    """max_{p,q in P} |v_p - v_q| - L d(p,q).  <= tol  iff  vals is L-Lipschitz on P.
    (Self-pairs give 0, so the result is >= 0.)"""
    worst = 0.0
    step = _step(P.shape[0], max_elems)
    for s in range(0, P.shape[0], step):
        D = weighted_l1(P[s:s + step], P, sigma)
        dv = (vals[s:s + step, None] - vals[None, :]).abs()
        worst = max(worst, float((dv - L * D).max()))
    return worst


# ------------------------------------------------------------- members of F_L(A)
class RandomLipschitz:
    """Random psi with Lip <= L w.r.t. weighted l1 (work in y = z / sigma).

    Convex mixture of four L-Lipschitz parts (a convex mixture keeps Lip <= L):
      cones : max_k [c_k - L ||y - p_k||_1]
      sep   : sum_j (L a_j / w_j) sin(w_j y_j + phi_j),  |a_j| <= 1   (separable => l1-Lip <= L)
      ridge : (L / w) sin(w <v, y> + phi),  ||v||_inf = 1             (l1-Lip <= L ||v||_inf)
      lin   : L <u, y - y_c>,  ||u||_inf <= 1
    """

    def __init__(self, center: Tensor, scale: float, L: float, sigma: Tensor,
                 gen: torch.Generator, *, offset: float = 0.0, n_cones: int = 6):
        d = center.numel()
        dt, dev = center.dtype, center.device
        # draw on the generator's device (CPU generators stay valid when the data lives on CUDA),
        # then move: the same seed gives the same member on every device
        gdev = gen.device if gen is not None else dev

        def unif(*shape):
            return torch.rand(*shape, generator=gen, dtype=dt, device=gdev).to(dev)

        def gauss(*shape):
            return torch.randn(*shape, generator=gen, dtype=dt, device=gdev).to(dev)

        self.L, self.sigma, self.offset = float(L), sigma, float(offset)
        self.yc = center / sigma
        s = float(scale)
        self.peaks = self.yc + s * gauss(n_cones, d)
        self.heights = self.L * s * gauss(n_cones)
        self.w_sep = (0.5 + 2.5 * unif(d)) / s
        self.a_sep = 2.0 * unif(d) - 1.0
        self.ph_sep = 2.0 * math.pi * unif(d)
        v = 2.0 * unif(d) - 1.0
        self.v_ridge = v / v.abs().max().clamp_min(1e-12)
        self.w_ridge = float((0.5 + 2.5 * unif(1)) / s)
        self.ph_ridge = float(2.0 * math.pi * unif(1))
        self.u_lin = 2.0 * unif(d) - 1.0
        e = -torch.log(unif(4).clamp_min(1e-12))
        self.mix = e / e.sum()  # Dirichlet(1,1,1,1)

    def __call__(self, Z: Tensor) -> Tensor:
        Y = Z / self.sigma
        L = self.L
        cones = (self.heights[None, :]
                 - L * (Y[:, None, :] - self.peaks[None, :, :]).abs().sum(-1)).max(dim=1).values
        sep = (L * self.a_sep / self.w_sep * torch.sin(self.w_sep * Y + self.ph_sep)).sum(-1)
        ridge = (L / self.w_ridge) * torch.sin(self.w_ridge * (Y @ self.v_ridge) + self.ph_ridge)
        lin = L * ((Y - self.yc) @ self.u_lin)
        parts = torch.stack([cones, sep, ridge, lin], dim=-1)
        return self.offset + parts @ self.mix


class Member:
    """g = min(h, max(ell, psi)): an exact member of F_L(A) for any L-Lipschitz psi.
    Envelopes are recomputed from (A, fA) on every call (independent of any LP box)."""

    def __init__(self, A: Tensor, fA: Tensor, L: float, sigma: Tensor,
                 psi: Callable[[Tensor], Tensor]):
        self.A, self.fA, self.L, self.sigma, self.psi = A, fA, float(L), sigma, psi

    def __call__(self, Z: Tensor) -> Tensor:
        lo, hi = envelopes(Z, self.A, self.fA, self.L, self.sigma)
        return clamp_to_envelope(self.psi(Z), lo, hi)
