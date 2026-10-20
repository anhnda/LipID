"""
lipid/pairwise.py — sharp pairwise Shapley-difference bounds and exact ranking certificates
(LipID v1.2, Revision A2/A3; reviewer Q3).

Setting (paper Secs. 3-4): weighted-l1 metric d_sigma, anchors A (containing x, and r in fixed mode), labels f|_A,
L >= L_A, McShane-Whitney envelopes ell_L <= g <= h_L. Players are the m ACTIVE features (delta_i > 0). Null features
(delta_j = 0) have phi_j = 0 for every feasible g; `pairwise_bounds(..., include_inactive=True)` keeps them as players
(duplicate hybrid vertices) and is used only for the null-feature consistency check.

Pairwise identity. For players i != j and S ranging over the subsets of the other m-2 players,
    phi_i - phi_j = sum_S beta_S [ g(z_{S+i}) - g(z_{S+j}) ],      beta_S = |S|! (m-|S|-2)! / (m-1)!.
P_S = z_{S+i} and Q_S = z_{S+j} differ exactly in coordinates i and j, so d(P_S, Q_S) = delta_i + delta_j and
    c_ij = L (delta_i + delta_j).
Corollary (sharp pairwise bounds; proof in paper/v12_theory.tex):
    lower_ij = sum_S beta_S max{ ell(P_S) - h(Q_S), -c_ij },     upper_ij = sum_S beta_S min{ h(P_S) - ell(Q_S), c_ij },
both ATTAINED by members of F_L(A); the attaining labelings are `extremal_pair_labels`.
Consequences:
  * phi_i > phi_j for every g in F_L(A)  <=>  lower_ij > 0       (exact: the infimum is attained);
  * lower_ij >= lower_i - upper_j, upper_ij <= upper_i - lower_j (never weaker than the marginal comparison, Prop. 2);
  * T is the strict signed top-k set of every feasible g  <=>  lower_ij > 0 for all i in T, j not in T  (exact);
  * |phi_i| > |phi_j| for every feasible g  <=>  [min D > 0 and min S > 0] or [max D < 0 and max S < 0],
    D = phi_i - phi_j, S = phi_i + phi_j (the feasible (D, S) set is a compact convex polytope: it avoids both axes
    iff it lies in one open quadrant). Bounds on S need the joint LP (lipid/joint.py).

Certificates use a strict normalized tolerance: a pair statement is certified when bound / c_ij exceeds `tol`. The
marginal comparison is normalized by the SAME c_ij, so "pairwise certified" always contains "marginal certified".
No response-function evaluation happens in this module.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

import lipid  # noqa: F401  (repository root on sys.path)
from bounds import all_masks, vertex_points
from envelope import envelopes
from lipid.core import edge_tensors

Tensor = torch.Tensor
SIGN_TOL = 1e-6          # same strict normalized tolerance as the P0'' pipeline (phase1.Config().tol)


# ============================================================================================ index sets
def pair_list(m: int) -> list[tuple[int, int]]:
    """Unordered pairs (a < b) of player positions 0..m-1, in a fixed order shared by every table."""
    return [(a, b) for a in range(m) for b in range(a + 1, m)]


@lru_cache(maxsize=None)
def _pair_index_np(m: int):
    """For every pair (a, b): P = S | bit a, Q = S | bit b, beta_S, over the 2^(m-2) masks S without bits a, b.
    Bit order is the one of bounds.vertex_points / t6_directional.edge_sets (bit jj <-> active position jj)."""
    pairs = pair_list(m)
    if m < 2:
        e = np.zeros((0, 0), dtype=np.int64)
        return pairs, e, e.copy(), np.zeros((0, 0)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    idx = np.arange(1 << m, dtype=np.int64)
    pop = np.array([bin(int(v)).count("1") for v in idx], dtype=np.int64)
    beta = np.array([math.factorial(s) * math.factorial(m - s - 2) / math.factorial(m - 1) for s in range(m - 1)])
    R = 1 << (m - 2)
    P = np.empty((len(pairs), R), dtype=np.int64)
    Q = np.empty((len(pairs), R), dtype=np.int64)
    B = np.empty((len(pairs), R), dtype=np.float64)
    for p, (a, b) in enumerate(pairs):
        ba, bb = 1 << a, 1 << b
        S = idx[(idx & (ba | bb)) == 0]
        P[p], Q[p], B[p] = S | ba, S | bb, beta[pop[S]]
    ia = np.array([a for a, _ in pairs], dtype=np.int64)
    ib = np.array([b for _, b in pairs], dtype=np.int64)
    for arr in (P, Q, B, ia, ib):
        arr.setflags(write=False)
    return pairs, P, Q, B, ia, ib


def pair_tensors(m: int, dtype=torch.float64, device="cpu") -> dict:
    """Torch versions of the pair index sets: P, Q (npairs, 2^(m-2)) long; B same shape; ia, ib (npairs,)."""
    pairs, P, Q, B, ia, ib = _pair_index_np(m)
    return dict(pairs=pairs, P=torch.as_tensor(np.array(P), device=device),
                Q=torch.as_tensor(np.array(Q), device=device),
                B=torch.as_tensor(np.array(B), dtype=dtype, device=device),
                ia=torch.as_tensor(np.array(ia), device=device), ib=torch.as_tensor(np.array(ib), device=device))


def pair_index_np(m: int):
    """(pairs, P, Q, B, ia, ib) as read-only numpy arrays (used by the LP code)."""
    return _pair_index_np(m)


# ============================================================================================ bounds
def marginal_from_envelopes(ell: Tensor, h: Tensor, Sx: Tensor, Px: Tensor, W: Tensor, Ld: Tensor):
    """Theorem 1 in raw units. ell, h: (..., N); Sx, Px, W: (m, 2^(m-1)); Ld: (..., m) = L * delta_active.
    Returns (lower, upper), each (..., m)."""
    a = torch.maximum(ell[..., Px] - h[..., Sx], -Ld[..., None])
    b = torch.minimum(h[..., Px] - ell[..., Sx], Ld[..., None])
    return (W * a).sum(-1), (W * b).sum(-1)


def pairwise_from_envelopes(ell: Tensor, h: Tensor, pt: dict, c: Tensor):
    """Sharp pairwise bounds in raw units. ell, h: (..., N); c: (..., npairs) = L (delta_i + delta_j).
    Returns (lower_ij, upper_ij), each (..., npairs), for phi_{pair[0]} - phi_{pair[1]}."""
    P, Q, B = pt["P"], pt["Q"], pt["B"]
    lo = (B * torch.maximum(ell[..., P] - h[..., Q], -c[..., None])).sum(-1)
    hi = (B * torch.minimum(h[..., P] - ell[..., Q], c[..., None])).sum(-1)
    return lo, hi


def pair_diff_from_vertices(vals: Tensor, pt: dict) -> Tensor:
    """phi_a - phi_b of the game with vertex values `vals` (..., N), via the pairwise identity."""
    return (pt["B"] * (vals[..., pt["P"]] - vals[..., pt["Q"]])).sum(-1)


def pairwise_bounds(x: Tensor, r: Tensor, A: Tensor, fA: Tensor, L: float, sigma: Tensor, *,
                    include_inactive: bool = False, max_players: int = 20, max_elems: int = 2 ** 24) -> dict:
    """Marginal (Theorem 1) and sharp pairwise intervals for all players, by exact enumeration of the 2^m hybrids.
    A must already contain x (domain) or x and r (fixed). Raw units throughout (phi units)."""
    d = x.numel()
    delta = (x - r).abs() / sigma
    active = list(range(d)) if include_inactive else torch.nonzero(delta > 0).flatten().tolist()
    m = len(active)
    out = dict(delta=delta, active=active, m=m, L=float(L), pairs=[])
    if m == 0:
        return out
    if m > max_players:
        raise ValueError(f"{m} players > max_players={max_players}")
    Z = vertex_points(x, r, active, all_masks(m, x.device))
    ell, h = envelopes(Z, A, fA, float(L), sigma, max_elems=max_elems)
    Sx, Px, W = edge_tensors(m, x.dtype, x.device)
    dl = delta[active]
    Ld = float(L) * dl
    lo, hi = marginal_from_envelopes(ell, h, Sx, Px, W, Ld)
    pt = pair_tensors(m, x.dtype, x.device)
    c = float(L) * (dl[pt["ia"]] + dl[pt["ib"]])
    plo, phi = pairwise_from_envelopes(ell, h, pt, c)
    out.update(Z=Z, ell=ell, h=h, Sx=Sx, Px=Px, W=W, Ld=Ld, lo=lo, hi=hi, pt=pt, c=c, pair_lo=plo, pair_hi=phi,
               pairs=[(active[a], active[b]) for a, b in pt["pairs"]])
    return out


# ============================================================================================ certificates
def _i8(t: Tensor) -> Tensor:
    return t.to(torch.int8)


def signed_cert(lo: Tensor, hi: Tensor, scale: Tensor, tol: float = SIGN_TOL) -> Tensor:
    """+1 if lo/scale > tol, -1 if hi/scale < -tol, else 0 (scale > 0)."""
    return _i8(lo / scale > tol) - _i8(hi / scale < -tol)


def marginal_pair_diff(lo: Tensor, hi: Tensor, pt: dict):
    """Outer bounds on phi_a - phi_b implied by the marginal intervals alone: [lo_a - hi_b, hi_a - lo_b]."""
    ia, ib = pt["ia"], pt["ib"]
    return lo[..., ia] - hi[..., ib], hi[..., ia] - lo[..., ib]


def abs_range(lo: Tensor, hi: Tensor):
    """(min |t|, max |t|) over t in [lo, hi]."""
    absmin = torch.where((lo <= 0) & (hi >= 0), torch.zeros_like(lo), torch.minimum(lo.abs(), hi.abs()))
    return absmin, torch.maximum(lo.abs(), hi.abs())


def abs_marginal_cert(lo: Tensor, hi: Tensor, pt: dict, c: Tensor, tol: float = SIGN_TOL) -> Tensor:
    """Proposition 2 (sufficient): +1 certifies |phi_a| > |phi_b|, -1 certifies |phi_b| > |phi_a|, 0 otherwise."""
    amin, amax = abs_range(lo, hi)
    ia, ib = pt["ia"], pt["ib"]
    return _i8((amin[..., ia] - amax[..., ib]) / c > tol) - _i8((amin[..., ib] - amax[..., ia]) / c > tol)


def abs_exact_cert(dlo, dhi, slo, shi, c, tol: float = SIGN_TOL):
    """Exact |phi_a| vs |phi_b| certificate from sharp bounds on D = phi_a - phi_b and S = phi_a + phi_b.
    Works on torch tensors or numpy arrays; returns +1 / -1 / 0 (int8)."""
    pos = lambda v: (v / c) > tol      # noqa: E731
    neg = lambda v: (v / c) < -tol     # noqa: E731
    a_over = (pos(dlo) & pos(slo)) | (neg(dhi) & neg(shi))
    b_over = (neg(dhi) & pos(slo)) | (pos(dlo) & neg(shi))
    if isinstance(a_over, torch.Tensor):
        return _i8(a_over) - _i8(b_over)
    return a_over.astype(np.int8) - b_over.astype(np.int8)


def dominance_matrix(cert, m: int) -> np.ndarray:
    """G[a, b] = True iff 'phi_a > phi_b' (or |.|) is certified; `cert` is the +1/-1/0 vector over pair_list(m)."""
    G = np.zeros((m, m), dtype=bool)
    cert = np.asarray(cert)
    for p, (a, b) in enumerate(pair_list(m)):
        if cert[p] > 0:
            G[a, b] = True
        elif cert[p] < 0:
            G[b, a] = True
    return G


def certified_topk(G: np.ndarray, k: int):
    """The set T (|T| = k) with G[i, j] for all i in T, j not in T, or None. For an antisymmetric relation such a T
    is unique and equals {i : wins(i) >= m - k} (members dominate all m-k outsiders; outsiders win at most m-k-1)."""
    m = G.shape[0]
    if not 1 <= k < m:
        return list(range(m)) if k == m else None
    T = np.flatnonzero(G.sum(1) >= m - k)
    if T.size != k:
        return None
    out = np.setdiff1d(np.arange(m), T)
    return T.tolist() if bool(G[np.ix_(T, out)].all()) else None


def top1(G: np.ndarray) -> int:
    """Position of the certified top-1 player, -1 if none (m = 1 -> 0, trivially)."""
    if G.shape[0] == 1:
        return 0
    T = certified_topk(G, 1)
    return -1 if T is None else int(T[0])


# ============================================================================================ attainment
def extremal_pair_labels(ell: Tensor, h: Tensor, Pp: Tensor, Qp: Tensor, c: float, side: str):
    """Labels on the two faces {P_S} and {Q_S} of ONE pair that attain the closed-form bound (proof of the corollary).
        upper:  v(Q_S) = ell(Q_S),  v(P_S) = min{ h(P_S), ell(Q_S) + c }
        lower:  v(Q_S) = h(Q_S),    v(P_S) = max{ ell(P_S), h(Q_S) - c }
    Together with the anchor labels they form an L-Lipschitz labeling of a finite set (checked by
    lipid.joint.labeling_violation); a McShane extension is then a member of F_L(A) attaining the bound."""
    if side == "upper":
        vQ = ell[Qp]
        vP = torch.minimum(h[Pp], vQ + c)
    elif side == "lower":
        vQ = h[Qp]
        vP = torch.maximum(ell[Pp], vQ - c)
    else:
        raise ValueError(side)
    return vP, vQ


__all__ = ["SIGN_TOL", "pair_list", "pair_tensors", "pair_index_np", "marginal_from_envelopes",
           "pairwise_from_envelopes", "pair_diff_from_vertices", "pairwise_bounds", "signed_cert",
           "marginal_pair_diff", "abs_range", "abs_marginal_cert", "abs_exact_cert", "dominance_matrix",
           "certified_topk", "top1", "extremal_pair_labels"]
