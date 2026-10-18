"""
bounds.py — pairwise coalition intervals and model-evaluation-free certificates.

For feature i, coalition S not containing i, z+ = z_{S+i}, z- = z_S, delta_i = |x_i - r_i| / sigma_i:
    a_S = max(ell(z+) - h(z-), -L delta_i),     b_S = min(h(z+) - ell(z-), L delta_i)
For every g in F_L(A):  a_S <= g(z+) - g(z-) <= b_S, hence
    E_{S~pi_i}[a_S] <= phi_i(g) <= E_{S~pi_i}[b_S]          (pairwise bound; B_pair = E[b - a])

a_S, b_S depend only on (x, r, A, f|_A, L, sigma): NOT on g and NOT on f at hybrid inputs.
Monte Carlo confidence bounds on E[a], E[b] therefore hold uniformly over F_L(A)  (Theorem 3).

Validity conditions for `mc_pair_bounds` (state them in the paper):
  * the K candidate references are fixed BEFORE any coalition is drawn;
  * n_per_stratum is fixed in advance — no optional stopping
    (adaptive n requires anytime-valid bounds or an explicit alpha-spending scheme);
  * union bound over K references x d features x 2 sides (x strata for Bernstein);
  * common random numbers across references are allowed (the union bound needs no independence).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from envelope import dist_to_set, envelopes

Tensor = torch.Tensor


# ------------------------------------------------------------------- combinatorics
def shapley_weights(m: int, dtype=torch.float64, device="cpu") -> Tensor:
    """w[s] = s! (m-1-s)! / m!  for s = 0..m-1."""
    s = torch.arange(m, dtype=dtype, device=device)
    return torch.exp(torch.lgamma(s + 1) + torch.lgamma(m - s) - math.lgamma(m + 1))


def all_masks(m: int, device="cpu") -> Tensor:
    """(2^m, m) bool; row v has column j True iff bit j of v is set."""
    idx = torch.arange(2 ** m, device=device)
    return ((idx.unsqueeze(1) >> torch.arange(m, device=device)) & 1).bool()


def popcount(idx: Tensor, m: int) -> Tensor:
    return ((idx.unsqueeze(1) >> torch.arange(m, device=idx.device)) & 1).sum(1)


def vertex_points(x: Tensor, r: Tensor, active: list[int], masks: Tensor) -> Tensor:
    """Hybrid vertices z_S over the active features (inactive features have x_j = r_j)."""
    Z = r.unsqueeze(0).repeat(masks.shape[0], 1)
    if active:
        Z[:, active] = torch.where(masks, x[active].unsqueeze(0), r[active].unsqueeze(0))
    return Z


def shapley_from_vertex_values(vals: Tensor, m: int) -> Tensor:
    """Exact Shapley values of the m-player game whose value at bitmask v is vals[v]."""
    if m == 0:
        return vals.new_zeros(0)
    idx = torch.arange(2 ** m, device=vals.device)
    w = shapley_weights(m, vals.dtype, vals.device)
    pop = popcount(idx, m)
    out = []
    for i in range(m):
        bit = 1 << i
        S = idx[(idx & bit) == 0]
        out.append((w[pop[S]] * (vals[S | bit] - vals[S])).sum())
    return torch.stack(out)


def effective_sparsity(x: Tensor, r: Tensor, sigma: Tensor, eps: float) -> int:
    """s_eps(x, r) = #{i : |x_i - r_i| / sigma_i > eps}  (eps fixed before experiments)."""
    return int(((x - r).abs() / sigma > eps).sum())


def hybrid_vertices(x: Tensor, r: Tensor, *, max_players: int = 16) -> tuple[Tensor, list[int]]:
    """All 2^m hybrid vertices z_S over the active features (x_j != r_j), in the bit order shared by
    pair_bounds_exact, HypercubeLP and shapley_from_vertex_values. Returns (Z, active)."""
    active = torch.nonzero(x != r).flatten().tolist()
    m = len(active)
    if m > max_players:
        raise ValueError(f"{m} active features > max_players={max_players}")
    return vertex_points(x, r, active, all_masks(m, x.device)), active


def shapley_full(vals: Tensor, active: list[int], d: int) -> Tensor:
    """Exact Shapley values (full d; inactive features = 0) of the game whose values at the hybrid
    vertices (hybrid_vertices order) are `vals`."""
    out = vals.new_zeros(d)
    if active:
        out[active] = shapley_from_vertex_values(vals, len(active))
    return out


def geometric_rho_inf(x: Tensor, r: Tensor, A: Tensor, sigma: Tensor, *,
                      max_players: int = 16) -> Tensor:
    """Large-L limit of the normalized AIW (amendment v3), purely geometric (no model values):
        rho_inf_i = sum_S w_S min(1, [d(z+, A) + d(z-, A)] / delta_i)
    For fixed (A, x, r):  rho_i(L) is non-decreasing in L,  rho_i(L) <= rho_inf_i,  rho_i(L) -> rho_inf_i.
    (At t = 1/L = 0 the class is {Lip <= 1, = 0 on A}; apply the exact pairwise formula with
    ell = -d(., A), h = d(., A).)  rho_inf <= min(1, D_i / delta_i).
    CAUTION: an UPPER bound only — rho_inf ~ 1 does not imply rho(L) ~ 1 at finite L
    (f(u, v) = u + v, anchors {(0,0), (1,1)}: rho(L) = 1 - 1/L while rho_inf = 1).
    Inactive features (x_j = r_j) get 0 (rho undefined there; filter by delta > 0)."""
    d = x.numel()
    delta = (x - r).abs() / sigma
    active = torch.nonzero(delta > 0).flatten().tolist()
    m = len(active)
    out = x.new_zeros(d)
    if m == 0:
        return out
    if m > max_players:
        raise ValueError(f"{m} active features > max_players={max_players}")
    masks = all_masks(m, x.device)
    Z = vertex_points(x, r, active, masks)
    q = dist_to_set(Z, A, sigma)
    w = shapley_weights(m, x.dtype, x.device)
    idx = torch.arange(2 ** m, device=x.device)
    pop = masks.sum(1)
    for jj, j in enumerate(active):
        bit = 1 << jj
        S = idx[(idx & bit) == 0]
        P = S | bit
        out[j] = (w[pop[S]] * torch.clamp((q[P] + q[S]) / delta[j], max=1.0)).sum()
    return out


# --------------------------------------------------------- exact pairwise bounds
def pair_bounds_exact(x: Tensor, r: Tensor, A: Tensor, fA: Tensor, L: float, sigma: Tensor, *,
                      max_players: int = 16) -> tuple[Tensor, Tensor, Tensor]:
    """Deterministic pairwise bounds by full enumeration (small d).

    Returns (lo, hi, D), each shape (d,):
      lo_i = E[a_S], hi_i = E[b_S]  -> phi_i(g) in [lo_i, hi_i] for all g in F_L(A)
      D_i  = sum_S w_S (d(z+, A) + d(z-, A))   (Shapley-weighted distance baseline)
    Inactive features (x_j = r_j) are null players: lo = hi = D = 0, and the active
    features' values equal those of the reduced game (Shapley consistency).
    The anchor set A must already contain x (and r in endpoint-fixed mode).
    """
    d = x.numel()
    delta = (x - r).abs() / sigma
    active = torch.nonzero(delta > 0).flatten().tolist()
    m = len(active)
    lo, hi, D = x.new_zeros(d), x.new_zeros(d), x.new_zeros(d)
    if m == 0:
        return lo, hi, D
    if m > max_players:
        raise ValueError(f"{m} active features > max_players={max_players}; use mc_pair_bounds")
    masks = all_masks(m, x.device)
    Z = vertex_points(x, r, active, masks)
    ell, h = envelopes(Z, A, fA, L, sigma)
    dA = dist_to_set(Z, A, sigma)
    w = shapley_weights(m, x.dtype, x.device)
    idx = torch.arange(2 ** m, device=x.device)
    pop = masks.sum(1)
    for jj, j in enumerate(active):
        bit = 1 << jj
        S = idx[(idx & bit) == 0]
        P = S | bit
        Ld = L * delta[j]
        a = torch.maximum(ell[P] - h[S], -Ld)
        b = torch.minimum(h[P] - ell[S], Ld)
        ws = w[pop[S]]
        lo[j] = (ws * a).sum()
        hi[j] = (ws * b).sum()
        D[j] = (ws * (dA[P] + dA[S])).sum()
    return lo, hi, D


# ------------------------------------------------------ Monte Carlo certificates
def coalition_masks(i: int, d: int, n_per_stratum: int, gen: torch.Generator | None,
                    device="cpu") -> Tensor:
    """Stratified Shapley coalitions for feature i: shape (d, n0, d) bool.
    Stratum k (k = 0..d-1) holds n0 uniform subsets of size k of [d] minus {i}.
    P(S) = (1/d) / C(d-1, k) = Shapley weight, so the stratum-average of stratum means is unbiased.
    Keys are drawn on the generator's device, then moved (CPU generator + CUDA data is allowed)."""
    gdev = gen.device if gen is not None else device
    keys = torch.rand(d, n_per_stratum, d, generator=gen, device=gdev).to(device)
    keys[..., i] = 2.0  # largest key -> never among the first k (k <= d-1)
    ranks = keys.argsort(dim=-1).argsort(dim=-1)
    k = torch.arange(d, device=device).view(d, 1, 1)
    return ranks < k


def _eb_radius(X: Tensor, R: Tensor, delta_conf: float) -> Tensor:
    """One-sided empirical-Bernstein radius (Maurer & Pontil 2009, Thm 4), per stratum.
    X: (..., n) samples in an interval of length R (broadcast to X[..., 0])."""
    n = X.shape[-1]
    V = X.var(dim=-1, unbiased=True)
    lg = math.log(2.0 / delta_conf)
    return torch.sqrt(2.0 * V * lg / n) + 7.0 * R * lg / (3.0 * (n - 1))


def _add_ref_anchor(Z, ell, h, refs, fr, ref_idx, L, sigma):
    dr = ((Z - refs[ref_idx]).abs() / sigma).sum(1)
    return (torch.maximum(ell, fr[ref_idx] - L * dr),
            torch.minimum(h, fr[ref_idx] + L * dr))


@dataclass
class MCResult:
    lo: Tensor      # (K, d): w.p. >= 1-eta, simultaneously lo <= phi_i^exact(g; x, r_k) for all g, i, k
    hi: Tensor      # (K, d): ... and phi_i^exact(g; x, r_k) <= hi
    a_hat: Tensor   # (K, d) stratified estimate of E[a]
    b_hat: Tensor   # (K, d) stratified estimate of E[b]
    method: str
    eta: float
    n_per_stratum: int


def mc_pair_bounds(x: Tensor, refs: Tensor, A: Tensor, fA: Tensor, L: float, sigma: Tensor, *,
                   fx: Tensor, fr: Tensor | None = None, mode: str = "domain",
                   n_per_stratum: int = 64, eta: float = 0.05, method: str = "best",
                   gen: torch.Generator | None = None, radius_scale: float = 1.0) -> MCResult:
    """High-probability bounds on exact Shapley values, uniformly over F_L(anchors),
    over all d features and all K references. No model evaluation at hybrid inputs.

    A, fA : base anchors (calibration set) WITHOUT x; x is always added (explanandum).
    mode  : 'domain' -> anchors A ∪ {x};  'fixed' -> anchors A ∪ {x, r_k} (needs fr).
    method: 'hoeffding' | 'bernstein' | 'best' (eta split in half, tighter side kept).
    The caller must ensure L >= L_A of every anchor set used (see envelope.lipschitz_after_adding).
    Strata k = 0 and k = d-1 contain a single coalition and are handled exactly (radius 0).
    radius_scale: multiplies every confidence radius. 1.0 = the valid bound. Values < 1 are a
    NEGATIVE CONTROL only (deliberately invalid bounds that a coverage test must detect).
    """
    if mode not in ("domain", "fixed"):
        raise ValueError(mode)
    if radius_scale < 0:
        raise ValueError("radius_scale must be >= 0")
    if method not in ("hoeffding", "bernstein", "best"):
        raise ValueError(method)
    if mode == "fixed" and fr is None:
        raise ValueError("mode='fixed' needs fr = f(refs)")
    if n_per_stratum < 2:
        raise ValueError("n_per_stratum must be >= 2")
    K, d = refs.shape
    n0 = n_per_stratum
    dev, dt = x.device, x.dtype
    A0 = torch.cat([A, x.unsqueeze(0)])
    f0 = torch.cat([fA, fx.reshape(1)])
    delta = (x.unsqueeze(0) - refs).abs() / sigma  # (K, d)

    nondeg = torch.zeros(d, dtype=dt, device=dev)
    if d >= 3:
        nondeg[1:d - 1] = 1.0
    s_nd = int(nondeg.sum())
    eta_h = eta if method == "hoeffding" else eta / 2.0
    eta_b = eta if method == "bernstein" else eta / 2.0

    lo, hi = x.new_zeros(K, d), x.new_zeros(K, d)
    a_hat, b_hat = x.new_zeros(K, d), x.new_zeros(K, d)
    ref_idx = torch.arange(K, device=dev).repeat_interleave(d * n0)
    shape = (K, d, n0)

    for i in range(d):
        M = coalition_masks(i, d, n0, gen, dev)          # (d strata, n0, d)  — shared by all refs (CRN)
        Mp = M.clone()
        Mp[..., i] = True
        Zm = torch.where(M.unsqueeze(0), x, refs[:, None, None, :]).reshape(-1, d)
        Zp = torch.where(Mp.unsqueeze(0), x, refs[:, None, None, :]).reshape(-1, d)
        ell_m, h_m = envelopes(Zm, A0, f0, L, sigma)
        ell_p, h_p = envelopes(Zp, A0, f0, L, sigma)
        if mode == "fixed":
            ell_m, h_m = _add_ref_anchor(Zm, ell_m, h_m, refs, fr, ref_idx, L, sigma)
            ell_p, h_p = _add_ref_anchor(Zp, ell_p, h_p, refs, fr, ref_idx, L, sigma)

        Ld = (L * delta[:, i]).view(K, 1, 1)
        a = torch.maximum(ell_p.view(shape) - h_m.view(shape), -Ld)
        b = torch.minimum(h_p.view(shape) - ell_m.view(shape), Ld)
        am, bm = a.mean(-1), b.mean(-1)                  # (K, d strata)
        a_hat[:, i], b_hat[:, i] = am.mean(-1), bm.mean(-1)
        R = 2.0 * L * delta[:, i]                         # (K,) range of a_S and of b_S

        lo_i = torch.full((K,), -math.inf, dtype=dt, device=dev)
        hi_i = torch.full((K,), math.inf, dtype=dt, device=dev)
        if method in ("hoeffding", "best"):
            eta_side = eta_h / (2 * K * d)
            # estimator = (1/d) sum_k mean_k; random part = s_nd strata x n0 samples of weight 1/(d n0)
            t = radius_scale * R * math.sqrt(s_nd * math.log(1.0 / eta_side) / (2.0 * d ** 2 * n0))
            lo_i = torch.maximum(lo_i, a_hat[:, i] - t)
            hi_i = torch.minimum(hi_i, b_hat[:, i] + t)
        if method in ("bernstein", "best"):
            if s_nd == 0:
                rad_a = rad_b = torch.zeros_like(am)
            else:
                dconf = eta_b / (2 * K * d * s_nd)
                rad_a = radius_scale * _eb_radius(a, R.view(K, 1), dconf) * nondeg
                rad_b = radius_scale * _eb_radius(b, R.view(K, 1), dconf) * nondeg
            lo_i = torch.maximum(lo_i, (am - rad_a).mean(-1))
            hi_i = torch.minimum(hi_i, (bm + rad_b).mean(-1))
        # a_S >= -L delta and b_S <= L delta deterministically
        lo[:, i] = torch.maximum(lo_i, -L * delta[:, i])
        hi[:, i] = torch.minimum(hi_i, L * delta[:, i])

    return MCResult(lo=lo, hi=hi, a_hat=a_hat, b_hat=b_hat, method=method, eta=eta,
                    n_per_stratum=n0)
