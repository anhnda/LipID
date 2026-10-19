"""
t6_directional.py — directional decomposition of the exact normalized AIW (torch, any device).

Feature i, coalition S not containing i, z+ = z_{S+i}, z- = z_S, Ld = L * delta_i,
ell / h = McShane–Whitney envelopes of (A, f|_A, L):

    B+ = h(z+) - ell(z-) = min_{u,v} [ f(u) - f(v) + L (d(z+,u) + d(z-,v)) ]
    B- = h(z-) - ell(z+) = min_{u,v} [ f(v) - f(u) + L (d(z+,u) + d(z-,v)) ]
    c+ = B+ / Ld,  c- = B- / Ld          (non-empty class:  c± >= -1,  c+ + c- >= 0)

Exact (P3):          b_S = Ld min(c+, 1),  a_S = -Ld min(c-, 1)
                     rho_S = (min(c+,1) + min(c-,1)) / 2
Directional-min (former 'box', T6):
                     T_box = min_{u,v} [L R_uv - |f(u) - f(v)|] = min(B+, B-)        (exact identity)
                     dmin_S = clamp(min(c+, c-), 0, 1)

Deterministic consequences, per feature-instance (Shapley weights sum to 1):
    (I1)  2 rho - 1  <=  rho_dmin  <=  rho
    (I2)  rho - rho_dmin = asym_pos + neg_part, where on each edge (a = min(c+,1), b = min(c-,1))
              min(a,b) >= 0:  |a - b| / 2      (directional asymmetry)
              min(a,b) <  0:  (a + b) / 2      (one direction forced; the sign-relevant regime)
    (I3)  certified sign  =>  rho < 1/2      (general: [lo, hi] inside [-L delta, L delta])
    (I4)  s_S = (q+ + q-)/delta_i <= 1 + 2 D_S / delta_i  (x in A);  fixed mode: 1 + 2 min(B_S, D_S)/delta_i,
          D_S = sum_{j not in S+i} delta_j,  B_S = sum_{j in S} delta_j
    (I5)  fixed mode at L = L_min:  rho_det = 0 on every edge  (the pair (x, r) gives R - d = -delta_i)
    (I6)  fixed mode, efficiency cap: with D = d(x, r), Delta = f(x) - f(r), X_i = (D - |Delta|/L)/delta_i - 1:
              rho_dmin_i <= clamp(X_i, 0, 1),   rho_i <= (1 + min(1, X_i)) / 2,
              X_i < 0  =>  sign(phi_i) = sign(Delta) is certified for every g in F_L(A)
          (phi_i = Delta - sum_{j != i} phi_j and |phi_j| <= L delta_j.)

Per-pair margin (at level L, normalized by L delta_i):  g = (R - d)/delta_i >= -1,  h = d/delta_i, t = lambda/L_min,
k = L_min / L:
    det   : V = g + (1 - k) h
    box   : V = g + (1 - k t) h                          (min over pairs = min(c+, c-))
    K     : V = g + (1 - k q_K(t)) h,  q_K(t) = max(ceil(K t), 1) / K  >= t
An edge is certified saturated by a bound iff min_{u,v} V >= 1 (M1). Pairs with g >= 1 never block, so only
B_S = {g < 1} matters:  q_K(t) <= (1/k)(1 - (1 - g)/h)  on B_S.  For an exact straddler (g = -1) that is
q_K(t) <= (1/k)(1 - 2/h)  — with q_K(t), not t, for finite K (t only in the box limit K -> inf).
Decoupled sufficient certificates (geometry vs slope): k max_{B_S} q <= 1 - max_{B_S} (1 - g)/h  (psi-cert),
weaker uniform form (1 - k Q) H >= 2 with H = min_{B_S} h.  'Straddlers' are pairs with g = -1
(exact l1-betweenness of u, z+, z-, v); many pairs can tie there, so per-edge statistics below use the
WHOLE tied / near-minimizing set, never a single argmin (argmin among ties is decided by round-off).
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
    rho: float       # exact normalized AIW
    dmin: float      # directional-min bound (former rho_box)
    asym: float      # rho - dmin  (= asym_pos + neg_part)
    asym_pos: float  # part from edges with both directions >= 0: sum w |a - b| / 2
    neg_part: float  # part from edges with one direction < 0:    sum w (a + b) / 2
    negmass: float   # Shapley weight of edges with min(c+, c-) < 0
    lo_n: float      # lo / Ld
    hi_n: float      # hi / Ld

    def sign(self, Ld: float, tol: float) -> int:
        if Ld * self.lo_n > tol:
            return 1
        if Ld * self.hi_n < -tol:
            return -1
        return 0


def feature_stats(w: Tensor, cp: Tensor, cm: Tensor) -> FeatureStats:
    hp, hm = cp.clamp(max=1.0), cm.clamp(max=1.0)
    rho_e = (hp + hm) / 2.0
    mn = torch.minimum(cp, cm)
    neg = mn < 0
    rho = float((w * rho_e).sum())
    dmin = float((w * mn.clamp(0.0, 1.0)).sum())
    asym_pos = float((w * (hp - hm).abs() / 2.0)[~neg].sum())
    neg_part = float((w * rho_e)[neg].sum())
    return FeatureStats(rho=rho, dmin=dmin, asym=rho - dmin, asym_pos=asym_pos, neg_part=neg_part,
                        negmass=float(w[neg].sum()), lo_n=-float((w * hm).sum()), hi_n=float((w * hp).sum()))


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
    """t_uv = lambda_uv / L_min in [0, 1]; 0 on the diagonal."""
    lam = torch.where(DA > 1e-12, dF / DA.clamp_min(1e-12), torch.zeros_like(DA))
    return (lam / L_min).clamp(0.0, 1.0)


def quantized_T(R: Tensor, DA: Tensor, ratio: Tensor, L: float, L_min: float, K: int) -> Tensor:
    """T_K with K slope levels. K = 1 reproduces det exactly (Lam = L_min everywhere, also when t = 0)."""
    q = torch.clamp(torch.ceil(K * ratio), min=1.0) / K
    return (L * R - (L_min * q * DA)[None]).amin(dim=(1, 2))


def det_T(R: Tensor, DA: Tensor, L: float, L_min: float) -> Tensor:
    return (L * R - L_min * DA[None]).amin(dim=(1, 2))


# ------------------------------------------------- exact straddlers and the psi certificate
def straddler_exact(A: Tensor, x: Tensor, r: Tensor, Zp: Tensor, i: int) -> Tensor:
    """Exact weighted-l1 betweenness (no tolerance): pair (u, v) straddles edge e iff
      coordinate i : u_i, x_i, r_i, v_i are monotone in the direction x -> r   (z+_i = x_i, z-_i = r_i)
      every j != i : z+_j (= z-_j) lies in [min(u_j, v_j), max(u_j, v_j)]
    which is exactly g = (R - d)/delta_i = -1 (the triangle-inequality bound is attained coordinate-wise).
    A: (N, d) anchors, Zp: (E, d) the z+ vertex of each edge, i: ORIGINAL coordinate of the feature.
    Returns (E, N, N) bool."""
    s = torch.sign(x[i] - r[i])
    coord_i = (((A[:, i] - x[i]) * s >= 0)[:, None]) & (((r[i] - A[:, i]) * s >= 0)[None, :])
    lo = torch.minimum(A[:, None, :], A[None, :, :])
    hi = torch.maximum(A[:, None, :], A[None, :, :])
    keep = torch.ones(A.shape[1], dtype=torch.bool, device=A.device)
    keep[i] = False
    Zk, lo, hi = Zp[:, keep], lo[..., keep], hi[..., keep]
    between = ((lo[None] <= Zk[:, None, None, :]) & (Zk[:, None, None, :] <= hi[None])).all(-1)
    return coord_i[None] & between


def psi_certificate(g: Tensor, h: Tensor, qv: Tensor, k: float) -> tuple[Tensor, Tensor, Tensor]:
    """Decoupled sufficient certificate for FULL saturation of each edge (any L >= L_min, any K, box: qv = t).
      B_S   = {(u, v): g_uv,S < 1}                       (pairs with g >= 1 can never block, since (1 - k q) h >= 0)
      psi_S = max_{B_S} (1 - g)/h                         (geometry only: detour deficit per unit span)
      Q_S   = max_{B_S} q(t_uv)                           (slope only)
      cert  : B_S empty  or  k Q_S <= 1 - psi_S
    Proof: on B_S, (1 - k q) h >= (1 - k Q_S) h >= psi_S h >= 1 - g, so V = g + (1 - k q) h >= 1.
    It is a relaxation of box (pairwise coupling replaced by the product of two maxima, possibly attained at
    different pairs): cert => min V >= 1, not conversely.  g: (E,N,N), h: (1,N,N), qv: (N,N).
    Returns (cert, empty, psi) per edge.
    Round-off: B_S uses g < 1 - 1e-12 (u = v pairs have g >= 1 exactly, but can come out as 1 - 1e-16 with h = 0);
    pairs with g in [1 - 1e-12, 1) keep V >= g >= 1 - 1e-12, the same tolerance as the full-saturation test."""
    blk = g < 1.0 - 1e-12
    empty = ~blk.flatten(1).any(1)
    hh = h.expand_as(g)
    psi = torch.where(blk, (1.0 - g) / hh.clamp_min(1e-12), torch.full_like(g, float("-inf"))).flatten(1).max(1).values
    Q = torch.where(blk, qv[None].expand_as(g), torch.zeros_like(g)).flatten(1).max(1).values
    return empty | (k * Q <= 1.0 - psi), empty, psi


def separation_decomposition(g: Tensor, h: Tensor, qv: Tensor, k: float, tol: float = 1e-12) -> dict:
    """Exact decomposition of the psi certificate (B_S nonempty). On B_S put a = (1 - g)/h, b = k q:
        exact saturation        <=>  max_B (a + b) <= 1           i.e.  mu_S = 1 - max_B (a + b) >= 0
        psi certificate         <=>  max_B a + max_B b <= 1       i.e.  C_S <= mu_S
        separation loss  C_S    =    max_B a + max_B b - max_B (a + b)  >= 0
    (per pair: V - 1 = h (1 - a - b), so mu_S is the saturation margin per unit span, not min V - 1).
    C_S = 0 whenever |B_S| = 1, so C_S must be read together with nB = |B_S|.
    Empty B_S: C = 0, mu = +inf (trivial). Returns per-edge tensors."""
    blk = g < 1.0 - tol
    empty = ~blk.flatten(1).any(1)
    neg_inf = torch.full_like(g, float("-inf"))
    a = torch.where(blk, (1.0 - g) / h.expand_as(g).clamp_min(1e-12), neg_inf)
    b = torch.where(blk, (k * qv)[None].expand_as(g), neg_inf)
    max_a = a.flatten(1).max(1).values
    max_b = b.flatten(1).max(1).values
    max_ab = (a + b).flatten(1).max(1).values
    zero = torch.zeros_like(max_a)
    C = torch.where(empty, zero, max_a + max_b - max_ab)
    mu = torch.where(empty, torch.full_like(max_a, float("inf")), 1.0 - max_ab)
    return dict(empty=empty, psi=max_a, kQ=max_b, C=C, mu=mu, nB=blk.flatten(1).sum(1))


# ------------------------------------------------- tie-robust geometry / slope diagnostic
def margin_diagnostic(R: Tensor, DA: Tensor, ratio: Tensor, cp: Tensor, cm: Tensor, w: Tensor,
                      L: float, L_min: float, delta_i: float, endpoints: list[int], sat: float = 0.8,
                      eta: float = 1e-9, nm_eps: float = 0.02, strad_exact: Tensor | None = None) -> dict:
    """Tie-robust replacement of the old argmin diagnostic. Per edge:
      straddlers  = pairs with g <= -1 + eta (1 + h)   (exact l1-betweenness up to round-off)
      box near-minimizers NM = pairs with V_box <= min V_box + nm_eps
    Two edge groups (Shapley-weighted means; None when the group is empty):
      rescued = det < sat <= box :  frac_strad   (edge has >= 1 straddler),
                                    frac_ep_strad (edge has a straddler that involves x or r),
                                    log10 of the straddler count (median over edges)
      blocked = box < sat        :  NM share that are straddlers / involve x or r, mean t and h on NM
    Also returns m_err = max |min V_box - min(c+, c-)| (identity check, must be ~1e-13)."""
    g = (R - DA[None]) / delta_i
    h = (DA / delta_i)[None]
    N = DA.shape[0]
    ep = torch.zeros(N, dtype=torch.bool, device=DA.device)
    ep[endpoints] = True
    ep_pair = (ep[:, None] | ep[None, :])[None]
    strad_tol = g <= -1.0 + eta * (1.0 + h)          # round-off tolerance only (eta = 1e-9)
    if strad_exact is not None:                       # exact coordinate-order test; tolerance version cross-checked
        strad = strad_exact
        strad_mismatch = int((strad_tol != strad_exact).sum())
    else:
        strad, strad_mismatch = strad_tol, None
    n_strad = strad.flatten(1).sum(1)
    has_ep_strad = (strad & ep_pair).flatten(1).any(1)

    k = L_min / L
    e_det = (g + (1.0 - k) * h).flatten(1).min(1).values.clamp(0.0, 1.0)
    V = g + (1.0 - k * ratio[None]) * h
    M = V.flatten(1).min(1).values
    m_err = float((M - torch.minimum(cp, cm)).abs().max())
    e_box = M.clamp(0.0, 1.0)
    nm = V <= (M + nm_eps)[:, None, None]
    n_nm = nm.flatten(1).sum(1).clamp_min(1).to(V.dtype)
    sh_strad = (nm & strad).flatten(1).sum(1) / n_nm
    sh_ep = (nm & ep_pair).flatten(1).sum(1) / n_nm
    t_nm = (nm * ratio[None]).flatten(1).sum(1) / n_nm
    h_nm = (nm * h).flatten(1).sum(1) / n_nm

    def wmean(x, mask):
        wm = float(w[mask].sum())
        return None if wm <= 0 else float((w * x.to(w.dtype))[mask].sum()) / wm

    # geometry-only minimum G_S = min g (L-independent). At L_min det_S = clamp(G_S, 0, 1); an exact straddler
    # gives G_S = -1, but losing all straddlers (G_S > -1) does NOT make det positive (that needs G_S > 0).
    # G_S = 1 is NOT a null event: in domain mode the top edge (z+ = x) always has the pair (x, x) with g = 1,
    # so G_S = 1 exactly whenever no other pair has g < 1. Same tolerance as the empty-B_S test.
    G = g.flatten(1).min(1).values
    has_strad = n_strad > 0
    ge1 = G >= 1.0 - 1e-12
    G_bins = dict(G_strad=float(w[has_strad].sum()),
                  G_neg=float(w[~has_strad & (G <= 0)].sum()),
                  G_01=float(w[(G > 0) & ~ge1].sum()),
                  G_ge1=float(w[ge1].sum()))

    # decoupled sufficient certificates for FULL edge saturation (min V >= 1); see psi_certificate.
    #   uniform (H, Q)  : (1 - k Q) H >= 2, H = min_{g<1} h      (weaker; implies the psi certificate)
    # empty B_S (G_S >= 1, i.e. det already saturated at L_min) certifies trivially -> reported separately.
    q4 = torch.clamp(torch.ceil(4 * ratio), min=1.0) / 4
    cert_psi_box, empty, _ = psi_certificate(g, h, ratio, k)
    cert_psi_K4, _, _ = psi_certificate(g, h, q4, k)
    blk = g < 1.0 - 1e-12
    hh = h.expand_as(g)
    H = torch.where(blk, hh, torch.full_like(g, float("inf"))).flatten(1).min(1).values
    Qt = torch.where(blk, ratio[None].expand_as(g), torch.zeros_like(g)).flatten(1).max(1).values
    M4 = (g + (1.0 - k * q4[None]) * h).flatten(1).min(1).values
    full_box, full_K4 = M >= 1.0 - 1e-12, M4 >= 1.0 - 1e-12
    cert_unif_box = empty | ((1.0 - k * Qt) * torch.where(empty, torch.ones_like(H), H) >= 2.0)
    cert_viol = int(((cert_psi_box | cert_unif_box) & ~full_box).sum() + (cert_psi_K4 & ~full_K4).sum())

    def share(c, full):
        wm = float(w[full].sum())
        return None if wm <= 0 else float(w[full & c].sum()) / wm

    # (C_S, mu_S) decomposition on full-saturated edges with nonempty B_S; coverage stratified on a
    # pre-registered mu grid (the margin in the certificate's own units, replacing the old 1 <= min V < 2 cut)
    sd_box = separation_decomposition(g, h, ratio, k)
    sd_K4 = separation_decomposition(g, h, q4, k)
    ne = full_box & ~sd_box["empty"]
    # identity check: psi certificate <=> C <= mu (away from round-off at the boundary)
    far = (sd_box["psi"] + sd_box["kQ"] - 1.0).abs() > 1e-9
    c1_viol = int((far & ~sd_box["empty"] & ((sd_box["C"] <= sd_box["mu"]) != cert_psi_box)).sum())
    mu = sd_box["mu"]
    MU_GRID = (0.0, 0.1, 0.25, 0.5, float("inf"))
    sep = {}
    for b_ in range(4):
        inb = ne & (mu >= MU_GRID[b_] - 1e-12) & (mu < MU_GRID[b_ + 1])
        sep[f"sep_mass_mu{b_}"] = share(inb, ne)
        sep[f"sep_cov_mu{b_}"] = share(cert_psi_box, inb)
    wne = float(w[ne].sum())
    ne4 = full_K4 & ~sd_K4["empty"]
    wne4 = float(w[ne4].sum())
    sep.update(
        sep_C_mean_box=None if wne <= 0 else float((w * sd_box["C"])[ne].sum()) / wne,
        sep_C_mean_K4=None if wne4 <= 0 else float((w * sd_K4["C"])[ne4].sum()) / wne4,
        sep_C_zero_share=share(sd_box["C"] <= 1e-12, ne),
        sep_med_log10_nB=(float(torch.log10(sd_box["nB"][ne].to(w.dtype).clamp_min(1)).median())
                          if bool(ne.any()) else None))
    # degeneracies of C_S = 0  (C_S = 0 <=> argmax a and argmax b intersect):
    #   |B_S| = 1 (trivial); K = 1 (q = 1 constant -> C = 0 always, never reported as a finding);
    #   K = 4 ties among the 4 quantized slope levels. For box, report C = 0 only among edges with |B_S| >= 2.
    ne_multi = ne & (sd_box["nB"] >= 2)
    lv4 = torch.stack([(blk & (q4[None] == l_ / 4.0)).flatten(1).any(1) for l_ in (1, 2, 3, 4)]).sum(0)
    # round-off ambiguity: some pair with g in [1 - 1e-12, 1) or min V in [1 - 1e-12, 1). Certificates are claims
    # of saturation up to 1e-12; these edges are flagged rather than silently counted as exact.
    ambig = ((g >= 1.0 - 1e-12) & (g < 1.0)).flatten(1).any(1) | ((M >= 1.0 - 1e-12) & (M < 1.0))
    sep.update(sep_C_zero_share_nB2=share(sd_box["C"] <= 1e-12, ne_multi),
               sep_C_zero_share_K4=share(sd_K4["C"] <= 1e-12, ne4),
               sep_K4_levels_med=(float(lv4[ne4].to(w.dtype).median()) if bool(ne4.any()) else None),
               ambig_mass=float(w[ambig].sum()))

    rescued = (e_det < sat) & (e_box >= sat)
    blocked = e_box < sat
    out = dict(m_err=m_err, cert_viol=cert_viol, strad_mismatch=strad_mismatch, c1_viol=c1_viol, **G_bins, **sep,
               full_box_mass=float(w[full_box].sum()), full_K4_mass=float(w[full_K4].sum()),
               psi_empty_share=share(empty, full_box),
               cert_psi_box_share=share(cert_psi_box, full_box),
               cert_psi_box_share_nontriv=share(cert_psi_box & ~empty, full_box & ~empty),
               cert_unif_box_share=share(cert_unif_box, full_box),
               cert_psi_K4_share=share(cert_psi_K4, full_K4),
               rescued_mass=float(w[rescued].sum()), blocked_mass=float(w[blocked].sum()),
               rescued_frac_strad=wmean(n_strad > 0, rescued),
               rescued_frac_ep_strad=wmean(has_ep_strad, rescued),
               rescued_med_log10_nstrad=(float(torch.log10(n_strad[rescued].to(w.dtype).clamp_min(1)).median())
                                         if bool(rescued.any()) else None),
               blocked_nm_share_strad=wmean(sh_strad, blocked),
               blocked_nm_share_ep=wmean(sh_ep, blocked),
               blocked_nm_t=wmean(t_nm, blocked),
               blocked_nm_h=wmean(h_nm, blocked))
    return out


# --------------------------------------------------------------------- fixed-mode caps
def efficiency_cap(D_xr: float, Delta: float, L: float, delta_i: float, tol: float = 0.0) -> dict:
    """(I6), fixed mode only. X = (D - |Delta|/L)/delta_i - 1.
    X < 0 gives lo >= -L delta_i X (Delta > 0); the sign is reported as forced only when that margin exceeds
    tol, so that it is comparable with the stored sign (lo > tol)."""
    X = (D_xr - abs(Delta) / L) / delta_i - 1.0
    forced = 0 if (-X * L * delta_i) <= tol or Delta == 0 else (1 if Delta > 0 else -1)
    return dict(X=X, rho_cap=(1.0 + min(1.0, X)) / 2.0, box_cap=min(1.0, max(0.0, X)), forced_sign=forced)


# --------------------------------------------------------------------- endpoint band
def endpoint_band_bound(delta_active: Tensor, S: Tensor, jj: int, fixed: bool = False) -> Tensor:
    """(I4) upper bound on s_S: 1 + 2 D_S / delta_i (x in A); fixed mode 1 + 2 min(B_S, D_S) / delta_i."""
    m = delta_active.numel()
    bits = ((S.unsqueeze(1) >> torch.arange(m, device=S.device)) & 1).bool()
    comp = ~bits
    comp[:, jj] = False
    D_S = (comp.to(delta_active.dtype) * delta_active[None, :]).sum(1)
    if fixed:
        B_S = (bits.to(delta_active.dtype) * delta_active[None, :]).sum(1)
        D_S = torch.minimum(D_S, B_S)
    return 1.0 + 2.0 * D_S / delta_active[jj]
