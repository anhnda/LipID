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


# ------------------------------------------------- tie-robust geometry / slope diagnostic
def margin_diagnostic(R: Tensor, DA: Tensor, ratio: Tensor, cp: Tensor, cm: Tensor, w: Tensor,
                      L: float, L_min: float, delta_i: float, endpoints: list[int], sat: float = 0.8,
                      eta: float = 1e-9, nm_eps: float = 0.02) -> dict:
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
    strad = g <= -1.0 + eta * (1.0 + h)
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
    G = g.flatten(1).min(1).values
    has_strad = n_strad > 0
    G_bins = dict(G_strad=float(w[has_strad].sum()),
                  G_neg=float(w[~has_strad & (G <= 0)].sum()),
                  G_01=float(w[(G > 0) & (G <= 1)].sum()),
                  G_gt1=float(w[G > 1].sum()))

    # decoupled sufficient certificates for FULL edge saturation (min V >= 1). Only pairs with g < 1 can block.
    #   psi-certificate : k Q <= 1 - psi,  psi = max_{g<1} (1 - g)/h,  Q = max_{g<1} q   (geometry vs slope)
    #   uniform (H, Q)  : (1 - k Q) H >= 2, H = min_{g<1} h                              (implies psi-cert)
    blk = g < 1.0
    empty = ~blk.flatten(1).any(1)
    hh = h.expand_as(g)
    neg_inf = torch.full_like(g, float("-inf"))
    psi = torch.where(blk, (1.0 - g) / hh.clamp_min(1e-12), neg_inf).flatten(1).max(1).values
    H = torch.where(blk, hh, torch.full_like(g, float("inf"))).flatten(1).min(1).values
    q4 = torch.clamp(torch.ceil(4 * ratio), min=1.0) / 4
    Qt = torch.where(blk, ratio[None].expand_as(g), torch.zeros_like(g)).flatten(1).max(1).values
    Q4 = torch.where(blk, q4[None].expand_as(g), torch.zeros_like(g)).flatten(1).max(1).values
    M4 = (g + (1.0 - k * q4[None]) * h).flatten(1).min(1).values
    full_box, full_K4 = M >= 1.0 - 1e-12, M4 >= 1.0 - 1e-12
    cert_psi_box = empty | (k * Qt <= 1.0 - psi)
    cert_unif_box = empty | ((1.0 - k * Qt) * torch.where(empty, torch.ones_like(H), H) >= 2.0)
    cert_psi_K4 = empty | (k * Q4 <= 1.0 - psi)
    cert_viol = int(((cert_psi_box | cert_unif_box) & ~full_box).sum() + (cert_psi_K4 & ~full_K4).sum())

    def share(c, full):
        wm = float(w[full].sum())
        return None if wm <= 0 else float(w[full & c].sum()) / wm

    rescued = (e_det < sat) & (e_box >= sat)
    blocked = e_box < sat
    out = dict(m_err=m_err, cert_viol=cert_viol, **G_bins,
               full_box_mass=float(w[full_box].sum()), full_K4_mass=float(w[full_K4].sum()),
               cert_psi_box_share=share(cert_psi_box, full_box),
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
