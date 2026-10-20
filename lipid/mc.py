"""
lipid/mc.py — stratified Monte Carlo bounds on the sharp marginal endpoints (paper appendix F; SCALE-01).

For feature i and coalition-size stratum k = 0..m-1, draw n_k coalitions uniformly (with replacement) among the
size-k subsets of the other m-1 features. The deterministic edge bounds a_{i,S}, b_{i,S} (eq. 8) lie in
[-L delta_i, L delta_i]. With R = 2 L delta_i and the union bound over 2 m^2 (stratum, feature, side) events,

    eps_ik = R sqrt( log(C / alpha) / (2 n_k) ),   lower_i = (1/m) sum_k (a_hat_ik - eps_ik),
                                                   upper_i = (1/m) sum_k (b_hat_ik + eps_ik),
clipped to [-L delta_i, L delta_i]. The paper states C = 4 m^2; C = 2 m^2 already suffices for one-sided Hoeffding
bounds on 2 m^2 events (the paper's constant is conservative). `log_constant` selects 'paper' (4 m^2, default)
or 'tight' (2 m^2). With exact_singletons=True the strata k = 0 and k = m-1 (a single coalition each) are
evaluated exactly (eps = 0) — a valid refinement, reported as a separate method.

The bounds hold uniformly over F_L(A) with probability >= 1 - alpha over the coalition draw (a and b do not depend
on g). Strata sizes must be fixed before sampling. No response-function evaluation happens here: the anchor labels
fA are inputs.
"""
from __future__ import annotations

import math
import time

import torch

import lipid  # noqa: F401
from envelope import envelopes


def sample_coalitions(i: int, m: int, k: int, n: int, gen: torch.Generator, device="cpu") -> torch.Tensor:
    """(n, m) bool: n uniform size-k subsets of [m] minus {i} (with replacement across draws)."""
    keys = torch.rand(n, m, generator=gen).to(device)
    keys[:, i] = 2.0
    ranks = keys.argsort(dim=1).argsort(dim=1)
    return ranks < k


def stratified_bounds(x, r, A, fA, L, sigma, *, n_per_stratum: int, alpha: float, gen: torch.Generator,
                      log_constant: str = "paper", exact_singletons: bool = False,
                      radius_scale: float = 1.0, max_elems: int = 2 ** 24) -> dict:
    """All features of x must be active (x_j != r_j). A must contain x (domain) or x and r (fixed)."""
    t0 = time.perf_counter()
    d = x.numel()
    delta = (x - r).abs() / sigma
    if bool((delta <= 0).any()):
        raise ValueError("stratified_bounds assumes every feature is active")
    m = d
    C = {"paper": 4.0 * m * m, "tight": 2.0 * m * m}[log_constant]
    lo, hi = x.new_zeros(m), x.new_zeros(m)
    a_hat, b_hat = x.new_zeros(m), x.new_zeros(m)
    n_points = 0
    for i in range(m):
        Ld = float(L * delta[i])
        R = 2.0 * Ld
        acc_lo, acc_hi, acc_a, acc_b = 0.0, 0.0, 0.0, 0.0
        for k in range(m):
            single = (k == 0 or k == m - 1)
            nk = 1 if (single and exact_singletons) else n_per_stratum
            M = sample_coalitions(i, m, k, nk, gen, x.device)
            Mp = M.clone()
            Mp[:, i] = True
            Zm = torch.where(M, x, r)
            Zp = torch.where(Mp, x, r)
            ell, h = envelopes(torch.cat([Zm, Zp]), A, fA, L, sigma, max_elems=max_elems)
            n_points += 2 * nk
            ell_m, ell_p = ell[:nk], ell[nk:]
            h_m, h_p = h[:nk], h[nk:]
            a = torch.maximum(ell_p - h_m, torch.full_like(ell_p, -Ld)).mean()
            b = torch.minimum(h_p - ell_m, torch.full_like(ell_p, Ld)).mean()
            eps = 0.0 if (single and exact_singletons) else radius_scale * R * math.sqrt(math.log(C / alpha) / (2 * nk))
            acc_a += float(a)
            acc_b += float(b)
            acc_lo += float(a) - eps
            acc_hi += float(b) + eps
        lo[i] = max(acc_lo / m, -Ld)
        hi[i] = min(acc_hi / m, Ld)
        a_hat[i], b_hat[i] = acc_a / m, acc_b / m
    return dict(lo=lo, hi=hi, a_hat=a_hat, b_hat=b_hat, n_hybrid_points=n_points,
                n_anchor_distances=n_points * A.shape[0], seconds=time.perf_counter() - t0,
                coalition_budget_per_feature=sum(1 if ((k in (0, m - 1)) and exact_singletons) else n_per_stratum
                                                 for k in range(m)))
