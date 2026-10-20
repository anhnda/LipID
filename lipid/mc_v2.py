"""
lipid/mc_v2.py — valid stratified Monte Carlo bounds on the sharp marginal endpoints, v1.2 variants (SCL-02; Q5).

Common structure (paper appendix F). For feature i and coalition-size stratum k = 0..m-1, the stratum mean of the
deterministic edge bounds a_{i,S}, b_{i,S} in [-L delta_i, L delta_i] (range R = 2 L delta_i) is estimated from coalitions
S of size k not containing i; the Shapley weight of every stratum is 1/m, so
    lower_i = (1/m) sum_k LCB_k(a),    upper_i = (1/m) sum_k UCB_k(b).
Simultaneous validity over all features, strata and both sides by a union bound over the SAMPLED events only; which
strata are enumerated exhaustively depends only on (m, n_k), never on data, so the event count is fixed in advance.

Methods (protocols/v12_preregistration.json, SCL-02):
  hoeffding_paper   as published: i.i.d. draws, eps = R sqrt(log(4 m^2/alpha) / (2 n)), clipping of the final endpoint
                    only (reproduces lipid.mc.stratified_bounds draw for draw; tests/test_v12_mc.py)
  hoeffding_tight   delta_e = alpha / E with E = 2 m (#sampled strata); strata with C(m-1, k) <= n_k enumerated
                    exhaustively (eps = 0; this includes k = 0 and k = m-1); per-stratum clipping to [-L d_i, L d_i]
  bernstein         as hoeffding_tight but eps from the empirical Bernstein bound of Maurer & Pontil (2009, Thm 4):
                    with prob. >= 1 - delta, E X - mean <= sqrt(2 V_n log(2/delta) / n) + 7 R log(2/delta) / (3 (n - 1)),
                    V_n the unbiased sample variance; applied per side (to a and to -b); needs n >= 2
  serfling_wor      sampling WITHOUT replacement, eps = R sqrt((1 - (n-1)/N) log(1/delta_e) / (2 n)) (Serfling 1974,
                    N = C(m-1, k)); exhaustive small strata; per-stratum clipping
Per-stratum clipping is valid because every stratum mean lies in [-L delta_i, L delta_i].
`radius_scale = 0` gives the zero-radius negative control (point estimates, invalid by design).
No response-function evaluation happens here: anchor labels fA are inputs.
"""
from __future__ import annotations

import itertools
import math
import time

import torch

import lipid  # noqa: F401
from envelope import envelopes
from lipid.mc import sample_coalitions

METHODS = {
    "hoeffding_paper": dict(radius="hoeffding", events="paper", exhaustive=False, wor=False, clip="overall"),
    "hoeffding_tight": dict(radius="hoeffding", events="count", exhaustive=True, wor=False, clip="stratum"),
    "bernstein": dict(radius="bernstein", events="count", exhaustive=True, wor=False, clip="stratum"),
    "serfling_wor": dict(radius="serfling", events="count", exhaustive=True, wor=True, clip="stratum"),
}


def all_coalitions(i: int, m: int, k: int, device="cpu") -> torch.Tensor:
    others = [j for j in range(m) if j != i]
    combos = list(itertools.combinations(others, k))
    M = torch.zeros(len(combos), m, dtype=torch.bool)
    for row, cmb in enumerate(combos):
        if cmb:
            M[row, list(cmb)] = True
    return M.to(device)


def sample_coalitions_wor(i: int, m: int, k: int, n: int, gen: torch.Generator, device="cpu") -> torch.Tensor:
    """n DISTINCT uniform size-k subsets of [m] minus {i}: the first n distinct values of an i.i.d. uniform sequence
    form a uniformly random n-subset (in uniformly random order)."""
    N = math.comb(m - 1, k)
    if n > N:
        raise ValueError("n > number of coalitions; enumerate instead")
    seen, rows = set(), []
    while len(rows) < n:
        M = sample_coalitions(i, m, k, 2 * (n - len(rows)) + 4, gen, "cpu")
        for row in M:
            key = int(sum(1 << j for j in torch.nonzero(row).flatten().tolist()))
            if key not in seen:
                seen.add(key)
                rows.append(row)
                if len(rows) == n:
                    break
    return torch.stack(rows).to(device)


def _edge_values(M, i, x, r, A, fA, L, sigma, Ld, max_elems):
    Mp = M.clone()
    Mp[:, i] = True
    Zm = torch.where(M, x, r)
    Zp = torch.where(Mp, x, r)
    n = M.shape[0]
    ell, h = envelopes(torch.cat([Zm, Zp]), A, fA, L, sigma, max_elems=max_elems)
    a = torch.maximum(ell[n:] - h[:n], torch.full_like(ell[n:], -Ld))
    b = torch.minimum(h[n:] - ell[:n], torch.full_like(ell[n:], Ld))
    return a, b


def stratified_bounds_v2(x, r, A, fA, L, sigma, *, n_per_stratum: int, alpha: float, gen: torch.Generator,
                         method: str = "hoeffding_tight", radius_scale: float = 1.0, max_elems: int = 2 ** 24) -> dict:
    """All features of x must be active. A must contain x (domain) or x and r (fixed)."""
    cfg = METHODS[method]
    t0 = time.perf_counter()
    m = x.numel()
    delta = (x - r).abs() / sigma
    if bool((delta <= 0).any()):
        raise ValueError("stratified_bounds_v2 assumes every feature is active")
    if cfg["radius"] == "bernstein" and n_per_stratum < 2:
        raise ValueError("empirical Bernstein needs n >= 2")
    Nk = [math.comb(m - 1, k) for k in range(m)]
    exh = [bool(cfg["exhaustive"] and Nk[k] <= n_per_stratum) for k in range(m)]
    n_sampled = sum(not e for e in exh)
    if cfg["events"] == "paper":
        delta_e = alpha / (4.0 * m * m)
    else:
        delta_e = alpha / max(1, 2 * m * n_sampled)
    lo, hi = x.new_zeros(m), x.new_zeros(m)
    n_coal = 0
    for i in range(m):
        Ld = float(L * delta[i])
        R = 2.0 * Ld
        acc_lo = acc_hi = 0.0
        for k in range(m):
            if exh[k]:
                M = all_coalitions(i, m, k, x.device)
            elif cfg["wor"]:
                M = sample_coalitions_wor(i, m, k, n_per_stratum, gen, x.device)
            else:
                M = sample_coalitions(i, m, k, n_per_stratum, gen, x.device)
            n = M.shape[0]
            n_coal += n
            a, b = _edge_values(M, i, x, r, A, fA, L, sigma, Ld, max_elems)
            am, bm = float(a.mean()), float(b.mean())
            if exh[k]:
                ea = eb = 0.0
            elif cfg["radius"] == "hoeffding":
                ea = eb = R * math.sqrt(math.log(1.0 / delta_e) / (2.0 * n))
            elif cfg["radius"] == "serfling":
                ea = eb = R * math.sqrt((1.0 - (n - 1.0) / Nk[k]) * math.log(1.0 / delta_e) / (2.0 * n))
            else:  # empirical Bernstein (Maurer & Pontil 2009, Theorem 4), one-sided per side
                lg = math.log(2.0 / delta_e)
                va, vb = float(a.var(correction=1)), float(b.var(correction=1))
                ea = math.sqrt(2.0 * va * lg / n) + 7.0 * R * lg / (3.0 * (n - 1.0))
                eb = math.sqrt(2.0 * vb * lg / n) + 7.0 * R * lg / (3.0 * (n - 1.0))
            lk, hk = am - radius_scale * ea, bm + radius_scale * eb
            if cfg["clip"] == "stratum":
                lk, hk = max(lk, -Ld), min(hk, Ld)
            acc_lo += lk
            acc_hi += hk
        lo[i] = max(acc_lo / m, -Ld)
        hi[i] = min(acc_hi / m, Ld)
    return dict(lo=lo, hi=hi, delta_e=delta_e, n_events=(4 * m * m if cfg["events"] == "paper" else 2 * m * n_sampled),
                exhaustive_strata=[k for k in range(m) if exh[k]], coalitions_per_feature=n_coal / m,
                seconds=time.perf_counter() - t0)


__all__ = ["METHODS", "stratified_bounds_v2", "all_coalitions", "sample_coalitions_wor"]
