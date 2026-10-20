"""
lipid/metric_lp.py — metric sensitivity of the edgewise aggregate (MET-01, paper section 7.2 and appendix B).

Weighted l_p metric  d_p(u, v) = || (u - v) / sigma ||_p,  p in {1, 2}.

* edgewise_interval:  the Theorem-1 aggregate  [sum_S w_S a_S, sum_S w_S b_S]  with l_p McShane-Whitney envelopes.
  For ANY metric it is a valid OUTER interval (each edge bound is necessary). It is sharp for p = 1 (Theorem 1);
  for p = 2 it need not be sharp (appendix B) and must never be reported as sharp.
* lp_interval:  the exact (sharp) marginal interval for ANY metric, by a linear program over the finite set of
  hybrid vertices: variables t_z (z in the 2^m hybrids), constraints
        ell_p(z) <= t_z <= h_p(z)                 (L-Lipschitz to every anchor; equality on anchors)
        |t_u - t_v| <= L d_p(u, v)                 (all hybrid pairs — l_2 is not path-additive on the cube)
  A feasible t together with the anchor labels is an L-Lipschitz function on a finite subset of a metric space,
  so its McShane extension belongs to F_L(A); conversely every member restricts to a feasible t. Hence the LP
  optimum of  sum_S w_S (t_{S+i} - t_S)  is the sharp endpoint. Solved with scipy HiGHS.
No response-function evaluation happens in this module.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

import lipid  # noqa: F401
from bounds import all_masks, vertex_points
from lipid.core import edge_tensors


def wdist(U: torch.Tensor, V: torch.Tensor, sigma: torch.Tensor, p: float) -> torch.Tensor:
    return torch.cdist(U / sigma, V / sigma, p=float(p))


def anchor_L_p(A, fA, sigma, p) -> float:
    D = wdist(A, A, sigma, p)
    dF = (fA[:, None] - fA[None, :]).abs()
    ok = D > 1e-12
    if bool(((~ok) & (dF > 1e-10)).any()):
        return math.inf
    return float(torch.where(ok, dF / D.clamp_min(1e-12), torch.zeros_like(D)).max()) if A.shape[0] > 1 else 0.0


def envelopes_p(Z, A, fA, L, sigma, p):
    LD = L * wdist(Z, A, sigma, p)
    return (fA[None, :] - LD).max(1).values, (fA[None, :] + LD).min(1).values


def _setup(x, r, sigma):
    delta = (x - r).abs() / sigma
    active = torch.nonzero(delta > 0).flatten().tolist()
    m = len(active)
    if m == 0:
        raise ValueError("x == r: no active feature")
    Z = vertex_points(x, r, active, all_masks(m, x.device))
    Sx, Px, W = edge_tensors(m, x.dtype, x.device)
    return delta, active, m, Z, Sx, Px, W


def edgewise_interval(x, r, A, fA, L, sigma, p) -> dict:
    """Theorem-1 aggregate with l_p envelopes. d_p(z+, z-) = delta_i for every p (one coordinate differs)."""
    t0 = time.perf_counter()
    delta, active, m, Z, Sx, Px, W = _setup(x, r, sigma)
    ell, h = envelopes_p(Z, A, fA, L, sigma, p)
    Ld = (L * delta[active])[:, None]
    b = torch.minimum(h[Px] - ell[Sx], Ld)
    a = torch.maximum(ell[Px] - h[Sx], -Ld)
    lo, hi = (W * a).sum(1), (W * b).sum(1)
    return dict(active=active, delta=delta, lo=lo.cpu().numpy(), hi=hi.cpu().numpy(),
                seconds=time.perf_counter() - t0)


def lp_interval(x, r, A, fA, L, sigma, p, *, tol_feas: float = 1e-9, verify: bool = True) -> dict:
    """Exact marginal interval of every active feature over F_L(A) under the weighted l_p metric."""
    t0 = time.perf_counter()
    delta, active, m, Z, Sx, Px, W = _setup(x, r, sigma)
    N = Z.shape[0]
    ell, h = envelopes_p(Z, A, fA, L, sigma, p)
    ell_np, h_np = ell.cpu().numpy(), h.cpu().numpy()
    if np.any(ell_np > h_np + tol_feas):
        raise ValueError("empty class: L < L_A (envelopes cross)")
    hi_b = np.maximum(h_np, ell_np)
    DZ = wdist(Z, Z, sigma, p).cpu().numpy()
    iu, ju = np.triu_indices(N, 1)
    npair = iu.size
    rows = np.repeat(np.arange(2 * npair), 2)
    cols = np.empty(4 * npair, dtype=np.int64)
    vals = np.empty(4 * npair)
    cols[0::4], vals[0::4] = iu, 1.0
    cols[1::4], vals[1::4] = ju, -1.0
    cols[2::4], vals[2::4] = ju, 1.0
    cols[3::4], vals[3::4] = iu, -1.0
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(2 * npair, N))
    b_ub = np.repeat(L * DZ[iu, ju], 2)
    bounds = list(zip(ell_np.tolist(), hi_b.tolist()))
    Sx_np, Px_np, W_np = Sx.cpu().numpy(), Px.cpu().numpy(), W.cpu().numpy()
    lo, hi, status, worst = np.zeros(m), np.zeros(m), [], 0.0
    for jj in range(m):
        c = np.zeros(N)
        np.add.at(c, Px_np[jj], W_np[jj])
        np.add.at(c, Sx_np[jj], -W_np[jj])
        out = []
        for sgn in (1.0, -1.0):          # min c.t  then  max c.t
            res = linprog(sgn * c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            status.append(res.status)
            if res.status != 0:
                raise RuntimeError(f"HiGHS failed: {res.message}")
            t = res.x
            if verify:
                lip = np.max(np.abs(t[iu] - t[ju]) - L * DZ[iu, ju]) if npair else 0.0
                box = max(float(np.max(ell_np - t)), float(np.max(t - hi_b)))
                worst = max(worst, lip, box)
            out.append(float(c @ t))
        lo[jj], hi[jj] = out[0], out[1]
    return dict(active=active, delta=delta, lo=lo, hi=hi, n_vertices=N, n_constraints=2 * npair,
                max_constraint_violation=worst, seconds=time.perf_counter() - t0)


def euclidean_counterexample() -> dict:
    """Appendix B: d = 2, x = (1,1), r = (0,0), A = {x, r} with f = 0, unit scales, L = 1."""
    x = torch.tensor([1.0, 1.0])
    r = torch.tensor([0.0, 0.0])
    A = torch.stack([x, r])
    fA = torch.zeros(2)
    s = torch.ones(2)
    out = {}
    for p in (1, 2):
        e = edgewise_interval(x, r, A, fA, 1.0, s, p)
        lpv = lp_interval(x, r, A, fA, 1.0, s, p)
        out[f"l{p}"] = dict(edgewise_upper_phi1=float(e["hi"][0]), lp_upper_phi1=float(lpv["hi"][0]),
                            edgewise_lower_phi1=float(e["lo"][0]), lp_lower_phi1=float(lpv["lo"][0]))
    return out
