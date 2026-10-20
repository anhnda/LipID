"""
lipid/joint.py — exact joint identified set of the attribution vector (LipID v1.2, Revision A1; reviewer Q3).

Proposition J (paper/v12_theory.tex). Let m = number of active players, z_S (S in 2^[m]) the hybrid vertices and
    P(A, L) = { t in R^{2^m} : ell_L(z_S) <= t_S <= h_L(z_S)  for all S,
                               |t_{S+j} - t_S| <= L delta_j    for every hypercube edge (S, S+j) }.
Under the weighted-l1 metric, d(z_S, z_T) = sum_{j in S xor T} delta_j is a shortest-path length on the hypercube, so
the edge constraints imply all vertex-pair constraints; the box is equivalent to Lipschitz consistency with every anchor.
Hence every t in P(A, L), together with the anchor labels, is an L-Lipschitz labeling of a finite set and has a McShane
extension in F_L(A); conversely every member restricts to a point of P(A, L). Therefore
    I_joint(A, L) = { phi(g) : g in F_L(A) } = { M t : t in P(A, L) },     M = Shapley matrix (m x 2^m),
a linear image of a polytope: an exact LP formulation, NOT a polynomial-time algorithm in m.

`JointPolytope(constraints="edges")` implements P(A, L); `constraints="all_pairs"` replaces the edge constraints by all
vertex pairs (the generic finite-metric formulation, used as an independent ground truth that does not rely on the
shortest-path argument). Every LP solution is re-verified in raw units, directly against the anchors and all vertex
pairs (`violation`), independently of how the LP was built.

LP numerics: values are divided by s = L * sum_i delta_i before solving (O(1) scale); HiGHS feasibility tolerances
are tightened to 1e-10 (scaled units). The objective is recomputed in raw units from the returned vertex values.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

import lipid  # noqa: F401
from bounds import all_masks, shapley_weights, vertex_points
from envelope import envelopes, weighted_l1
from lipid.pairwise import pair_index_np

LP_TOL = 1e-10


class Infeasible(RuntimeError):
    """F_L(A) is empty (L < L_A): the envelopes cross."""


def shapley_matrix(m: int) -> np.ndarray:
    """M (m x 2^m): phi_jj(t) = M[jj] @ t for the m-player game with vertex values t (bit jj <-> player jj)."""
    N = 1 << m
    idx = np.arange(N)
    pop = np.array([bin(int(v)).count("1") for v in idx])
    w = shapley_weights(m).cpu().numpy() if m > 0 else np.zeros(0)
    M = np.zeros((m, N))
    for jj in range(m):
        bit = 1 << jj
        S = idx[(idx & bit) == 0]
        np.add.at(M[jj], S | bit, w[pop[S]])
        np.add.at(M[jj], S, -w[pop[S]])
    return M


class JointPolytope:
    def __init__(self, x: torch.Tensor, r: torch.Tensor, A: torch.Tensor, fA: torch.Tensor, L: float,
                 sigma: torch.Tensor, *, constraints: str = "edges", max_players: int = 12, feas_tol: float = 1e-9,
                 lp_tol: float = LP_TOL, method: str = "highs"):
        """A must already contain x (domain) or x and r (fixed)."""
        if constraints not in ("edges", "all_pairs"):
            raise ValueError(constraints)
        self.x, self.r, self.A, self.fA, self.L, self.sigma = x, r, A, fA, float(L), sigma
        self.constraints, self.lp_tol, self.method = constraints, lp_tol, method
        delta = (x - r).abs() / sigma
        self.delta = delta
        self.active = torch.nonzero(delta > 0).flatten().tolist()
        m = self.m = len(self.active)
        if m == 0:
            raise ValueError("no active feature")
        if m > max_players:
            raise ValueError(f"{m} active features > max_players={max_players}")
        self.N = N = 1 << m
        self.Z = vertex_points(x, r, self.active, all_masks(m, x.device))
        ell, h = envelopes(self.Z, A, fA, self.L, sigma)
        ell_np, h_np = ell.detach().cpu().numpy().astype(np.float64), h.detach().cpu().numpy().astype(np.float64)
        gap = float(np.max(ell_np - h_np))
        if gap > feas_tol * (1.0 + float(fA.abs().max())):
            raise Infeasible(f"empty class: max(ell - h) = {gap:.3e}; need L >= L_A")
        self.ell, self.h = ell_np, np.maximum(h_np, ell_np)
        dl = delta[self.active].detach().cpu().numpy().astype(np.float64)
        self.dl = dl
        s = float(self.L * dl.sum())
        self.s = s if s > 0 else 1.0
        self._bounds = list(zip((self.ell / self.s).tolist(), (self.h / self.s).tolist()))
        if constraints == "edges":
            idx = np.arange(N)
            rows, cols, vals, rhs = [], [], [], []
            row = 0
            for jj in range(m):
                bit = 1 << jj
                S = idx[(idx & bit) == 0]
                P = S | bit
                k = S.size
                r1 = np.arange(row, row + k)
                r2 = r1 + k
                rows += [r1, r1, r2, r2]
                cols += [P, S, S, P]
                vals += [np.ones(k), -np.ones(k), np.ones(k), -np.ones(k)]
                rhs += [np.full(2 * k, self.L * dl[jj] / self.s)]
                row += 2 * k
            self.A_ub = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                      shape=(row, N))
            self.b_ub = np.concatenate(rhs)
        else:
            DZ = weighted_l1(self.Z, self.Z, sigma).detach().cpu().numpy().astype(np.float64)
            iu, ju = np.triu_indices(N, 1)
            npair = iu.size
            rws = np.repeat(np.arange(2 * npair), 2)
            cls = np.empty(4 * npair, dtype=np.int64)
            vls = np.empty(4 * npair)
            cls[0::4], vls[0::4] = iu, 1.0
            cls[1::4], vls[1::4] = ju, -1.0
            cls[2::4], vls[2::4] = ju, 1.0
            cls[3::4], vls[3::4] = iu, -1.0
            self.A_ub = sp.csr_matrix((vls, (rws, cls)), shape=(2 * npair, N))
            self.b_ub = np.repeat(self.L * DZ[iu, ju] / self.s, 2)
        self.M = shapley_matrix(m)
        self.n_lp = 0
        self.lp_seconds = 0.0

    # ------------------------------------------------------------------------------------------ LP core
    def solve(self, obj: np.ndarray, maximize: bool = False):
        """Optimise obj @ t over P(A, L). Returns (value in raw units, t in raw units)."""
        t0 = time.perf_counter()
        res = linprog(-obj if maximize else obj, A_ub=self.A_ub, b_ub=self.b_ub, bounds=self._bounds,
                      method=self.method,
                      options=dict(primal_feasibility_tolerance=self.lp_tol, dual_feasibility_tolerance=self.lp_tol))
        self.lp_seconds += time.perf_counter() - t0
        self.n_lp += 1
        if res.status != 0:
            raise RuntimeError(f"HiGHS failed (status {res.status}): {res.message}")
        t = res.x * self.s
        return float(obj @ t), t

    def interval(self, obj: np.ndarray):
        """Sharp [min, max] of the linear functional obj @ t (obj over vertices). Returns (lo, hi, t_lo, t_hi)."""
        lo, t_lo = self.solve(obj, maximize=False)
        hi, t_hi = self.solve(obj, maximize=True)
        return lo, hi, t_lo, t_hi

    # ------------------------------------------------------------------------------------- contrasts
    def contrast(self, coef) -> np.ndarray:
        """Vertex objective of sum_jj coef[jj] phi_jj (coef over ACTIVE positions)."""
        return np.asarray(coef, dtype=np.float64) @ self.M

    def marginal(self):
        """Sharp marginal intervals of every active player (must equal Theorem 1)."""
        lo, hi = np.zeros(self.m), np.zeros(self.m)
        for jj in range(self.m):
            lo[jj], hi[jj], _, _ = self.interval(self.M[jj])
        return lo, hi

    def pair_diff(self, a: int, b: int):
        """Sharp interval of phi_a - phi_b (a, b active positions)."""
        return self.interval(self.M[a] - self.M[b])

    def pair_sum(self, a: int, b: int):
        """Sharp interval of phi_a + phi_b (no closed form in general: mixed-sign weights on four faces)."""
        return self.interval(self.M[a] + self.M[b])

    def pair_diff_obj_closed(self, a: int, b: int) -> np.ndarray:
        """The pairwise-identity objective sum_S beta_S (e_{S+a} - e_{S+b}); must equal M[a] - M[b]."""
        pairs, P, Q, B, _, _ = pair_index_np(self.m)
        p = pairs.index((min(a, b), max(a, b)))
        o = np.zeros(self.N)
        np.add.at(o, P[p], B[p])
        np.add.at(o, Q[p], -B[p])
        return o if a < b else -o

    # ------------------------------------------------------------------------------------ verification
    def violation(self, t: np.ndarray) -> dict:
        """Raw-unit check of a vertex labeling t, independent of the LP construction:
        box (envelopes), direct anchor consistency |t_S - f(a)| <= L d(z_S, a), and ALL vertex pairs."""
        tt = torch.as_tensor(t, dtype=self.Z.dtype, device=self.Z.device)
        box = float(max(np.max(self.ell - t), np.max(t - self.h)))
        D = weighted_l1(self.Z, self.A, self.sigma)
        anc = float(((tt[:, None] - self.fA[None, :]).abs() - self.L * D).max())
        DZ = weighted_l1(self.Z, self.Z, self.sigma)
        pair = float(((tt[:, None] - tt[None, :]).abs() - self.L * DZ).max())
        return dict(box=max(box, 0.0), anchors=max(anc, 0.0), pairs=max(pair, 0.0))


# ============================================================================== finite labelings and McShane
def labeling_violation(points: torch.Tensor, vals: torch.Tensor, A: torch.Tensor, fA: torch.Tensor, L: float,
                       sigma: torch.Tensor) -> dict:
    """Is (points -> vals) together with (A -> fA) an L-Lipschitz labeling? Max violations in raw units."""
    D = weighted_l1(points, points, sigma)
    pp = float(((vals[:, None] - vals[None, :]).abs() - L * D).max()) if points.shape[0] else 0.0
    DA = weighted_l1(points, A, sigma)
    pa = float(((vals[:, None] - fA[None, :]).abs() - L * DA).max()) if points.shape[0] else 0.0
    DAA = weighted_l1(A, A, sigma)
    aa = float(((fA[:, None] - fA[None, :]).abs() - L * DAA).max())
    return dict(points=max(pp, 0.0), points_anchors=max(pa, 0.0), anchors=max(aa, 0.0))


def mcshane(points: torch.Tensor, vals: torch.Tensor, L: float, sigma: torch.Tensor):
    """Upper McShane extension G(z) = min_k [vals_k + L d(z, p_k)] of a Lipschitz labeling (an L-Lipschitz function
    that agrees with the labeling wherever the labeling is L-Lipschitz)."""
    def G(Z: torch.Tensor) -> torch.Tensor:
        return (vals[None, :] + L * weighted_l1(Z, points, sigma)).min(dim=1).values
    return G


__all__ = ["Infeasible", "JointPolytope", "shapley_matrix", "labeling_violation", "mcshane", "LP_TOL"]
