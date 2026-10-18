"""
hypercube_lp.py — exact class-conditional extrema of exact Shapley values (Theorems 1 + 2).

Under weighted l1 the 2^m hybrid vertices z_S (S ⊆ active features) are the corners of a box,
and their l1 distances equal shortest-path lengths on the hypercube graph with edge weights
delta_j. Hence
    L-Lipschitz on all vertex pairs  <=>  L-Lipschitz on the m 2^{m-1} hypercube edges,
and {anchor agreement + Lipschitz to every anchor}  <=>  box ell(z_S) <= t_S <= h(z_S).
Every feasible t extends to a member of F_L(A) (McShane), so LP optima are exact.

Solver: scipy HiGHS (torch has no LP solver). Torch is used for geometry and verification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

from bounds import all_masks, shapley_from_vertex_values, shapley_weights, vertex_points
from envelope import envelopes, lipschitz_violation, weighted_l1


class Infeasible(RuntimeError):
    """F_L(A) is empty for this anchor set (L < L_A)."""


@dataclass
class PairCert:
    i: int
    j: int
    certified: bool   # min_g (|phi_i| - |phi_j|) > tol
    margin: float     # see `kind`
    kind: str         # 'exact' | 'lower' (pre-screen proved certification) | 'upper' (pre-screen refuted it)
    n_lp: int


class HypercubeLP:
    def __init__(self, x: torch.Tensor, r: torch.Tensor, A: torch.Tensor, fA: torch.Tensor,
                 L: float, sigma: torch.Tensor, *, active_tol: float = 0.0,
                 feas_tol: float = 1e-9, max_players: int = 20):
        """A must already contain x (and r for endpoint-fixed mode)."""
        self.x, self.r, self.A, self.fA, self.L, self.sigma = x, r, A, fA, float(L), sigma
        self.d = x.numel()
        self.delta = (x - r).abs() / sigma
        self.active = torch.nonzero(self.delta > active_tol).flatten().tolist()
        self._pos = {j: jj for jj, j in enumerate(self.active)}
        self.m = m = len(self.active)
        if m > max_players:
            raise ValueError(f"{m} active features > max_players={max_players}")
        self.n_lp = 0
        self.masks = all_masks(m, x.device)
        self.Z = vertex_points(x, r, self.active, self.masks)
        ell, h = envelopes(self.Z, A, fA, self.L, sigma)
        gap = float((ell - h).max())
        if gap > feas_tol * (1.0 + float(fA.abs().max())):
            raise Infeasible(f"empty class: max(ell - h) = {gap:.3e}; need L >= L_A")
        self.ell, self.h = ell, torch.maximum(h, ell)
        self._build()
        self._interval = None
        self._solutions: dict = {}

    # ------------------------------------------------------------------ setup
    def _build(self):
        m, n = self.m, 2 ** self.m
        ell, h = self.ell.cpu().numpy(), self.h.cpu().numpy()
        self._bounds = list(zip(ell.tolist(), h.tolist()))
        if m == 0:
            self.A_ub, self.b_ub, self.C = None, None, np.zeros((0, 1))
            return
        idx = np.arange(n)
        delta = self.delta.cpu().numpy()
        rows, cols, vals, rhs = [], [], [], []
        row = 0
        for jj, j in enumerate(self.active):
            bit = 1 << jj
            S = idx[(idx & bit) == 0]
            P = S | bit
            k = S.size
            Ld = self.L * delta[j]
            r1 = np.arange(row, row + k)
            r2 = r1 + k
            rows += [r1, r1, r2, r2]
            cols += [P, S, S, P]
            vals += [np.ones(k), -np.ones(k), np.ones(k), -np.ones(k)]
            rhs += [np.full(k, Ld), np.full(k, Ld)]   # t_P - t_S <= Ld ; t_S - t_P <= Ld
            row += 2 * k
        self.A_ub = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(row, n))
        self.b_ub = np.concatenate(rhs)
        w = shapley_weights(m).numpy()
        pop = self.masks.sum(1).cpu().numpy()
        C = np.zeros((m, n))
        for jj in range(m):
            bit = 1 << jj
            S = idx[(idx & bit) == 0]
            C[jj, S | bit] += w[pop[S]]
            C[jj, S] -= w[pop[S]]
        self.C = C   # phi_j(t) = C[pos(j)] @ t  (exact Shapley of the reduced game)

    def _lp(self, obj: np.ndarray, *, maximize: bool = False, extra=None):
        """Optimise obj @ t over the feasible set; extra = (row, rhs) adds row @ t <= rhs.
        Returns (value, argopt) or (None, None) if infeasible."""
        A_ub, b_ub = self.A_ub, self.b_ub
        if extra is not None:
            A_ub = sp.vstack([A_ub, sp.csr_matrix(extra[0][None, :])], format="csr")
            b_ub = np.append(b_ub, extra[1])
        res = linprog(-obj if maximize else obj, A_ub=A_ub, b_ub=b_ub,
                      bounds=self._bounds, method="highs")
        self.n_lp += 1
        if res.status == 2:
            return None, None
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        return (float(-res.fun) if maximize else float(res.fun)), res.x

    # ------------------------------------------------------------ intervals
    def interval(self) -> tuple[np.ndarray, np.ndarray]:
        """[phi_i^min, phi_i^max] over F_L(A) for every feature (0 for inactive ones)."""
        if self._interval is None:
            lo, hi = np.zeros(self.d), np.zeros(self.d)
            for jj, j in enumerate(self.active):
                v, t = self._lp(self.C[jj])
                if v is None:
                    raise Infeasible("interval LP infeasible")
                lo[j] = v
                self._solutions[(j, "min")] = t
                v, t = self._lp(self.C[jj], maximize=True)
                hi[j] = v
                self._solutions[(j, "max")] = t
            self._interval = (lo, hi)
        return self._interval

    def width(self) -> np.ndarray:
        lo, hi = self.interval()
        return hi - lo

    def signs(self, tol: float = 1e-7) -> np.ndarray:
        """+1 / -1 if the sign is certified over the class, 0 otherwise (inactive -> 0, 'unchanged')."""
        lo, hi = self.interval()
        s = np.zeros(self.d, dtype=int)
        s[lo > tol] = 1
        s[hi < -tol] = -1
        return s

    def shapley(self, vertex_vals: torch.Tensor) -> np.ndarray:
        """Exact Shapley values (full d) of a function given by its values on self.Z."""
        out = np.zeros(self.d)
        if self.m:
            out[self.active] = shapley_from_vertex_values(vertex_vals, self.m).cpu().numpy()
        return out

    # ------------------------------------------------- |phi| ranking certificates
    def abs_margin(self, i: int, j: int, tol: float = 1e-7) -> PairCert:
        """min over F_L(A) of |phi_i| - |phi_j|, exactly (<= 4 LPs), with interval pre-screen.
        Uses  -|a| = min(-a, a)  and a split on the sign region of phi_i."""
        lo, hi = self.interval()

        def min_abs(k):
            return 0.0 if lo[k] <= 0.0 <= hi[k] else min(abs(lo[k]), abs(hi[k]))

        def max_abs(k):
            return max(abs(lo[k]), abs(hi[k]))

        if i not in self._pos:   # phi_i ≡ 0
            mg = -max_abs(j)
            return PairCert(i, j, mg > tol, mg, "exact", 0)
        if j not in self._pos:   # phi_j ≡ 0
            mg = min_abs(i)
            return PairCert(i, j, mg > tol, mg, "exact", 0)
        lb = min_abs(i) - max_abs(j)
        if lb > tol:
            return PairCert(i, j, True, lb, "lower", 0)
        ub = max_abs(i) - min_abs(j)
        if ub <= tol:
            return PairCert(i, j, False, ub, "upper", 0)

        ci, cj = self.C[self._pos[i]], self.C[self._pos[j]]
        sg = self.signs(tol)
        s_set = (int(sg[i]),) if sg[i] else (1, -1)
        t_set = (int(sg[j]),) if sg[j] else (1, -1)
        best, n0 = np.inf, self.n_lp
        for s in s_set:
            extra = None if sg[i] else (-s * ci, 0.0)   # region s*phi_i >= 0
            for t in t_set:
                v, _ = self._lp(s * ci - t * cj, extra=extra)
                if v is not None:
                    best = min(best, v)
        if not np.isfinite(best):
            raise RuntimeError("all sign branches infeasible (should be impossible)")
        return PairCert(i, j, best > tol, float(best), "exact", self.n_lp - n0)

    def candidate_topk(self, k: int, phi: np.ndarray | None = None) -> list[int]:
        """Candidate top-k by |phi| (default: interval midpoints)."""
        if phi is None:
            lo, hi = self.interval()
            phi = 0.5 * (lo + hi)
        return [int(v) for v in np.argsort(-np.abs(phi), kind="stable")[:k]]

    def certify_topk_abs(self, T, tol: float = 1e-7, stop_early: bool = False) -> dict:
        """T is the top-k set (by |phi|) of EVERY g in F_L(A)  <=>  all k(d-k) pairs certified."""
        T = list(T)
        outside = [j for j in range(self.d) if j not in T]
        fails, pairs, n0 = [], [], self.n_lp
        for i in T:
            for j in outside:
                pc = self.abs_margin(i, j, tol)
                pairs.append(pc)
                if not pc.certified:
                    fails.append((i, j))
                    if stop_early:
                        return dict(certified=False, fails=fails, pairs=pairs, n_lp=self.n_lp - n0)
        return dict(certified=not fails, fails=fails, pairs=pairs, n_lp=self.n_lp - n0)

    def certify_order_abs(self, order, tol: float = 1e-7) -> dict:
        """Ordered statement |phi_{o1}| > |phi_{o2}| > ... for every g  <=>  all adjacent pairs certified."""
        order = list(order)
        pairs = [self.abs_margin(order[t], order[t + 1], tol) for t in range(len(order) - 1)]
        return dict(certified=all(p.certified for p in pairs), pairs=pairs)

    # ------------------------------------------------------------ verification
    def verify_solutions(self, all_pairs: bool = True) -> float:
        """Independent check of every stored LP optimum (Theorem 2 sanity check):
        Lipschitz to every anchor directly (not via the box) and, optionally, on ALL vertex
        pairs (not only hypercube edges). Returns the max violation (<= tol expected)."""
        if not self._solutions:
            return 0.0
        D = weighted_l1(self.Z, self.A, self.sigma)
        worst = 0.0
        for t in self._solutions.values():
            tt = torch.as_tensor(t, dtype=self.Z.dtype, device=self.Z.device)
            worst = max(worst, float(((tt[:, None] - self.fA[None, :]).abs() - self.L * D).max()))
            if all_pairs:
                worst = max(worst, lipschitz_violation(self.Z, tt, self.L, self.sigma))
        return worst
