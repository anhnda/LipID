"""
phase1.py — Phase 1: mathematical validation of Attribution Identifiability Width (AIW).

    python -m pytest -q tests.py                      # unit tests first
    python phase1.py --out phase1_quick --quick       # pilot run (seed 0)
    python phase1.py --out phase1_out \
        --sci-rho-target 0.8 --sci-min-frac 0.25 --sci-min-sign-rate 0.10   # thresholds: YOUR choice,
                                                      # fixed BEFORE the run (written to config.json)
    python phase1.py --out phase1_out --device cpu    # default 'auto' = cuda if available

Outputs in --out:
    config.json      the exact configuration (pre-registration record, incl. scientific thresholds)
    records.jsonl    one line per (d, rho, n_anchor, model, kappa, x, r, anchor mode)
    optimistic.jsonl (amendment v3) one line per (d, rho, n_anchor, model, x, r, anchor mode), NOT filtered
                     by feasibility: rho and certificates at L_min = L_A(A), L_V = Lip(f on hybrids ∪ A),
                     L_true (when known), geometric limit rho_inf, deterministic theorem checks
    stress.json      saturated anchors, sparse references, 2-D counterexample (+ fa at L_V),
                     u+v counterexample (rho = 1 - 1/L), stable-OOD
    mc_coverage.json coverage on non-saturated references + negative control (shrunken radius)
    runtime.json     cost of pairwise bound vs hypercube LP vs Monte Carlo
    summary.json     aggregated metrics, technical GO (PHASE1_GO) and the separate scientific verdict

Record fields: LA_C = L_A(C) (calibration anchors only, sets L = kappa * LA_C);
LA_used = L_A(A) with A = C ∪ {x} (domain) or C ∪ {x, r} (fixed). infeasible <=> LA_used > L.

Data splits are disjoint draws: train (sigma only) | C_anchor | R_candidate | D_eval.
sigma = 1.4826 * MAD on the train split, fixed before anything else.
"OOD" here means low-density mismatch: a non-degenerate Gaussian has full support.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from scipy.stats import binomtest, spearmanr

from bounds import (effective_sparsity, geometric_rho_inf, hybrid_vertices, mc_pair_bounds,
                    pair_bounds_exact, shapley_full)
from envelope import (Member, RandomLipschitz, anchor_lipschitz, clamp_to_envelope,
                      envelope_violation, lipschitz_after_adding, lipschitz_violation)
from hypercube_lp import HypercubeLP, Infeasible

torch.set_default_dtype(torch.float64)


def _json_default(o):
    """Make numpy / torch scalars and arrays JSON-serializable (np.bool_, np.float64, tensors, ...)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# =============================================================================== config
PILOT_SEEDS = (0,)   # seeds already used by the pilot (--quick) run; never reused for the confirmatory grid

AMENDMENT = {
    "type": "post_pilot_pre_confirmatory_grid",
    "reason": ("For d=2 with endpoint-fixed anchors and weighted L1 geometry, LP width equals the "
               "pairwise bound by structural equivalence of the constraints: for feature i every "
               "vertex lies in exactly one (z_S, z_{S+i}) pair, and every cross-direction edge "
               "touches an anchored endpoint, so it is already encoded by the envelopes."),
    "observed_before_amendment": "pilot GO flag only (lp_tightness=false); no tightness numbers",
    "primary_lp_tightness_min_d": 3,
    "median_width_ratio_threshold": 0.8,
    "alternative_sign_rate_gain_pp": 10,
    "report_by_dimension": True,
    "report_by_anchor_mode": True,
    "exclude_zero_denominator_from_ratios": True,
    "secondary_analysis": ("features with >= 1 cross-direction hypercube edge joining two "
                           "non-anchor hybrid vertices (necessary, not sufficient, for LP gain)"),
    "confirmatory_seeds_disjoint_from_pilot": True,
    "pilot_seeds": list(PILOT_SEEDS),
    "status": "superseded by v2 (its premise — LP gain via cross-direction free edges — is false)",
}

AMENDMENT_V2 = {
    "type": "post_pilot_pre_confirmatory_grid",
    "data_driven": True,
    "observed_before_amendment": ("pilot 2 numbers: Gamma_LP / B_pair = 1 (to 1e-16) in every instance, "
                                  "all d and modes, incl. features with cross-direction free edges; "
                                  "MC width / B_pair = 1; certified-sign rates 11-39%, top-k 1.6%"),
    "finding": ("Proposition (bounds.py): under weighted l1 the pairwise interval equals the exact class "
                "interval for every feature, any anchor set, any L >= L_A, any non-negative semivalue. "
                "Discovered from pilot data, then proved by explicit construction."),
    "lp_interval_tightness_criterion": "void_by_theorem (ratio == 1 identically); not counted as pass or fail",
    "theorem_check": "max |interval_LP - interval_pair| over all feasible instances <= tol (deterministic)",
    "new_lp_question": ("joint certificates: among ordered pairs (i above j by |phi_f|, both active), "
                        "rate certified by exact LP (<= 4 LPs) minus rate certified by interval separation alone"),
    "lp_retention_threshold_pp": 10,
    "phase1_go_definition": ("deterministic correctness AND MC coverage. The LP question decides whether the LP "
                             "stays in the method; it no longer gates Phase 1 (failing it = drop LP, not the framework)"),
    "anchor_informativeness_report": ("rho_i = B_pair_i / (2 L delta_i): quantiles and fraction < 0.9 by d, n_anchor, "
                                      "kappa, mode — report only, the key usefulness signal for Phase 2"),
    "confirmatory_seeds_disjoint_from_pilot": True,
    "pilot_seeds": list(PILOT_SEEDS),
}

AMENDMENT_V3 = {
    "type": "post_confirmatory_grid_pre_P0",
    "data_driven": True,
    "observed_before_amendment": ("confirmatory grid (seed 1): median rho at kappa=1 = 0.374 / 0.725 / 0.936 / 0.979 "
                                  "for d = 2..5; frozen-model violation 39% at kappa=1, 0.7% at kappa=4; MC coverage "
                                  "config saturated (median exact rho ~ 1), hence a vacuous coverage test"),
    "theory_added": [
        "for fixed (A, x, r): rho_i(L) is non-decreasing in L and certified intervals are nested",
        "rho_i(L) -> rho_inf_i (purely geometric) as L -> inf, and rho_i(L) <= rho_inf_i (upper bound ONLY)",
        "optimistic ceiling: for every valid L >= L_min = L_A(A): rho(L) >= rho(L_min) and the signs / top-k "
        "certified at L are a subset of those certified at L_min — no knowledge of L_true needed",
        "local validity: at L_V = Lip(f on hybrids ∪ A) the interval provably contains phi(f) "
        "(McShane extension of f restricted to hybrids ∪ A); L_min <= L_V <= L_true",
    ],
    "new_outputs": ("optimistic.jsonl: every instance at L_min, L_V and L_true (when known), no filtering by "
                    "feasibility under L = kappa * L_A(C); deterministic theorem checks gate technical GO"),
    "mc_changes": ("coverage run on references chosen BEFORE any sampling, by exact quantities only, as the least "
                   "saturated feasible candidates; vacuity (median exact rho >= mc_rho_max) blocks GO; negative "
                   "control: radius multiplied by mc_neg_scales, the mc_neg_required scale MUST be detected"),
    "go_changes": ("PHASE1_GO = technical GO = stress checks AND grid correctness AND optimistic theorem checks AND "
                   "MC coverage AND MC non-vacuous AND MC negative control detected; a skipped part is "
                   "'not_evaluated' and blocks GO. The scientific verdict (S1/S2/S3) is separate and needs "
                   "thresholds registered in config.json BEFORE the run"),
    "failure_reporting": "infeasible rate, conditional violation rate, total failure rate — reported separately",
    "confirmatory_seeds_disjoint_from_pilot": True,
    "pilot_seeds": list(PILOT_SEEDS),
}

STRESS_GATED = ("saturated", "sparse_reference", "counterexample_2d", "uv_counterexample")
TECH_GO_KEYS = ("stress_checks", "grid_correctness", "optimistic_theorem_checks",
                "mc_coverage", "mc_non_vacuous", "mc_negative_control_detected")


@dataclass
class Config:
    seed: int = 1                    # confirmatory default; the pilot used seed 0
    dims: tuple = (2, 3, 4, 5)
    rhos: tuple = (0.0, 0.6, 0.9, 0.99)
    n_anchor: tuple = (16, 64, 256)
    models: tuple = ("linear", "fa", "stable_ood", "unstable_id")
    kappas: tuple = (1.0, 1.25, 1.5, 2.0, 4.0)
    a: float = 4.0
    n_train: int = 2000
    n_candidate: int = 200
    n_eval_x: int = 5
    refs_per_x: int = 4
    n_members: int = 100
    n_member_lipcheck: int = 5
    top_k: int = 2
    eff_eps: float = 0.1
    tol: float = 1e-6
    # Monte Carlo coverage experiment (amendment v3: non-saturated references + negative control)
    mc_d: int = 4
    mc_rho: float = 0.0
    mc_n_anchor: int = 256
    mc_model: str = "linear"
    mc_kappa: float = 1.25
    mc_K: int = 4
    mc_runs: int = 200
    mc_n_per_stratum: int = 32
    mc_eta: float = 0.05
    mc_rho_max: float = 0.9              # coverage test counts only if median exact rho < this, per mode
    mc_neg_scales: tuple = (0.0, 0.25)   # negative control: confidence radius multiplied by these
    mc_neg_required: float = 0.0         # this scale MUST be detected (binomial test rejects at go_mc_pvalue)
    # amendment v3: optimistic ceiling (L_min), local validity (L_V), geometric limit (rho_inf)
    opt_topk_lp: bool = True             # top-k ceilings use the hypercube LP (<= 4 LPs per pair)
    # scientific verdict: thresholds MUST be fixed before the run (CLI flags -> config.json)
    sci_rho_target: "float | None" = None     # a feature-instance is informative if rho <= this
    sci_min_frac: "float | None" = None       # fraction of informative feature-instances required
    sci_min_sign_rate: "float | None" = None  # certified-sign rate at L_V required for S3
    device: str = "auto"                 # 'auto' -> cuda if available; LPs always run on CPU (scipy)
    # runtime benchmark
    rt_dims: tuple = (4, 6, 8, 10, 12)
    rt_n_anchor: int = 256
    rt_n_per_stratum: int = 64
    # pre-registered GO thresholds (Phase 1)
    go_lp_ratio: float = 0.8        # median Gamma_LP / B_pair <= this ...
    go_lp_sign_gain: float = 0.10   # ... or certified-sign rate gain (LP - pair) >= this
    go_mc_pvalue: float = 0.01      # binomial test of failure rate > eta must NOT reject at this level
    go_lp_min_d: int = 3            # amendment v1 (superseded): kept only for the legacy report
    go_lp_joint_gain: float = 0.10  # amendment v2: LP kept iff joint-certificate rate gain >= this
    rho_informative: float = 0.9    # report: fraction of features with rho_i < this
    run_kind: str = "confirmatory"
    amendment: dict = field(default_factory=lambda: {"v1": dict(AMENDMENT), "v2": dict(AMENDMENT_V2),
                                                     "v3": dict(AMENDMENT_V3)})

    @staticmethod
    def quick() -> "Config":
        return Config(seed=0, dims=(2, 3), rhos=(0.0, 0.9), n_anchor=(32,), models=("linear", "fa"),
                      kappas=(1.0, 2.0), n_eval_x=2, refs_per_x=2, n_members=20,
                      mc_runs=40, rt_dims=(4, 6), run_kind="pilot")


# ============================================================================== helpers
def seed_of(*key) -> int:
    return zlib.crc32(repr(key).encode()) & 0x7FFFFFFF


def gen_of(*key) -> torch.Generator:
    return torch.Generator().manual_seed(seed_of(*key))


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is not available")
    return name


def gaussian_sampler(d: int, rho: float, device="cpu"):
    """Draws on CPU with the CPU generator, then moves to `device`: identical numbers on every device."""
    cov = (1.0 - rho) * torch.eye(d) + rho * torch.ones(d, d)
    chol = torch.linalg.cholesky(cov)

    def sample(n: int, gen: torch.Generator) -> torch.Tensor:
        return (torch.randn(n, d, generator=gen) @ chol.T).to(device)
    return sample


def robust_sigma(X: torch.Tensor) -> torch.Tensor:
    med = X.median(dim=0).values
    mad = (X - med).abs().median(dim=0).values
    return (1.4826 * mad).clamp_min(1e-8)


def q(t: torch.Tensor) -> torch.Tensor:
    return t * t / (1.0 + t * t)


def make_model(name: str, a: float):
    """All models are globally Lipschitz. 'fa' is the bounded analogue of u + a u (u - v)^2:
    identical to z_1 on the diagonal, different off it, with asymmetric Shapley effect."""
    if name == "linear":
        def f(Z):
            return Z @ linear_weights(Z.shape[1], Z.dtype, Z.device)
    elif name == "fa":
        def f(Z):
            return Z[:, 0] + a * torch.tanh(Z[:, 0]) * q(Z[:, :1] - Z[:, 1:]).sum(1)
    elif name == "stable_ood":   # flat (saturated) but shifted away from the diagonal
        def f(Z):
            return Z[:, 0] + a * torch.tanh(((Z[:, :1] - Z[:, 1:]) ** 2).sum(1))
    elif name == "unstable_id":  # steep along the in-distribution (diagonal) direction
        def f(Z):
            return Z[:, 0] + a * torch.tanh(4.0 * Z.mean(1))
    else:
        raise ValueError(name)
    return f


def linear_weights(d: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    return torch.linspace(1.0, 0.2, d, dtype=dtype, device=device)


def lipschitz_true(name: str, d: int, sigma: torch.Tensor, a: float):
    """Exact GLOBAL Lipschitz constant w.r.t. weighted l1, when known in closed form:
    Lip = sup_z max_j |df/dz_j(z)| * sigma_j  (dual norm of weighted l1).
      linear      : max_j |w_j| sigma_j
      unstable_id : f = z_0 + a tanh(4 mean(z)); sup sech^2 = 1 at mean(z) = 0, so
                    max((1 + 4a/d) sigma_0, max_{j>=1} (4a/d) sigma_j)
    Other models -> None (use L_V, which needs no global bound)."""
    s = sigma.detach().cpu().to(torch.float64)
    if name == "linear":
        return float((linear_weights(d).abs() * s).max())
    if name == "unstable_id":
        g = torch.full((d,), 4.0 * a / d, dtype=torch.float64)
        g[0] += 1.0
        return float((g * s).max())
    return None


def _np(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)


def _f(v) -> float:
    return float(v)


# ======================================================== deterministic validation
def evaluate_instance(x, r, A, fAv, L, sigma, f, cfg: Config, gen: torch.Generator) -> dict:
    tol = cfg.tol
    t0 = time.perf_counter()
    lo_p, hi_p, D = (_np(v) for v in pair_bounds_exact(x, r, A, fAv, L, sigma))
    t_pair = time.perf_counter() - t0
    t0 = time.perf_counter()
    lp = HypercubeLP(x, r, A, fAv, L, sigma)
    lo_l, hi_l = lp.interval()
    t_lp = time.perf_counter() - t0
    d = lp.d
    delta = _np(lp.delta)
    twoLd = 2.0 * L * delta
    B, G = hi_p - lo_p, hi_l - lo_l

    # Gamma_LP <= B_pair <= 2 L delta,  and  [lo_lp, hi_lp] ⊆ [lo_pair, hi_pair]
    viol_chain = max(0.0, float(np.max(G - B, initial=0.0)), float(np.max(B - twoLd, initial=0.0)))
    viol_contain = max(0.0, float(np.max(lo_p - lo_l, initial=0.0)), float(np.max(hi_l - hi_p, initial=0.0)))

    # frozen model: empirical transfer (NOT a theorem check — f need not be in the class)
    fZ = f(lp.Z)
    phi_f = lp.shapley(fZ)
    VL = float(envelope_violation(fZ, lp.ell, lp.h).max())
    transfer_out = max(0.0, float(np.max(lo_l - phi_f)), float(np.max(phi_f - hi_l)))

    sg_lp = lp.signs(tol)
    sg_pair = np.zeros(d, dtype=int)
    sg_pair[lo_p > tol] = 1
    sg_pair[hi_p < -tol] = -1

    topk = None
    k = min(cfg.top_k, lp.m)
    if 1 <= k < d:
        T = lp.candidate_topk(k, phi=phi_f)
        res = lp.certify_topk_abs(T, tol)
        topk = dict(T=T, certified=bool(res["certified"]), n_lp=int(res["n_lp"]),
                    n_fail=len(res["fails"]))

    # Proposition check: LP interval == pairwise interval (both directions, not only containment)
    exact_gap = max(float(np.max(np.abs(lo_l - lo_p), initial=0.0)),
                    float(np.max(np.abs(hi_l - hi_p), initial=0.0)))

    # joint certificates (amendment v2): ordered pairs i above j by |phi_f|, both active
    act = set(lp.active)
    order = [int(v) for v in np.argsort(-np.abs(phi_f), kind="stable") if int(v) in act]

    def _min_abs(k):
        return 0.0 if lo_l[k] <= 0.0 <= hi_l[k] else min(abs(lo_l[k]), abs(hi_l[k]))

    def _max_abs(k):
        return max(abs(lo_l[k]), abs(hi_l[k]))

    n_pairs = n_cert_int = n_cert_lp_only = 0
    joint_lb_viol = 0.0
    cert_pairs = []
    n_lp0 = lp.n_lp
    for a_ in range(len(order)):
        for b_ in range(a_ + 1, len(order)):
            i, j = order[a_], order[b_]
            pc = lp.abs_margin(i, j, tol)
            n_pairs += 1
            if pc.kind == "lower":
                n_cert_int += 1
                cert_pairs.append((i, j))
            elif pc.kind == "exact":
                # exact LP margin can never be below the interval-separation lower bound
                joint_lb_viol = max(joint_lb_viol, (_min_abs(i) - _max_abs(j)) - pc.margin)
                if pc.certified:
                    n_cert_lp_only += 1
                    cert_pairs.append((i, j))
    joint = dict(n_pairs=n_pairs, cert_interval=n_cert_int, cert_lp_only=n_cert_lp_only,
                 n_lp_calls=lp.n_lp - n_lp0)

    # members of F_L(A): ell, h, then clamped random Lipschitz functions
    center = 0.5 * (x + r)
    scale = 0.5 + 0.5 * float(lp.delta.mean())
    f_mu, f_sd = float(fAv.mean()), float(fAv.std())
    mv_lp = mv_pair = lip_viol = 0.0
    sign_contra = topk_contra = joint_contra = 0
    for mi in range(cfg.n_members):
        if mi == 0:
            vals = lp.ell
        elif mi == 1:
            vals = lp.h
        else:
            off = f_mu + f_sd * float(torch.randn(1, generator=gen))
            psi = RandomLipschitz(center, scale, L, sigma, gen, offset=off)
            if mi < 2 + cfg.n_member_lipcheck:   # independent envelope recomputation + direct check
                vals = Member(A, fAv, L, sigma, psi)(lp.Z)
                P = torch.cat([lp.Z, A])
                lip_viol = max(lip_viol, lipschitz_violation(P, torch.cat([vals, fAv]), L, sigma))
            else:
                vals = clamp_to_envelope(psi(lp.Z), lp.ell, lp.h)
        phi_g = lp.shapley(vals)
        mv_lp = max(mv_lp, float(np.max(lo_l - phi_g)), float(np.max(phi_g - hi_l)))
        mv_pair = max(mv_pair, float(np.max(lo_p - phi_g)), float(np.max(phi_g - hi_p)))
        if np.any((sg_lp == 1) & (phi_g < -tol)) or np.any((sg_lp == -1) & (phi_g > tol)):
            sign_contra += 1
        if topk and topk["certified"]:
            ab = np.abs(phi_g)
            Ts = topk["T"]
            outside = [j for j in range(d) if j not in Ts]
            if outside and ab[Ts].min() < ab[outside].max() - tol:
                topk_contra += 1
        if cert_pairs:
            ab = np.abs(phi_g)
            if any(ab[i] < ab[j] - tol for i, j in cert_pairs):
                joint_contra += 1

    sol_viol = lp.verify_solutions(all_pairs=lp.m <= 12)

    return dict(
        m=lp.m, n_lp=lp.n_lp, t_pair=t_pair, t_lp=t_lp,
        cross_free_edges=lp.cross_free_edges().tolist(),
        delta=delta.tolist(), twoLd=twoLd.tolist(), B_pair=B.tolist(), G_lp=G.tolist(), D=D.tolist(),
        lo_pair=lo_p.tolist(), hi_pair=hi_p.tolist(), lo_lp=lo_l.tolist(), hi_lp=hi_l.tolist(),
        phi_f=phi_f.tolist(), sign_lp=sg_lp.tolist(), sign_pair=sg_pair.tolist(), topk=topk,
        viol_chain=viol_chain, viol_contain=viol_contain, member_viol_lp=mv_lp,
        member_viol_pair=mv_pair, lip_viol=lip_viol, sol_viol=sol_viol,
        sign_contra=sign_contra, topk_contra=topk_contra, joint_contra=joint_contra,
        exact_gap=exact_gap, joint_lb_viol=max(0.0, joint_lb_viol), joint=joint,
        VL=VL, transfer_out=transfer_out,
    )


def run_deterministic(cfg: Config, out_dir: str) -> str:
    path = os.path.join(out_dir, "records.jsonl")
    with open(path, "w") as fh:
        for d, rho, nA in itertools.product(cfg.dims, cfg.rhos, cfg.n_anchor):
            gd = gen_of("data", cfg.seed, d, rho, nA)
            sample = gaussian_sampler(d, rho, cfg.device)
            sigma = robust_sigma(sample(cfg.n_train, gd))
            C, Rc, Xev = sample(nA, gd), sample(cfg.n_candidate, gd), sample(cfg.n_eval_x, gd)
            plan = [(xi, torch.randperm(cfg.n_candidate, generator=gd)[:cfg.refs_per_x].tolist())
                    for xi in range(cfg.n_eval_x)]
            for model in cfg.models:
                f = make_model(model, cfg.a)
                fC, fX, fR = f(C), f(Xev), f(Rc)
                LA_C = anchor_lipschitz(C, fC, sigma)
                for kappa in cfg.kappas:
                    L = kappa * LA_C
                    t_cfg, n_inf, n_tot = time.perf_counter(), 0, 0
                    for xi, ridx in plan:
                        x = Xev[xi]
                        A_dom = torch.cat([C, x[None]])
                        f_dom = torch.cat([fC, fX[xi:xi + 1]])
                        LA_dom = lipschitz_after_adding(LA_C, C, fC, x[None], fX[xi:xi + 1], sigma)
                        for rj in ridx:
                            r = Rc[rj]
                            for mode in ("domain", "fixed"):
                                if mode == "domain":
                                    A, fAv, LA = A_dom, f_dom, LA_dom
                                else:
                                    A = torch.cat([A_dom, r[None]])
                                    fAv = torch.cat([f_dom, fR[rj:rj + 1]])
                                    LA = lipschitz_after_adding(LA_dom, A_dom, f_dom, r[None],
                                                                fR[rj:rj + 1], sigma)
                                rec = dict(d=d, rho=rho, n_anchor=nA, model=model, kappa=kappa,
                                           L=L, LA_C=LA_C, LA_used=LA, x_idx=xi, r_idx=rj, mode=mode,
                                           eff_sparsity=effective_sparsity(x, r, sigma, cfg.eff_eps),
                                           contrast=_f((fX[xi] - fR[rj]).abs()))
                                n_tot += 1
                                if LA > L * (1.0 + 1e-9):
                                    rec["infeasible"] = True   # x (or r) inconsistent with the class
                                    n_inf += 1
                                else:
                                    gm = gen_of("members", cfg.seed, d, rho, nA, model, kappa, xi, rj, mode)
                                    try:
                                        rec.update(evaluate_instance(x, r, A, fAv, L, sigma, f, cfg, gm))
                                        rec["infeasible"] = False
                                    except Infeasible as e:
                                        rec["infeasible"], rec["note"] = True, str(e)
                                        n_inf += 1
                                fh.write(json.dumps(rec, default=_json_default) + "\n")
                    print(f"[det] d={d} rho={rho} nA={nA} {model:<11} kappa={kappa:<5} "
                          f"inst={n_tot} infeasible={n_inf} {time.perf_counter() - t_cfg:.1f}s", flush=True)
    return path


# ======================================== amendment v3: optimistic ceiling + local validity
def _topk_at(x, r, A, fAv, L, sigma, phi_f, cfg: Config):
    """Top-k (by |phi_f|) certificate at a given L via the hypercube LP; None when not applicable."""
    if not cfg.opt_topk_lp:
        return None
    d = x.numel()
    m = int(((x - r).abs() > 0).sum())
    k = min(cfg.top_k, m)
    if not (1 <= k < d):
        return None
    try:
        lp = HypercubeLP(x, r, A, fAv, L, sigma)
    except Infeasible:
        return dict(certified=None, infeasible=True)
    T = lp.candidate_topk(k, phi=phi_f)
    res = lp.certify_topk_abs(T, cfg.tol, stop_early=True)
    return dict(T=T, certified=bool(res["certified"]))


def evaluate_optimistic(x, r, A, fAv, L_min, L_true, sigma, f, cfg: Config) -> dict:
    """One instance, independent of kappa (amendment v3).
      L_min  = L_A(A): smallest feasible L. For EVERY valid L >= L_min: rho(L) >= rho(L_min) and the signs /
               top-k certified at L are a subset of those at L_min (monotonicity + nesting) -> optimistic ceiling.
      L_V    = Lip(f on hybrids ∪ A): the McShane extension of f restricted to hybrids ∪ A lies in F_{L_V}(A) and
               has the same hybrid values, so phi(f) is PROVABLY inside the interval at L_V; L_min <= L_V <= L_true.
      L_true = exact global constant when known (linear, unstable_id): cross-check only.
      rho_inf: purely geometric limit; rho(L) <= rho_inf for every L (upper bound only).
    'checks' are theorem checks: any value > tol is a bug (or a numerical problem), never a finding."""
    tol = cfg.tol
    d = x.numel()
    Z, active = hybrid_vertices(x, r)
    fZ = f(Z)
    L_V = lipschitz_after_adding(L_min, A, fAv, Z, fZ, sigma)
    phi_f = _np(shapley_full(fZ, active, d))
    delta = _np((x - r).abs() / sigma)
    rinf = _np(geometric_rho_inf(x, r, A, sigma))
    act = np.zeros(d, dtype=bool)
    act[active] = True

    def feat(v):
        return [float(v[j]) if act[j] else None for j in range(d)]

    levels = dict(Lmin=L_min, LV=L_V)
    if L_true is not None:
        levels["Ltrue"] = max(L_true, L_min)
    res = {}
    for name, L in levels.items():
        if not math.isfinite(L):
            res[name] = None
            continue
        lo, hi, _ = pair_bounds_exact(x, r, A, fAv, L, sigma)
        lo, hi = _np(lo), _np(hi)
        rho = np.zeros(d)
        rho[act] = (hi[act] - lo[act]) / (2.0 * L * delta[act])
        sg = np.zeros(d, dtype=int)
        sg[lo > tol] = 1
        sg[hi < -tol] = -1
        outside = max(0.0, float(np.max(lo - phi_f, initial=0.0)), float(np.max(phi_f - hi, initial=0.0)))
        res[name] = dict(L=L, rho=feat(rho), sign=sg.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                         outside=outside, topk=_topk_at(x, r, A, fAv, L, sigma, phi_f, cfg))

    chk = {}
    lm, lv, lt = res.get("Lmin"), res.get("LV"), res.get("Ltrue")
    if lm is not None and lv is not None:
        rl = np.array([lm["rho"][j] for j in active])
        rv = np.array([lv["rho"][j] for j in active])
        chk["LV_contains_frozen"] = lv["outside"]
        chk["rho_mono_Lmin_LV"] = float(np.max(rl - rv, initial=0.0))
        chk["rho_le_inf"] = float(np.max(rv - rinf[act], initial=0.0))
        chk["sign_nested"] = int(sum(1 for j in range(d) if lv["sign"][j] != 0 and lv["sign"][j] != lm["sign"][j]))
        tv, tm = lv["topk"], lm["topk"]
        chk["topk_nested"] = int(bool(tv and tm and tv.get("certified") and tm.get("certified") is False))
        if lt is not None:
            rt_ = np.array([lt["rho"][j] for j in active])
            chk["rho_mono_LV_Ltrue"] = float(np.max(rv - rt_, initial=0.0))
    if lt is not None:
        chk["Ltrue_contains_frozen"] = lt["outside"]
        chk["LV_le_Ltrue"] = max(0.0, (L_V - L_true) / L_true)   # relative
    return dict(L_V=L_V, kappa_V=L_V / L_min, active=active, delta=delta.tolist(), phi_f=phi_f.tolist(),
                rho_inf=feat(rinf), levels=res, checks=chk)


def run_optimistic(cfg: Config, out_dir: str) -> str:
    """Every (d, rho, n_anchor, model, x, r, mode) instance of the grid — the SAME draws as run_deterministic —
    evaluated at L_min, L_V (and L_true when known). Nothing is dropped for being infeasible under
    L = kappa * L_A(C): at L_min = L_A(A) the class is feasible by definition."""
    path = os.path.join(out_dir, "optimistic.jsonl")
    with open(path, "w") as fh:
        for d, rho, nA in itertools.product(cfg.dims, cfg.rhos, cfg.n_anchor):
            gd = gen_of("data", cfg.seed, d, rho, nA)          # identical sequence of draws to run_deterministic
            sample = gaussian_sampler(d, rho, cfg.device)
            sigma = robust_sigma(sample(cfg.n_train, gd))
            C, Rc, Xev = sample(nA, gd), sample(cfg.n_candidate, gd), sample(cfg.n_eval_x, gd)
            plan = [(xi, torch.randperm(cfg.n_candidate, generator=gd)[:cfg.refs_per_x].tolist())
                    for xi in range(cfg.n_eval_x)]
            for model in cfg.models:
                t_cfg, n_deg = time.perf_counter(), 0
                f = make_model(model, cfg.a)
                fC, fX, fR = f(C), f(Xev), f(Rc)
                LA_C = anchor_lipschitz(C, fC, sigma)
                L_true = lipschitz_true(model, d, sigma, cfg.a)
                for xi, ridx in plan:
                    x = Xev[xi]
                    A_dom = torch.cat([C, x[None]])
                    f_dom = torch.cat([fC, fX[xi:xi + 1]])
                    LA_dom = lipschitz_after_adding(LA_C, C, fC, x[None], fX[xi:xi + 1], sigma)
                    for rj in ridx:
                        r = Rc[rj]
                        for mode in ("domain", "fixed"):
                            if mode == "domain":
                                A, fAv, L_min = A_dom, f_dom, LA_dom
                            else:
                                A = torch.cat([A_dom, r[None]])
                                fAv = torch.cat([f_dom, fR[rj:rj + 1]])
                                L_min = lipschitz_after_adding(LA_dom, A_dom, f_dom, r[None],
                                                               fR[rj:rj + 1], sigma)
                            rec = dict(d=d, rho=rho, n_anchor=nA, model=model, x_idx=xi, r_idx=rj, mode=mode,
                                       eff_sparsity=effective_sparsity(x, r, sigma, cfg.eff_eps),
                                       contrast=_f((fX[xi] - fR[rj]).abs()),
                                       LA_cal=LA_C, L_min=L_min, L_true=L_true,
                                       feasible_at_kappa1=bool(L_min <= LA_C * (1.0 + 1e-9)))
                            if not (math.isfinite(L_min) and L_min > 0):
                                rec["degenerate"] = True    # duplicate anchors with different values, or f const on A
                                n_deg += 1
                            else:
                                rec.update(evaluate_optimistic(x, r, A, fAv, L_min, L_true, sigma, f, cfg))
                            fh.write(json.dumps(rec, default=_json_default) + "\n")
                print(f"[opt] d={d} rho={rho} nA={nA} {model:<11} degenerate={n_deg} "
                      f"{time.perf_counter() - t_cfg:.1f}s", flush=True)
    return path


# ================================================================ Monte Carlo coverage
def run_mc_coverage(cfg: Config, out_dir: str) -> dict:
    """Family-wise coverage of the Monte Carlo certificates (amendment v3).
    * The K references are fixed BEFORE any coalition is sampled, using exact quantities only: among feasible
      candidates, the K least saturated (max over the two modes of the median exact rho). The validity
      conditions of the MC bounds are untouched; the point is that a coverage test on saturated intervals
      ([-L delta, L delta] contains everything) is vacuous.
    * Vacuity is reported per mode (median exact rho) and gates technical GO via mc_rho_max.
    * Negative control: identical runs with every confidence radius multiplied by mc_neg_scales. A test that
      cannot detect a shrunken radius has no power; the mc_neg_required scale MUST be detected."""
    d, rho, nA = cfg.mc_d, cfg.mc_rho, cfg.mc_n_anchor
    gd = gen_of("mc-data", cfg.seed, d, rho, nA)
    sample = gaussian_sampler(d, rho, cfg.device)
    sigma = robust_sigma(sample(cfg.n_train, gd))
    C, Rc, x = sample(nA, gd), sample(cfg.n_candidate, gd), sample(1, gd)[0]
    f = make_model(cfg.mc_model, cfg.a)
    fC, fx = f(C), f(x[None])[0]
    LA_C = anchor_lipschitz(C, fC, sigma)
    L = cfg.mc_kappa * LA_C
    A_dom, f_dom = torch.cat([C, x[None]]), torch.cat([fC, fx[None]])
    LA_dom = lipschitz_after_adding(LA_C, C, fC, x[None], fx[None], sigma)
    if LA_dom > L:
        raise RuntimeError("x infeasible for the MC config; change seed or mc_kappa")

    def anchors(mode, r, fr):
        if mode == "domain":
            return A_dom, f_dom
        return torch.cat([A_dom, r[None]]), torch.cat([f_dom, fr.reshape(1)])

    def median_rho(lo, hi, r):
        dl = (x - r).abs() / sigma
        act = dl > 0
        return float(((hi - lo)[act] / (2.0 * L * dl[act])).median())

    # K references fixed BEFORE any coalition sampling: feasible in both modes, least saturated (exact rho only)
    scored = []
    for j in torch.randperm(cfg.n_candidate, generator=gd).tolist():
        r = Rc[j]
        fr = f(r[None])[0]
        if lipschitz_after_adding(LA_dom, A_dom, f_dom, r[None], fr.reshape(1), sigma) > L:
            continue
        worst = 0.0
        for mode in ("domain", "fixed"):
            A, fAv = anchors(mode, r, fr)
            lo, hi, _ = pair_bounds_exact(x, r, A, fAv, L, sigma)
            worst = max(worst, median_rho(lo, hi, r))
        scored.append((worst, j))
    if len(scored) < cfg.mc_K:
        raise RuntimeError("not enough feasible references for the MC config")
    scored.sort()
    refs = [j for _, j in scored[:cfg.mc_K]]
    R = Rc[refs]
    fR = f(R)

    out = dict(L=L, refs=refs, n_feasible_candidates=len(scored),
               selected_worst_median_rho=[w for w, _ in scored[:cfg.mc_K]])
    for mode in ("domain", "fixed"):
        lo_ex, hi_ex = [], []
        for k in range(cfg.mc_K):
            A, fAv = anchors(mode, R[k], fR[k])
            lo, hi, _ = pair_bounds_exact(x, R[k], A, fAv, L, sigma)
            lo_ex.append(lo)
            hi_ex.append(hi)
        lo_ex, hi_ex = torch.stack(lo_ex), torch.stack(hi_ex)
        B_ex = hi_ex - lo_ex
        mask = B_ex > 1e-9
        twoLd_ex = 2.0 * L * (x.unsqueeze(0) - R).abs() / sigma
        act_ex = twoLd_ex > 0
        rho_ex = (B_ex[act_ex] / twoLd_ex[act_ex]).cpu().numpy()
        med = float(np.median(rho_ex))
        out[f"rho_exact_{mode}"] = dict(
            quantiles_10_25_50_75_90=np.quantile(rho_ex, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            frac_below=float(np.mean(rho_ex < cfg.rho_informative)),
            median=med, non_vacuous=bool(med < cfg.mc_rho_max))

        def coverage(method, scale):
            fails, ratios = 0, []
            for run in range(cfg.mc_runs):
                g = gen_of("mc-run", cfg.seed, mode, run)   # same coalitions across methods and scales
                res = mc_pair_bounds(x, R, C, fC, L, sigma, fx=fx, fr=fR, mode=mode,
                                     n_per_stratum=cfg.mc_n_per_stratum, eta=cfg.mc_eta,
                                     method=method, gen=g, radius_scale=scale)
                fail = bool(((res.lo - lo_ex) > cfg.tol).any() or ((hi_ex - res.hi) > cfg.tol).any())
                fails += int(fail)
                if mask.any():
                    ratios.append(float(((res.hi - res.lo)[mask] / B_ex[mask]).median()))
            p = binomtest(fails, cfg.mc_runs, cfg.mc_eta, alternative="greater").pvalue
            return dict(fails=fails, runs=cfg.mc_runs, rate=fails / cfg.mc_runs, pvalue=float(p),
                        median_width_over_Bpair=float(np.median(ratios)) if ratios else None)

        for method in ("hoeffding", "bernstein", "best"):
            c = coverage(method, 1.0)
            c["ok"] = bool(c["pvalue"] >= cfg.go_mc_pvalue)
            out[f"{mode}/{method}"] = c
            print(f"[mc] {mode}/{method}: fails={c['fails']}/{cfg.mc_runs} p={c['pvalue']:.3f} "
                  f"median exact rho={med:.3f}", flush=True)
        for scale in cfg.mc_neg_scales:
            c = coverage("best", float(scale))
            c["scale"] = float(scale)
            c["detected"] = bool(c["pvalue"] < cfg.go_mc_pvalue)
            out[f"negative/{mode}/scale={scale}"] = c
            print(f"[mc-neg] {mode} radius x{scale}: fails={c['fails']}/{cfg.mc_runs} "
                  f"detected={c['detected']}", flush=True)
    with open(os.path.join(out_dir, "mc_coverage.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    return out


# ======================================================================= stress tests
def stress_saturated(tol: float) -> dict:
    """f = u, anchors {x=(1,1), a=(0,1)}, domain mode (r NOT an anchor), sigma = 1.
    Closed form: min phi_1 = 1 - min(1 + L eps, L (1 - eps)).
    kappa = 1: certified for every eps > 0; kappa = 1 + gamma: certified iff eps > gamma/(1+gamma)."""
    sig = torch.ones(2)
    x, a = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])
    A, fA = torch.stack([x, a]), torch.tensor([1.0, 0.0])
    LA = anchor_lipschitz(A, fA, sig)
    rows, mism, verr = [], 0, 0.0
    for kappa in (1.0, 1.1, 1.5, 2.0):
        L = kappa * LA
        thr = (kappa - 1.0) / kappa
        for eps in np.linspace(0.02, 0.98, 49):
            if kappa > 1.0 and abs(eps - thr) < 1e-3:
                continue
            r = torch.tensor([1.0 - eps, 1.0])
            lo, _ = HypercubeLP(x, r, A, fA, L, sig).interval()
            pred = 1.0 - min(1.0 + L * eps, L * (1.0 - eps))
            verr = max(verr, abs(lo[0] - pred))
            cert, pred_cert = lo[0] > tol, (kappa == 1.0) or (eps > thr)
            mism += int(cert != pred_cert)
            rows.append(dict(kappa=kappa, eps=float(eps), phi1_min=float(lo[0]), pred=pred,
                             certified=bool(cert)))
    return dict(ok=bool(mism == 0 and verr <= tol), mismatches=int(mism), max_value_err=float(verr), rows=rows)


def stress_sparse_reference(cfg: Config) -> dict:
    """One-feature references (r = x except feature 0) in endpoint-fixed mode: phi_0 = f(x) - f(r),
    all others 0, so top-1 = {0} is certified by construction whenever f(x) != f(r)."""
    d, rho, nA = 4, 0.9, 64
    gd = gen_of("sparse", cfg.seed)
    sample = gaussian_sampler(d, rho, cfg.device)
    sigma = robust_sigma(sample(cfg.n_train, gd))
    C, Rc, Xev = sample(nA, gd), sample(50, gd), sample(10, gd)
    f = make_model("fa", cfg.a)
    fC = f(C)
    LA_C = anchor_lipschitz(C, fC, sigma)
    n = n_inf = n_pos = n_cert = mism = 0
    for xi in range(Xev.shape[0]):
        x = Xev[xi]
        for j in range(10):
            r = x.clone()
            r[0] = Rc[j, 0]
            fx, fr = f(x[None]), f(r[None])
            A = torch.cat([C, x[None], r[None]])
            fAv = torch.cat([fC, fx, fr])
            LA = lipschitz_after_adding(LA_C, C, fC, torch.stack([x, r]), torch.cat([fx, fr]), sigma)
            L = 2.0 * LA
            n += 1
            try:
                lp = HypercubeLP(x, r, A, fAv, L, sigma)
            except Infeasible:
                n_inf += 1
                continue
            pos = float((fx - fr).abs()) > cfg.tol
            cert = lp.certify_topk_abs([0], cfg.tol)["certified"]
            n_pos += int(pos)
            n_cert += int(cert)
            mism += int(cert != pos)
    return dict(ok=(mism == 0), n=n, n_infeasible=n_inf, n_contrast_pos=n_pos, n_certified=n_cert,
                mismatches=mism,
                note="degeneracy is structural: restrict R_valid to a predefined population "
                     "and stratify by effective sparsity")


def stress_counterexample_2d(cfg: Config) -> dict:
    """f0 = u vs bounded f_a = u + a tanh(u) q(u - v): identical on the diagonal (all anchors),
    different Shapley values at x=(1,1), r=(0,0). For L >= Lip(f0) = 1, f0 is in the class,
    so its phi MUST lie in the LP interval (deterministic check)."""
    sig = torch.ones(2)
    ts = torch.linspace(-2.0, 2.0, 41)
    x, r = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    A = torch.cat([torch.stack([ts, ts], 1), x[None], r[None]])

    def f0(Z):
        return Z[:, 0]
    fa = make_model("fa", cfg.a)
    fA = f0(A)
    LA = anchor_lipschitz(A, fA, sig)   # 0.5 under l1
    rows, ok = [], True
    for kappa in (1.0, 2.0, 4.0, 8.0):
        L = kappa * LA
        lp = HypercubeLP(x, r, A, fA, L, sig)
        lo, hi = lp.interval()
        phi0, phia = lp.shapley(f0(lp.Z)), lp.shapley(fa(lp.Z))
        in0 = bool(np.all(phi0 >= lo - cfg.tol) and np.all(phi0 <= hi + cfg.tol))
        ina = bool(np.all(phia >= lo - cfg.tol) and np.all(phia <= hi + cfg.tol))
        if L >= 1.0 and not in0:
            ok = False
        rows.append(dict(kappa=kappa, L=L, lo=lo.tolist(), hi=hi.tolist(), phi_f0=phi0.tolist(),
                         phi_fa=phia.tolist(), f0_inside=in0, fa_inside=ina))
    # amendment v3: fa agrees with f0 on every anchor (q(0) = 0 on the diagonal), so at
    # L_V(fa) = Lip(fa on hybrids ∪ A) phi(fa) MUST be inside the interval (theorem, gated)
    Zv = HypercubeLP(x, r, A, fA, LA, sig).Z
    LV_fa = lipschitz_after_adding(LA, A, fA, Zv, fa(Zv), sig)
    lpv = HypercubeLP(x, r, A, fA, LV_fa, sig)
    lo, hi = lpv.interval()
    phia = lpv.shapley(fa(lpv.Z))
    fa_inside_LV = bool(np.all(phia >= lo - cfg.tol) and np.all(phia <= hi + cfg.tol))
    rho_LV = ((hi - lo) / (2.0 * LV_fa * _np(lpv.delta))).tolist()
    return dict(ok=bool(ok and fa_inside_LV), LA=LA, rows=rows,
                fa_at_LV=dict(L_V=LV_fa, kappa_needed=LV_fa / LA, lo=lo.tolist(), hi=hi.tolist(),
                              phi_fa=phia.tolist(), inside=fa_inside_LV, rho=rho_LV))


def stress_uv_counterexample(tol: float) -> dict:
    """f(u, v) = u + v, anchors {(0,0), (1,1)}, x = (1,1), r = (0,0), sigma = 1 (L_A = L_true = 1).
    Closed form for L >= 1: rho_1(L) = rho_2(L) = 1 - 1/L, while rho_inf = 1.
    Guards against reading rho_inf ~ 1 as 'no valid L can be informative' (it is an upper bound only)."""
    sig = torch.ones(2)
    x, r = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    A = torch.stack([r, x])
    fA = A.sum(1)
    LA = anchor_lipschitz(A, fA, sig)
    rows, err = [], 0.0
    for L in (1.0, 1.1, 1.5, 2.0, 5.0, 10.0, 100.0):
        lo, hi, _ = pair_bounds_exact(x, r, A, fA, L, sig)
        rho = ((hi - lo) / (2.0 * L)).tolist()
        pred = 1.0 - 1.0 / L
        err = max(err, max(abs(v - pred) for v in rho))
        rows.append(dict(L=L, rho=rho, pred=pred))
    rinf = geometric_rho_inf(x, r, A, sig).tolist()
    ok = err <= tol and abs(LA - 1.0) <= tol and max(abs(v - 1.0) for v in rinf) <= tol
    return dict(ok=bool(ok), LA=LA, max_err=err, rho_inf=rinf, rows=rows)


def stress_stable_ood(cfg: Config) -> dict:
    """Descriptive: local output variance (a variance-based baseline) vs AIW for a model that is
    flat but shifted away from the data diagonal. Expectation: variance ~ 0 while AIW is not."""
    d, rho, nA, kappa, noise = 3, 0.99, 64, 2.0, 0.05
    gd = gen_of("stable-ood", cfg.seed)
    sample = gaussian_sampler(d, rho, cfg.device)
    sigma = robust_sigma(sample(cfg.n_train, gd))
    C, Rc, Xev = sample(nA, gd), sample(40, gd), sample(5, gd)
    f = make_model("stable_ood", cfg.a)
    fC = f(C)
    LA_C = anchor_lipschitz(C, fC, sigma)
    U, Gs, TO = [], [], []
    for xi in range(Xev.shape[0]):
        x = Xev[xi]
        A, fAv = torch.cat([C, x[None]]), torch.cat([fC, f(x[None])])
        LA = lipschitz_after_adding(LA_C, C, fC, x[None], f(x[None]), sigma)
        L = kappa * LA
        for j in range(8):
            lp = HypercubeLP(x, Rc[j], A, fAv, L, sigma)
            lo, hi = lp.interval()
            Zn = lp.Z[:, None, :] + noise * sigma * torch.randn(lp.Z.shape[0], 32, d, generator=gd).to(lp.Z.device)
            U.append(float(f(Zn.reshape(-1, d)).reshape(lp.Z.shape[0], 32).var(dim=1).mean()))
            Gs.append(float((hi - lo).sum()))
            phi = lp.shapley(f(lp.Z))
            TO.append(max(0.0, float(np.max(lo - phi)), float(np.max(phi - hi))))
    rho_s = spearmanr(U, Gs)[0]
    return dict(local_variance=U, aiw_sum=Gs, transfer_out=TO,
                spearman_variance_vs_aiw=float(rho_s),
                median_local_variance=float(np.median(U)), median_aiw_sum=float(np.median(Gs)))


def run_stress(cfg: Config, out_dir: str) -> dict:
    out = dict(saturated=stress_saturated(cfg.tol),
               sparse_reference=stress_sparse_reference(cfg),
               counterexample_2d=stress_counterexample_2d(cfg),
               uv_counterexample=stress_uv_counterexample(cfg.tol),
               stable_ood=stress_stable_ood(cfg))
    with open(os.path.join(out_dir, "stress.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    for k in STRESS_GATED:
        print(f"[stress] {k}: ok={out[k]['ok']}", flush=True)
    return out


# ============================================================================== runtime
def run_runtime(cfg: Config, out_dir: str) -> list:
    rows = []
    for d in cfg.rt_dims:
        gd = gen_of("rt", cfg.seed, d)
        sample = gaussian_sampler(d, 0.9, cfg.device)
        sigma = robust_sigma(sample(cfg.n_train, gd))
        C, x, r = sample(cfg.rt_n_anchor, gd), sample(1, gd)[0], sample(1, gd)[0]
        f = make_model("fa", cfg.a)
        fC, fx = f(C), f(x[None])
        LA = lipschitz_after_adding(anchor_lipschitz(C, fC, sigma), C, fC, x[None], fx, sigma)
        L = 2.0 * LA
        A, fAv = torch.cat([C, x[None]]), torch.cat([fC, fx])
        t0 = time.perf_counter()
        pair_bounds_exact(x, r, A, fAv, L, sigma)
        t_pair = time.perf_counter() - t0
        t0 = time.perf_counter()
        lp = HypercubeLP(x, r, A, fAv, L, sigma)
        lp.interval()
        t_lp = time.perf_counter() - t0
        t0 = time.perf_counter()
        mc_pair_bounds(x, r[None], C, fC, L, sigma, fx=fx[0], mode="domain",
                       n_per_stratum=cfg.rt_n_per_stratum, eta=0.05, method="best",
                       gen=gen_of("rt-mc", cfg.seed, d))
        t_mc = time.perf_counter() - t0
        rows.append(dict(d=d, m=lp.m, n_vertices=2 ** lp.m, n_edges=lp.m * 2 ** max(lp.m - 1, 0),
                         n_lp=lp.n_lp, t_pair_exact=t_pair, t_lp_interval=t_lp, t_mc_best=t_mc,
                         hybrid_model_evals_fast=0))
        print(f"[rt] d={d}: pair {t_pair:.3f}s | LP {t_lp:.3f}s ({lp.n_lp} LPs) | MC {t_mc:.3f}s", flush=True)
    with open(os.path.join(out_dir, "runtime.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=_json_default)
    return rows


# ============================================================================== summary
def _tightness(recs: list, feature_filter=None) -> dict:
    """LP vs pairwise tightness over active features (optionally filtered per (record, feature)).
    Ratios exclude B_pair = 0 (zero denominator); those cases are counted separately."""
    ratios, n_act, n_zero, c_lp, c_pair, n_topk, c_topk = [], 0, 0, 0, 0, 0, 0
    for r in recs:
        for j in range(r["d"]):
            if r["delta"][j] <= 0.0:
                continue
            if feature_filter is not None and not feature_filter(r, j):
                continue
            n_act += 1
            if r["B_pair"][j] > 1e-9:
                ratios.append(r["G_lp"][j] / r["B_pair"][j])
            else:
                n_zero += 1
            c_lp += int(r["sign_lp"][j] != 0)
            c_pair += int(r["sign_pair"][j] != 0)
        if r.get("topk"):
            n_topk += 1
            c_topk += int(r["topk"]["certified"])
    q = np.quantile(ratios, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if ratios else None
    return dict(n_features=n_act, n_zero_denominator=n_zero,
                median_G_over_B=float(np.median(ratios)) if ratios else None,
                quantiles_G_over_B_10_25_50_75_90=q,
                frac_ratio_below_0_99=float(np.mean(np.array(ratios) < 0.99)) if ratios else None,
                cert_sign_rate_lp=c_lp / n_act if n_act else None,
                cert_sign_rate_pair=c_pair / n_act if n_act else None,
                cert_sign_gain=(c_lp - c_pair) / n_act if n_act else None,
                cert_topk_rate=c_topk / n_topk if n_topk else None)


def _lp_criterion(t: dict, cfg: Config) -> bool:
    return bool((t["median_G_over_B"] is not None and t["median_G_over_B"] <= cfg.go_lp_ratio)
                or (t["cert_sign_gain"] is not None and t["cert_sign_gain"] >= cfg.go_lp_sign_gain))


def _has_cross_free_edge(r: dict, j: int) -> bool:
    return r.get("cross_free_edges", [0] * r["d"])[j] > 0


def _joint(recs: list, cfg: Config) -> dict:
    """Amendment v2: joint |phi_i| > |phi_j| certificates, interval separation vs exact LP."""
    n = sum(r["joint"]["n_pairs"] for r in recs)
    ci = sum(r["joint"]["cert_interval"] for r in recs)
    cl = sum(r["joint"]["cert_lp_only"] for r in recs)
    calls = sum(r["joint"]["n_lp_calls"] for r in recs)
    gain = cl / n if n else None
    return dict(n_pairs=n, rate_interval=ci / n if n else None, rate_lp=(ci + cl) / n if n else None,
                gain_lp_minus_interval=gain, lp_calls_per_pair=calls / n if n else None,
                lp_retained=bool(gain is not None and gain >= cfg.go_lp_joint_gain))


def _rho(recs: list, cfg: Config) -> dict:
    """Anchor informativeness rho_i = B_pair_i / (2 L delta_i) over active features (1 = anchors add nothing)."""
    v = [r["B_pair"][j] / r["twoLd"][j] for r in recs for j in range(r["d"]) if r["twoLd"][j] > 0]
    if not v:
        return dict(n_features=0)
    v = np.asarray(v)
    return dict(n_features=int(v.size),
                quantiles_10_25_50_75_90=np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                frac_below=float(np.mean(v < cfg.rho_informative)), threshold=cfg.rho_informative)


def _failures(recs: list, keys: tuple, cfg: Config) -> dict:
    """Amendment v3: infeasible rate, conditional violation rate (among feasible), total failure rate.
    Per instance the total-failure indicator is non-increasing in kappa (nested intervals); the conditional
    rate is not (its denominator changes with kappa), so compare kappas with the total rate."""
    groups: dict = {}
    for r in recs:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = {}
    for g in sorted(groups, key=lambda t: tuple(str(v) for v in t)):
        rs = groups[g]
        n = len(rs)
        n_inf = sum(bool(r.get("infeasible")) for r in rs)
        feas = [r for r in rs if not r.get("infeasible")]
        n_viol = sum(r["transfer_out"] > cfg.tol for r in feas)
        out["/".join(f"{k}={v}" for k, v in zip(keys, g))] = dict(
            n=n, infeasible_rate=n_inf / n,
            conditional_violation_rate=n_viol / len(feas) if feas else None,
            total_failure_rate=(n_inf + n_viol) / n)
    return out


OPT_LEVELS = ("Lmin", "LV", "Ltrue")
OPT_CHECKS = ("LV_contains_frozen", "Ltrue_contains_frozen", "rho_mono_Lmin_LV", "rho_mono_LV_Ltrue",
              "rho_le_inf", "sign_nested", "topk_nested", "LV_le_Ltrue")


def _quant(v) -> "list | None":
    v = np.asarray(v, dtype=float)
    return np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if v.size else None


def _opt_group(rows: list, cfg: Config) -> dict:
    """Optimistic-ceiling summary of a group of optimistic.jsonl rows (feature-instance level).
    Lmin: ceiling for EVERY valid L (rho can only be larger, certificates only fewer).
    LV  : the tightest interval that provably contains phi(f).
    Ltrue: only for models with a closed-form global constant."""
    rows = [r for r in rows if not r.get("degenerate")]
    out: dict = dict(n_instances=len(rows))
    if not rows:
        return out
    rinf = np.array([r["rho_inf"][j] for r in rows for j in r["active"]], dtype=float)
    out["rho_inf"] = dict(quantiles_10_25_50_75_90=_quant(rinf),
                          frac_below=float(np.mean(rinf < cfg.rho_informative)) if rinf.size else None)
    for lev in OPT_LEVELS:
        rs = [r for r in rows if r["levels"].get(lev)]
        if not rs:
            continue
        rho = np.array([r["levels"][lev]["rho"][j] for r in rs for j in r["active"]], dtype=float)
        sg = [r["levels"][lev]["sign"][j] != 0 for r in rs for j in r["active"]]
        tk = [r["levels"][lev]["topk"]["certified"] for r in rs
              if r["levels"][lev].get("topk") and r["levels"][lev]["topk"].get("certified") is not None]
        e = dict(n_instances=len(rs), n_features=int(rho.size),
                 rho_quantiles_10_25_50_75_90=_quant(rho),
                 rho_frac_below=float(np.mean(rho < cfg.rho_informative)) if rho.size else None,
                 cert_sign_rate=float(np.mean(sg)) if sg else None,
                 cert_topk_rate=float(np.mean(tk)) if tk else None,
                 frozen_outside_rate=float(np.mean([r["levels"][lev]["outside"] > cfg.tol for r in rs])))
        if cfg.sci_rho_target is not None:
            e["rho_frac_le_target"] = float(np.mean(rho <= cfg.sci_rho_target)) if rho.size else None
        out[lev] = e
    out["kappa_V_quantiles_10_25_50_75_90"] = _quant([r["kappa_V"] for r in rows])
    out["L_min_over_LA_cal_quantiles_10_25_50_75_90"] = _quant(
        [r["L_min"] / r["LA_cal"] for r in rows if r["LA_cal"] > 0])
    out["feasible_at_kappa1_rate"] = float(np.mean([r["feasible_at_kappa1"] for r in rows]))
    return out


def _opt_checks(rows: list, cfg: Config) -> dict:
    """Theorem checks from optimistic.jsonl (relative for LV_le_Ltrue): any value > tol is a bug."""
    worst = {k: 0.0 for k in OPT_CHECKS}
    n_over = {k: 0 for k in OPT_CHECKS}
    for r in rows:
        for k, v in r.get("checks", {}).items():
            worst[k] = max(worst[k], float(v))
            n_over[k] += int(float(v) > cfg.tol)
    return dict(max=worst, n_over_tol=n_over, ok=all(n == 0 for n in n_over.values()))


def _scientific(opt_by_d: dict, cfg: Config) -> dict:
    """Pre-registered scenarios per d (thresholds from config.json, fixed before the run):
      S3 valid_and_useful        : frac(rho(L_V) <= target) >= min_frac AND cert-sign rate at L_V >= min_sign_rate
      S2 lost_to_model_roughness : not S3, but frac(rho(L_min) <= target) >= min_frac
      S1 not_identifiable        : otherwise — even the optimistic ceiling at L_min fails"""
    th = dict(rho_target=cfg.sci_rho_target, min_frac=cfg.sci_min_frac, min_sign_rate=cfg.sci_min_sign_rate)
    if any(v is None for v in th.values()):
        return dict(status="thresholds_not_registered", thresholds=th,
                    note="pass --sci-rho-target, --sci-min-frac and --sci-min-sign-rate BEFORE the run")
    verdict = {}
    for key, g in opt_by_d.items():
        lmin, lv = g.get("Lmin"), g.get("LV")
        if not lmin or not lv or lv.get("rho_frac_le_target") is None:
            verdict[key] = "no_data"
        elif (lv["rho_frac_le_target"] >= cfg.sci_min_frac
              and (lv["cert_sign_rate"] or 0.0) >= cfg.sci_min_sign_rate):
            verdict[key] = "S3_valid_and_useful"
        elif (lmin.get("rho_frac_le_target") or 0.0) >= cfg.sci_min_frac:
            verdict[key] = "S2_lost_to_model_roughness"
        else:
            verdict[key] = "S1_not_identifiable"
    return dict(status="evaluated", thresholds=th, verdict_by_d=verdict)


def summarize(cfg: Config, out_dir: str, rec_path, opt_path, mc, stress, rt) -> dict:
    S: dict = dict(run_kind=cfg.run_kind, seed=cfg.seed, device=cfg.device, amendment=cfg.amendment, go={})
    stress_ok = all(stress[k]["ok"] for k in STRESS_GATED)
    S["go"]["stress_checks"] = bool(stress_ok)
    grid_ok = None
    joint_all = None
    if rec_path:
        with open(rec_path) as fh:
            recs = [json.loads(line) for line in fh]
        feas = [r for r in recs if not r.get("infeasible")]
        S["n_instances"], S["n_feasible"] = len(recs), len(feas)
        S["infeasible_rate_by_kappa_mode"] = {
            f"{k}/{m}": float(np.mean([r["infeasible"] for r in recs if r["kappa"] == k and r["mode"] == m]))
            for k in cfg.kappas for m in ("domain", "fixed")}
        fields = ["viol_chain", "viol_contain", "member_viol_lp", "member_viol_pair", "lip_viol", "sol_viol"]
        S["deterministic"] = {k: dict(max=max((r[k] for r in feas), default=0.0),
                                      n_over_tol=sum(r[k] > cfg.tol for r in feas)) for k in fields}
        fields += ["exact_gap", "joint_lb_viol"]   # Proposition check + LP-vs-interval sanity
        for k in ("exact_gap", "joint_lb_viol"):
            S["deterministic"][k] = dict(max=max((r[k] for r in feas), default=0.0),
                                         n_over_tol=sum(r[k] > cfg.tol for r in feas))
        for k in ("sign_contra", "topk_contra", "joint_contra"):
            S["deterministic"][k] = sum(r[k] for r in feas)
        grid_ok = all(S["deterministic"][k]["n_over_tol"] == 0 for k in fields) \
            and all(S["deterministic"][k] == 0 for k in ("sign_contra", "topk_contra", "joint_contra"))

        # Proposition: LP interval == pairwise interval => interval-tightness criterion is void
        S["lp_interval_tightness"] = dict(
            status="void_by_theorem", proposition="bounds.py: pairwise interval is exact under weighted l1",
            max_abs_interval_gap=S["deterministic"]["exact_gap"]["max"],
            legacy_reports=dict(pooled_all_d=_tightness(feas),
                                d_ge_3=_tightness([r for r in feas if r["d"] >= cfg.go_lp_min_d]),
                                cross_free_edge=_tightness(feas, _has_cross_free_edge)))

        # amendment v2: joint certificates decide whether the LP stays in the method
        joint_all = _joint(feas, cfg)
        S["lp_joint_certificates"] = dict(
            overall=joint_all,
            by_d_mode={f"d={d}/{m}": _joint([r for r in feas if r["d"] == d and r["mode"] == m], cfg)
                       for d in cfg.dims for m in ("domain", "fixed")},
            by_mode_kappa={f"{m}/kappa={k}": _joint([r for r in feas if r["mode"] == m and r["kappa"] == k], cfg)
                           for m in ("domain", "fixed") for k in cfg.kappas})

        # anchor informativeness (report only; key usefulness signal for Phase 2)
        S["anchor_informativeness_rho"] = dict(
            overall=_rho(feas, cfg),
            by_d={f"d={d}": _rho([r for r in feas if r["d"] == d], cfg) for d in cfg.dims},
            by_n_anchor={f"nA={n}": _rho([r for r in feas if r["n_anchor"] == n], cfg) for n in cfg.n_anchor},
            by_kappa={f"kappa={k}": _rho([r for r in feas if r["kappa"] == k], cfg) for k in cfg.kappas},
            by_mode={m: _rho([r for r in feas if r["mode"] == m], cfg) for m in ("domain", "fixed")},
            by_d_nA_mode={f"d={d}/nA={n}/{m}": _rho([r for r in feas if r["d"] == d and r["n_anchor"] == n
                                                     and r["mode"] == m], cfg)
                          for d in cfg.dims for n in cfg.n_anchor for m in ("domain", "fixed")})
        S["certified_sign_rate"] = {f"d={d}/{m}": _tightness([r for r in feas if r["d"] == d and r["mode"] == m])
                                    ["cert_sign_rate_lp"] for d in cfg.dims for m in ("domain", "fixed")}
        S["transfer"] = dict(  # empirical, not a theorem check
            frozen_phi_outside_rate=float(np.mean([r["transfer_out"] > cfg.tol for r in feas])) if feas else None,
            envelope_violation_rate=float(np.mean([r["VL"] > cfg.tol for r in feas])) if feas else None)
        # amendment v3: three failure rates, reported separately
        S["failure_by_kappa"] = _failures(recs, ("kappa",), cfg)
        S["failure_by_d_kappa"] = _failures(recs, ("d", "kappa"), cfg)
        S["failure_by_model_kappa"] = _failures(recs, ("model", "kappa"), cfg)
    S["go"]["grid_correctness"] = bool(grid_ok) if rec_path else "not_evaluated"
    S["go"]["deterministic_correctness"] = bool(stress_ok and grid_ok) if rec_path else "not_evaluated"

    # amendment v3: optimistic ceiling (L_min), local validity (L_V), geometric limit — every instance
    if opt_path:
        with open(opt_path) as fh:
            rows = [json.loads(line) for line in fh]
        modes = ("domain", "fixed")
        by_d = {f"d={d}": _opt_group([r for r in rows if r["d"] == d], cfg) for d in cfg.dims}
        S["optimistic"] = dict(
            n_rows=len(rows), n_degenerate=sum(bool(r.get("degenerate")) for r in rows),
            overall=_opt_group(rows, cfg), by_d=by_d,
            by_d_mode={f"d={d}/{m}": _opt_group([r for r in rows if r["d"] == d and r["mode"] == m], cfg)
                       for d in cfg.dims for m in modes},
            by_d_model={f"d={d}/{mo}": _opt_group([r for r in rows if r["d"] == d and r["model"] == mo], cfg)
                        for d in cfg.dims for mo in cfg.models},
            by_d_rho={f"d={d}/rho={rh}": _opt_group([r for r in rows if r["d"] == d and r["rho"] == rh], cfg)
                      for d in cfg.dims for rh in cfg.rhos},
            by_d_nA={f"d={d}/nA={n}": _opt_group([r for r in rows if r["d"] == d and r["n_anchor"] == n], cfg)
                     for d in cfg.dims for n in cfg.n_anchor})
        chk = _opt_checks(rows, cfg)
        S["optimistic_theorem_checks"] = chk
        S["go"]["optimistic_theorem_checks"] = chk["ok"]
        S["scientific"] = _scientific(by_d, cfg)
    else:
        S["go"]["optimistic_theorem_checks"] = "not_evaluated"
        S["scientific"] = dict(status="not_evaluated")

    if mc:
        cov = {k: v for k, v in mc.items() if "/" in k and not k.startswith("negative/")}
        neg = {k: v for k, v in mc.items() if k.startswith("negative/")}
        S["mc"], S["mc_negative_control"] = cov, neg
        S["mc_rho_exact"] = {k: v for k, v in mc.items() if k.startswith("rho_exact_")}
        S["go"]["mc_coverage"] = all(v["ok"] for v in cov.values())
        S["go"]["mc_non_vacuous"] = all(v["non_vacuous"] for v in S["mc_rho_exact"].values())
        req = [v for v in neg.values() if v["scale"] == cfg.mc_neg_required]
        S["go"]["mc_negative_control_detected"] = bool(req) and all(v["detected"] for v in req)
    else:
        for k in ("mc_coverage", "mc_non_vacuous", "mc_negative_control_detected"):
            S["go"][k] = "not_evaluated"
    if rt:
        S["runtime"] = rt
    # amendment v3: technical GO; any skipped part is 'not_evaluated' and blocks it
    S["go"]["PHASE1_GO"] = all(S["go"].get(k) is True for k in TECH_GO_KEYS)
    S["go"]["PHASE1_GO_meaning"] = ("technical GO only: correctness + theorem checks + a valid, non-vacuous, "
                                    "sensitive MC test. Usefulness = the separate 'scientific' verdict")
    S["go"]["lp_interval_tightness"] = "void_by_theorem"
    if joint_all is not None:
        S["go"]["lp_retained_for_joint_certificates"] = joint_all["lp_retained"]
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=2, default=_json_default)
    print(json.dumps(S["go"], indent=2, default=_json_default))
    print(json.dumps(S["scientific"], indent=2, default=_json_default))
    return S


# ================================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="phase1_out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: 0 for --quick (pilot), 1 for the confirmatory grid")
    ap.add_argument("--skip", nargs="*", default=[], choices=["det", "opt", "mc", "runtime"],
                    help="skipped parts are 'not_evaluated' and block PHASE1_GO")
    ap.add_argument("--device", default=None, help="auto (default) | cpu | cuda | cuda:N")
    ap.add_argument("--sci-rho-target", type=float, default=None,
                    help="scientific verdict: informative iff rho <= this (register BEFORE the run)")
    ap.add_argument("--sci-min-frac", type=float, default=None,
                    help="scientific verdict: required fraction of informative feature-instances")
    ap.add_argument("--sci-min-sign-rate", type=float, default=None,
                    help="scientific verdict: required certified-sign rate at L_V for S3")
    args = ap.parse_args()
    cfg = Config.quick() if args.quick else Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    cfg.device = resolve_device(cfg.device)
    for name in ("sci_rho_target", "sci_min_frac", "sci_min_sign_rate"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if cfg.run_kind == "confirmatory" and cfg.seed in PILOT_SEEDS:
        ap.error(f"seed {cfg.seed} was used by the pilot; the confirmatory grid needs a fresh seed "
                 f"(amendment: confirmatory_seeds_disjoint_from_pilot)")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as fh:   # written BEFORE anything runs
        json.dump(asdict(cfg), fh, indent=2, default=_json_default)
    print(f"[cfg] device={cfg.device} seed={cfg.seed} run_kind={cfg.run_kind}", flush=True)
    stress = run_stress(cfg, args.out)
    rec_path = None if "det" in args.skip else run_deterministic(cfg, args.out)
    opt_path = None if "opt" in args.skip else run_optimistic(cfg, args.out)
    mc = None if "mc" in args.skip else run_mc_coverage(cfg, args.out)
    rt = None if "runtime" in args.skip else run_runtime(cfg, args.out)
    summarize(cfg, args.out, rec_path, opt_path, mc, stress, rt)


if __name__ == "__main__":
    main()