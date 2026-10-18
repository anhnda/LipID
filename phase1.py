"""
phase1.py — Phase 1: mathematical validation of Attribution Identifiability Width (AIW).

    python -m pytest -q tests.py                      # unit tests first
    python phase1.py --out phase1_quick --quick       # smoke run
    python phase1.py --out phase1_out                 # full pre-registered grid

Outputs in --out:
    config.json      the exact configuration (pre-registration record)
    records.jsonl    one line per (d, rho, n_anchor, model, kappa, x, r, anchor mode)
    stress.json      saturated anchors, sparse references, 2-D counterexample, stable-OOD
    mc_coverage.json Hoeffding / empirical-Bernstein family-wise coverage (binomial test)
    runtime.json     cost of pairwise bound vs hypercube LP vs Monte Carlo
    summary.json     aggregated metrics + GO / NO-GO checks

Data splits are disjoint draws: train (sigma only) | C_anchor | R_candidate | D_eval.
sigma = 1.4826 * MAD on the train split, fixed before anything else.
"OOD" here means low-density mismatch: a non-degenerate Gaussian has full support.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.stats import binomtest, spearmanr

from bounds import effective_sparsity, mc_pair_bounds, pair_bounds_exact
from envelope import (Member, RandomLipschitz, anchor_lipschitz, clamp_to_envelope,
                      envelope_violation, lipschitz_after_adding, lipschitz_violation)
from hypercube_lp import HypercubeLP, Infeasible

torch.set_default_dtype(torch.float64)


# =============================================================================== config
@dataclass
class Config:
    seed: int = 0
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
    # Monte Carlo coverage experiment
    mc_d: int = 5
    mc_rho: float = 0.9
    mc_n_anchor: int = 64
    mc_model: str = "fa"
    mc_kappa: float = 1.5
    mc_K: int = 4
    mc_runs: int = 200
    mc_n_per_stratum: int = 32
    mc_eta: float = 0.05
    # runtime benchmark
    rt_dims: tuple = (4, 6, 8, 10, 12)
    rt_n_anchor: int = 256
    rt_n_per_stratum: int = 64
    # pre-registered GO thresholds (Phase 1)
    go_lp_ratio: float = 0.8        # median Gamma_LP / B_pair <= this ...
    go_lp_sign_gain: float = 0.10   # ... or certified-sign rate gain (LP - pair) >= this
    go_mc_pvalue: float = 0.01      # binomial test of failure rate > eta must NOT reject at this level

    @staticmethod
    def quick() -> "Config":
        return Config(dims=(2, 3), rhos=(0.0, 0.9), n_anchor=(32,), models=("linear", "fa"),
                      kappas=(1.0, 2.0), n_eval_x=2, refs_per_x=2, n_members=20,
                      mc_runs=40, rt_dims=(4, 6))


# ============================================================================== helpers
def seed_of(*key) -> int:
    return zlib.crc32(repr(key).encode()) & 0x7FFFFFFF


def gen_of(*key) -> torch.Generator:
    return torch.Generator().manual_seed(seed_of(*key))


def gaussian_sampler(d: int, rho: float):
    cov = (1.0 - rho) * torch.eye(d) + rho * torch.ones(d, d)
    chol = torch.linalg.cholesky(cov)

    def sample(n: int, gen: torch.Generator) -> torch.Tensor:
        return torch.randn(n, d, generator=gen) @ chol.T
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
            return Z @ torch.linspace(1.0, 0.2, Z.shape[1], dtype=Z.dtype)
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

    # members of F_L(A): ell, h, then clamped random Lipschitz functions
    center = 0.5 * (x + r)
    scale = 0.5 + 0.5 * float(lp.delta.mean())
    f_mu, f_sd = float(fAv.mean()), float(fAv.std())
    mv_lp = mv_pair = lip_viol = 0.0
    sign_contra = topk_contra = 0
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

    sol_viol = lp.verify_solutions(all_pairs=lp.m <= 12)

    return dict(
        m=lp.m, n_lp=lp.n_lp, t_pair=t_pair, t_lp=t_lp,
        delta=delta.tolist(), twoLd=twoLd.tolist(), B_pair=B.tolist(), G_lp=G.tolist(), D=D.tolist(),
        lo_pair=lo_p.tolist(), hi_pair=hi_p.tolist(), lo_lp=lo_l.tolist(), hi_lp=hi_l.tolist(),
        phi_f=phi_f.tolist(), sign_lp=sg_lp.tolist(), sign_pair=sg_pair.tolist(), topk=topk,
        viol_chain=viol_chain, viol_contain=viol_contain, member_viol_lp=mv_lp,
        member_viol_pair=mv_pair, lip_viol=lip_viol, sol_viol=sol_viol,
        sign_contra=sign_contra, topk_contra=topk_contra,
        VL=VL, transfer_out=transfer_out,
    )


def run_deterministic(cfg: Config, out_dir: str) -> str:
    path = os.path.join(out_dir, "records.jsonl")
    with open(path, "w") as fh:
        for d, rho, nA in itertools.product(cfg.dims, cfg.rhos, cfg.n_anchor):
            gd = gen_of("data", cfg.seed, d, rho, nA)
            sample = gaussian_sampler(d, rho)
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
                                fh.write(json.dumps(rec) + "\n")
                    print(f"[det] d={d} rho={rho} nA={nA} {model:<11} kappa={kappa:<5} "
                          f"inst={n_tot} infeasible={n_inf} {time.perf_counter() - t_cfg:.1f}s", flush=True)
    return path


# ================================================================ Monte Carlo coverage
def run_mc_coverage(cfg: Config, out_dir: str) -> dict:
    d, rho, nA = cfg.mc_d, cfg.mc_rho, cfg.mc_n_anchor
    gd = gen_of("mc-data", cfg.seed, d, rho, nA)
    sample = gaussian_sampler(d, rho)
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

    # K references fixed BEFORE any coalition sampling; feasible in both modes
    refs = []
    for j in torch.randperm(cfg.n_candidate, generator=gd).tolist():
        r = Rc[j]
        if lipschitz_after_adding(LA_dom, A_dom, f_dom, r[None], f(r[None]), sigma) <= L:
            refs.append(j)
        if len(refs) == cfg.mc_K:
            break
    if len(refs) < cfg.mc_K:
        raise RuntimeError("not enough feasible references for the MC config")
    R = Rc[refs]
    fR = f(R)

    out = dict(L=L, refs=refs)
    for mode in ("domain", "fixed"):
        lo_ex, hi_ex = [], []
        for k in range(cfg.mc_K):
            A = A_dom if mode == "domain" else torch.cat([A_dom, R[k:k + 1]])
            fAv = f_dom if mode == "domain" else torch.cat([f_dom, fR[k:k + 1]])
            lo, hi, _ = pair_bounds_exact(x, R[k], A, fAv, L, sigma)
            lo_ex.append(lo)
            hi_ex.append(hi)
        lo_ex, hi_ex = torch.stack(lo_ex), torch.stack(hi_ex)
        B_ex = hi_ex - lo_ex
        mask = B_ex > 1e-9
        for method in ("hoeffding", "bernstein", "best"):
            fails, ratios = 0, []
            for run in range(cfg.mc_runs):
                g = gen_of("mc-run", cfg.seed, mode, run)   # same coalitions across methods
                res = mc_pair_bounds(x, R, C, fC, L, sigma, fx=fx, fr=fR, mode=mode,
                                     n_per_stratum=cfg.mc_n_per_stratum, eta=cfg.mc_eta,
                                     method=method, gen=g)
                fail = bool(((res.lo - lo_ex) > cfg.tol).any() or ((hi_ex - res.hi) > cfg.tol).any())
                fails += int(fail)
                if mask.any():
                    ratios.append(float(((res.hi - res.lo)[mask] / B_ex[mask]).median()))
            p = binomtest(fails, cfg.mc_runs, cfg.mc_eta, alternative="greater").pvalue
            out[f"{mode}/{method}"] = dict(
                fails=fails, runs=cfg.mc_runs, rate=fails / cfg.mc_runs, pvalue=float(p),
                ok=bool(p >= cfg.go_mc_pvalue),
                median_width_over_Bpair=float(np.median(ratios)) if ratios else None)
            print(f"[mc] {mode}/{method}: fails={fails}/{cfg.mc_runs} p={p:.3f}", flush=True)
    with open(os.path.join(out_dir, "mc_coverage.json"), "w") as fh:
        json.dump(out, fh, indent=2)
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
    return dict(ok=(mism == 0 and verr <= tol), mismatches=mism, max_value_err=verr, rows=rows)


def stress_sparse_reference(cfg: Config) -> dict:
    """One-feature references (r = x except feature 0) in endpoint-fixed mode: phi_0 = f(x) - f(r),
    all others 0, so top-1 = {0} is certified by construction whenever f(x) != f(r)."""
    d, rho, nA = 4, 0.9, 64
    gd = gen_of("sparse", cfg.seed)
    sample = gaussian_sampler(d, rho)
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
    return dict(ok=ok, LA=LA, rows=rows)


def stress_stable_ood(cfg: Config) -> dict:
    """Descriptive: local output variance (a variance-based baseline) vs AIW for a model that is
    flat but shifted away from the data diagonal. Expectation: variance ~ 0 while AIW is not."""
    d, rho, nA, kappa, noise = 3, 0.99, 64, 2.0, 0.05
    gd = gen_of("stable-ood", cfg.seed)
    sample = gaussian_sampler(d, rho)
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
            Zn = lp.Z[:, None, :] + noise * sigma * torch.randn(lp.Z.shape[0], 32, d, generator=gd)
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
               stable_ood=stress_stable_ood(cfg))
    with open(os.path.join(out_dir, "stress.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    for k in ("saturated", "sparse_reference", "counterexample_2d"):
        print(f"[stress] {k}: ok={out[k]['ok']}", flush=True)
    return out


# ============================================================================== runtime
def run_runtime(cfg: Config, out_dir: str) -> list:
    rows = []
    for d in cfg.rt_dims:
        gd = gen_of("rt", cfg.seed, d)
        sample = gaussian_sampler(d, 0.9)
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
        json.dump(rows, fh, indent=2)
    return rows


# ============================================================================== summary
def _tightness(recs: list) -> dict:
    ratios, n_act, c_lp, c_pair, n_topk, c_topk = [], 0, 0, 0, 0, 0
    for r in recs:
        for j in range(r["d"]):
            if r["delta"][j] <= 0.0:
                continue
            n_act += 1
            if r["B_pair"][j] > 1e-9:
                ratios.append(r["G_lp"][j] / r["B_pair"][j])
            c_lp += int(r["sign_lp"][j] != 0)
            c_pair += int(r["sign_pair"][j] != 0)
        if r.get("topk"):
            n_topk += 1
            c_topk += int(r["topk"]["certified"])
    return dict(n_features=n_act,
                median_G_over_B=float(np.median(ratios)) if ratios else None,
                cert_sign_rate_lp=c_lp / n_act if n_act else None,
                cert_sign_rate_pair=c_pair / n_act if n_act else None,
                cert_sign_gain=(c_lp - c_pair) / n_act if n_act else None,
                cert_topk_rate=c_topk / n_topk if n_topk else None)


def summarize(cfg: Config, out_dir: str, rec_path, mc, stress, rt) -> dict:
    S: dict = dict(go={})
    det_ok = all(stress[k]["ok"] for k in ("saturated", "sparse_reference", "counterexample_2d"))
    lp_ok = None
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
        S["deterministic"]["sign_contra"] = sum(r["sign_contra"] for r in feas)
        S["deterministic"]["topk_contra"] = sum(r["topk_contra"] for r in feas)
        det_ok = det_ok and all(S["deterministic"][k]["n_over_tol"] == 0 for k in fields) \
            and S["deterministic"]["sign_contra"] == 0 and S["deterministic"]["topk_contra"] == 0
        S["tightness_overall"] = _tightness(feas)
        S["tightness_by"] = {f"{m}/kappa={k}/nA={n}": _tightness(
            [r for r in feas if r["mode"] == m and r["kappa"] == k and r["n_anchor"] == n])
            for m in ("domain", "fixed") for k in cfg.kappas for n in cfg.n_anchor}
        t = S["tightness_overall"]
        lp_ok = bool((t["median_G_over_B"] is not None and t["median_G_over_B"] <= cfg.go_lp_ratio)
                     or (t["cert_sign_gain"] is not None and t["cert_sign_gain"] >= cfg.go_lp_sign_gain))
        S["transfer"] = dict(  # empirical, not a theorem check
            frozen_phi_outside_rate=float(np.mean([r["transfer_out"] > cfg.tol for r in feas])) if feas else None,
            envelope_violation_rate=float(np.mean([r["VL"] > cfg.tol for r in feas])) if feas else None)
    S["go"]["deterministic_correctness"] = bool(det_ok)
    if mc:
        S["mc"] = {k: v for k, v in mc.items() if "/" in k}
        S["go"]["mc_coverage"] = all(v["ok"] for v in S["mc"].values())
    if lp_ok is not None:
        S["go"]["lp_tightness"] = lp_ok
    if rt:
        S["runtime"] = rt
    S["go"]["PHASE1_GO"] = all(S["go"].values())
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=2)
    print(json.dumps(S["go"], indent=2))
    return S


# ================================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="phase1_out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], choices=["det", "mc", "runtime"])
    args = ap.parse_args()
    cfg = Config.quick() if args.quick else Config()
    cfg.seed = args.seed
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    stress = run_stress(cfg, args.out)
    rec_path = None if "det" in args.skip else run_deterministic(cfg, args.out)
    mc = None if "mc" in args.skip else run_mc_coverage(cfg, args.out)
    rt = None if "runtime" in args.skip else run_runtime(cfg, args.out)
    summarize(cfg, args.out, rec_path, mc, stress, rt)


if __name__ == "__main__":
    main()
