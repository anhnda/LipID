#!/usr/bin/env python3
"""
jnt01_validate_pairwise.py — JNT-01: independent validation of the sharp pairwise Shapley-difference bounds (Q3).

    python scripts/jnt01_validate_pairwise.py --workers 8            # full pre-specified grid (CPU; LPs in HiGHS)
    python scripts/jnt01_validate_pairwise.py --quick --workers 4    # d <= 4, 1 rep

For every instance, Lipschitz level and unordered active pair (a, b) three independent checks are made:
  (1) LP ground truth over ALL vertex pairs (generic finite-metric formulation; does not use the hypercube
      shortest-path reduction) and the edge-only LP of Proposition J: |closed form - LP| / c_ab <= 1e-8.
      Every LP solution is re-verified in raw units against the anchors and all vertex pairs.
  (2) Solver-free attainment (proof-linked): the extremal labelings of the corollary are built on the two faces
      {P_S}, {Q_S}; together with the anchor labels they must be an L-Lipschitz labeling (violation <= 1e-12 relative),
      and the McShane extension G evaluated on ALL 2^m hybrids must give, through the full Shapley formula (not the
      pairwise identity), phi_a(G) - phi_b(G) = closed-form endpoint (relative error <= 1e-10).
  (3) Identity: the LP objective M[a] - M[b] equals the pairwise-identity objective sum_S beta_S (e_P - e_Q).
Also recorded: dominance over the marginal comparison, containment of the true difference when L >= L_V.

Grid (protocols/v12_preregistration.json, JNT-01): d in {2..6} x {domain, fixed} x n in {0, 2, 6, 16, 64} x
{unit, nonuniform} scales x 5 generators x levels {L_A, 1.5 L_A, 3 L_A, L_V, L_cert} x 3 reps, plus ten deterministic
edge cases. Levels with L = 0 (single anchor) are skipped and listed.
Outputs (results_v12/): jnt01_rows.csv.gz, jnt01_summary.csv, jnt01_edgecases.csv, jnt01_table.tex,
jnt01_skips.csv, jnt01_manifest.json. Exit code 1 if any pre-specified tolerance is exceeded.
"""
import argparse
import itertools
import math
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

TOL_LP = 1e-8          # |closed - LP| / c_ab
TOL_LP_VIOL = 1e-9     # LP solution violation / c_ab
TOL_CONSTR = 1e-12     # extremal labeling violation, relative to L * D
TOL_VALUE = 1e-10      # McShane extension value vs closed form, relative to c_ab
DIMS = (2, 3, 4, 5, 6)
NS = (0, 2, 6, 16, 64)
MODELS = ("linear_random", "fa", "stable_ood", "unstable_id", "random_lipschitz")
LEVELS = ("LA", "1.5LA", "3LA", "LV", "cert")


# ================================================================================================ instances
def make_model(name, d, sigma, gen):
    """Returns (f, L_cert). L_cert is a valid global weighted-l1 Lipschitz level independent of anchors/hybrids."""
    import phase1 as P
    from envelope import RandomLipschitz
    from lipid.calibration import L_cert
    if name == "linear_random":
        w = torch.randn(d, generator=gen)
        return (lambda Z: Z @ w), float((w.abs() * sigma).max())
    if name == "random_lipschitz":
        L0 = 1.0
        psi = RandomLipschitz(torch.zeros(d), 1.0, L0, sigma, gen)
        return psi, L0
    a = P.Config().a
    return P.make_model(name, a), L_cert(name, d, sigma, a)


def grid_instance(d, mode, n, scale, model, rep):
    import phase1 as P
    g = P.gen_of("jnt01", d, mode, n, scale, model, rep)
    smp = P.gaussian_sampler(d, 0.6)
    x, r = smp(1, g)[0], smp(1, g)[0]
    C = smp(n, g) if n > 0 else torch.zeros(0, d)
    sigma = torch.ones(d) if scale == "unit" else 0.45 + 1.55 * torch.rand(d, generator=g)
    f, Lc = make_model(model, d, sigma, g)
    A = torch.cat([C, x[None]] + ([r[None]] if mode == "fixed" else []))
    return x, r, A, f(A), sigma, f, Lc


def levels_for(x, r, A, fA, sigma, f, Lc):
    from bounds import all_masks, vertex_points
    from envelope import anchor_lipschitz, max_ratio
    LA = anchor_lipschitz(A, fA, sigma)
    delta = (x - r).abs() / sigma
    act = torch.nonzero(delta > 0).flatten().tolist()
    Z = vertex_points(x, r, act, all_masks(len(act)))
    fZ = f(Z)
    LV = max(LA, max_ratio(Z, fZ, A, fA, sigma), max_ratio(Z, fZ, Z, fZ, sigma))
    return dict(LA=LA, LV=LV, cert=Lc), {"LA": LA, "1.5LA": 1.5 * LA, "3LA": 3.0 * LA, "LV": LV, "cert": Lc}


# ================================================================================================ checks
def check_instance(x, r, A, fA, L, sigma, f=None, *, LV=None, include_lp=True, include_inactive=False):
    """All checks for every unordered pair of players at level L. Returns a list of row dicts."""
    from bounds import shapley_from_vertex_values
    from lipid import pairwise as PW
    from lipid.joint import JointPolytope, labeling_violation, mcshane
    t0 = time.perf_counter()
    res = PW.pairwise_bounds(x, r, A, fA, L, sigma, include_inactive=include_inactive)
    t_closed = time.perf_counter() - t0
    m = res["m"]
    if m < 2:
        return []
    pt = res["pt"]
    P_, Q_ = pt["P"], pt["Q"]
    npair = len(pt["pairs"])
    D = float(res["delta"][res["active"]].sum())
    phi_true = None
    if f is not None:
        phi_true = shapley_from_vertex_values(f(res["Z"]), m)
    jpA = jpE = None
    if include_lp and not include_inactive:
        jpA = JointPolytope(x, r, A, fA, L, sigma, constraints="all_pairs")
        jpE = JointPolytope(x, r, A, fA, L, sigma, constraints="edges")
    rows = []
    for p, (a, b) in enumerate(pt["pairs"]):
        c = float(res["c"][p])
        lo, hi = float(res["pair_lo"][p]), float(res["pair_hi"][p])
        row = dict(a=res["active"][a], b=res["active"][b], m=m, L=float(L), c=c, closed_lo=lo, closed_hi=hi,
                   t_closed_per_pair=t_closed / npair)
        # ---- (1) LPs
        if jpA is not None:
            obj = jpA.M[a] - jpA.M[b]
            ident = float(np.max(np.abs(obj - jpA.pair_diff_obj_closed(a, b))))
            tA = time.perf_counter()
            lA, hA, tlA, thA = jpA.interval(obj)
            tA = time.perf_counter() - tA
            tE = time.perf_counter()
            lE, hE, tlE, thE = jpE.interval(obj)
            tE = time.perf_counter() - tE
            viol = max(max(jpA.violation(tlA).values()), max(jpA.violation(thA).values()),
                       max(jpE.violation(tlE).values()), max(jpE.violation(thE).values()))
            row.update(lpA_lo=lA, lpA_hi=hA, lpE_lo=lE, lpE_hi=hE, ident_err=ident,
                       err_allpairs=max(abs(lA - lo), abs(hA - hi)) / c, err_edges=max(abs(lE - lo), abs(hE - hi)) / c,
                       lp_viol=viol / c, t_lp_allpairs=tA / 2, t_lp_edges=tE / 2)
        # ---- (2) solver-free attainment
        cv, ve, ae = 0.0, 0.0, 0.0
        for side, target in (("lower", lo), ("upper", hi)):
            vP, vQ = PW.extremal_pair_labels(res["ell"], res["h"], P_[p], Q_[p], c, side)
            pts = torch.cat([res["Z"][P_[p]], res["Z"][Q_[p]]])
            vals = torch.cat([vP, vQ])
            lv = labeling_violation(pts, vals, A, fA, float(L), sigma)
            cv = max(cv, lv["points"], lv["points_anchors"]) / max(float(L) * D, 1e-300)
            G = mcshane(torch.cat([pts, A]), torch.cat([vals, fA]), float(L), sigma)
            phiG = shapley_from_vertex_values(G(res["Z"]), m)
            ve = max(ve, abs(float(phiG[a] - phiG[b]) - target) / c)
            ae = max(ae, float((G(A) - fA).abs().max()) / max(float(L) * D, 1e-300))
        row.update(constr_viol=cv, constr_value_err=ve, ext_anchor_err=ae)
        # ---- dominance and containment
        mlo = float(res["lo"][a] - res["hi"][b])
        mhi = float(res["hi"][a] - res["lo"][b])
        row.update(marg_lo=mlo, marg_hi=mhi, dom_viol=max(0.0, (mlo - lo) / c, (hi - mhi) / c))
        if phi_true is not None:
            td = float(phi_true[a] - phi_true[b])
            row.update(true_diff=td, contained=int(lo - 1e-9 * c <= td <= hi + 1e-9 * c),
                       level_valid=int(LV is not None and float(L) >= LV * (1 - 1e-12)))
        rows.append(row)
    return rows


def grid_task(task):
    torch.set_num_threads(1)
    d, mode, n, scale, model, rep = task
    out, skips = [], []
    x, r, A, fA, sigma, f, Lc = grid_instance(d, mode, n, scale, model, rep)
    Ls, levels = levels_for(x, r, A, fA, sigma, f, Lc)
    if Lc < Ls["LV"] * (1 - 1e-9):
        skips.append(dict(task=str(task), level="cert", reason=f"INVALID certificate L_cert={Lc} < L_V={Ls['LV']}"))
    for lev, L in levels.items():
        if not (L > 1e-12):
            skips.append(dict(task=str(task), level=lev, reason=f"L={L} (single anchor: L_A = 0)"))
            continue
        for row in check_instance(x, r, A, fA, L, sigma, f, LV=Ls["LV"]):
            row.update(case="grid", d=d, mode=mode, n_anchor=n, scale=scale, model=model, rep=rep, level=lev,
                       L_A=Ls["LA"], L_V=Ls["LV"], L_cert=Lc)
            out.append(row)
    return out, skips


# ================================================================================================ edge cases
def edge_cases():
    """Deterministic edge cases. Each yields (name, x, r, A, fA, sigma, f, [levels], extra_checks)."""
    import phase1 as P
    from bounds import all_masks, vertex_points
    from envelope import anchor_lipschitz
    g = P.gen_of("jnt01-edge")
    smp = P.gaussian_sampler(4, 0.6)
    w4 = torch.tensor([2.0, -1.5, 0.5, -0.2])
    lin4 = lambda Z: Z @ w4                           # noqa: E731
    x4, r4, C4 = smp(1, g)[0], smp(1, g)[0], smp(16, g)
    s1 = torch.ones(4)
    cases = []

    def add(name, x, r, A, f, sigma, levels, kind="std"):
        """levels: (label, value, absolute) — value is a multiple of L_A unless absolute=True."""
        fA = f(A)
        LA = anchor_lipschitz(A, fA, sigma)
        Ls = [(lab, (float(v) if absolute else LA * float(v))) for lab, v, absolute in levels]
        cases.append(dict(name=name, x=x, r=r, A=A, fA=fA, sigma=sigma, f=f, levels=Ls, LA=LA, kind=kind))

    add("E1_duplicated_anchors", x4, r4, torch.cat([C4, C4[:3], x4[None], x4[None]]), lin4, s1,
        [("LA", 1.0, False), ("2LA", 2.0, False)])
    x5, r5 = smp(1, g)[0], smp(1, g)[0]
    x5 = torch.cat([x5, torch.tensor([0.7])])
    r5 = torch.cat([r5, torch.tensor([0.7])])           # coordinate 4 null
    r5[1] = x5[1]                                        # coordinate 1 null
    w5 = torch.tensor([1.0, 0.8, -0.6, 0.4, 0.3])
    C5 = torch.randn(12, 5, generator=g)
    add("E2_null_players", x5, r5, torch.cat([C5, x5[None]]), lambda Z: Z @ w5, torch.ones(5),
        [("1.5LA", 1.5, False)], kind="null")
    add("E3_L_equals_LA", x4, r4, torch.cat([C4, x4[None]]), lin4, s1, [("LA", 1.0, False)])
    add("E4_L_above_LA", x4, r4, torch.cat([C4, x4[None], r4[None]]), lin4, s1, [("3LA", 3.0, False)])
    xs, rs = torch.ones(4), torch.zeros(4)               # phi = w: both signs of differences
    add("E5_signed_differences", xs, rs, torch.cat([smp(64, g), xs[None], rs[None]]), lin4, s1,
        [("LA", 1.0, False), ("1.5LA", 1.5, False)])
    Z4 = vertex_points(x4, r4, [0, 1, 2, 3], all_masks(4))
    add("E6_anchors_on_hybrids", x4, r4, torch.cat([Z4[::2], C4[:4], x4[None]]), lin4, s1,
        [("LA", 1.0, False), ("2LA", 2.0, False)])
    add("E7_all_hybrids_observed", x4, r4, Z4.clone(), lin4, s1, [("Ltrue", float(w4.abs().max()), True)],
        kind="point")
    add("E8_single_anchor", x4, r4, x4[None].clone(), lin4, s1, [("L=1", 1.0, True)], kind="single")
    sx = torch.tensor([0.05, 5.0, 1.0, 0.5])
    add("E9_scale_ratio_100", x4, r4, torch.cat([C4, x4[None]]), lin4, sx, [("1.5LA", 1.5, False)])
    x2, r2 = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    add("E10_d2_closed_form", x2, r2, torch.cat([torch.randn(5, 2, generator=g), x2[None]]),
        lambda Z: Z[:, 0] + 0.3 * Z[:, 1], torch.ones(2), [("LA", 1.0, False), ("2LA", 2.0, False)], kind="d2")
    return cases


def run_edge_cases():
    from envelope import envelopes
    from lipid import pairwise as PW
    rows, checks = [], []
    for cse in edge_cases():
        for lab, L in cse["levels"]:
            rr = check_instance(cse["x"], cse["r"], cse["A"], cse["fA"], L, cse["sigma"], cse["f"], LV=None)
            for row in rr:
                row.update(case=cse["name"], level=lab, L_A=cse["LA"])
            rows += rr
            name = cse["name"]
            if cse["kind"] == "null":
                act = PW.pairwise_bounds(cse["x"], cse["r"], cse["A"], cse["fA"], L, cse["sigma"])
                full = PW.pairwise_bounds(cse["x"], cse["r"], cse["A"], cse["fA"], L, cse["sigma"],
                                          include_inactive=True)
                pos = {j: jj for jj, j in enumerate(act["active"])}
                null = [j for j in range(cse["x"].numel()) if j not in pos]
                err_m = max(abs(float(full["lo"][j] - act["lo"][pos[j]])) + abs(float(full["hi"][j] - act["hi"][pos[j]]))
                            for j in pos)
                err_n0 = max(abs(float(full["lo"][j])) + abs(float(full["hi"][j])) for j in null)
                err_pair = 0.0
                for p, (a, b) in enumerate(full["pairs"]):
                    if a in pos and b in null:
                        err_pair = max(err_pair, abs(float(full["pair_lo"][p] - act["lo"][pos[a]])),
                                       abs(float(full["pair_hi"][p] - act["hi"][pos[a]])))
                    elif a in null and b in pos:
                        err_pair = max(err_pair, abs(float(full["pair_lo"][p] + act["hi"][pos[b]])),
                                       abs(float(full["pair_hi"][p] + act["lo"][pos[b]])))
                    elif a in null and b in null:
                        err_pair = max(err_pair, abs(float(full["pair_lo"][p])), abs(float(full["pair_hi"][p])))
                scale = L * float(act["delta"].sum())
                for chk, v in (("marginal_with_null_players_equal", err_m), ("null_marginal_is_zero", err_n0),
                               ("pair_with_null_equals_marginal", err_pair)):
                    checks.append(dict(case=name, level=lab, check=chk, value=v / scale, passed=bool(v / scale < 1e-12)))
            if cse["kind"] == "point":
                worst = max((max(abs(r_["closed_lo"] - r_["true_diff"]), abs(r_["closed_hi"] - r_["true_diff"])) / r_["c"]
                             for r_ in rr), default=0.0)
                checks.append(dict(case=name, level=lab, check="point_identified_bounds_equal_truth", value=worst,
                                   passed=bool(worst < 1e-12)))
            if cse["kind"] == "d2":
                res = PW.pairwise_bounds(cse["x"], cse["r"], cse["A"], cse["fA"], L, cse["sigma"])
                Z = res["Z"]
                ell, h = envelopes(Z, cse["A"], cse["fA"], L, cse["sigma"])
                c = L * 2.0
                lo = max(float(ell[1] - h[2]), -c)          # z_{1} = bit 0, z_{2} = bit 1
                hi = min(float(h[1] - ell[2]), c)
                v = max(abs(lo - float(res["pair_lo"][0])), abs(hi - float(res["pair_hi"][0]))) / c
                checks.append(dict(case=name, level=lab, check="d2_beta_equals_1_formula", value=v, passed=bool(v < 1e-13)))
            for chk, col, tol in (("closed_vs_lp_allpairs", "err_allpairs", TOL_LP), ("closed_vs_lp_edges", "err_edges", TOL_LP),
                                  ("attainment_value", "constr_value_err", TOL_VALUE),
                                  ("attainment_labeling", "constr_viol", TOL_CONSTR),
                                  ("dominance_over_marginal", "dom_viol", 1e-12)):
                vals = [r_[col] for r_ in rr if col in r_]
                if vals:
                    checks.append(dict(case=name, level=lab, check=chk, value=float(max(vals)),
                                       passed=bool(max(vals) <= tol)))
    return rows, checks


# ================================================================================================ main
def main():
    from lipid.provenance import base_manifest, write_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v12"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dims = (2, 3, 4) if args.quick else DIMS
    reps = 1 if args.quick else args.reps
    tasks = list(itertools.product(dims, ("domain", "fixed"), NS, ("unit", "nonuniform"), MODELS, range(reps)))
    t0 = time.perf_counter()
    rows, skips = [], []
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for k, (rr, sk) in enumerate(pool.imap_unordered(grid_task, tasks, chunksize=4)):
                rows += rr
                skips += sk
                if k % 200 == 0:
                    print(f"[JNT-01] {k + 1}/{len(tasks)} tasks, {len(rows)} pair rows, {time.perf_counter() - t0:.0f}s",
                          flush=True)
    else:
        for k, t in enumerate(tasks):
            rr, sk = grid_task(t)
            rows += rr
            skips += sk
    erows, checks = run_edge_cases()
    df = pd.DataFrame(rows + erows)
    df.to_csv(os.path.join(args.out, "jnt01_rows.csv.gz"), index=False, compression="gzip", float_format="%.17g")
    pd.DataFrame(skips, columns=["task", "level", "reason"]).to_csv(os.path.join(args.out, "jnt01_skips.csv"), index=False)
    ck = pd.DataFrame(checks)
    ck.to_csv(os.path.join(args.out, "jnt01_edgecases.csv"), index=False)
    G = df[df.case == "grid"]
    S = G.groupby(["d", "mode", "level"]).agg(
        n_pairs=("c", "size"), max_err_allpairs=("err_allpairs", "max"), max_err_edges=("err_edges", "max"),
        n_fail=("err_allpairs", lambda s: int(np.sum((s > TOL_LP) | (G.loc[s.index, "err_edges"] > TOL_LP)))),
        max_lp_viol=("lp_viol", "max"), max_constr_viol=("constr_viol", "max"),
        max_constr_value_err=("constr_value_err", "max"), max_dom_viol=("dom_viol", "max"),
        max_ident_err=("ident_err", "max"),
        containment_valid=("contained", lambda s: float(s[G.loc[s.index, "level_valid"] == 1].mean())
                           if (G.loc[s.index, "level_valid"] == 1).any() else math.nan),
        us_closed=("t_closed_per_pair", lambda s: 1e6 * float(s.median())),
        ms_lp_allpairs=("t_lp_allpairs", lambda s: 1e3 * float(s.median())),
        ms_lp_edges=("t_lp_edges", lambda s: 1e3 * float(s.median()))).reset_index()
    S.to_csv(os.path.join(args.out, "jnt01_summary.csv"), index=False)
    verdict = dict(
        n_pair_comparisons=int(len(df)), n_grid=int(len(G)),
        fail_lp=int(((df.err_allpairs > TOL_LP) | (df.err_edges > TOL_LP)).sum()),
        fail_lp_viol=int((df.lp_viol > TOL_LP_VIOL).sum()), fail_constr=int((df.constr_viol > TOL_CONSTR).sum()),
        fail_value=int((df.constr_value_err > TOL_VALUE).sum()), fail_dom=int((df.dom_viol > 1e-12).sum()),
        fail_ident=int((df.ident_err > 1e-12).sum()),
        fail_containment_valid=int(((df.get("level_valid", 0) == 1) & (df.get("contained", 1) == 0)).sum()),
        fail_edge_checks=int((~ck.passed).sum()) if len(ck) else 0, n_skips=len(skips),
        invalid_certificates=int(sum("INVALID" in s["reason"] for s in skips)))
    verdict["accepted"] = all(v == 0 for k, v in verdict.items() if k.startswith(("fail_", "invalid_")))
    by_d = G.groupby("d").agg(N=("c", "size"), e_all=("err_allpairs", "max"), e_edge=("err_edges", "max"),
                              viol=("lp_viol", "max"), cval=("constr_value_err", "max"),
                              us=("t_closed_per_pair", lambda s: 1e6 * float(s.median())),
                              ms=("t_lp_allpairs", lambda s: 1e3 * float(s.median()))).reset_index()
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{JNT-01: closed-form sharp pairwise bounds versus the finite-hybrid LP over all vertex pairs and the "
             "edge-only LP of Proposition~\\ref{prop:joint} (grid of " + f"{len(G)}" + " pair comparisons plus "
             f"{len(erows)} edge-case comparisons). Errors are normalized by $c_{{ij}}=L(\\delta_i+\\delta_j)$; "
             "\\emph{attain} is the solver-free check: the McShane extension of the extremal labeling, evaluated through "
             "the full Shapley formula, reproduces the closed form. Pre-specified tolerance $10^{-8}$; failures: "
             f"{verdict['fail_lp']} (LP), {verdict['fail_value']} (attainment), {verdict['fail_edge_checks']} (edge-case checks).}}",
             "\\label{tab:jnt01}", "\\small", "\\begin{tabular}{rrrrrrr}", "\\toprule",
             "$d$ & pairs & max err (all-pairs LP) & max err (edge LP) & max LP viol. & max err (attain) & "
             "closed [$\\mu$s] / LP [ms] \\\\", "\\midrule"]
    for _, rr in by_d.iterrows():
        lines.append(f"{int(rr.d)} & {int(rr.N)} & {rr.e_all:.1e} & {rr.e_edge:.1e} & {rr.viol:.1e} & {rr.cval:.1e} & "
                     f"{rr.us:.0f} / {rr.ms:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(args.out, "jnt01_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    write_json(os.path.join(args.out, "jnt01_manifest.json"),
               base_manifest(experiment="JNT-01", verdict=verdict, quick=args.quick, reps=reps,
                             tolerances=dict(lp=TOL_LP, lp_viol=TOL_LP_VIOL, constr=TOL_CONSTR, value=TOL_VALUE),
                             seconds=time.perf_counter() - t0))
    print(S.to_string(index=False))
    print(ck.to_string(index=False))
    print("VERDICT:", verdict)
    sys.exit(0 if verdict["accepted"] else 1)


if __name__ == "__main__":
    main()
