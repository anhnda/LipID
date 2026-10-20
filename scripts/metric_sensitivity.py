#!/usr/bin/env python3
"""
metric_sensitivity.py — MET-01: edgewise aggregate vs exact finite-hybrid LP under weighted l1 and l2.

    python scripts/metric_sensitivity.py            # full grid (CPU; LPs run in scipy HiGHS)
    python scripts/metric_sensitivity.py --quick    # small grid

Grid: p in {1,2} x d in {2,3,4,5} x n_anchor in {4,16,64} x mode {domain, fixed} x kappa = L/L_A^{(p)} in {1,1.5,3}
x scale law {unit, nonuniform} x generator {linear, fa} x reps. Anchors / x / r from N(0, Sigma_.6).
Under l1 the two must agree (Theorem 1); under l2 the edgewise interval is only an outer interval.
Outputs: results/metric_counterexamples.csv (appendix-B example + every feature-instance),
         results/metric_sensitivity.csv (summary by p, d), results/metric_sensitivity.tex.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import lipid  # noqa: E402
import phase1 as P  # noqa: E402
from lipid import metric_lp as ML  # noqa: E402
from lipid.provenance import base_manifest, write_json  # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    dims = (2, 3) if args.quick else (2, 3, 4, 5)
    nanchors = (4, 16) if args.quick else (4, 16, 64)
    reps = 2 if args.quick else args.reps
    os.makedirs(args.out, exist_ok=True)
    rows = []
    ce = ML.euclidean_counterexample()
    for p, v in ce.items():
        rows.append(dict(kind="appendix_B", p=int(p[1]), d=2, n_anchor=0, mode="explicit", kappa=1.0, scale="unit",
                         model="zero", rep=0, feature=0, lp_lo=v["lp_lower_phi1"], lp_hi=v["lp_upper_phi1"],
                         edge_lo=v["edgewise_lower_phi1"], edge_hi=v["edgewise_upper_phi1"], Ldelta=1.0,
                         lp_seconds=np.nan, edge_seconds=np.nan, lp_viol=np.nan))
    for p in (1, 2):
        for d in dims:
            for n in nanchors:
                for mode in ("domain", "fixed"):
                    for kappa in (1.0, 1.5, 3.0):
                        for scale in ("unit", "nonuniform"):
                            for model in ("linear", "fa"):
                                f = P.make_model(model, P.Config().a)
                                for rep in range(reps):
                                    g = P.gen_of("met01", p, d, n, mode, kappa, scale, model, rep)
                                    smp = P.gaussian_sampler(d, 0.6)
                                    C, x, r = smp(n, g), smp(1, g)[0], smp(1, g)[0]
                                    sigma = (torch.ones(d) if scale == "unit" else 0.3 + 2.0 * torch.rand(d, generator=g))
                                    A = torch.cat([C, x[None]] + ([r[None]] if mode == "fixed" else []))
                                    fA = f(A)
                                    LA = ML.anchor_L_p(A, fA, sigma, p)
                                    L = kappa * LA
                                    e = ML.edgewise_interval(x, r, A, fA, L, sigma, p)
                                    lpv = ML.lp_interval(x, r, A, fA, L, sigma, p)
                                    for jj, j in enumerate(lpv["active"]):
                                        rows.append(dict(kind="random", p=p, d=d, n_anchor=n, mode=mode, kappa=kappa,
                                                         scale=scale, model=model, rep=rep, feature=j,
                                                         lp_lo=lpv["lo"][jj], lp_hi=lpv["hi"][jj],
                                                         edge_lo=e["lo"][jj], edge_hi=e["hi"][jj],
                                                         Ldelta=float(L * lpv["delta"][j]),
                                                         lp_seconds=lpv["seconds"] / len(lpv["active"]),
                                                         edge_seconds=e["seconds"] / len(lpv["active"]),
                                                         lp_viol=lpv["max_constraint_violation"]))
                print(f"[MET-01] p={p} d={d} done ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df["gap_upper"] = df.edge_hi - df.lp_hi
    df["gap_lower"] = df.lp_lo - df.edge_lo
    df["gap_norm"] = (df.gap_upper + df.gap_lower) / (2 * df.Ldelta)
    df["outer_valid"] = (df.gap_upper >= -1e-7) & (df.gap_lower >= -1e-7)
    df.to_csv(os.path.join(args.out, "metric_counterexamples.csv"), index=False)
    R = df[df.kind == "random"]
    S = R.groupby(["p", "d"]).agg(n_feature_instances=("gap_norm", "size"),
                                  max_abs_gap=("gap_upper", lambda s: float(np.max(np.abs(s)))),
                                  median_gap_norm=("gap_norm", "median"),
                                  q90_gap_norm=("gap_norm", lambda s: float(s.quantile(.9))),
                                  max_gap_norm=("gap_norm", "max"),
                                  frac_gap_gt_1e7=("gap_norm", lambda s: float(np.mean(s > 1e-7))),
                                  all_outer_valid=("outer_valid", "all"),
                                  median_lp_ms=("lp_seconds", lambda s: 1e3 * float(s.median())),
                                  median_edge_ms=("edge_seconds", lambda s: 1e3 * float(s.median())),
                                  max_lp_violation=("lp_viol", "max")).reset_index()
    S.to_csv(os.path.join(args.out, "metric_sensitivity.csv"), index=False)
    l1 = S[S.p == 1]
    if (l1.max_abs_gap > 1e-6).any():
        print("WARNING: l1 edgewise != LP beyond 1e-6 — check solver tolerance / bug")
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Edgewise envelope aggregate versus the exact finite-hybrid LP. Gap = (edgewise width $-$ LP width)"
             "$/(2L\\delta_i)$ over feature-instances; $\\ell_1$ gaps are solver round-off (Theorem~1), $\\ell_2$ "
             "edgewise intervals are valid outer intervals but not sharp. Appendix~B example: $\\ell_2$ edgewise upper "
             f"{ce['l2']['edgewise_upper_phi1']:.6f}, LP {ce['l2']['lp_upper_phi1']:.6f}; $\\ell_1$ both "
             f"{ce['l1']['lp_upper_phi1']:.6f}.}}", "\\label{tab:metric}", "\\small",
             "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Metric & $d$ & $N$ & median gap & q90 gap & max gap & LP ms & edge ms \\\\", "\\midrule"]
    for _, r in S.iterrows():
        lines.append(f"$\\ell_{int(r.p)}$ & {int(r.d)} & {int(r.n_feature_instances)} & {r.median_gap_norm:.2e} & "
                     f"{r.q90_gap_norm:.2e} & {r.max_gap_norm:.2e} & {r.median_lp_ms:.1f} & {r.median_edge_ms:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(args.out, "metric_sensitivity.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    write_json(os.path.join(args.out, "metric_sensitivity_manifest.json"), base_manifest(quick=args.quick, reps=reps))
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
