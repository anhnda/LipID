#!/usr/bin/env python3
"""
jnt03_absolute_lp.py — JNT-03: exact absolute-ranking certificates from joint constraints (reviewer Q3, secondary).

    python scripts/jnt03_absolute_lp.py --device cuda                 # pre-specified subset (protocol JNT-03)
    python scripts/jnt03_absolute_lp.py --analyze-only                # rebuild tables from results_v12/jnt03_rows.csv.gz

Exact rule (lipid/pairwise.abs_exact_cert): |phi_i| > |phi_j| for every g in F_L(A) iff
    [min D > 0 and min S > 0]  or  [max D < 0 and max S < 0],    D = phi_i - phi_j (closed form), S = phi_i + phi_j (LP).
Compared with Proposition 2 (marginal, sufficient). In the (+,+) quadrant min(|phi_i| - |phi_j|) = min(min D, min S),
which is cross-checked against hypercube_lp.HypercubeLP.abs_margin (existing <= 4-LP method) on every
--crosscheck-every-th pair. The closed-form D bounds are also re-checked against the edge LP (--no-diff-lp to skip).

Subset (fixed by index before results): seeds 8-11, x_id in {0, 1} (8 of 20 query/reference pairs per block),
n in {256, 4096}, levels cert and oracle, all d, rho_c, generators, laws and modes.
Outputs: results_v12/jnt03_rows.csv.gz, jnt03_summary.csv, jnt03_table.tex, numbers_jnt03.csv, jnt03_manifest.json.
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from lipid import pairwise as PW  # noqa: E402
from lipid.instances import iter_chains, level_values  # noqa: E402
from lipid.joint import JointPolytope  # noqa: E402
from lipid.provenance import base_manifest, write_json  # noqa: E402
from lipid.report import Numbers, across_seeds, spread, tex_table  # noqa: E402

TOL = PW.SIGN_TOL


def run(args):
    import hypercube_lp as HL
    nargs = SimpleNamespace(dims=args.dims, rhos=args.rhos, models=args.models, arms=["data", "prod"], n_x=5, refs=4,
                            block=2048)
    ns = [16, 64, 256, 1024, 4096]
    keep_n = set(args.nA)
    rows, k_pair, n_chain = [], 0, 0
    t_start = time.perf_counter()
    for seed in args.seeds:
        for ch in iter_chains(seed, nargs, ns, args.device, select_x=lambda d, rh, xi: xi in args.x_ids):
            if ch["m"] < 2:
                continue
            n_chain += 1
            if args.max_chains and n_chain > args.max_chains:
                break
            x, r, sigma = ch["x"], ch["r"], ch["sigma"]
            phi = ch["phi"].detach().cpu().numpy()
            act = ch["active"]
            for lev in ch["levels"]:
                if lev["n"] not in keep_n:
                    continue
                Lval = level_values(ch, lev)
                for kind in args.levels:
                    L = Lval[kind]
                    t0 = time.perf_counter()
                    res = PW.pairwise_bounds(x, r, lev["A"], lev["fA"], L, sigma)
                    t_closed = time.perf_counter() - t0
                    jp = JointPolytope(x.cpu(), r.cpu(), lev["A"].cpu(), lev["fA"].cpu(), L, sigma.cpu())
                    c = res["c"].cpu()
                    am = PW.abs_marginal_cert(res["lo"], res["hi"], res["pt"], res["c"], TOL).cpu().numpy()
                    hl = None
                    for p, (a, b) in enumerate(res["pt"]["pairs"]):
                        cc = float(c[p])
                        dlo, dhi = float(res["pair_lo"][p]), float(res["pair_hi"][p])
                        n0, s0 = jp.n_lp, jp.lp_seconds
                        slo, shi, _, _ = jp.pair_sum(a, b)
                        row = dict(seed=seed, model=ch["model"], d=ch["d"], rho_corr=ch["rhoc"], mode=ch["mode"],
                                   anchor_law=ch["arm"], n_anchor=lev["n"], L_kind=kind, L_used=L,
                                   instance_id=ch["instance_id"], m_active=ch["m"], feat_i=act[a], feat_j=act[b], c_ij=cc,
                                   dlo=dlo, dhi=dhi, slo=slo, shi=shi,
                                   true_diff=float(phi[a] - phi[b]), true_sum=float(phi[a] + phi[b]),
                                   abs_marg=int(am[p]),
                                   abs_exact=int(PW.abs_exact_cert(np.array(dlo), np.array(dhi), np.array(slo),
                                                                   np.array(shi), cc, TOL)),
                                   signed_pair=int((dlo / cc > TOL)) - int((dhi / cc < -TOL)),
                                   t_closed_s=t_closed / len(res["pt"]["pairs"]),
                                   t_lp_sum_s=jp.lp_seconds - s0)
                        row["contained_sum"] = int(slo - 1e-9 * cc <= row["true_sum"] <= shi + 1e-9 * cc)
                        if args.diff_lp:
                            dl_lp, dh_lp, _, _ = jp.pair_diff(a, b)
                            row.update(dlo_lp=dl_lp, dhi_lp=dh_lp, err_diff_lp=max(abs(dl_lp - dlo), abs(dh_lp - dhi)) / cc)
                        if args.crosscheck_every and k_pair % args.crosscheck_every == 0:
                            if hl is None:
                                hl = HL.HypercubeLP(x.cpu(), r.cpu(), lev["A"].cpu(), lev["fA"].cpu(), L, sigma.cpu())
                            pc = hl.abs_margin(act[a], act[b], tol=0.0)
                            if row["abs_exact"] == 1:
                                expected = min(dlo, slo) if (dlo > 0 and slo > 0) else min(-dhi, -shi)
                            else:
                                expected = np.nan
                            row.update(hl_margin=pc.margin, hl_kind=pc.kind,
                                       hl_err=(abs(pc.margin - expected) / cc if pc.kind == "exact"
                                               and np.isfinite(expected) else np.nan),
                                       hl_disagree=int((pc.margin > 1e-5 * cc and row["abs_exact"] != 1)
                                                       or (row["abs_exact"] == 1 and pc.margin < -1e-9 * cc)))
                        rows.append(row)
                        k_pair += 1
        print(f"[JNT-03] seed {seed} done: {len(rows)} pair rows, {time.perf_counter() - t_start:.0f}s", flush=True)
    return pd.DataFrame(rows)


def top1_abs(g: pd.DataFrame, col: str) -> int:
    feats = sorted(set(g.feat_i) | set(g.feat_j))
    pos = {f: k for k, f in enumerate(feats)}
    G = np.zeros((len(feats), len(feats)), dtype=bool)
    for a, b, v in zip(g.feat_i, g.feat_j, g[col]):
        if v > 0:
            G[pos[a], pos[b]] = True
        elif v < 0:
            G[pos[b], pos[a]] = True
    return int(PW.top1(G) >= 0)


def analyze(df: pd.DataFrame, out: str):
    keys = ["seed", "model", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "L_kind", "instance_id"]
    inst = df.groupby(keys).apply(lambda g: pd.Series(dict(
        n_pairs=len(g), n_abs_marg=int((g.abs_marg != 0).sum()), n_abs_exact=int((g.abs_exact != 0).sum()),
        top1_abs_marg=top1_abs(g, "abs_marg"), top1_abs_exact=top1_abs(g, "abs_exact"),
        t_closed=float(g.t_closed_s.sum()), t_lp=float(g.t_lp_sum_s.sum())))).reset_index()
    SL = inst.groupby(["seed", "d", "mode", "anchor_law", "n_anchor", "L_kind"]).agg(
        n_pairs=("n_pairs", "sum"), n_abs_marg=("n_abs_marg", "sum"), n_abs_exact=("n_abs_exact", "sum"),
        top1_abs_marg=("top1_abs_marg", "mean"), top1_abs_exact=("top1_abs_exact", "mean"),
        t_closed=("t_closed", "sum"), t_lp=("t_lp", "sum")).reset_index()
    SL["rate_abs_marg"] = SL.n_abs_marg / SL.n_pairs
    SL["rate_abs_exact"] = SL.n_abs_exact / SL.n_pairs
    SL["gain_abs"] = SL.rate_abs_exact - SL.rate_abs_marg
    SL["lp_overhead_ratio"] = SL.t_lp / SL.t_closed.clip(lower=1e-12)
    SL.to_csv(os.path.join(out, "jnt03_seed_level.csv"), index=False)
    S = across_seeds(SL, ["d", "mode", "anchor_law", "n_anchor", "L_kind"],
                     ["rate_abs_marg", "rate_abs_exact", "gain_abs", "top1_abs_marg", "top1_abs_exact",
                      "lp_overhead_ratio"])
    S.to_csv(os.path.join(out, "jnt03_summary.csv"), index=False)
    checks = dict(n_rows=len(df), exact_weaker_than_prop2=int(((df.abs_marg != 0) & (df.abs_exact != df.abs_marg)).sum()),
                  sum_not_contained_valid=int((df.contained_sum == 0).sum()),
                  max_err_diff_lp=float(df.err_diff_lp.max()) if "err_diff_lp" in df else None,
                  hl_disagree=int(df.hl_disagree.sum()) if "hl_disagree" in df else None,
                  max_hl_err=float(np.nanmax(df.hl_err)) if "hl_err" in df and df.hl_err.notna().any() else None)
    rows = []
    for _, r in S[(S.anchor_law == "data") & (S["mode"] == "domain")].sort_values(["L_kind", "n_anchor", "d"]).iterrows():
        rows.append([r.L_kind, int(r.n_anchor), int(r.d), spread(r, "rate_abs_marg"), spread(r, "rate_abs_exact"),
                     spread(r, "gain_abs", 2, pct=True), spread(r, "top1_abs_marg"), spread(r, "top1_abs_exact"),
                     spread(r, "lp_overhead_ratio", 0)])
    cap = ("JNT-03: absolute-attribution ranking. Proposition~2 (marginal, sufficient) versus the exact joint certificate "
           "from sharp bounds on $\\phi_i-\\phi_j$ (closed form) and $\\phi_i+\\phi_j$ (LP); data anchors, domain mode, "
           "confirmation seeds 8--11, subset fixed by index. Seed means (min--max). Overhead = LP time / closed-form time. "
           f"Checks: exact never weaker than Prop.~2 ({checks['exact_weaker_than_prop2']} violations); "
           f"cross-check with the 4-LP $|\\phi|$-margin method disagreements: {checks['hl_disagree']}.")
    with open(os.path.join(out, "jnt03_table.tex"), "w") as fh:
        fh.write(tex_table(["level", "$n$", "$d$", "Prop.~2", "exact", "gain [pp]", "top-1 Prop.~2", "top-1 exact",
                            "LP overhead"], rows, cap, "tab:jnt03", align="lrrllllll", size="\\scriptsize"))
    N = Numbers("JNT-03", out)
    for _, r in S[(S.anchor_law == "data") & (S["mode"] == "domain") & (S.n_anchor == 4096)].iterrows():
        N.add(f"{r.L_kind}.d{int(r.d)}.gain_abs_pp", r.gain_abs_mean, "exact minus Prop. 2 absolute-pair rate (pp)",
              "jnt03_summary.csv", pct=True)
    N.add("checks.exact_weaker", str(checks["exact_weaker_than_prop2"]), "must be 0", "jnt03_manifest.json")
    N.write()
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 9, 10, 11])
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6, 0.9, 0.99])
    ap.add_argument("--models", nargs="+", default=["linear", "fa", "stable_ood", "unstable_id"])
    ap.add_argument("--x-ids", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--nA", type=int, nargs="+", default=[256, 4096])
    ap.add_argument("--levels", nargs="+", default=["cert", "oracle"])
    ap.add_argument("--no-diff-lp", dest="diff_lp", action="store_false")
    ap.add_argument("--crosscheck-every", type=int, default=25)
    ap.add_argument("--max-chains", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v12"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "jnt03_rows.csv.gz")
    t0 = time.perf_counter()
    if args.analyze_only:
        df = pd.read_csv(path)
    else:
        df = run(args)
        df.to_csv(path, index=False, compression="gzip", float_format="%.17g")
    checks = analyze(df, args.out)
    write_json(os.path.join(args.out, "jnt03_manifest.json"),
               base_manifest(experiment="JNT-03", args=vars(args), checks=checks, seconds=time.perf_counter() - t0))
    print(checks)


if __name__ == "__main__":
    main()
