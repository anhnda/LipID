#!/usr/bin/env python3
"""
cert02_diagnostics.py — CERT-02: diagnosing saturation-certificate failures at d = 5, 6 (reviewer Q6).

    python scripts/cert02_diagnostics.py --device cuda                       # seeds 8-11, d 5-6, n 1024/4096
    python scripts/cert02_diagnostics.py --device cuda --seeds 8 --nA 1024   # quicker diagnostic
    python scripts/cert02_diagnostics.py --analyze-only

Uses the Section-6 instances (lipid/instances.py, archived RNG streams) and the relaxation chain of lipid/certdiag.py:
    exact >= ABS (= min(c+, c-) per edge) >= max(GEO, RANGE) >= UNI = max(T3, RANGE)
which splits the gap between the exact normalized AIW and the paper's Proposition-7 certificate into three nonnegative
terms (direction asymmetry, slope magnitude, decoupling) and classifies every saturated-but-uncertified feature.
Diagnostic only (confirmation seeds, descriptive): no claim of a new certificate unless it is a proven lower bound AND
cheaper than the O(|A| 2^m) exact envelope (GEO is O(|A|^2 2^m), so it is not).

Outputs (results_v12/): cert02_features.csv.gz (one row per feature-instance x n x level), cert02_summary.csv,
cert02_by_bins.csv (by D/delta bin and kappa bin), cert02_table.tex, cert02_checks.json, numbers_cert02.csv,
figures_v12/fig_cert02_classes.pdf.
"""
import argparse
import json
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

from envelope import weighted_l1  # noqa: E402
from lipid.certdiag import CLASSES, classify, relaxation_chain  # noqa: E402
from lipid.instances import iter_chains, level_values  # noqa: E402
from lipid.provenance import base_manifest, write_json  # noqa: E402
from lipid.report import Numbers, tex_table  # noqa: E402

KBINS = (1.05, 1.5, 3.0)
KLAB = ("<1.05", "1.05-1.5", "1.5-3", ">=3")
TOLC = 1e-10


def run(args):
    nargs = SimpleNamespace(dims=args.dims, rhos=args.rhos, models=args.models, arms=["data", "prod"], n_x=5, refs=4,
                            block=2048)
    ns = [16, 64, 256, 1024, 4096]
    keep = set(args.nA)
    rows = []
    t0 = time.perf_counter()
    for seed in args.seeds:
        for ch in iter_chains(seed, nargs, ns, args.device):
            for lev in ch["levels"]:
                if lev["n"] not in keep:
                    continue
                Lval = level_values(ch, lev)
                kinds = list(args.levels)
                Lvec = torch.tensor([Lval[k] for k in kinds], dtype=ch["Z"].dtype, device=ch["Z"].device)
                D_AA = weighted_l1(lev["A"], lev["A"], ch["sigma"])
                geo = "none" if lev["n"] > args.geo_max_n else args.geo
                R = relaxation_chain(lev["Dk"], lev["fA"], Lvec, float(lev["L_n"]), float(lev["Omega"]), ch["Sx"],
                                     ch["Px"], ch["W"], ch["dl"], D_AA=D_AA, n_background=lev["n"], geo=geo)
                R = {k: v.detach().cpu().numpy() for k, v in R.items()}
                for kk, kind in enumerate(kinds):
                    for jj, feat in enumerate(ch["active"]):
                        g = lambda key: float(R[key][kk, jj])   # noqa: E731
                        rows.append(dict(seed=seed, model=ch["model"], d=ch["d"], rho_corr=ch["rhoc"], mode=ch["mode"],
                                         anchor_law=ch["arm"], n_anchor=lev["n"], instance_id=ch["instance_id"],
                                         feature_i=feat, L_kind=kind, L_used=float(Lvec[kk]), L_A=float(lev["L_n"]),
                                         kappa=float(R["kappa"][kk]), D_over_delta=float(ch["dod"][jj]),
                                         Dbin=str(ch["dbin"][jj]), rho_exact=g("rho_exact"), rho_abs=g("rho_abs"),
                                         rho_geo=g("rho_geo"), rho_pairbest=g("rho_pairbest"), rho_range=g("rho_range"),
                                         rho_t3=g("rho_t3"), rho_uni=g("rho_uni"), asym=g("asym"),
                                         share_endpoint_binding=g("share_endpoint_binding"),
                                         mean_h_binding=g("mean_h_binding"), mean_slope_binding=g("mean_slope_binding")))
        print(f"[CERT-02] seed {seed}: {len(rows)} rows, {time.perf_counter() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def analyze(df, out, figs):
    df = df.copy()
    df["cls"] = [classify(a, b, c, d) for a, b, c, d in zip(df.rho_exact, df.rho_uni, df.rho_abs, df.rho_pairbest)]
    df["kbin"] = np.asarray(KLAB)[np.digitize(df.kappa, KBINS)]
    pb = df.rho_pairbest
    checks = dict(
        n_rows=int(len(df)),
        v_exact_ge_abs=int((df.rho_abs > df.rho_exact + TOLC).sum()),
        v_abs_ge_pairbest=int((pb > df.rho_abs + TOLC).sum()),
        v_pairbest_ge_uni=int((df.rho_uni > pb + TOLC).sum()),
        v_geo_ge_t3=int((df.rho_t3 > df.rho_geo + TOLC).sum()),
        uncomputed_geo_in_target=int(((df.rho_exact > 0.8) & (df.rho_uni <= 0.8) & df.rho_pairbest.isna()).sum()))
    sat = df[df.rho_exact > 0.8].copy()
    unc = sat[sat.rho_uni <= 0.8].copy()
    unc["term_sign"] = unc.rho_exact - unc.rho_abs
    unc["term_slope"] = unc.rho_abs - unc.rho_pairbest
    unc["term_decouple"] = unc.rho_pairbest - unc.rho_uni
    keys = ["d", "mode", "anchor_law", "n_anchor", "L_kind"]

    def summarize(keys_):
        a = df.groupby(keys_).agg(n=("rho_exact", "size"), p_sat=("rho_exact", lambda s: float(np.mean(s > 0.8))),
                                  p_uni=("rho_uni", lambda s: float(np.mean(s > 0.8)))).reset_index()
        b = sat.groupby(keys_).cls.value_counts(normalize=True).unstack(fill_value=0.0).reset_index()
        c = unc.groupby(keys_).agg(n_uncertified=("rho_exact", "size"), term_sign=("term_sign", "mean"),
                                   term_slope=("term_slope", "mean"), term_decouple=("term_decouple", "mean"),
                                   endpoint_binding=("share_endpoint_binding", "mean"),
                                   asym=("asym", "mean")).reset_index()
        return a.merge(b, on=keys_, how="left").merge(c, on=keys_, how="left")

    S = summarize(keys)
    S.to_csv(os.path.join(out, "cert02_summary.csv"), index=False)
    pd.concat([summarize(keys + ["Dbin"]).assign(binning="Dbin"), summarize(keys + ["kbin"]).assign(binning="kappa"),
               summarize(keys + ["model"]).assign(binning="model")]).to_csv(os.path.join(out, "cert02_by_bins.csv"),
                                                                            index=False)
    rows = []
    for _, r in S[(S.n_anchor == S.n_anchor.max())].sort_values(["L_kind", "anchor_law", "mode", "d"]).iterrows():
        rows.append([r.L_kind, r.anchor_law, r["mode"], int(r.d), f"{r.p_sat:.2f}", f"{r.p_uni:.2f}"]
                    + [f"{r.get(c, 0.0):.2f}" if np.isfinite(r.get(c, np.nan)) else "--"
                       for c in ("certified", "sign_asymmetry", "slope_magnitude", "decoupling")]
                    + [f"{r.term_sign:.2f}/{r.term_slope:.2f}/{r.term_decouple:.2f}" if np.isfinite(r.term_sign) else "--"])
    cap = ("CERT-02: where the Proposition-7 certificate loses saturation ($n=" f"{int(S.n_anchor.max())}" "$, seeds "
           f"{','.join(str(s) for s in sorted(df.seed.unique()))}, descriptive). Among saturated features ($\\rho>0.8$): share "
           "certified by UNI, and shares first lost by direction asymmetry ($\\rho_{\\mathrm{abs}}\\le .8$), slope "
           "magnitude ($\\max(\\rho_{\\mathrm{geo}},\\rho_{\\mathrm{range}})\\le .8$) or decoupling. Last column: mean gap "
           "terms (sign/slope/decoupling) among uncertified saturated features; they add up to $\\rho-\\rho_{\\mathrm{uni}}$. "
           f"Chain-order violations: {checks['v_exact_ge_abs'] + checks['v_abs_ge_pairbest'] + checks['v_pairbest_ge_uni'] + checks['v_geo_ge_t3']}.")
    with open(os.path.join(out, "cert02_table.tex"), "w") as fh:
        fh.write(tex_table(["level", "law", "mode", "$d$", "$p_{\\mathrm{sat}}$", "UNI cert.", "certified", "asym.",
                            "slope", "decoupl.", "gap terms"], rows, cap, "tab:cert02", align="lllrrrrrrrl",
                           size="\\scriptsize"))
    with open(os.path.join(out, "cert02_checks.json"), "w") as fh:
        json.dump(checks, fh, indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = S[(S.n_anchor == S.n_anchor.max()) & (S.anchor_law == "data")]
    lv = sorted(sel.L_kind.unique())
    fig, axes = plt.subplots(1, len(lv), figsize=(4.8 * len(lv), 3.2), sharey=True, squeeze=False)
    cols = dict(certified="C2", sign_asymmetry="C3", slope_magnitude="C1", decoupling="C0")
    for ax, L in zip(axes[0], lv):
        v = sel[sel.L_kind == L].sort_values(["mode", "d"])
        lab = [f"{m}\nd={d}" for m, d in zip(v["mode"], v.d)]
        bottom = np.zeros(len(v))
        for c in ("certified", "sign_asymmetry", "slope_magnitude", "decoupling"):
            h = v[c].fillna(0.0).to_numpy() if c in v else np.zeros(len(v))
            ax.bar(range(len(v)), h, bottom=bottom, color=cols[c], label=c.replace("_", " "))
            bottom += h
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(lab, fontsize=7)
        ax.set_title(f"level {L}, data anchors")
    axes[0][0].set_ylabel("share of saturated features")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, "fig_cert02_classes.pdf"))
    N = Numbers("CERT-02", out)
    for _, r in S[(S.n_anchor == S.n_anchor.max()) & (S.anchor_law == "data") & (S["mode"] == "domain")].iterrows():
        for c in CLASSES[2:]:
            if c in r and np.isfinite(r[c]):
                N.add(f"{r.L_kind}.d{int(r.d)}.{c}", r[c], f"share of saturated features first lost by {c}",
                      "cert02_summary.csv")
    N.write()
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 9, 10, 11])
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6, 0.9, 0.99])
    ap.add_argument("--models", nargs="+", default=["linear", "fa", "stable_ood", "unstable_id"])
    ap.add_argument("--nA", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--levels", nargs="+", default=["oracle", "cert"])
    ap.add_argument("--geo", choices=["none", "target", "all"], default="target")
    ap.add_argument("--geo-max-n", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v12"))
    ap.add_argument("--figs", default=os.path.join(ROOT, "figures_v12"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "cert02_features.csv.gz")
    t0 = time.perf_counter()
    df = pd.read_csv(path) if args.analyze_only else run(args)
    if not args.analyze_only:
        df.to_csv(path, index=False, compression="gzip", float_format="%.17g")
    checks = analyze(df, args.out, args.figs)
    write_json(os.path.join(args.out, "cert02_manifest.json"),
               base_manifest(experiment="CERT-02", args=vars(args), checks=checks, seconds=time.perf_counter() - t0))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
