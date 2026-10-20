#!/usr/bin/env python3
"""
jnt02_analyze.py — JNT-02 tables and figures: how much ranking information sharp pairwise bounds recover (Q3).

    python scripts/jnt02_analyze.py --run explore=runs/pairs_explore --run confirm=runs/pairs_confirm \
                                    [--run stat01=runs/pairs_stat01] [--main confirm]

Reads pairs_seed*/inst_seed* written by scripts/run_pairwise.py, one seed at a time (pair tables are aggregated to
cells immediately; nothing is pooled across seeds before the seed-level summary). Partitions are reported separately;
--main selects the partition used in the main-text table/figure (default: stat01 if present, else confirm).

Outputs (results_v12/, figures_v12/):
  jnt02_cells.csv            per partition x seed x d x rho_c x mode x law x n x L_kind x generator: counts
  jnt02_seed_level.csv       per seed x d x mode x law x n x L_kind (pooled over rho_c, generators): rates, gains,
                             instance-level gain distribution (share of zero-gain instances, mean, q90, max), top-1
  jnt02_summary.csv          across seeds within each partition (mean, min, max; t interval if >= 10 seeds)
  jnt02_main.tex             main table: n = 4096 and 256, data anchors, domain mode, L_cert and L*
  jnt02_subgroups.tex        supplementary: every d x mode x law x level at n = 4096
  jnt02_by_generator.csv     generator-level seed summaries (supplementary)
  jnt02_integrity.json       containment at valid L and dominance (must be all zero)
  fig_jnt02_certification.pdf   certification rate vs n (pairwise solid, marginal dashed), by d, L_cert and L*
  numbers_jnt02.csv          numbers quoted in the text
Instances with fewer than two active features carry no pair and are excluded from top-1 rates.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lipid.io import read_table, seeds_in  # noqa: E402
from lipid.report import Numbers, across_seeds, savefig, spread, tex_table  # noqa: E402

CELL = ["seed", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "L_kind", "model"]
SEEDKEY = ["partition", "seed", "d", "mode", "anchor_law", "n_anchor", "L_kind"]
VALID = ("LV", "oracle", "cert")
PCOLS = ["seed", "model", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "L_kind", "cert_pair", "cert_marg",
         "cert_new", "pair_rho", "marg_rho", "contained", "v_dom"]
ICOLS = ["seed", "model", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "L_kind", "m_active", "n_pairs",
         "n_cert_marg", "n_cert_pair", "n_cert_new", "any_new", "top1_marg", "top1_pair", "top1_pair_correct",
         "n_abs_marg"]


def load_partition(tag, run_dir):
    cells, insts = [], []
    for s in seeds_in(run_dir, "pairs"):
        P = read_table(os.path.join(run_dir, f"pairs_seed{s}"), columns=PCOLS)
        P["cp"] = (P.cert_pair != 0).astype(np.int64)
        P["cm"] = (P.cert_marg != 0).astype(np.int64)
        P["valid"] = P.L_kind.astype(str).isin(VALID)
        P["nc_valid"] = ((P.contained == 0) & P.valid).astype(np.int64)
        g = P.groupby(CELL, observed=True)
        c = g.agg(n_pairs=("cp", "size"), n_cert_pair=("cp", "sum"), n_cert_marg=("cm", "sum"),
                  n_new=("cert_new", "sum"), sum_pair_rho=("pair_rho", "sum"), sum_marg_rho=("marg_rho", "sum"),
                  n_not_contained_valid=("nc_valid", "sum"), n_v_dom=("v_dom", "sum")).reset_index()
        c["partition"] = tag
        cells.append(c)
        I = read_table(os.path.join(run_dir, f"inst_seed{s}"), columns=ICOLS)
        I["partition"] = tag
        insts.append(I)
        print(f"[JNT-02] {tag}: seed {s} loaded ({len(P)} pair rows)", flush=True)
    if not cells:
        raise SystemExit(f"no pairs_seed* in {run_dir}")
    return pd.concat(cells, ignore_index=True), pd.concat(insts, ignore_index=True)


def seed_level(cells, insts):
    for c in ("mode", "anchor_law", "L_kind", "model"):
        cells[c] = cells[c].astype(str)
        insts[c] = insts[c].astype(str)
    a = cells.groupby(SEEDKEY).agg(n_pairs=("n_pairs", "sum"), n_cert_pair=("n_cert_pair", "sum"),
                                   n_cert_marg=("n_cert_marg", "sum"), n_new=("n_new", "sum"),
                                   sum_pair_rho=("sum_pair_rho", "sum"), sum_marg_rho=("sum_marg_rho", "sum")
                                   ).reset_index()
    a["rate_pair"] = a.n_cert_pair / a.n_pairs
    a["rate_marg"] = a.n_cert_marg / a.n_pairs
    a["gain"] = a.rate_pair - a.rate_marg
    a["mean_pair_rho"] = a.sum_pair_rho / a.n_pairs
    a["mean_marg_rho"] = a.sum_marg_rho / a.n_pairs
    I = insts[insts.m_active >= 2].copy()
    I["inst_gain"] = I.n_cert_new / I.n_pairs
    I["t1p"] = (I.top1_pair >= 0).astype(float)
    I["t1m"] = (I.top1_marg >= 0).astype(float)
    b = I.groupby(SEEDKEY).agg(n_inst=("inst_gain", "size"), frac_any_new=("any_new", "mean"),
                               share_zero_gain=("inst_gain", lambda s: float(np.mean(s == 0))),
                               mean_inst_gain=("inst_gain", "mean"),
                               q90_inst_gain=("inst_gain", lambda s: float(np.quantile(s, 0.9))),
                               max_inst_gain=("inst_gain", "max"), top1_pair=("t1p", "mean"),
                               top1_marg=("t1m", "mean")).reset_index()
    out = a.merge(b, on=SEEDKEY, how="left")
    out["top1_gain"] = out.top1_pair - out.top1_marg
    return out


METRICS = ["rate_marg", "rate_pair", "gain", "frac_any_new", "share_zero_gain", "mean_inst_gain", "q90_inst_gain",
           "top1_marg", "top1_pair", "top1_gain", "mean_pair_rho", "mean_marg_rho"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="partition=run_dir (repeatable)")
    ap.add_argument("--main", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v12"))
    ap.add_argument("--figs", default=os.path.join(ROOT, "figures_v12"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parts = dict(r.split("=", 1) for r in args.run)
    main_tag = args.main or ("stat01" if "stat01" in parts else ("confirm" if "confirm" in parts else list(parts)[0]))
    C, I = [], []
    for tag, d in parts.items():
        c, i = load_partition(tag, d)
        C.append(c)
        I.append(i)
    cells, insts = pd.concat(C, ignore_index=True), pd.concat(I, ignore_index=True)
    cells.to_csv(os.path.join(args.out, "jnt02_cells.csv"), index=False)
    SL = seed_level(cells, insts)
    SL.to_csv(os.path.join(args.out, "jnt02_seed_level.csv"), index=False)
    S = across_seeds(SL, ["partition", "d", "mode", "anchor_law", "n_anchor", "L_kind"], METRICS)
    S.to_csv(os.path.join(args.out, "jnt02_summary.csv"), index=False)

    # generator-level (supplementary)
    cg = cells.groupby(["partition", "seed", "model", "d", "mode", "anchor_law", "n_anchor", "L_kind"]).agg(
        n_pairs=("n_pairs", "sum"), n_cert_pair=("n_cert_pair", "sum"), n_cert_marg=("n_cert_marg", "sum")).reset_index()
    cg["rate_pair"], cg["rate_marg"] = cg.n_cert_pair / cg.n_pairs, cg.n_cert_marg / cg.n_pairs
    cg["gain"] = cg.rate_pair - cg.rate_marg
    across_seeds(cg, ["partition", "model", "d", "mode", "anchor_law", "n_anchor", "L_kind"],
                 ["rate_marg", "rate_pair", "gain"]).to_csv(os.path.join(args.out, "jnt02_by_generator.csv"), index=False)

    integ = dict(not_contained_valid_L=int(cells.n_not_contained_valid.sum()), v_dom=int(cells.n_v_dom.sum()),
                 top1_pair_wrong_valid_L=int(((insts.top1_pair_correct == 0) &
                                              insts.L_kind.astype(str).isin(VALID)).sum()),
                 partitions={t: sorted(int(s) for s in cells[cells.partition == t].seed.unique()) for t in parts})
    with open(os.path.join(args.out, "jnt02_integrity.json"), "w") as fh:
        json.dump(integ, fh, indent=1)

    # ---------------------------------------------------------------- main table
    M = S[(S.partition == main_tag) & (S.anchor_law == "data") & (S["mode"] == "domain")]
    rows = []
    for n in (4096, 256):
        for lev, lab in (("cert", "$L_{\\mathrm{cert}}$"), ("oracle", "$L^\\star$")):
            for d in sorted(M.d.unique()):
                r = M[(M.n_anchor == n) & (M.L_kind == lev) & (M.d == d)]
                if not len(r):
                    continue
                r = r.iloc[0].to_dict()
                rows.append([n, lab, d, spread(r, "rate_marg"), spread(r, "rate_pair"), spread(r, "gain", 2, pct=True),
                             spread(r, "frac_any_new"), spread(r, "top1_marg"), spread(r, "top1_pair")])
        rows.append("MIDRULE")
    rows = rows[:-1]
    nseed = int(M.n_seeds.max()) if len(M) else 0
    spread_txt = ("95\\% seed-level $t$ intervals in brackets" if nseed >= 10
                  else "seed min--max in parentheses (descriptive, not confidence intervals)")
    cap = ("JNT-02: strict signed ranking certificates from sharp pairwise bounds versus marginal-interval comparison "
           f"(partition \\texttt{{{main_tag}}}, {nseed} seeds, data anchors, domain mode, generators and $\\rho_c$ pooled "
           "within seed). Rates are fractions of unordered active feature pairs; gain in percentage points; "
           "\\emph{any new}: fraction of instances with at least one additional certificate; top-1: strict signed top-1 "
           f"certified. Seed means; {spread_txt}. Pairwise certification is never weaker by construction.")
    with open(os.path.join(args.out, "jnt02_main.tex"), "w") as fh:
        fh.write(tex_table(["$n$", "$L$", "$d$", "marginal", "pairwise", "gain [pp]", "any new", "top-1 marg.",
                            "top-1 pair"], rows, cap, "tab:jnt02", align="rlrllllll", size="\\scriptsize"))

    # ---------------------------------------------------------------- supplementary subgroup table
    sub = S[(S.partition == main_tag) & (S.n_anchor == 4096)].sort_values(["L_kind", "anchor_law", "mode", "d"])
    rows = [[r.L_kind, r.anchor_law, r["mode"], int(r.d), spread(r, "rate_marg"), spread(r, "rate_pair"),
             spread(r, "gain", 2, pct=True), spread(r, "share_zero_gain"), spread(r, "q90_inst_gain")]
            for _, r in sub.iterrows()]
    with open(os.path.join(args.out, "jnt02_subgroups.tex"), "w") as fh:
        fh.write(tex_table(["level", "law", "mode", "$d$", "marginal", "pairwise", "gain [pp]", "zero-gain inst.",
                            "q90 inst. gain"], rows,
                           f"JNT-02 subgroups at $n=4096$ (partition \\texttt{{{main_tag}}}); all cells are shown, "
                           "including those with zero gain.", "tab:jnt02sub", align="llllrllll", size="\\scriptsize"))

    # ---------------------------------------------------------------- figure
    levels = [lv for lv in ("cert", "oracle") if lv in set(M.L_kind)]
    fig, axes = plt.subplots(1, max(1, len(levels)), figsize=(5.2 * max(1, len(levels)), 3.4), sharey=True,
                             squeeze=False)
    for ax, lev in zip(axes[0], levels):
        for col, d in zip(("C0", "C1", "C2", "C3", "C4"), sorted(M.d.unique())):
            v = M[(M.L_kind == lev) & (M.d == d)].sort_values("n_anchor")
            for metric, ls in (("rate_pair", "-"), ("rate_marg", "--")):
                ax.plot(v.n_anchor, v[f"{metric}_mean"], ls, color=col, marker="o" if ls == "-" else None,
                        label=(f"d={d}" if metric == "rate_pair" else None))
                lo = v[f"{metric}_t_lo"].where(np.isfinite(v[f"{metric}_t_lo"]), v[f"{metric}_min"])
                hi = v[f"{metric}_t_hi"].where(np.isfinite(v[f"{metric}_t_hi"]), v[f"{metric}_max"])
                ax.fill_between(v.n_anchor, lo, hi, color=col, alpha=0.12 if ls == "-" else 0.06, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("background anchors $n$")
        ax.set_title({"cert": r"certified $L_{\mathrm{cert}}$", "oracle": r"oracle $L^\star$"}[lev])
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("certified signed pairs (fraction)")
    axes[0][0].legend(frameon=False, fontsize=7, title="solid: pairwise\ndashed: marginal", title_fontsize=7)
    fig.tight_layout()
    savefig(fig, args.figs, "fig_jnt02_certification.pdf")

    # ---------------------------------------------------------------- quoted numbers
    N = Numbers("JNT-02", args.out)
    for lev in ("cert", "oracle"):
        for d in sorted(M.d.unique()):
            r = M[(M.n_anchor == 4096) & (M.L_kind == lev) & (M.d == d)]
            if len(r):
                r = r.iloc[0]
                N.add(f"{lev}.d{d}.n4096.rate_marg", r.rate_marg_mean, "marginal signed-pair certification rate, seed mean",
                      "jnt02_summary.csv")
                N.add(f"{lev}.d{d}.n4096.rate_pair", r.rate_pair_mean, "pairwise signed-pair certification rate, seed mean",
                      "jnt02_summary.csv")
                N.add(f"{lev}.d{d}.n4096.gain_pp", r.gain_mean, "pairwise minus marginal, percentage points",
                      "jnt02_summary.csv", nd=2, pct=True)
                N.add(f"{lev}.d{d}.n4096.any_new", r.frac_any_new_mean, "instances with >= 1 new certificate",
                      "jnt02_summary.csv")
    N.add("integrity.not_contained_valid", str(integ["not_contained_valid_L"]), "true difference outside sharp pair "
          "bounds at LV/oracle/cert (must be 0)", "jnt02_integrity.json")
    N.add("integrity.v_dom", str(integ["v_dom"]), "rows where pairwise is weaker than marginal (must be 0)",
          "jnt02_integrity.json")
    N.write()
    print(json.dumps(integ, indent=1))
    print(M[["d", "n_anchor", "L_kind", "n_seeds", "rate_marg_mean", "rate_pair_mean", "gain_mean",
             "frac_any_new_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
