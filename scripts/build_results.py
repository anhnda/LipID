#!/usr/bin/env python3
"""
build_results.py — rebuild every table / figure / quoted number of the draft from raw run outputs.

    python scripts/build_results.py --explore runs/explore --confirm runs/confirm \
        --legacy-confirm legacy/nested2_summary_confirm.json

No empirical number is hardcoded: everything below is computed from runs/<tag>/{records,diag}_seed*.{parquet,csv.gz}
(row level) -> per-seed aggregates (CSV) -> LaTeX fragments / PDF figures. Optional sections (metric sensitivity,
scaling, real audit) are collected if their scripts have been run.

Seed partition: 'explore' = seeds 1-4, 'confirm' = seeds 8-11 (previously held-out P0'' confirmation). They are
reported separately; 'pooled8' appears only where the draft explicitly pooled eight seeds (Table 4) and is labeled.
Spreads are descriptive seed min--max (4 independent seeds per partition), never confidence intervals.
"""
import argparse
import glob
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

import lipid  # noqa: E402
import phase1 as P  # noqa: E402
from lipid import pipeline as PL  # noqa: E402
from lipid.analysis import CI_NOTE, RULE_TEXT, criteria, fmt, rng, seed_stats  # noqa: E402
from lipid.io import diag_columns, load_json, read_run, write_table  # noqa: E402
from lipid.provenance import now, sha256, write_json  # noqa: E402

DIMS = (3, 4, 5, 6)
NS = (16, 64, 256, 1024, 4096)
MODELS = ("linear", "fa", "stable_ood", "unstable_id")
KINDS = PL.L_KINDS
TOL = P.Config().tol
REC_COLS = ["seed", "model", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "chain_id", "instance_id", "fid",
            "Dbin", "L_kind", "rho", "Gamma", "cert_sign", "contained", "saturation_gt_08", "L_A", "L_V", "L_star",
            "L_cert", "L_used", "v_nest", "v_Lorder", "v_rhoL"]
LINK_COLS = ["seed", "fid", "fid_prev", "fid_pair"]


# ============================================================================================ small helpers
class Ctx:
    def __init__(self, out, figs):
        self.out, self.figs = out, figs
        self.tex, self.captions, self.numbers, self.issues = {}, [], [], []
        os.makedirs(out, exist_ok=True)
        os.makedirs(figs, exist_ok=True)
        os.makedirs(os.path.join(out, "matched"), exist_ok=True)

    def path(self, name):
        return os.path.join(self.out, name)

    def csv(self, df, name):
        df.to_csv(self.path(name), index=False)
        return self.path(name)

    def write_tex(self, name, body):
        with open(self.path(name), "w") as fh:
            fh.write(body)
        self.tex[name] = self.path(name)

    def number(self, key, value, definition, source):
        self.numbers.append(dict(key=key, value=value, definition=definition, source=source))

    def caption(self, fig, text):
        self.captions.append((fig, text))


def tabular(header, rows, align=None, caption=None, label=None):
    align = align or ("l" + "c" * (len(header) - 1))
    out = []
    if caption is not None:
        out += ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}"]
        if label:
            out.append(f"\\label{{{label}}}")
        out.append("\\small")
    out += [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    if caption is not None:
        out.append("\\end{table}")
    return "\n".join(out) + "\n"


def v3_summaries(run_dir):
    """nested_anchors_v2-style summary recomputed here from the diag rows (row level -> aggregate)."""
    D = read_run(run_dir, "diag", columns=list(PL.V2_COLS))
    S = dict(seeds={})
    for s, g in D.groupby("seed"):
        C = diag_columns(g.reset_index(drop=True))
        S["seeds"][str(int(s))] = PL.v2_summary_from_diag({c: C[c] for c in PL.V2_COLS}, TOL)
    return S


def seeds_of(S):
    return sorted(int(s) for s in S["seeds"])


# ============================================================================================ 1. crosswalk
def legacy_crosswalk(ctx, S, legacy, tag, tol=1e-9):
    mism, n_cmp, worst, missing = [], 0, 0.0, []
    for s, L in legacy["seeds"].items():
        if s not in S["seeds"]:
            missing.append(f"seed {s}")
            continue
        for tab in ("by_d_mode_arm_n", "by_d_mode_arm_model_n", "by_d_mode_arm_rhoc_n", "by_d_mode_arm_rhoc_Dbin_n"):
            new_t = S["seeds"][s][tab]
            for key, st in L[tab].items():
                nv = new_t.get(key)
                if nv is None:
                    missing.append(f"{s}/{tab}/{key}")
                    continue
                for stat, lv in st.items():
                    vv = nv.get(stat)
                    n_cmp += 1
                    if lv is None or vv is None:
                        diff = 0.0 if lv is None and vv is None else np.inf
                    else:
                        diff = abs(float(lv) - float(vv))
                    worst = max(worst, diff if np.isfinite(diff) else worst)
                    if diff > tol:
                        mism.append(dict(seed=s, table=tab, key=key, stat=stat, legacy=lv, v3=vv, abs_diff=diff))
            extra = set(new_t) - set(L[tab])
            if extra:
                missing.append(f"{s}/{tab}: {len(extra)} keys only in v3")
        for c, v in L["checks"].items():
            if S["seeds"][s]["checks"].get(c) != v:
                mism.append(dict(seed=s, table="checks", key=c, stat=c, legacy=v, v3=S["seeds"][s]["checks"].get(c),
                                 abs_diff=np.inf))
    pd.DataFrame(mism, columns=["seed", "table", "key", "stat", "legacy", "v3", "abs_diff"]).to_csv(
        ctx.path(f"legacy_crosswalk_{tag}_mismatches.csv"), index=False)
    summ = dict(tag=tag, n_compared=n_cmp, n_mismatch=len(mism), tolerance=tol, max_abs_diff_finite=worst,
                missing=missing[:200], n_missing=len(missing),
                verdict=("REPRODUCED" if not mism and not missing else "DEVIATIONS (see mismatches csv)"))
    write_json(ctx.path(f"legacy_crosswalk_{tag}.json"), summ)
    print(f"[crosswalk {tag}] compared={n_cmp} mismatches={len(mism)} missing={len(missing)} max|diff|={worst:.3g}")
    return summ


# ============================================================================================ 2. prevalence
def prevalence(ctx, R, tag):
    sel = R[(R.L_kind == "oracle") & (R.anchor_law == "data")]
    g = sel.groupby(["seed", "mode", "d", "n_anchor"], observed=True)
    df = g.agg(n_feature_instances=("rho", "size"), p_sat=("saturation_gt_08", "mean"),
               sign_rate=("cert_sign", lambda s: float(np.mean(s != 0))), mean_rho=("rho", "mean")).reset_index()
    df["partition"] = tag
    return df


def prevalence_outputs(ctx, prev):
    ctx.csv(prev, "prevalence_by_seed.csv")
    c = prev[prev.partition == "confirm"]
    if c.empty:
        return
    rows = []
    for mode in ("domain", "fixed"):
        for d in DIMS:
            vals = [rng(c[(c["mode"] == mode) & (c.d == d) & (c.n_anchor == n)].p_sat) for n in NS]
            rows.append([f"{mode.capitalize()} $d={d}$"] + vals)
    ctx.write_tex("tab_prevalence.tex", tabular(["Setting"] + [f"$n={n}$" if i == 0 else str(n) for i, n in
                                                                 enumerate(NS)], rows,
                  caption=("Range of $p_{\\mathrm{sat}}=\\mathbb{P}(\\rho>0.8)$ over validation seeds 8--11, under "
                           "$L^\\star$ with data anchors. Descriptive seed min--max ranges, not confidence intervals. "
                           "Denominators: see results/prevalence\\_by\\_seed.csv."),
                  label="tab:prevalence"))
    # figure 3
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cm = plt.get_cmap("tab10")
    for k, d in enumerate(DIMS):
        sub = c[(c["mode"] == "domain") & (c.d == d)]
        st = sub.groupby("n_anchor").p_sat.agg(["mean", "min", "max"]).reindex(NS)
        ax.errorbar(NS, st["mean"], yerr=[st["mean"] - st["min"], st["max"] - st["mean"]], marker="o",
                    capsize=3, color=cm(k), label=f"$d={d}$")
        for s, gs in sub.groupby("seed"):
            gs = gs.set_index("n_anchor").reindex(NS)
            ax.plot(NS, gs.p_sat, color=cm(k), alpha=0.25, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of background anchors $n$")
    ax.set_ylabel(r"Saturation prevalence $P(\rho > 0.8)$")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.figs, "fig_saturation_prevalence.pdf"))
    plt.close(fig)
    ctx.caption("fig_saturation_prevalence.pdf",
                "Domain mode, data anchors, oracle-valid L*. Marker = mean over the 4 confirmation seeds (8-11) of the "
                "per-seed prevalence; whiskers = seed min-max (descriptive, NOT confidence intervals); thin lines = "
                "individual seeds. Aggregate unit within a seed: feature-instance (20 query/reference pairs x 4 "
                "generators x 4 rho_c x active features), which share one anchor chain per (d, rho_c).")


# ============================================================================================ 3. distribution
def dist_table(ctx, Ss):
    rows_csv = []
    for tag, S in Ss.items():
        for s in seeds_of(S):
            TR = S["seeds"][str(s)]["by_d_mode_arm_rhoc_n"]
            for rc in (0.0, 0.6, 0.9, 0.99):
                for mode in ("domain", "fixed"):
                    for d in DIMS:
                        for n in NS:
                            e = TR.get(f"{d}/{mode}/prod/{rc}/{n}")
                            if e is None:
                                continue
                            rows_csv.append(dict(partition=tag, seed=s, rho_corr=rc, mode=mode, d=d, n_anchor=n,
                                                 n_pairs=e["n"], mean_dpair=e["mean_dpair"],
                                                 frac_data_gt_prod=e["frac_data_gt_prod"]))
    df = pd.DataFrame(rows_csv)
    ctx.csv(df, "anchor_law_dpair_by_seed.csv")
    if df.empty:
        return
    sel = df[(df.rho_corr == 0.99) & (df.n_anchor == 4096)]
    for name, parts, cap in (("tab_dist.tex", ("explore", "confirm"), "pooled over the eight reported seeds (1--4 and 8--11)"),
                             ("tab_dist_confirm.tex", ("confirm",), "over confirmation seeds 8--11 only")):
        s2 = sel[sel.partition.isin(parts)]
        if s2.partition.nunique() < len(parts):
            continue
        rows = [[mode.capitalize()] + [rng(s2[(s2["mode"] == mode) & (s2.d == d)].mean_dpair) for d in DIMS]
                for mode in ("domain", "fixed")]
        ctx.write_tex(name, tabular(["Mode"] + [f"$d={d}$" for d in DIMS], rows,
                                    caption=("Range of seed-level $\\mathbb{E}[\\rho_{\\mathrm{data}}-\\rho_{\\mathrm{prod}}]$ "
                                             f"at $n=4096$, $\\rho_c=0.99$, {cap}; matched feature-instances, oracle $L^\\star$. "
                                             "Descriptive ranges."),
                                    label=("tab:dist" if name == "tab_dist.tex" else "tab:dist-confirm")))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    off = {"domain": -0.12, "fixed": 0.12}
    col = {"domain": "C0", "fixed": "C1"}
    for mode in ("domain", "fixed"):
        for k, d in enumerate(DIMS):
            v = sel[(sel["mode"] == mode) & (sel.d == d)]
            if v.empty:
                continue
            mu = v.mean_dpair.mean()
            ax.errorbar(k + off[mode], mu, yerr=[[mu - v.mean_dpair.min()], [v.mean_dpair.max() - mu]], fmt="o",
                        capsize=4, color=col[mode], label=mode if k == 0 else None)
            for part, mk in (("explore", "x"), ("confirm", ".")):
                vv = v[v.partition == part]
                ax.scatter(np.full(len(vv), k + off[mode] + 0.05), vv.mean_dpair, marker=mk, color=col[mode],
                           alpha=0.6, s=14)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(DIMS)), [f"d={d}" for d in DIMS])
    ax.set_ylabel(r"$E[\rho_{data}-\rho_{prod}]$")
    ax.set_xlabel("Dimension")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.figs, "fig_anchor_law_gap.pdf"))
    plt.close(fig)
    ctx.caption("fig_anchor_law_gap.pdf",
                "rho_c = 0.99, n = 4096, oracle L*. Marker = mean of seed-level matched mean differences over the eight "
                "seeds (1-4 'x', 8-11 '.'); whiskers = seed min-max (descriptive). Matching: same x, r, sigma, f, L*, "
                "mode, n; only the background anchors' law differs.")


# ============================================================================================ 4. criteria
def criteria_outputs(ctx, S_conf, legacy):
    rows, status = criteria(S_conf, seeds_of(S_conf))
    out = pd.DataFrame(rows)
    out["value"] = out["value"].map(lambda v: json.dumps(v))
    ctx.csv(out, "criteria_reconstructed_cells.csv")
    if legacy is not None:
        _, st_leg = criteria(legacy, seeds_of(legacy))
        for k in status:
            status[k]["mechanical_on_legacy_summary"] = st_leg[k]["mechanical"]
    st_rows = [dict(criterion=k, **v) for k, v in status.items()]
    ctx.csv(pd.DataFrame(st_rows), "criteria_status.csv")
    def tex_rule(s):
        return (s.replace("_", "\\_").replace("rho\\_c", "$\\rho_c$").replace("<=", "$\\le$")
                .replace(">=", "$\\ge$"))
    body = tabular(["ID", "Rule (abbreviated)", "Reported outcome", "Recomputed from raw rows"],
                   [[k, tex_rule(RULE_TEXT[k]), v["historical"], v["mechanical"]] for k, v in status.items()],
                   align="lp{6cm}p{3.6cm}p{3.2cm}",
                   caption=("Status of pre-specified P0'' criteria in the confirmation run (seeds 8--11). "
                            "``Recomputed'' applies the operationalization in results/preregistration\\_ledger.md "
                            "to the reproduced raw rows; historical labels are retained unchanged."),
                   label="tab:criteria")
    ctx.write_tex("tab_criteria.tex", body)
    return status


# ============================================================================================ 5. calibration
def calibration(ctx, R, tag):
    keys = ["seed", "model", "d", "mode", "anchor_law", "n_anchor"]
    piv = R.pivot_table(index=["seed", "fid"], columns="L_kind", values=["rho", "cert_sign", "contained"],
                        observed=True)
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    base = R[R.L_kind == "cert"][keys + ["fid", "L_A", "L_V", "L_star", "L_cert"]].set_index(["seed", "fid"])
    W = base.join(piv).reset_index()
    W["ratio_cert_LV"] = W.L_cert / W.L_V
    W["ratio_cert_star"] = W.L_cert / W.L_star
    W["ratio_LV_LA"] = W.L_V / W.L_A
    W["d_rho_cert_oracle"] = W.rho_cert - W.rho_oracle
    W["sign_lost"] = ((W.cert_sign_oracle != 0) & (W.cert_sign_cert == 0)).astype(float)
    g = W.groupby(keys, observed=True)
    agg = dict(n_feature_instances=("fid", "size"),
               ratio_cert_LV_median=("ratio_cert_LV", "median"),
               ratio_cert_LV_q25=("ratio_cert_LV", lambda s: s.quantile(.25)),
               ratio_cert_LV_q75=("ratio_cert_LV", lambda s: s.quantile(.75)),
               ratio_cert_LV_min=("ratio_cert_LV", "min"), ratio_cert_LV_max=("ratio_cert_LV", "max"),
               ratio_cert_star_median=("ratio_cert_star", "median"), ratio_cert_star_max=("ratio_cert_star", "max"),
               d_rho_cert_oracle_median=("d_rho_cert_oracle", "median"),
               d_rho_cert_oracle_mean=("d_rho_cert_oracle", "mean"),
               d_rho_cert_oracle_min=("d_rho_cert_oracle", "min"),
               sign_lost_rate=("sign_lost", "mean"))
    for k in KINDS:
        agg[f"median_rho_{k}"] = (f"rho_{k}", "median")
        agg[f"p_sat_{k}"] = (f"rho_{k}", lambda s: float(np.mean(s > 0.8)))
        agg[f"sign_rate_{k}"] = (f"cert_sign_{k}", lambda s: float(np.mean(s != 0)))
        agg[f"containment_{k}"] = (f"contained_{k}", "mean")
    out = g.agg(**agg).reset_index()
    out["partition"] = tag
    out["L_cert_scope"] = "global (R^d)"
    return out, W


def calibration_outputs(ctx, cal, Wc):
    ctx.csv(cal, "calibration_by_seed.csv")
    if Wc is None or Wc.empty:
        return
    sel = Wc[(Wc.anchor_law == "data") & (Wc.n_anchor == 4096)]
    rows = []
    for model in MODELS:
        v = sel[sel.model == model]
        if v.empty:
            continue
        per = v.groupby("seed")
        rq = v.ratio_cert_LV
        rows.append([model.replace("_", "\\_"),
                     f"{rq.median():.2f} [{rq.quantile(.25):.2f}, {rq.quantile(.75):.2f}]; max {rq.max():.2f}",
                     rng(per.contained_cert.mean(), 3),
                     rng(per.rho_cert.median()) + " (" + rng(per.rho_oracle.median()) + ")",
                     rng(per.cert_sign_cert.apply(lambda s: np.mean(s != 0)), 3) + " ("
                     + rng(per.cert_sign_oracle.apply(lambda s: np.mean(s != 0)), 3) + ")"])
    ctx.write_tex("calibration_table.tex", tabular(
        ["Generator", "$L_{\\mathrm{cert}}/L_V$ median [IQR]; max", "Containment at $L_{\\mathrm{cert}}$",
         "Median $\\rho$ cert (oracle $L^\\star$)", "Sign cert.\\ rate cert (oracle)"], rows,
        caption=("Operational calibration evaluation, confirmation seeds 8--11, data anchors, $n=4096$, pooled over "
                 "$d\\in\\{3,4,5,6\\}$ and both modes. $L_{\\mathrm{cert}}$ is a global certificate (appendix H) computed "
                 "from generator parameters and $\\sigma$ only. Ratio quantiles pooled over feature-instances; other "
                 "entries are seed min--max of seed-level values. Stratified values: results/calibration\\_by\\_seed.csv."),
        label="tab:calcert"))
    # figure: p_sat vs n per generator, oracle vs cert vs opt (domain, data, confirm)
    c = cal[(cal.partition == "confirm") & (cal["mode"] == "domain") & (cal.anchor_law == "data")]
    if c.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    cm = plt.get_cmap("tab10")
    for ax, model in zip(axes, MODELS):
        for k, d in enumerate(DIMS):
            sub = c[(c.model == model) & (c.d == d)]
            for kind, ls in (("oracle", "-"), ("cert", "--"), ("opt", ":")):
                st = sub.groupby("n_anchor")[f"p_sat_{kind}"].agg(["mean", "min", "max"]).reindex(NS)
                ax.plot(NS, st["mean"], ls=ls, color=cm(k), label=(f"d={d}" if kind == "oracle" else None))
                ax.fill_between(NS, st["min"], st["max"], color=cm(k), alpha=0.08)
        ax.set_xscale("log", base=2)
        ax.set_title(model)
        ax.set_xlabel("n")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$P(\rho>0.8)$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.figs, "fig_certified_calibration.pdf"))
    plt.close(fig)
    ctx.caption("fig_certified_calibration.pdf",
                "Domain mode, data anchors, confirmation seeds. Solid: oracle L* = max(L_true, L_V); dashed: certified "
                "global L_cert; dotted: optimistic L_A(A_n). Lines = mean of seed-level prevalences; bands = seed "
                "min-max (descriptive). For linear and unstable_id (a = 4) L* = L_cert, so solid and dashed coincide.")


# ============================================================================================ 6. paired effects
def matched_tables(ctx, R, links, tag):
    """Nesting (n -> next n, same chain, same L) and anchor-law (data vs prod, same everything else) matched tables."""
    keep = ["seed", "fid", "model", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "chain_id", "instance_id",
            "Dbin", "rho", "Gamma", "cert_sign"]
    out_nest, out_law = [], []
    L = links.set_index(["seed", "fid"])
    for kind in ("LV", "oracle", "cert", "opt"):
        X = R[R.L_kind == kind][keep].copy()
        X["sig"] = (X.cert_sign != 0).astype(float)
        X = X.join(L, on=["seed", "fid"])
        Y = X[["seed", "fid", "rho", "Gamma", "sig", "n_anchor"]].rename(
            columns={"fid": "fid_l", "rho": "rho_l", "Gamma": "Gamma_l", "sig": "sig_l", "n_anchor": "n_l"})
        if kind != "opt":   # nesting at FIXED L only (opt moves with n; its change is the N2 decomposition)
            M = X[X.fid_prev >= 0].merge(Y, left_on=["seed", "fid_prev"], right_on=["seed", "fid_l"])
            M["L_kind"] = kind
            out_nest.append(M)
        P_ = X[X.fid_pair >= 0].merge(Y, left_on=["seed", "fid_pair"], right_on=["seed", "fid_l"])
        P_["L_kind"] = kind
        out_law.append(P_)
    N = pd.concat(out_nest, ignore_index=True)
    A = pd.concat(out_law, ignore_index=True)
    common = ["seed", "chain_id", "instance_id", "model", "d", "rho_corr", "mode", "Dbin", "L_kind"]
    # long matched tables: id_left / id_right / metric / difference
    nest_long = []
    for metric, a, b in (("rho", "rho_l", "rho"), ("Gamma", "Gamma_l", "Gamma"), ("sign_certified", "sig", "sig_l")):
        t = N[common + ["anchor_law", "n_l", "n_anchor", "fid_l", "fid"]].copy()
        t["metric"] = metric
        # rho/Gamma: left (smaller n) minus right (larger n) >= 0;  sign: right minus left >= 0  (Proposition 4)
        t["difference"] = (N[a] - N[b]) if metric != "sign_certified" else (N["sig"] - N["sig_l"])
        nest_long.append(t.rename(columns={"fid_l": "id_left", "fid": "id_right", "n_l": "n_left",
                                           "n_anchor": "n_right"}))
    nest_long = pd.concat(nest_long, ignore_index=True)
    law_long = []
    for metric, a, b in (("rho", "rho_l", "rho"), ("Gamma", "Gamma_l", "Gamma"), ("sign_certified", "sig_l", "sig")):
        t = A[common + ["n_anchor", "fid_l", "fid"]].copy()
        t["metric"] = metric
        t["difference"] = A[a] - A[b]          # data minus prod (left = data, right = prod)
        law_long.append(t.rename(columns={"fid_l": "id_left", "fid": "id_right"}))
    law_long = pd.concat(law_long, ignore_index=True)
    for t in (nest_long, law_long):
        for c in ("L_kind", "metric", "model", "mode", "anchor_law", "chain_id", "Dbin"):
            if c in t.columns and str(t[c].dtype) != "category":
                t[c] = t[c].astype("category")
    del N, A, out_nest, out_law
    write_table(nest_long, os.path.join(ctx.out, "matched", f"{tag}_nesting"))
    write_table(law_long, os.path.join(ctx.out, "matched", f"{tag}_anchor_law"))
    return nest_long, law_long


def n2_table(ctx, D, tag):
    t = D[D.fid_prev >= 0]
    rows = []
    for metric in ("N2_info", "N2_calib", "N2_total"):
        x = t[["seed", "chain_id", "inst", "model", "d", "rhoc", "arm", "mode", "n", "fid_prev", "fid"]].copy()
        x["metric"] = metric
        x["difference"] = t[metric].to_numpy()
        rows.append(x)
    out = pd.concat(rows, ignore_index=True).rename(columns={"fid_prev": "id_left", "fid": "id_right", "inst":
                                                             "instance_id", "rhoc": "rho_corr", "arm": "anchor_law",
                                                             "n": "n_right"})
    write_table(out, os.path.join(ctx.out, "matched", f"{tag}_N2"))
    return out


def paired_summaries(ctx, nest, law, n2, tag):
    res = []
    def summarize(df, effect, keys, expect_sign=None):
        g = df.groupby(["seed"] + keys, observed=True)["difference"]
        a = g.agg(n_pairs="size", mean="mean", median="median", min="min", max="max").reset_index()
        a["frac_negative"] = g.apply(lambda s: float(np.mean(s < -TOL))).to_numpy()
        a["effect"] = effect
        a["expected_sign"] = expect_sign
        a["partition"] = tag
        return a
    k_nest = ["L_kind", "metric", "d", "mode", "anchor_law", "n_left", "n_right"]
    res.append(summarize(nest, "nesting (left n minus right n; sign: gain)", k_nest, ">= 0 (Prop. 4)"))
    res.append(summarize(law, "anchor law (data minus prod)", ["L_kind", "metric", "rho_corr", "d", "mode",
                                                                "n_anchor"], None))
    res.append(summarize(n2, "N2 decomposition (optimistic)", ["metric", "d", "mode", "anchor_law", "n_right"],
                         "info <= 0, calib >= 0 (Prop. 5)"))
    out = pd.concat(res, ignore_index=True)
    return out


def across_seeds(df, keys):
    rows = []
    for k, g in df.groupby(keys, observed=True, dropna=False):
        st = seed_stats(g["mean"])
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), n_seeds=st["n_seeds"],
                         seed_means=json.dumps([round(x, 6) for x in g.sort_values("seed")["mean"]]),
                         mean_of_seed_means=st["mean"], sd_of_seed_means=st["sd"], min=st["min"], max=st["max"],
                         n_pairs_total=int(g.n_pairs.sum()), interval=st["note"]))
    return pd.DataFrame(rows)


# ============================================================================================ 7. subgroups
def subgroup_outputs(ctx, law, D, tag):
    t = law[(law.metric == "rho") & (law.L_kind.isin(["oracle", "cert"]))]
    g = t.groupby(["seed", "L_kind", "rho_corr", "d", "mode", "n_anchor", "Dbin"], observed=True)["difference"]
    sub = g.agg(n_pairs="size", mean_diff="mean", median_diff="median").reset_index()
    sub["reversal"] = sub.mean_diff < 0
    sub["partition"] = tag
    # Mahalanobis tail share: identical across models / laws / modes / n -> one copy per (x, r, feature)
    m = D[(D.arm == "data") & (D["mode"] == "domain") & (D.n == min(NS)) & (D.model == D.model.iloc[0])]
    mt = m.groupby(["seed", "rhoc", "d"], observed=True).lowdens.agg(n_feature_instances="size",
                                                                   mean_tail_share="mean").reset_index()
    mt["partition"] = tag
    return sub, mt


def subgroup_figure(ctx, sub):
    c = sub[(sub.partition == "confirm") & (sub.L_kind == "oracle") & (sub.rho_corr == 0.99) & (sub.n_anchor == 4096)]
    if c.empty:
        return
    order = list(PL.V2.DLAB)
    fig, axes = plt.subplots(1, len(DIMS), figsize=(12, 3.0), sharey=True)
    for ax, d in zip(axes, DIMS):
        for mode, col, off in (("domain", "C0", -0.1), ("fixed", "C1", 0.1)):
            v = c[(c.d == d) & (c["mode"] == mode)]
            for k, b in enumerate(order):
                vb = v[v.Dbin == b]
                if vb.empty:
                    continue
                ax.scatter(np.full(len(vb), k + off), vb.mean_diff, color=col, s=12, alpha=0.7,
                           label=mode if (k == 0 and d == DIMS[0]) else None)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(order)), order)
        ax.set_title(f"d={d}")
        ax.set_xlabel(r"$D/\delta_i$ bin")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"seed mean of $\rho_{data}-\rho_{prod}$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ctx.figs, "fig_anchor_law_strata.pdf"))
    plt.close(fig)
    ctx.caption("fig_anchor_law_strata.pdf",
                "rho_c = 0.99, n = 4096, oracle L*, confirmation seeds. Each point is one seed's matched mean difference "
                "within a pre-defined D/delta_i bin (endpoints 2, 4, 8 fixed in nested_anchors_v2 before seeds 8-11). "
                "Points below zero are sign reversals and are retained. Bins with few pairs are noisy: see n_pairs.")


# ============================================================================================ 8. text numbers
def text_numbers(ctx, Ss, prev, mt, sub):
    for part, parts in (("confirm", ["confirm"]), ("explore", ["explore"]), ("pooled8", ["explore", "confirm"])):
        c = prev[prev.partition.isin(parts)]
        if c.partition.nunique() < len(parts):
            continue
        for mode in ("domain", "fixed"):
            for d in DIMS:
                v = c[(c["mode"] == mode) & (c.d == d) & (c.n_anchor == 4096)]
                ctx.number(f"p_sat_{mode}_d{d}_n4096_{part}", rng(v.p_sat), "P(rho>0.8), oracle L*, data anchors",
                           f"{part} seeds")
                ctx.number(f"sign_rate_{mode}_d{d}_n4096_{part}", rng(v.sign_rate),
                           "strict sign certified at L*, data anchors", f"{part} seeds")
    S = Ss.get("confirm")
    if S:
        sd = seeds_of(S)
        T = lambda s: S["seeds"][str(s)]
        for mode in ("domain", "fixed"):
            for d in DIMS:
                ctx.number(f"geom_range_cert_{mode}_d{d}_n4096",
                           rng([T(s)["by_d_mode_arm_n"][f"{d}/{mode}/data/4096"]["p_uni_com"] for s in sd]),
                           "share of feature-instances certified saturated by the unified geometry/range bound (eq. 17)",
                           "confirm seeds")
        for n in (16, 4096):
            vals = [T(s)["by_d_mode_arm_model_n"][f"{d}/{mode}/data/{mo}/{n}"]["cover_opt"]
                    for s in sd for d in DIMS for mode in ("domain", "fixed") for mo in MODELS]
            ctx.number(f"cover_opt_all_n{n}", rng(vals), "true-attribution containment at L_A(A_n), range over seeds x d "
                       "x mode x generator (data anchors)", "confirm seeds")
        for d in (5, 6):
            for mode in ("domain", "fixed"):
                e = [T(s)["by_d_mode_arm_n"][f"{d}/{mode}/data/64"] for s in sd]
                ctx.number(f"N2_calib_16to64_{mode}_d{d}", rng([x["N2_calib"] for x in e]), "mean recalibration term",
                           "confirm seeds")
                ctx.number(f"N2_info_16to64_{mode}_d{d}", rng([x["N2_info"] for x in e]), "mean information term",
                           "confirm seeds")
        e = [T(s)["by_d_mode_arm_n"][f"{d}/{mode}/data/4096"] for s in sd for d in DIMS for mode in ("domain", "fixed")]
        ctx.number("N2_info_1024to4096_all", rng([x["N2_info"] for x in e]), "mean information term, all d and modes",
                   "confirm seeds")
        ctx.number("N2_calib_1024to4096_max", fmt(max(x["N2_calib"] for x in e)), "max mean recalibration term",
                   "confirm seeds")
    m = mt[mt.partition == "confirm"]
    if not m.empty:
        ctx.number("maha_tail_rho099", rng(m[m.rhoc == 0.99].mean_tail_share),
                   "Shapley-weighted share of hybrid endpoints with Mahalanobis M(z) above the 99% data quantile, rho_c=.99, "
                   "range over seeds x d", "confirm seeds")
        ctx.number("maha_tail_rho0_max", fmt(m[m.rhoc == 0.0].mean_tail_share.max()),
                   "same share at rho_c = 0 (independent coordinates), max over seeds x d", "confirm seeds")
    s = sub[(sub.partition == "confirm") & (sub.L_kind == "oracle") & (sub.rho_corr == 0.99) & (sub.reversal)]
    ctx.number("dbin_reversals_rho099", rng(s.mean_diff), "seed-level negative data-prod differences within D/delta "
               "bins, rho_c=.99, any n, d, mode", f"confirm seeds; {len(s)} reversed cells")
    pd.DataFrame(ctx.numbers).to_csv(ctx.path("text_numbers.csv"), index=False)


# ============================================================================================ 9. documents
def ledger(ctx, status, manifests):
    L = ["# Pre-registration ledger (PRE-01)", "",
         f"Generated {now()} by scripts/build_results.py from raw rows. Historical outcomes are NOT relabeled.", "",
         "## Seed partition", "", "* exploratory: seeds 1-4 (protocol development; predictions E1-E4 in "
         "nested_anchors_v2.py docstring were written before these runs)",
         "* follow-up (P0', nested_anchors.py): seeds 5-7 — not part of this release",
         "* confirmation (P0''): seeds 8-11, run after the criteria were set", "",
         "## Original locked protocol text", "",
         "**NOT AVAILABLE IN THE ARCHIVE.** The rules below are the abbreviations printed in Table 5 of the draft. "
         "Attach the frozen protocol file (with timestamp / hash) here before submission. Until then the "
         "operationalizations are a reconstruction; they reproduce the historically reported outcomes "
         "(C2b 4/160, C4b d=4 failing in 2/4 seeds) on the archived summary, which is evidence but not proof that "
         "they match the locked text.", "",
         "## Criteria", "",
         "| ID | Rule (abbreviated) | Operationalization used here | Historical outcome | Recomputed (v3 raw rows) |"
         " On archived summary |", "|---|---|---|---|---|---|"]
    for k, v in status.items():
        L.append(f"| {k} | {v['rule']} | {v['operationalization']} | {v['historical']} | {v['mechanical']} | "
                 f"{v.get('mechanical_on_legacy_summary', 'n/a')} |")
    L += ["", "Notes:", "* C3 is partly structural: at n = 4096 the linear-generator coverage 1 - 1/d follows from "
          "Proposition 6 for the feature with the maximal weighted coefficient; it is a consistency check.",
          "* C5 was an invalid specification (domain-mode certificates are not uniformly zero); it is neither counted "
          "as success nor repaired by restricting it retrospectively to one mode.",
          "* C2b failed; C4b was only partly met. These outcomes stay in the paper.", "", "## Runs", ""]
    for tag, man in manifests.items():
        if man:
            L.append(f"* {tag}: protocol {man.get('protocol')}, git {man.get('git', {}).get('commit')} "
                     f"(dirty={man.get('git', {}).get('dirty')}), started {man.get('started')}, finished "
                     f"{man.get('finished')}, seeds {man.get('seeds')}, host {man.get('environment', {}).get('hostname')}")
    L.append("* archived P0'' summary: legacy/nested2_summary_confirm.json (sha256 "
             + (sha256(os.path.join(ROOT, "legacy", "nested2_summary_confirm.json"))
                if os.path.exists(os.path.join(ROOT, "legacy", "nested2_summary_confirm.json")) else "missing") + ")")
    with open(ctx.path("preregistration_ledger.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def provenance_md(ctx, manifests, crosswalks, counts):
    L = ["# Provenance (REP-03)", "", f"Built {now()}.", ""]
    for tag, man in manifests.items():
        if not man:
            continue
        env = man.get("environment", {})
        L += [f"## run `{tag}`", "",
              f"* argv: `{' '.join(man.get('argv', []))}`",
              f"* git: {man.get('git')}",
              f"* python {env.get('python')}; torch {env.get('torch')} (CUDA {env.get('torch_cuda_version')}, "
              f"devices {env.get('cuda_devices')}); numpy {env.get('numpy')}; scipy {env.get('scipy')}; pandas "
              f"{env.get('pandas')}; pyarrow {env.get('pyarrow')}",
              f"* host {env.get('hostname')}, {env.get('platform')}, {env.get('cpu_count')} CPUs",
              f"* rows: {counts.get(tag)}",
              f"* output sha256: see runs/{tag}/manifest.json (outputs)", ""]
    for tag, c in crosswalks.items():
        L += [f"## legacy crosswalk `{tag}`", "", f"* {c}", ""]
    L += ["## RNG", "", "All randomness comes from torch CPU generators keyed by crc32(repr(key)) (phase1.seed_of); "
          "the keys per (seed, d, rho_c) are listed in runs/<tag>/rng_keys.csv. Draws happen on CPU and are moved to "
          "the device, so CPU and CUDA runs see identical inputs; reductions on CUDA may differ in the last ulp.", ""]
    with open(ctx.path("provenance.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def uncertainty_md(ctx, counts):
    txt = f"""# Uncertainty methods (REP-01)

* Independent unit: the **seed**. Within a seed, each (d, rho_c) has one sigma draw, one set of 20 query/reference
  pairs (5 x points x 4 references) and one nested anchor chain per anchor law, shared by the 4 generators and both
  modes. Feature-instances, generators and (x, r) pairs are therefore NOT independent observations.
* Reported spreads are seed-level min--max over 4 seeds per partition (exploratory 1-4, confirmation 8-11; the
  pooled 8-seed range appears only for Table 4, labeled). {CI_NOTE}.
* Denominators: every per-seed aggregate CSV carries the number of feature-instances / matched pairs
  (columns n_feature_instances, n_pairs). Per seed and (d, mode, law, n): 20 pairs x 4 generators x 4 rho_c x d
  active features (minus skipped units listed in runs/<tag>/skips_seed*.csv).
* Matched effects (results/matched/*.parquet) keep id_left / id_right / seed / chain_id so any re-analysis can
  respect the clustering. No bootstrap across features is used anywhere.
* Row counts: {counts}
"""
    with open(ctx.path("uncertainty_methods.md"), "w") as fh:
        fh.write(txt)


def crosswalk_paper_md(ctx):
    rows = [("Table 2 tab:calcert", "results/calibration_table.tex", "CAL-02", "records L_kind in {oracle, cert}"),
            ("Table 3 tab:prevalence", "results/tab_prevalence.tex", "REP-03", "records L_kind=oracle, data"),
            ("Table 4 tab:dist", "results/tab_dist.tex (+ tab_dist_confirm.tex)", "REP-02", "diag dpair"),
            ("Table 5 tab:criteria", "results/tab_criteria.tex, preregistration_ledger.md", "PRE-01", "diag summaries"),
            ("Table 6 (scaling)", "results/scaling.tex", "SCALE-01", "scripts/benchmark_scaling.py"),
            ("Metric sensitivity", "results/metric_sensitivity.tex", "MET-01", "scripts/metric_sensitivity.py"),
            ("Real audit", "results/real_audit.tex, figures/fig_real_audit.pdf", "REAL-01", "scripts/real_audit.py"),
            ("Figure 2", "figures/fig_counterexample.pdf", "-", "scripts/fig2_counterexample.py"),
            ("Figure 3", "figures/fig_saturation_prevalence.pdf", "REP-03", "prevalence_by_seed.csv"),
            ("Figure 4", "figures/fig_anchor_law_gap.pdf", "REP-02", "anchor_law_dpair_by_seed.csv"),
            ("New: certified calibration figure", "figures/fig_certified_calibration.pdf", "CAL-02",
             "calibration_by_seed.csv"),
            ("New: anchor-law strata figure", "figures/fig_anchor_law_strata.pdf", "REP-02",
             "subgroups_d_over_delta.csv"),
            ("Numbers quoted in sections 6.3-6.6, 7", "results/text_numbers.csv", "REP-01", "diag summaries")]
    L = ["# Table / figure crosswalk to lipid_v1_1.tex", "", "| Paper object | File | TODO | Source |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    L += ["", "## Figure captions (aggregate unit and whisker meaning)", ""]
    L += [f"* **{f}**: {t}" for f, t in ctx.captions]
    with open(ctx.path("crosswalk_paper.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main_tables(ctx):
    for extra in ("scaling.tex", "metric_sensitivity.tex", "real_audit.tex"):
        if os.path.exists(ctx.path(extra)):
            ctx.tex[extra] = ctx.path(extra)
    body = ["% generated by scripts/build_results.py — do not edit by hand", f"% built {now()}"]
    for name in sorted(ctx.tex):
        body += [f"% ---- {name}", open(ctx.tex[name]).read()]
    with open(ctx.path("main_tables.tex"), "w") as fh:
        fh.write("\n".join(body) + "\n")


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", default=os.path.join(ROOT, "runs", "explore"))
    ap.add_argument("--confirm", default=os.path.join(ROOT, "runs", "confirm"))
    ap.add_argument("--legacy-confirm", default=os.path.join(ROOT, "legacy", "nested2_summary_confirm.json"))
    ap.add_argument("--legacy-explore", default=os.path.join(ROOT, "legacy", "nested2_summary_explore.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--figs", default=os.path.join(ROOT, "figures"))
    args = ap.parse_args()
    ctx = Ctx(args.out, args.figs)
    runs = {t: d for t, d in (("explore", args.explore), ("confirm", args.confirm)) if glob.glob(os.path.join(d, "diag_seed*"))}
    if "confirm" not in runs:
        raise SystemExit(f"no confirmation run found in {args.confirm}")
    manifests = {t: (load_json(os.path.join(d, "manifest.json")) if os.path.exists(os.path.join(d, "manifest.json"))
                     else None) for t, d in runs.items()}
    legacy = {t: (load_json(p) if os.path.exists(p) else None)
              for t, p in (("confirm", args.legacy_confirm), ("explore", args.legacy_explore))}

    Ss, crosswalks, counts = {}, {}, {}
    prevs, cals, Wconf, nest_all, law_all, n2_all, subs, mts = [], [], None, [], [], [], [], []
    for tag, d in runs.items():
        print(f"== {tag}: {d}")
        Ss[tag] = v3_summaries(d)
        if legacy.get(tag):
            crosswalks[tag] = legacy_crosswalk(ctx, Ss[tag], legacy[tag], tag)
        R = read_run(d, "records", columns=REC_COLS)
        D = read_run(d, "diag", columns=list(PL.V2_COLS) + ["fid", "fid_prev", "fid_pair", "chain_id"])
        counts[tag] = dict(record_rows=len(R), diag_rows=len(D), seeds=sorted(int(s) for s in D.seed.unique()))
        for kind in ("LV", "oracle", "cert"):
            nc = int(((R.L_kind == kind) & (R.contained == 0)).sum())
            if nc:
                ctx.issues.append(f"{tag}: {nc} rows with true phi outside the interval at {kind}")
        for flag in ("v_nest", "v_Lorder", "v_rhoL"):
            nv = int((R[flag] == 1).sum())
            if nv:
                ctx.issues.append(f"{tag}: {nv} rows with {flag} = 1")
        prevs.append(prevalence(ctx, R, tag))
        cal, W = calibration(ctx, R, tag)
        cals.append(cal)
        if tag == "confirm":
            Wconf = W
        nest, law = matched_tables(ctx, R, D[LINK_COLS], tag)
        n2 = n2_table(ctx, D, tag)
        # consistency: oracle anchor-law differences equal diag dpair
        chk = law[(law.L_kind == "oracle") & (law.metric == "rho")].set_index(["seed", "id_right"])["difference"]
        dp = D[D.fid_pair >= 0].set_index(["seed", "fid"])["dpair"]
        err = float((chk - dp.reindex(chk.index)).abs().max()) if len(chk) else 0.0
        if err > 1e-12:
            ctx.issues.append(f"{tag}: matched anchor-law table differs from diag dpair by {err}")
        ps = paired_summaries(ctx, nest, law, n2, tag)
        nest_all.append(ps)
        sub, mt = subgroup_outputs(ctx, law, D, tag)
        subs.append(sub)
        mts.append(mt)
        del R, D, nest, law, n2

    prev = pd.concat(prevs, ignore_index=True)
    prevalence_outputs(ctx, prev)
    dist_table(ctx, Ss)
    status = criteria_outputs(ctx, Ss["confirm"], legacy.get("confirm"))
    calibration_outputs(ctx, pd.concat(cals, ignore_index=True), Wconf)
    pe = pd.concat(nest_all, ignore_index=True)
    ctx.csv(pe, "paired_effects_by_seed.csv")
    keys = ["partition", "effect", "L_kind", "metric", "d", "mode", "anchor_law", "n_left", "n_right", "rho_corr",
            "n_anchor"]
    keys = [k for k in keys if k in pe.columns]
    ctx.csv(across_seeds(pe, keys), "paired_effects_summary.csv")
    ctx.csv(pe[pe.effect.astype(str).str.startswith("anchor law")], "anchor_law_by_seed.csv")
    sub = pd.concat(subs, ignore_index=True)
    ctx.csv(sub, "subgroups_d_over_delta.csv")
    ctx.csv(sub.groupby(["partition", "seed", "L_kind", "rho_corr", "d", "mode", "n_anchor"], observed=True).apply(
        lambda g: pd.Series(dict(n_bins=len(g), n_reversed_bins=int(g.reversal.sum()),
                                 min_bin_mean=float(g.mean_diff.min())))).reset_index(),
        "anchor_law_reversal_counts.csv")
    mt = pd.concat(mts, ignore_index=True)
    ctx.csv(mt, "mahalanobis_tail.csv")
    subgroup_figure(ctx, sub)
    text_numbers(ctx, Ss, prev, mt, sub)
    for tag, S in Ss.items():
        write_json(ctx.path(f"nested2_summary_from_raw_{tag}.json"), S)
    ledger(ctx, status, manifests)
    uncertainty_md(ctx, counts)
    provenance_md(ctx, manifests, crosswalks, counts)
    crosswalk_paper_md(ctx)
    main_tables(ctx)
    write_json(ctx.path("build_report.json"), dict(built=now(), runs=runs, counts=counts, issues=ctx.issues,
                                                   crosswalk=crosswalks, criteria=status))
    print("\nissues (must be empty):", ctx.issues or "none")
    print(f"tables: {sorted(ctx.tex)}")


if __name__ == "__main__":
    main()
