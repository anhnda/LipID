#!/usr/bin/env python3
"""
stat01_analyze.py — STAT-01: seed-level effect estimates with uncertainty for the pre-specified comparisons (Q7).

    python scripts/stat01_analyze.py \
        --partition stat01:runs/stat01:runs/pairs_stat01 \
        --partition confirm:runs/confirm:runs/pairs_confirm \
        --partition explore:runs/explore:runs/pairs_explore \
        --lock protocols/locks/STAT-01.lock.json

Inference is computed ONLY for the partition named in --inferential (default stat01, the new locked independent seeds).
The historical partitions (seeds 1-4, 8-11) are summarized descriptively (mean, min, max) with the same code, so the
reader sees continuity without any re-labelling of the historical analysis.

Estimands (protocols/v12_preregistration.json, STAT-01; one value per seed, matched within seed):
  P1   p_sat(d=6) - p_sat(d=3)                       L*, data anchors, domain mode, n = 4096, pooled generators/rho_c
  P2.d mean(rho_data - rho_prod), matched           L*, rho_c = .99, n = 4096, domain mode; d = 3..6 (family of 4)
  P3.g mean(rho_cert - rho_oracle), matched          data, domain, n = 4096; g in {fa, stable_ood} (family of 2);
       linear / unstable_id are structural zeros (L_cert = L*): reported, not tested
  P4   pairwise - marginal signed-pair certification rate   L_cert, data, domain, n = 4096, pooled
Secondary estimands are listed with primary = False (no multiplicity claim).
Outputs: results_v12/stat01_seed_values.csv, stat01_effects.csv, stat01_primary.tex, stat01_secondary.tex,
uncertainty_methods_v12.md, numbers_stat01.csv, stat01_manifest.json (includes the lock verification result).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lipid.io import read_table, seeds_in  # noqa: E402
from lipid.lock import verify_lock  # noqa: E402
from lipid.provenance import base_manifest, write_json  # noqa: E402
from lipid.report import Numbers, fnum, tex_table  # noqa: E402
from lipid.stats import MIN_SEEDS_FOR_CI, holm, summarize  # noqa: E402

NMAX = 4096
RCOLS = ["seed", "model", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "instance_id", "feature_i", "fid",
         "L_kind", "rho", "saturation_gt_08", "cert_sign"]
PCOLS = ["seed", "model", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "L_kind", "cert_pair", "cert_marg"]
MKEY = ["model", "d", "rho_corr", "mode", "n_anchor", "instance_id", "feature_i"]


def _s(df):
    for c in ("model", "mode", "anchor_law", "L_kind"):
        if c in df:
            df[c] = df[c].astype(str)
    return df


def seed_values_records(R: pd.DataFrame, seed: int) -> list:
    out = []

    def add(eid, fam, stratum, primary, value, desc):
        out.append(dict(estimand=eid, family=fam, stratum=stratum, primary=primary, seed=seed, value=float(value),
                        description=desc))

    # ---- P1 (+ secondary)
    for lev, mode, prim in (("oracle", "domain", True), ("oracle", "fixed", False), ("cert", "domain", False)):
        q = R[(R.L_kind == lev) & (R.anchor_law == "data") & (R["mode"] == mode)]
        p = q.groupby("d").saturation_gt_08.mean()
        if 3 in p.index and 6 in p.index:
            add(f"P1{'' if prim else f'.{lev}.{mode}'}", "P1" if prim else "P1-sec", f"{lev}/{mode}", prim,
                p[6] - p[3], "p_sat(d=6) - p_sat(d=3), data anchors, n=4096")
    # ---- P2 (+ secondary)
    for lev, mode, rho, prim in (("oracle", "domain", 0.99, True), ("oracle", "fixed", 0.99, False),
                                 ("cert", "domain", 0.99, False), ("oracle", "domain", 0.0, False)):
        q = R[(R.L_kind == lev) & (R["mode"] == mode) & np.isclose(R.rho_corr, rho)]
        D = q[q.anchor_law == "data"].merge(q[q.anchor_law == "prod"], on=MKEY, suffixes=("_data", "_prod"))
        D["diff"] = D.rho_data - D.rho_prod
        for d, v in D.groupby("d")["diff"].mean().items():
            tag = "" if prim else f".{lev}.{mode}.rho{rho}"
            add(f"P2.d{d}{tag}", "P2" if prim else "P2-sec", f"d={d}/{lev}/{mode}/rho={rho}", prim, v,
                "mean matched rho_data - rho_prod, n=4096")
    # ---- P3 (+ secondary sign-rate difference)
    q = R[(R.anchor_law == "data") & (R["mode"] == "domain")]
    D = q[q.L_kind == "cert"].merge(q[q.L_kind == "oracle"], on=["fid"], suffixes=("_c", "_o"))
    D["diff"] = D.rho_c - D.rho_o
    D["sdiff"] = (D.cert_sign_c != 0).astype(float) - (D.cert_sign_o != 0).astype(float)
    for g, v in D.groupby("model_c")["diff"].mean().items():
        prim = g in ("fa", "stable_ood")
        add(f"P3.{g}", "P3" if prim else "P3-structural", f"model={g}", prim, v, "mean matched rho_cert - rho_oracle")
    for g, v in D.groupby("model_c")["sdiff"].mean().items():
        add(f"P3s.{g}", "P3-sec", f"model={g}", False, v, "sign-certification rate cert - oracle")
    return out


def seed_values_pairs(Pp: pd.DataFrame, seed: int) -> list:
    out = []
    for lev, prim in (("cert", True), ("oracle", False)):
        q = Pp[(Pp.L_kind == lev) & (Pp.anchor_law == "data") & (Pp["mode"] == "domain")]
        if not len(q):
            continue
        v = float(np.mean(q.cert_pair != 0) - np.mean(q.cert_marg != 0))
        out.append(dict(estimand="P4" if prim else f"P4.{lev}", family="P4" if prim else "P4-sec", stratum=f"{lev}/pooled",
                        primary=prim, seed=seed, value=v, description="pairwise - marginal signed-pair rate, n=4096"))
        for d, g in q.groupby("d"):
            out.append(dict(estimand=f"P4.d{d}.{lev}", family="P4-sec", stratum=f"{lev}/d={d}", primary=False, seed=seed,
                            value=float(np.mean(g.cert_pair != 0) - np.mean(g.cert_marg != 0)),
                            description="pairwise - marginal signed-pair rate, n=4096, per d"))
    return out


def collect(tag, rec_dir, pair_dir):
    vals = []
    for s in seeds_in(rec_dir, "records"):
        R = _s(read_table(os.path.join(rec_dir, f"records_seed{s}"), columns=RCOLS, filters=[("n_anchor", "==", NMAX)]))
        vals += seed_values_records(R, s)
        print(f"[STAT-01] {tag}: records seed {s}", flush=True)
    if pair_dir and os.path.isdir(pair_dir):
        for s in seeds_in(pair_dir, "pairs"):
            Pp = _s(read_table(os.path.join(pair_dir, f"pairs_seed{s}"), columns=PCOLS,
                               filters=[("n_anchor", "==", NMAX)]))
            vals += seed_values_pairs(Pp, s)
            print(f"[STAT-01] {tag}: pairs seed {s}", flush=True)
    df = pd.DataFrame(vals)
    df["partition"] = tag
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", action="append", required=True, help="tag:records_dir:pairs_dir")
    ap.add_argument("--inferential", default="stat01")
    ap.add_argument("--lock", default=os.path.join(ROOT, "protocols", "locks", "STAT-01.lock.json"))
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v12"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lock = verify_lock(args.lock) if os.path.exists(args.lock) else dict(ok=False, missing=[args.lock], changed=[])
    if not lock["ok"]:
        print("WARNING: protocol lock missing or changed — results are labelled as DEVIATING from the locked protocol:",
              lock, flush=True)
    V = []
    for spec in args.partition:
        tag, rec, pair = (spec.split(":") + [None, None])[:3]
        V.append(collect(tag, rec, pair))
    V = pd.concat(V, ignore_index=True)
    V.to_csv(os.path.join(args.out, "stat01_seed_values.csv"), index=False)

    rows = []
    for (tag, eid), g in V.groupby(["partition", "estimand"]):
        fam = g.family.iloc[0]
        fam_size = V[(V.partition == tag) & (V.family == fam)].estimand.nunique()
        s = summarize(g.value.to_numpy(), family_size=fam_size, B=args.B)
        inferential = (tag == args.inferential) and s["n_seeds"] >= MIN_SEEDS_FOR_CI
        if not inferential:
            for k in ("t_lo", "t_hi", "p_two_sided", "boot_lo", "boot_hi", "bonf_lo", "bonf_hi"):
                s[k] = np.nan
        rows.append(dict(partition=tag, estimand=eid, family=fam, stratum=g.stratum.iloc[0],
                         primary=bool(g.primary.iloc[0]), description=g.description.iloc[0], family_size=fam_size,
                         inferential=inferential, lock_ok=bool(lock["ok"]), **s))
    E = pd.DataFrame(rows)
    E["p_holm"] = np.nan
    for (tag, fam), g in E[E.inferential].groupby(["partition", "family"]):
        E.loc[g.index, "p_holm"] = holm(g.p_two_sided.to_numpy())
    E.to_csv(os.path.join(args.out, "stat01_effects.csv"), index=False)

    def table(sel, label, caption):
        rows = []
        for _, r in sel.sort_values(["family", "estimand"]).iterrows():
            rows.append([r.estimand.replace("_", "\\_"), r.stratum.replace("_", "\\_"), int(r.n_seeds), fnum(r["mean"], 3),
                         f"[{fnum(r.t_lo, 3)}, {fnum(r.t_hi, 3)}]", f"[{fnum(r.boot_lo, 3)}, {fnum(r.boot_hi, 3)}]",
                         f"[{fnum(r.bonf_lo, 3)}, {fnum(r.bonf_hi, 3)}]",
                         ("--" if not np.isfinite(r.p_holm) else f"{r.p_holm:.2g}"),
                         f"{fnum(r['min'], 3)}--{fnum(r['max'], 3)}"])
        return tex_table(["estimand", "stratum", "seeds", "mean", "95\\% $t$", "bootstrap", "Bonferroni", "Holm $p$",
                          "seed range"], rows, caption, label, align="llrrllllr", size="\\scriptsize")

    inf = E[E.partition == args.inferential]
    note = ("" if lock["ok"] else " \\textbf{The protocol lock did not verify; these results deviate from the locked "
            "protocol (see stat01\\_manifest.json).}")
    cap = (f"STAT-01 primary comparisons on the new independent seeds (partition \\texttt{{{args.inferential}}}). "
           "Unit: seed (one matched aggregate per seed); feature-instances are never resampled. Intervals describe "
           "variability over the random-seed design of the synthetic study, not population uncertainty. Holm adjustment "
           "within each family; P3 for \\texttt{linear}/\\texttt{unstable\\_id} is a structural zero and is not tested."
           + note)
    with open(os.path.join(args.out, "stat01_primary.tex"), "w") as fh:
        fh.write(table(inf[inf.primary], "tab:stat01", cap))
    with open(os.path.join(args.out, "stat01_secondary.tex"), "w") as fh:
        fh.write(table(inf[~inf.primary], "tab:stat01sec", "STAT-01 secondary estimands (no multiplicity claim)." + note))
    desc = E[E.partition != args.inferential]
    desc[["partition", "estimand", "stratum", "n_seeds", "mean", "min", "max"]].to_csv(
        os.path.join(args.out, "stat01_historical_descriptive.csv"), index=False)

    with open(os.path.join(args.out, "uncertainty_methods_v12.md"), "w") as fh:
        fh.write(f"""# Uncertainty methods (v1.2, STAT-01)

* Independent unit: the seed. Within a seed each (d, rho_c) block shares one sigma draw, one set of 20 (x, r) pairs and
  one nested anchor chain per anchor law across generators and modes, so feature-instances are dependent and are never
  resampled.
* Each estimand is reduced to one matched aggregate per seed (definitions in protocols/v12_preregistration.json).
* Inferential partition: `{args.inferential}` (new seeds, locked before running; lock verified: {lock['ok']}).
  95% t interval over seeds (df = S - 1); percentile bootstrap over seeds (B = {args.B}) as a sensitivity analysis;
  two-sided one-sample t test of mean zero with Holm adjustment within each pre-specified family; Bonferroni-
  simultaneous t intervals within family also reported. Intervals require at least {MIN_SEEDS_FOR_CI} seeds.
* Historical partitions (seeds 1-4 exploration, 8-11 confirmation) are reported with the same estimand code but
  descriptively (mean, min, max) — no intervals, no re-labelling of v1.1 outcomes.
* Interpretation: intervals quantify variability over the random design of this synthetic study (x, r, sigma, anchor
  chains); they are not statements about real-world models or datasets.
* P4 is nonnegative by construction (pairwise bounds dominate marginal comparisons); its interval measures magnitude.
""")
    N = Numbers("STAT-01", args.out)
    for _, r in inf[inf.primary].iterrows():
        key = r.estimand.replace(".", "_")
        N.add(f"{key}.mean", r["mean"], f"{r.description} ({r.stratum}), seed mean", "stat01_effects.csv", nd=3)
        N.add(f"{key}.ci", f"[{fnum(r.t_lo, 3)}, {fnum(r.t_hi, 3)}]", "95% seed-level t interval", "stat01_effects.csv")
    N.add("n_seeds", str(int(inf.n_seeds.max()) if len(inf) else 0), "independent seeds in the inferential partition",
          "stat01_effects.csv")
    N.write()
    write_json(os.path.join(args.out, "stat01_manifest.json"),
               base_manifest(experiment="STAT-01", lock=lock, partitions=args.partition, inferential=args.inferential,
                             B=args.B))
    print(inf[["estimand", "n_seeds", "mean", "t_lo", "t_hi", "p_holm"]].to_string(index=False))
    print("lock:", json.dumps(lock))


if __name__ == "__main__":
    main()
