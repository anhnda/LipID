#!/usr/bin/env python3
"""
build_revision.py — collect the v1.2 outputs for the manuscript and the response to reviewers.

    python scripts/build_revision.py

Reads whatever the v1.2 steps have produced in results_v12/ and figures_v12/ (missing steps are listed, not faked):
  numbers_v12.tex     \\lipnum{<key>} for every number quoted in the text (from numbers_*.csv)
  numbers_v12.csv     the same with definitions and source files
  main_tables_v12.tex \\input lines for every LaTeX fragment that exists
  crosswalk_v12.md    reviewer question -> experiment -> output files -> manuscript location, with status
  build_report_v12.json  integrity checks gathered from every experiment; `issues` must be empty before submission
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from lipid.provenance import git_info, now  # noqa: E402

OUT = os.path.join(ROOT, "results_v12")
FIG = os.path.join(ROOT, "figures_v12")

CROSSWALK = [
    ("Q3", "A1 Proposition J (joint polytope)", "theory", [], "Sec. 4 (new Prop. 4.2) + App. A.2",
     "paper/v12_theory.tex"),
    ("Q3", "A2 Corollary (sharp pairwise bounds)", "theory", [], "Sec. 4 (Cor. 4.3) + App. A.3", "paper/v12_theory.tex"),
    ("Q3", "A3 exact ranking certificates", "theory", [], "Sec. 4 (Prop. 4.4, replaces the 'conservative' remark)",
     "paper/v12_theory.tex"),
    ("Q3", "JNT-01 validation", "verification", ["jnt01_table.tex", "jnt01_summary.csv", "jnt01_edgecases.csv"],
     "App. A.4 (Table A1); one sentence in Sec. 4", "scripts/jnt01_validate_pairwise.py"),
    ("Q3", "JNT-02 ranking gains", "descriptive", ["jnt02_main.tex", "jnt02_subgroups.tex", "jnt02_summary.csv",
                                                    "../figures_v12/fig_jnt02_certification.pdf"],
     "Sec. 6.x new subsection 'Joint ranking information' (Table + Fig.); subgroups in supplement",
     "scripts/run_pairwise.py, scripts/jnt02_analyze.py"),
    ("Q3", "JNT-03 absolute ranking", "descriptive", ["jnt03_table.tex", "jnt03_summary.csv"],
     "Supplement (main text only if material)", "scripts/jnt03_absolute_lp.py"),
    ("Q7", "STAT-01 independent seeds", "confirmatory", ["stat01_primary.tex", "stat01_secondary.tex",
                                                          "stat01_effects.csv", "uncertainty_methods_v12.md"],
     "Sec. 6.1 (unit of uncertainty) + new Table in Sec. 6; methods note in supplement",
     "scripts/stat01_analyze.py"),
    ("Q7", "REP-01 artifacts", "verification", ["verification_report.md"], "Reproducibility section",
     "scripts/verify_release.py, scripts/clean_env_reproduce.sh, scripts/make_release.sh"),
    ("Q2", "MET-02 scale sensitivity", "descriptive", ["met02_table.tex", "met02_real.tex", "met02_summary.csv",
                                                        "../figures_v12/fig_met02.pdf"],
     "Sec. 7.2 (one paragraph + supplementary table)", "scripts/met02_scale_sensitivity.py"),
    ("Q5", "SCL-02 non-saturated MC", "descriptive", ["scl02_table.tex", "scl02_summary.csv",
                                                       "../figures_v12/fig_scl02_excess.pdf"],
     "Sec. 7.1 (replaces the 'needs non-saturated benchmark' remark)", "scripts/scl02_nonsaturated.py"),
    ("Q6", "CERT-02 certificate diagnostics", "diagnostic", ["cert02_table.tex", "cert02_summary.csv",
                                                              "../figures_v12/fig_cert02_classes.pdf"],
     "Sec. 7 'Observed limitations' + Prop. (relaxation identity) in App. E", "scripts/cert02_diagnostics.py"),
    ("Q4", "conditional / manifold Shapley", "discussion", [], "Sec. 2 + new Sec. 8 paragraph",
     "paper/v12_discussion.tex"),
    ("Q1", "operational calibration", "discussion", [], "Sec. 5.1 + App. H wording", "paper/v12_discussion.tex"),
]


def gather_checks():
    issues, checks = [], {}

    def load(name):
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    m = load("jnt01_manifest.json")
    if m:
        checks["JNT-01"] = m.get("verdict")
        if not m.get("verdict", {}).get("accepted", False):
            issues.append("JNT-01 verdict not accepted (see jnt01_manifest.json)")
    j = load("jnt02_integrity.json")
    if j:
        checks["JNT-02"] = j
        for k in ("not_contained_valid_L", "v_dom", "top1_pair_wrong_valid_L"):
            if j.get(k):
                issues.append(f"JNT-02 {k} = {j[k]} (must be 0)")
    m = load("jnt03_manifest.json")
    if m:
        c = m.get("checks", {})
        checks["JNT-03"] = c
        for k in ("exact_weaker_than_prop2", "sum_not_contained_valid", "hl_disagree"):
            if c.get(k):
                issues.append(f"JNT-03 {k} = {c[k]}")
    c = load("met02_checks.json")
    if c:
        checks["MET-02"] = c
        if (c.get("max_invariance_violation") or 0) > 1e-9:
            issues.append("MET-02 scale invariance violated beyond 1e-9")
        if c.get("containment_cert_min", 1.0) < 1.0 or c.get("pair_not_contained_valid", 0):
            issues.append("MET-02 containment failure at a valid level")
    c = load("cert02_checks.json")
    if c:
        checks["CERT-02"] = c
        if any(c.get(k) for k in ("v_exact_ge_abs", "v_abs_ge_pairbest", "v_pairbest_ge_uni", "v_geo_ge_t3")):
            issues.append("CERT-02 relaxation chain order violated")
    m = load("stat01_manifest.json")
    if m:
        checks["STAT-01"] = dict(lock=m.get("lock"))
        if not (m.get("lock") or {}).get("ok"):
            issues.append("STAT-01 analysed with a lock that does not verify: report as deviation")
    v = load("verification_report.json")
    if v:
        checks["REP-01"] = dict(ok=v.get("ok"), scope=v.get("scope"))
        if not v.get("ok"):
            issues.append("REP-01 verification failed")
    return checks, issues


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(OUT, "numbers_*.csv")))
              if not p.endswith("numbers_v12.csv")]
    nums = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["key", "value", "definition", "source"])
    nums.to_csv(os.path.join(OUT, "numbers_v12.csv"), index=False)
    lines = ["% generated by scripts/build_revision.py — do not edit", "\\makeatletter",
             "\\providecommand{\\lipnum}[1]{\\@ifundefined{lipnum@#1}{\\textbf{??#1??}}{\\@nameuse{lipnum@#1}}}"]
    for _, r in nums.iterrows():
        val = str(r["value"]).replace("%", "\\%")
        lines.append(f"\\@namedef{{lipnum@{r['key']}}}{{{val}}}")
    lines.append("\\makeatother")
    with open(os.path.join(OUT, "numbers_v12.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    frags = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "*.tex"))
                   if not p.endswith(("numbers_v12.tex", "main_tables_v12.tex")))
    with open(os.path.join(OUT, "main_tables_v12.tex"), "w") as fh:
        fh.write("% generated by scripts/build_revision.py\n" + "".join(f"\\input{{results_v12/{f}}}\n" for f in frags))
    checks, issues = gather_checks()
    md = [f"# LipID v1.2 crosswalk ({now()}, commit {git_info().get('commit')})", "",
          "| reviewer | item | type | status | outputs | manuscript location | code |", "|---|---|---|---|---|---|---|"]
    for q, item, kind, files, loc, code in CROSSWALK:
        have = [f for f in files if os.path.exists(os.path.join(OUT, f))]
        status = ("text" if not files else ("complete" if len(have) == len(files) else
                                             ("partial" if have else "NOT RUN")))
        md.append(f"| {q} | {item} | {kind} | {status} | {', '.join(files) or '-'} | {loc} | {code} |")
    md += ["", "## Integrity checks", "", "```json", json.dumps(checks, indent=1, default=str), "```", "",
           "## Issues (must be empty before submission)", ""] + ([f"* {i}" for i in issues] or ["* none"])
    with open(os.path.join(OUT, "crosswalk_v12.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "build_report_v12.json"), "w") as fh:
        json.dump(dict(created=now(), git=git_info(), checks=checks, issues=issues, fragments=frags,
                       n_numbers=int(len(nums))), fh, indent=1, default=str)
    print("\n".join(md))
    if issues:
        print("\nISSUES:\n" + "\n".join(issues))


if __name__ == "__main__":
    main()
