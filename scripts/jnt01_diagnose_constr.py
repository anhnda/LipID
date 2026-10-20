#!/usr/bin/env python3
"""
jnt01_diagnose_constr.py — POST-HOC diagnosis of the JNT-01 `fail_constr` rows. It does NOT change the JNT-01 verdict.

    python scripts/jnt01_diagnose_constr.py            # reads results_v12/jnt01_rows.csv.gz

Pre-specified check that failed: the extremal labeling of the pairwise corollary, together with the anchor labels, must
be L-Lipschitz with violation <= 1e-12 * L * D (float64). This script re-evaluates every failing row in EXACT rational
arithmetic (fractions.Fraction built from the float64 inputs that the run used: x, r, sigma, anchors, labels, L), so
floating-point round-off is removed. Each row is classified as
  L_below_exact_LA   the float level L is smaller than the exact anchor slope L_A of the same data: the class F_L(A)
                     is EMPTY in exact arithmetic, the corollary's hypothesis L >= L_A fails, the check is not applicable
  roundoff_only      L >= exact L_A and the exact violation is <= 0: the float violation was round-off
  GENUINE            L >= exact L_A and the exact violation is > 0: a counterexample to the construction (proof bug)
Outputs: results_v12/jnt01_constr_diagnosis.csv and .md. The md states explicitly that this analysis is post hoc.
"""
import os
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

import jnt01_validate_pairwise as J  # noqa: E402
from lipid.pairwise import pair_index_np  # noqa: E402


def F(t):
    return [Fraction(float(v)) for v in t.reshape(-1).tolist()]


def dist(u, v, s):
    return sum(abs(a - b) / c for a, b, c in zip(u, v, s))


def exact_check(x, r, A, fA, L, sigma, a_feat, b_feat):
    xs, rs, ss = F(x), F(r), F(sigma)
    As = [F(row) for row in A]
    fs = F(fA)
    Lq = Fraction(float(L))
    d = len(xs)
    active = [j for j in range(d) if xs[j] != rs[j]]
    m = len(active)
    # exact anchor slope
    LA = Fraction(0)
    for i in range(len(As)):
        for k in range(i + 1, len(As)):
            dd = dist(As[i], As[k], ss)
            if dd > 0:
                LA = max(LA, abs(fs[i] - fs[k]) / dd)
            elif fs[i] != fs[k]:
                LA = None
                break
    verts = []
    for mask in range(1 << m):
        z = list(rs)
        for jj, j in enumerate(active):
            if (mask >> jj) & 1:
                z[j] = xs[j]
        verts.append(z)

    def ell(z):
        return max(f - Lq * dist(z, a, ss) for a, f in zip(As, fs))

    def h(z):
        return min(f + Lq * dist(z, a, ss) for a, f in zip(As, fs))

    pairs, P, Q, B, _, _ = pair_index_np(m)
    pa, pb = active.index(a_feat), active.index(b_feat)
    p = pairs.index((min(pa, pb), max(pa, pb)))
    delta = [abs(xs[j] - rs[j]) / ss[j] for j in active]
    c = Lq * (delta[pa] + delta[pb])
    worst = {}
    for side in ("lower", "upper"):
        pts, vals = [], []
        for s_p, s_q in zip(P[p].tolist(), Q[p].tolist()):
            zP, zQ = verts[s_p], verts[s_q]
            if side == "upper":
                vQ = ell(zQ)
                vP = min(h(zP), vQ + c)
            else:
                vQ = h(zQ)
                vP = max(ell(zP), vQ - c)
            pts += [zP, zQ]
            vals += [vP, vQ]
        v_pp = max(abs(vals[i] - vals[k]) - Lq * dist(pts[i], pts[k], ss)
                   for i in range(len(pts)) for k in range(len(pts)))
        v_pa = max(abs(v - f) - Lq * dist(z, a, ss) for z, v in zip(pts, vals) for a, f in zip(As, fs))
        worst[side] = max(v_pp, v_pa)
    return LA, Lq, max(worst.values()), sum(delta)


def main():
    out = os.path.join(ROOT, "results_v12")
    df = pd.read_csv(os.path.join(out, "jnt01_rows.csv.gz"))
    bad = df[df.constr_viol > J.TOL_CONSTR].copy()
    print(f"{len(bad)} rows with constr_viol > {J.TOL_CONSTR} (of {len(df)})")
    rows = []
    for _, rw in bad.iterrows():
        if rw.case != "grid":
            rows.append(dict(rw.to_dict(), cls="edge_case_not_rechecked"))
            continue
        x, r, A, fA, sigma, f, Lc = J.grid_instance(int(rw.d), rw["mode"], int(rw.n_anchor), rw.scale, rw.model, int(rw.rep))
        _, levels = J.levels_for(x, r, A, fA, sigma, f, Lc)
        L = levels[rw.level]
        assert abs(L - rw.L) <= 1e-15 * max(1.0, abs(L)), "instance not reproduced"
        LA, Lq, viol, D = exact_check(x, r, A, fA, L, sigma, int(rw.a), int(rw.b))
        if LA is None:
            cls = "duplicate_anchor_conflict"
        elif Lq < LA:
            cls = "L_below_exact_LA"
        elif viol <= 0:
            cls = "roundoff_only"
        else:
            cls = "GENUINE"
        rows.append(dict(case=rw.case, d=int(rw.d), mode=rw["mode"], n_anchor=int(rw.n_anchor), scale=rw.scale,
                         model=rw.model, rep=int(rw.rep), level=rw.level, a=int(rw.a), b=int(rw.b),
                         float_constr_viol_rel=float(rw.constr_viol),
                         L_float=float(Lq), L_A_exact=(float(LA) if LA is not None else np.nan),
                         L_minus_LA_rel=(float((Lq - LA) / LA) if LA else np.nan),
                         exact_violation_rel=float(viol / (Lq * D)) if Lq > 0 else np.nan, cls=cls))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "jnt01_constr_diagnosis.csv"), index=False)
    by_cls = R.cls.value_counts().to_dict() if len(R) else {}
    by_level = R.groupby(["level", "cls"]).size().to_dict() if len(R) else {}
    md = ["# JNT-01 `fail_constr`: post-hoc diagnosis", "",
          "**Status of the pre-specified check: NOT MET** (verdict in jnt01_manifest.json is unchanged).",
          "This diagnosis was written after seeing the failure; it explains it but does not re-label it.", "",
          f"* rows failing the float check (> {J.TOL_CONSTR} relative): {len(bad)} of {len(df)}",
          f"* failing values (relative to L*D): min {bad.constr_viol.min():.3g}, max {bad.constr_viol.max():.3g}"
          if len(bad) else "* none", f"* classes (exact rational re-check): {by_cls}", f"* by level: {by_level}", "",
          "Interpretation rules (fixed in this script before reading its output):",
          "* `L_below_exact_LA`: the float L_A (level `LA`) is below the exact anchor slope; the hypothesis L >= L_A of",
          "  the corollary fails in exact arithmetic, so the check is not applicable to that row. The pre-specified",
          "  tolerance did not anticipate this; that is a specification error of the check, to be reported as such.",
          "* `roundoff_only`: float64 round-off exceeded 1e-12 relative; the exact construction is valid.",
          "* `GENUINE`: the attaining construction is wrong for that row; the proof must be fixed before submission,",
          "  even though the LP agreed with the closed form."]
    with open(os.path.join(out, "jnt01_constr_diagnosis.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    if len(R):
        print(R.to_string(index=False))


if __name__ == "__main__":
    main()
