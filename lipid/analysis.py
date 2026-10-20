"""
lipid/analysis.py — seed-level summaries, pre-registered criteria (reconstructed), text numbers.

Uncertainty convention (REP-01): the independent unit is the SEED (one anchor chain per (seed, d, rho_corr, law),
shared by all 20 query/reference pairs, generators and modes). Only 4 confirmation seeds exist, so every reported
spread is the descriptive seed min--max; no confidence interval is computed (MIN_CLUSTERS_FOR_CI = 10).
"""
from __future__ import annotations

import math

import numpy as np

MIN_CLUSTERS_FOR_CI = 10
CI_NOTE = "descriptive seed min-max; CI unreliable (fewer than 10 independent seeds)"

HISTORICAL_STATUS = {   # Table 5 of the draft (as reported for the P0'' confirmation run, seeds 8-11)
    "C1": "Met",
    "C2a": "Met",
    "C2b": "Not met; 4/160 cells outside tolerance",
    "C3": "Met; partly structural",
    "C4a": "Met",
    "C4b": "Partly met; d = 4 fails in 2/4 seeds",
    "C5": "Invalid specification; contradicted by domain results",
}
RULE_TEXT = {   # abbreviated rules as printed in Table 5 (the original locked protocol text is NOT in the archive)
    "C1": "1024-4096 width reduction decreases strictly with d in both modes",
    "C2a": "For rho_c = .99, data-product width gap > 0.1 at d = 3, positive at all d",
    "C2b": "At rho_c = 0, absolute gap < .03 in every cell",
    "C3": "Optimistic containment in linear generator <= 1 - 1/d; equality at n = 4096",
    "C4a": "For 1024-4096, anchor information dominates recalibration",
    "C4b": "For 16-64, fixed d >= 4, total optimistic width increases",
    "C5": "Uniformly zero geometry/range certification for linear/unstable in both modes",
}
OPERATIONALIZATION = {   # how each abbreviated rule is evaluated here (data anchors, L* unless stated)
    "C1": "per seed and mode: mean dRho at n=4096 (rho(A_1024;L*) - rho(A_4096;L*)), strictly decreasing over d=3..6",
    "C2a": "per seed and mode, rho_c=.99, n=4096: mean dpair(d=3) > 0.1 and mean dpair(d) > 0 for all d",
    "C2b": "per seed x d x mode x n (160 cells), rho_c=0: |mean dpair| < 0.03",
    "C3": "per seed x d x mode x n, linear: cover_opt <= 1-1/d (+1e-12); |cover_opt - (1-1/d)| < 1e-9 at n=4096",
    "C4a": "per seed x d x mode at n=4096: |mean N2_info| > |mean N2_calib|",
    "C4b": "per seed x d in {4,5,6}, fixed mode, n=64: mean N2_total > 0",
    "C5": "per seed x d x mode x n, generators linear and unstable_id: p_uni_com == 0",
}


def seed_stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(n_seeds=0, mean=math.nan, sd=math.nan, min=math.nan, max=math.nan, note=CI_NOTE)
    return dict(n_seeds=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else math.nan,
                min=float(v.min()), max=float(v.max()),
                note=(CI_NOTE if v.size < MIN_CLUSTERS_FOR_CI else "seed-level t interval admissible"))


def _tab(S, seed, name):
    seeds = S["seeds"]
    key = str(seed) if str(seed) in seeds else seed
    return seeds[key][name]


def criteria(S: dict, seeds, dims=(3, 4, 5, 6), ns=(16, 64, 256, 1024, 4096)) -> tuple[list, dict]:
    """Mechanical evaluation of the reconstructed C1-C5 rules on a nested_anchors_v2-style summary.
    Returns (cell rows, status per criterion)."""
    rows = []

    def add(cid, seed, cell, value, ok):
        rows.append(dict(criterion=cid, seed=seed, cell=cell, value=value, passed=bool(ok)))

    for s in seeds:
        T = _tab(S, s, "by_d_mode_arm_n")
        TR = _tab(S, s, "by_d_mode_arm_rhoc_n")
        TM = _tab(S, s, "by_d_mode_arm_model_n")
        for mode in ("domain", "fixed"):
            v = [T[f"{d}/{mode}/data/4096"]["mean_dRho"] for d in dims]
            add("C1", s, f"{mode}", v, all(v[i] > v[i + 1] for i in range(len(v) - 1)))
            g = [TR[f"{d}/{mode}/prod/0.99/4096"]["mean_dpair"] for d in dims]
            add("C2a", s, f"{mode}", g, g[0] > 0.1 and all(x > 0 for x in g))
            for d in dims:
                for n in ns:
                    x = TR[f"{d}/{mode}/prod/0.0/{n}"]["mean_dpair"]
                    add("C2b", s, f"d={d}/{mode}/n={n}", x, abs(x) < 0.03)
                    c = TM[f"{d}/{mode}/data/linear/{n}"]["cover_opt"]
                    ok = c <= 1 - 1 / d + 1e-12 and (n != max(ns) or abs(c - (1 - 1 / d)) < 1e-9)
                    add("C3", s, f"d={d}/{mode}/n={n}", c, ok)
                    for mo in ("linear", "unstable_id"):
                        u = TM[f"{d}/{mode}/data/{mo}/{n}"]["p_uni_com"]
                        add("C5", s, f"d={d}/{mode}/{mo}/n={n}", u, u == 0)
                e = T[f"{d}/{mode}/data/4096"]
                add("C4a", s, f"d={d}/{mode}", [e["N2_info"], e["N2_calib"]], abs(e["N2_info"]) > abs(e["N2_calib"]))
        for d in dims:
            if d >= 4:
                x = T[f"{d}/fixed/data/64"]["N2_total"]
                add("C4b", s, f"d={d}/fixed", x, x > 0)
    status = {}
    for cid in RULE_TEXT:
        r = [x for x in rows if x["criterion"] == cid]
        nf = sum(not x["passed"] for x in r)
        status[cid] = dict(n_cells=len(r), n_fail=nf,
                           mechanical=("met" if nf == 0 else f"not met ({nf}/{len(r)} cells fail)"),
                           historical=HISTORICAL_STATUS[cid], rule=RULE_TEXT[cid],
                           operationalization=OPERATIONALIZATION[cid])
    return rows, status


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{nd}f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def rng(values, nd=2):
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return "--"
    a, b = fmt(min(v), nd), fmt(max(v), nd)
    return a if a == b else f"{a}--{b}"
