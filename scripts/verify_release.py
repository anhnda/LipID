#!/usr/bin/env python3
"""
verify_release.py — REP-01: artifact verification report (reviewer Q7).

    # 1) integrity of what is on disk: every manifest's output checksums, and source hashes vs the manifests
    python scripts/verify_release.py --check-manifests
    # 2) a clean-environment rerun (scripts/clean_env_reproduce.sh) compared with the archived outputs
    python scripts/verify_release.py --check-manifests --compare results repro_check/results \
        --compare results_v12 repro_check/results_v12 --scope "tests + JNT-01 + metric + real (clean venv, CPU)"

Writes results_v12/verification_report.{json,md}. The report states the SCOPE of what was independently re-run;
anything outside the scope is listed as "not re-run" rather than implied. Exit code 1 on any mismatch.
Numeric comparison of CSV files: same columns and row count, |a - b| <= atol + rtol |b| for numeric columns
(rtol 1e-9, atol 1e-12 by default), exact equality otherwise. Parquet/CSV.gz row tables are compared the same way.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lipid.provenance import environment, git_info, now, sha256, source_checksums  # noqa: E402

TABLE_EXT = (".csv", ".csv.gz", ".parquet")
SKIP_COLS = {"runtime_sec", "max_rss_mb", "cuda_peak_mb", "seconds", "time_median_s", "time_max_s", "peak_rss_mb",
             "rss_after_setup_mb", "t_closed_per_pair", "t_lp_allpairs", "t_lp_edges", "t_closed_s", "t_lp_sum_s",
             "lp_seconds", "edge_seconds", "median_lp_ms", "median_edge_ms", "us_closed", "ms_lp_allpairs",
             "ms_lp_edges", "time_s", "cuda_peak_mb_config"}


def check_manifests(root):
    out = []
    for man in sorted(glob.glob(os.path.join(root, "runs", "*", "manifest.json"))
                      + glob.glob(os.path.join(root, "results", "*manifest*.json"))
                      + glob.glob(os.path.join(root, "results_v12", "*manifest*.json"))):
        with open(man) as fh:
            M = json.load(fh)
        bad, missing = [], []
        for rel, meta in (M.get("outputs") or {}).items():
            p = rel if os.path.isabs(rel) else os.path.join(root, rel)
            if not os.path.isfile(p):
                missing.append(rel)
            elif sha256(p) != meta["sha256"]:
                bad.append(rel)
        src = M.get("source_sha256") or {}
        cur = source_checksums(root)
        changed_src = sorted(f for f, h in src.items() if cur.get(f) not in (None, h))
        out.append(dict(manifest=os.path.relpath(man, root), commit=(M.get("git") or {}).get("commit"),
                        dirty=(M.get("git") or {}).get("dirty"), n_outputs=len(M.get("outputs") or {}),
                        checksum_mismatch=bad, missing=missing, source_changed_since=changed_src))
    return out


def read_any(p):
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def compare_tables(a, b, rtol, atol):
    A, B = read_any(a), read_any(b)
    res = dict(file=b, rows=(len(A), len(B)), ok=True, issues=[])
    if list(A.columns) != list(B.columns):
        res["ok"] = False
        res["issues"].append(f"columns differ: {sorted(set(A.columns) ^ set(B.columns))}")
        return res
    if len(A) != len(B):
        res["ok"] = False
        res["issues"].append("row count differs")
        return res
    worst = {}
    for c in A.columns:
        if c in SKIP_COLS:
            continue
        x, y = A[c], B[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            diff = np.where(both_nan, 0.0, np.abs(xv - yv))
            lim = atol + rtol * np.abs(yv)
            bad = (diff > lim) | (np.isnan(xv) != np.isnan(yv))
            worst[c] = float(np.nanmax(diff)) if diff.size else 0.0
            if bad.any():
                res["ok"] = False
                res["issues"].append(f"{c}: {int(bad.sum())} values beyond tolerance (max diff {worst[c]:.3g})")
        else:
            ne = (x.astype(str).to_numpy() != y.astype(str).to_numpy())
            if ne.any():
                res["ok"] = False
                res["issues"].append(f"{c}: {int(ne.sum())} non-numeric values differ")
    res["max_abs_diff"] = worst
    return res


def compare_dirs(da, db, rtol, atol):
    out, not_in_b = [], []
    for p in sorted(glob.glob(os.path.join(da, "**", "*"), recursive=True)):
        if not p.endswith(TABLE_EXT) or "/matched/" in p or "scaling_cache" in p:
            continue
        q = os.path.join(db, os.path.relpath(p, da))
        if not os.path.exists(q):
            not_in_b.append(os.path.relpath(p, da))
            continue
        try:
            out.append(compare_tables(p, q, rtol, atol))
        except Exception as e:  # pragma: no cover
            out.append(dict(file=q, ok=False, issues=[f"read error {e!r}"]))
    return out, not_in_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--check-manifests", action="store_true")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("ARCHIVED", "RERUN"))
    ap.add_argument("--scope", default="", help="plain-language description of what the rerun covered")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v12"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rep = dict(created=now(), git=git_info(args.root), environment=environment(), scope=args.scope,
               tolerances=dict(rtol=args.rtol, atol=args.atol))
    ok = True
    if args.check_manifests:
        rep["manifests"] = check_manifests(args.root)
        ok &= all(not m["checksum_mismatch"] and not m["missing"] for m in rep["manifests"])
    rep["comparisons"] = []
    for a, b in args.compare:
        res, nb = compare_dirs(a, b, args.rtol, args.atol)
        rep["comparisons"].append(dict(archived=a, rerun=b, n_files=len(res), n_ok=sum(r["ok"] for r in res),
                                       failures=[r for r in res if not r["ok"]], not_rerun=nb))
        ok &= all(r["ok"] for r in res)
    rep["ok"] = bool(ok)
    with open(os.path.join(args.out, "verification_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    L = [f"# Verification report ({rep['created']})", "",
         f"* commit: `{rep['git'].get('commit')}` (dirty: {rep['git'].get('dirty')})",
         f"* scope of the independent rerun: {args.scope or '(none declared)'}",
         f"* tolerances: rtol = {args.rtol}, atol = {args.atol} (timing / memory columns are not compared)",
         f"* overall: **{'PASS' if ok else 'FAIL'}**", ""]
    if args.check_manifests:
        L += ["## Manifests", "", "| manifest | outputs | checksum mismatches | missing | source changed since run |",
              "|---|---|---|---|---|"]
        for m in rep["manifests"]:
            L.append(f"| {m['manifest']} | {m['n_outputs']} | {len(m['checksum_mismatch'])} | {len(m['missing'])} | "
                     f"{len(m['source_changed_since'])} |")
        L.append("")
    for c in rep["comparisons"]:
        L += [f"## {c['archived']}  vs  {c['rerun']}", "", f"* tables compared: {c['n_files']}, identical within "
              f"tolerance: {c['n_ok']}", f"* tables NOT re-run (present only in the archive): {len(c['not_rerun'])}"]
        for nb in c["not_rerun"][:200]:
            L.append(f"  * not re-run: `{nb}`")
        for f in c["failures"]:
            L.append(f"* FAIL `{f['file']}`: " + "; ".join(f["issues"]))
        L.append("")
    with open(os.path.join(args.out, "verification_report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
