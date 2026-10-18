#!/usr/bin/env python3
"""
summarize_records.py — compress phase1 records.jsonl into small tables.

Usage:
    python summarize_records.py phase1_out/records.jsonl --out tables
Needs only numpy. Reads the file as a stream (low memory) and skips broken lines.

Outputs (in --out):
    by_config.csv   one row per (d, rho, n_anchor, model, kappa, mode)
    by_d.csv, by_kappa.csv, by_d_kappa.csv, by_rho_kappa.csv, by_model_kappa.csv  (roll-ups)
and prints the roll-ups.

Columns
    n_inst, n_infeasible, infeas_rate
    transfer_rate   P(frozen phi outside [lo, hi])   among feasible   (assumption check)
    VL_rate         P(V_L > tol)  f provably NOT in the class
    rho_med, rho_frac_lt09, rho_frac_eq1   rho_i = B_pair / (2 L delta_i)
    sign_rate, topk_rate                   certified sign / top-k
    sp_B_D, sp_B_delta, sp_B_2Ld           Spearman of B_pair with baselines D_i, delta_i, 2L delta_i
    psp_B_D_given_delta                    partial Spearman of B_pair with D_i controlling for delta_i
Caveat: instances share the same x (n_eval_x per config) so they are correlated;
rates are descriptive, not independent samples.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

TOL = 1e-6
KEY = ("d", "rho", "n_anchor", "model", "kappa", "mode")


def rank(a):
    """Average ranks (ties handled)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    s = a[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return corr(rank(x), rank(y)) if len(x) >= 3 else float("nan")


def partial_spearman(x, y, z):
    rx, ry, rz = rank(x), rank(y), rank(z)
    rxy, rxz, ryz = corr(rx, ry), corr(rx, rz), corr(ry, rz)
    den = np.sqrt(max(1e-300, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / den)


class Acc:
    def __init__(self):
        self.n = self.infeas = self.trans = self.vl = 0
        self.topk_n = self.topk_cert = 0
        self.sign_n = self.sign_cert = 0
        self.rho, self.B, self.D, self.delta, self.twoLd = [], [], [], [], []

    def add(self, r):
        self.n += 1
        if r.get("infeasible", False):
            self.infeas += 1
            return
        self.trans += r["transfer_out"] > TOL
        self.vl += r["VL"] > TOL
        for j in range(r["d"]):
            t = r["twoLd"][j]
            if t > 0:
                self.rho.append(r["B_pair"][j] / t)
                self.B.append(r["B_pair"][j])
                self.D.append(r["D"][j])
                self.delta.append(r["delta"][j])
                self.twoLd.append(t)
                self.sign_n += 1
                self.sign_cert += int(r["sign_pair"][j] != 0)
        if r.get("topk"):
            self.topk_n += 1
            self.topk_cert += int(r["topk"]["certified"])

    def merge(self, o):
        self.n += o.n
        self.infeas += o.infeas
        self.trans += o.trans
        self.vl += o.vl
        self.topk_n += o.topk_n
        self.topk_cert += o.topk_cert
        self.sign_n += o.sign_n
        self.sign_cert += o.sign_cert
        for k in ("rho", "B", "D", "delta", "twoLd"):
            getattr(self, k).extend(getattr(o, k))

    def row(self):
        feas = self.n - self.infeas
        rho = np.asarray(self.rho)
        B, D, de, t = map(np.asarray, (self.B, self.D, self.delta, self.twoLd))
        nan = float("nan")
        return dict(
            n_inst=self.n, n_infeasible=self.infeas,
            infeas_rate=self.infeas / self.n if self.n else nan,
            transfer_rate=self.trans / feas if feas else nan,
            VL_rate=self.vl / feas if feas else nan,
            rho_med=float(np.median(rho)) if rho.size else nan,
            rho_frac_lt09=float(np.mean(rho < 0.9)) if rho.size else nan,
            rho_frac_eq1=float(np.mean(rho > 1 - 1e-9)) if rho.size else nan,
            sign_rate=self.sign_cert / self.sign_n if self.sign_n else nan,
            topk_rate=self.topk_cert / self.topk_n if self.topk_n else nan,
            sp_B_D=spearman(B, D) if B.size >= 3 else nan,
            sp_B_delta=spearman(B, de) if B.size >= 3 else nan,
            sp_B_2Ld=spearman(B, t) if B.size >= 3 else nan,
            psp_B_D_given_delta=partial_spearman(B, D, de) if B.size >= 3 else nan,
        )


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_table(path, keys, groups):
    rows = []
    for k in sorted(groups, key=lambda z: tuple(str(a) for a in z)):
        row = dict(zip(keys, k))
        row.update(groups[k].row())
        rows.append(row)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def print_table(title, rows, keys):
    cols = list(keys) + ["n_inst", "infeas_rate", "transfer_rate", "VL_rate", "rho_med",
                         "rho_frac_lt09", "sign_rate", "topk_rate", "sp_B_D", "psp_B_D_given_delta"]
    print(f"\n== {title} ==")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(fmt(r[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--every", type=int, default=1, help="keep every k-th line (subsample)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = defaultdict(Acc)
    bad = 0
    with open(args.records) as fh:
        for i, line in enumerate(fh):
            if i % args.every:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            cfg[tuple(r[k] for k in KEY)].add(r)
    print(f"configs={len(cfg)} skipped_bad_lines={bad}")

    write_table(os.path.join(args.out, "by_config.csv"), KEY, cfg)

    def rollup(name, keys):
        idx = [KEY.index(k) for k in keys]
        g = defaultdict(Acc)
        for k, a in cfg.items():
            g[tuple(k[i] for i in idx)].merge(a)
        rows = write_table(os.path.join(args.out, f"{name}.csv"), keys, g)
        print_table(name, rows, keys)

    rollup("by_d", ("d",))
    rollup("by_kappa", ("kappa",))
    rollup("by_d_kappa", ("d", "kappa"))
    rollup("by_rho_kappa", ("rho", "kappa"))
    rollup("by_model_kappa", ("model", "kappa"))
    rollup("by_nA_d", ("n_anchor", "d"))


if __name__ == "__main__":
    main()
