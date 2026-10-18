#!/usr/bin/env python3
"""
lowerbound.py — check the geometric sandwich  rho_low(kappa) <= rho(L) <= rho_inf  on an EXISTING run.
No LP and no model calls: the data are regenerated with the same generator keys as
phase1.run_optimistic, hybrid-to-anchor distances are recomputed, and rho(L) is read from optimistic.jsonl.

Per active feature i, per coalition S (not containing i), with z+ = z_{S+i}, z- = z_S:
    s_S      = (d(z+, A) + d(z-, A)) / delta_i                     (NOT clipped)
    rho_low  = sum_S w_S [min(1, (1 - 1/k) s_S - 1/k)]_+   ,  k = L / L_A(A) = L / L_min
    p_sat    = sum_S w_S 1{ s_S >= (k + 1) / (k - 1) }            (saturation mass, k > 1; rho >= p_sat)
    rho_inf  = sum_S w_S min(1, s_S)                               (must equal the stored rho_inf)

    python lowerbound.py phase1_out phase1_seed2 phase1_seed3 phase1_seed4
Writes <run>/lowerbound_summary.json and prints tables.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import phase1 as P
from bounds import all_masks, shapley_weights, vertex_points
from envelope import dist_to_set

S_THRESH = (1.0, 3.0, 5.0, 2.18 / 0.18)   # saturation thresholds for kappa = inf, 2, 1.5, 1.18
LEVELS = ("Lmin", "LV", "Ltrue")


def coalition_geometry(x, r, A, sigma):
    """{feature j: (w_S, s_S)} over coalitions S not containing j (reduced-game Shapley weights)."""
    delta = (x - r).abs() / sigma
    active = torch.nonzero(delta > 0).flatten().tolist()
    m = len(active)
    if m == 0:
        return {}
    masks = all_masks(m, x.device)
    Z = vertex_points(x, r, active, masks)
    q = dist_to_set(Z, A, sigma)
    w = shapley_weights(m, x.dtype, x.device)
    idx = torch.arange(2 ** m, device=x.device)
    pop = masks.sum(1)
    out = {}
    for jj, j in enumerate(active):
        bit = 1 << jj
        S = idx[(idx & bit) == 0]
        Pm = S | bit
        out[j] = (w[pop[S]].cpu().numpy(), ((q[Pm] + q[S]) / delta[j]).cpu().numpy())
    return out


def rho_low(w, s, k):
    v = np.minimum(1.0, (1.0 - 1.0 / k) * s - 1.0 / k)
    return float((w * np.clip(v, 0.0, None)).sum())


def p_sat(w, s, k):
    if k <= 1.0:
        return 0.0
    return float(w[s >= (k + 1.0) / (k - 1.0)].sum())


def process(run, device):
    cfg = json.load(open(os.path.join(run, "config.json")))
    tol = cfg["tol"]
    rows = {}
    with open(os.path.join(run, "optimistic.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("degenerate"):
                rows[(r["d"], r["rho"], r["n_anchor"], r["model"], r["x_idx"], r["r_idx"], r["mode"])] = r
    recs, geo = [], []
    regen_err, n_missing = 0.0, 0
    for d in cfg["dims"]:
        for rho in cfg["rhos"]:
            for nA in cfg["n_anchor"]:
                gd = P.gen_of("data", cfg["seed"], d, rho, nA)          # same keys and draw order as run_optimistic
                sample = P.gaussian_sampler(d, rho, device)
                sigma = P.robust_sigma(sample(cfg["n_train"], gd))
                C, Rc, Xev = sample(nA, gd), sample(cfg["n_candidate"], gd), sample(cfg["n_eval_x"], gd)
                plan = [(xi, torch.randperm(cfg["n_candidate"], generator=gd)[:cfg["refs_per_x"]].tolist())
                        for xi in range(cfg["n_eval_x"])]
                for xi, ridx in plan:
                    x = Xev[xi]
                    for rj in ridx:
                        r = Rc[rj]
                        for mode in ("domain", "fixed"):
                            A = torch.cat([C, x[None]]) if mode == "domain" else torch.cat([C, x[None], r[None]])
                            g = coalition_geometry(x, r, A, sigma)
                            for j, (w, s) in g.items():
                                geo.append(dict(d=d, mode=mode, mass={t: float(w[s >= t].sum()) for t in S_THRESH}))
                            for model in cfg["models"]:
                                row = rows.get((d, rho, nA, model, xi, rj, mode))
                                if row is None:
                                    n_missing += 1
                                    continue
                                for lev in LEVELS:
                                    e = row["levels"].get(lev)
                                    if not e:
                                        continue
                                    k = e["L"] / row["L_min"]
                                    for j in row["active"]:
                                        w, s = g[j]
                                        ri = float((w * np.minimum(1.0, s)).sum())
                                        regen_err = max(regen_err, abs(ri - row["rho_inf"][j]))
                                        recs.append(dict(d=d, model=model, mode=mode, lev=lev, k=k,
                                                         rho=e["rho"][j], low=rho_low(w, s, k), inf=ri,
                                                         p=p_sat(w, s, k)))
    return cfg, tol, recs, geo, regen_err, n_missing


def table(recs, keyf, tol):
    G = defaultdict(list)
    for r in recs:
        G[keyf(r)].append(r)
    out = {}
    for key in sorted(G, key=lambda t: tuple(str(v) for v in t)):
        rs = G[key]
        rho = np.array([r["rho"] for r in rs])
        low = np.array([r["low"] for r in rs])
        inf_ = np.array([r["inf"] for r in rs])
        p = np.array([r["p"] for r in rs])
        kap = np.array([r["k"] for r in rs])
        sat = rho > 0.8
        out["/".join(str(v) for v in key)] = dict(
            n=len(rs),
            viol_low=int(np.sum(rho < low - tol)), viol_inf=int(np.sum(rho > inf_ + tol)),
            med_kappa=float(np.median(kap)), med_low=float(np.median(low)), med_rho=float(np.median(rho)),
            med_inf=float(np.median(inf_)), med_gap=float(np.median(rho - low)), med_p_sat=float(np.median(p)),
            # of the feature-instances with rho > 0.8 (non-informative), how many does the lower bound explain?
            explained=float(np.mean(low[sat] > 0.8)) if sat.any() else None)
    return out


def geo_table(geo):
    G = defaultdict(list)
    for g in geo:
        G[(g["d"], g["mode"])].append(g["mass"])
    return {f"d={d}/{m}": {f"mass_s_ge_{t:.2f}": float(np.mean([ms[t] for ms in v])) for t in S_THRESH}
            for (d, m), v in sorted(G.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    for run in args.runs:
        cfg, tol, recs, geo, regen_err, n_missing = process(run, args.device)
        res = dict(run=run, seed=cfg["seed"], regen_max_abs_err_rho_inf=regen_err, n_missing_rows=n_missing,
                   by_d_level=table(recs, lambda r: (r["d"], r["lev"]), tol),
                   by_d_model_level=table(recs, lambda r: (r["d"], r["model"], r["lev"]), tol),
                   geometry_by_d_mode=geo_table(geo))
        with open(os.path.join(run, "lowerbound_summary.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\n=== {run} (seed {cfg['seed']})  regen_err={regen_err:.2e}  missing={n_missing}")
        print("d/level | n | viol_low viol_inf | med k | med low | med rho | med inf | med gap | med p_sat | explained")
        for k, v in res["by_d_level"].items():
            ex = "-" if v["explained"] is None else f"{v['explained']:.2f}"
            print(f"{k} | {v['n']} | {v['viol_low']} {v['viol_inf']} | {v['med_kappa']:.3f} | {v['med_low']:.3f} | "
                  f"{v['med_rho']:.3f} | {v['med_inf']:.3f} | {v['med_gap']:.3f} | {v['med_p_sat']:.3f} | {ex}")
        print("geometry: Shapley-weight mass with s_S >= 1 / 3 / 5 / 12.1")
        for k, v in res["geometry_by_d_mode"].items():
            print(k, " ".join(f"{x:.3f}" for x in v.values()))


if __name__ == "__main__":
    main()
