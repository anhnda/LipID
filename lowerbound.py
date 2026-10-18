#!/usr/bin/env python3
"""
lowerbound.py (v3) — check lower bounds on the normalized AIW on an EXISTING run.
No LP: the data are regenerated with the same generator keys as phase1.run_optimistic, anchor values are
recomputed with the same model functions (no hybrid evaluations), and rho(L) is read from optimistic.jsonl.

Per active feature i, per coalition S (not containing i), z+ = z_{S+i}, z- = z_S, delta = delta_i,
R_uv = d(z+, u) + d(z-, v) over ordered anchor pairs, d_uv = d(u, v), dF_uv = |f(u) - f(v)|, kappa = L / L_min.

Exact (Prop. 3):  B = min_{u,v} [ f(u) - f(v) + L R_uv ]   (signed)  -> rho(L), read from optimistic.jsonl.
Every bound below is  rho_X = sum_S w_S clamp(T_X / (L delta), 0, 1)  with T_X <= B (and <= -A):
  box  (sign-blind):            T = min [ L R_uv - dF_uv ]                     uses |f(u)-f(v)|, no triangle ineq.
  loc  (v2, triangle-relaxed):  T = min [ L R_uv - lambda_uv (R_uv + delta) ]   lambda_uv = dF_uv / d_uv
  det  (pure geometry + kappa): T = min [ L R_uv - L_min d_uv ]                 uses only lambda_uv <= L_min
  low  (global, closed form):   rho_low = sum_S w_S [min(1, (1 - 1/k) s_S - 1/k)]_+,  s_S = (q+ + q-)/delta
  inf  (geometric limit):       rho_inf = sum_S w_S min(1, s_S)                 (must equal the stored rho_inf)
Theory (checked here):  low <= det <= box,  low <= loc <= box,  box <= rho(L) <= inf.
NOTE: loc is identically 0 at kappa = 1 (the steepest pair gives T <= -L delta); box and det are not.
Gap rho - box = information carried only by the SIGN of f(u) - f(v).

    python lowerbound.py phase1_out phase1_seed2 phase1_seed3 phase1_seed4 [--device cuda] [--no-local]
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
from envelope import weighted_l1

S_THRESH = (1.0, 3.0, 5.0, 2.18 / 0.18)   # saturation thresholds for kappa = inf, 2, 1.5, 1.18
LEVELS = ("Lmin", "LV", "Ltrue")
MODES = ("domain", "fixed")
LOCAL = ("det", "loc", "box")


def rho_low(w, s, k):
    v = np.minimum(1.0, (1.0 - 1.0 / k) * s - 1.0 / k)
    return float((w * np.clip(v, 0.0, None)).sum())


def p_sat(w, s, k):
    if k <= 1.0:
        return 0.0
    return float(w[s >= (k + 1.0) / (k - 1.0)].sum())


def process(run, device, use_local):
    cfg = json.load(open(os.path.join(run, "config.json")))
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
                fv = {}
                for model in cfg["models"]:
                    f = P.make_model(model, cfg["a"])
                    fv[model] = (f(C), f(Xev), f(Rc))                    # same batches as phase1 -> same floats
                for xi, ridx in plan:
                    x = Xev[xi]
                    for rj in ridx:
                        r = Rc[rj]
                        delta = (x - r).abs() / sigma
                        active = torch.nonzero(delta > 0).flatten().tolist()
                        m = len(active)
                        if m == 0:
                            continue
                        masks = all_masks(m, x.device)
                        Z = vertex_points(x, r, active, masks)
                        w_all = shapley_weights(m, x.dtype, x.device)
                        idx = torch.arange(2 ** m, device=x.device)
                        pop = masks.sum(1)
                        for mode in MODES:
                            A = torch.cat([C, x[None]]) if mode == "domain" else torch.cat([C, x[None], r[None]])
                            D = weighted_l1(Z, A, sigma)                 # (2^m, |A|) hybrid-to-anchor distances
                            q = D.min(dim=1).values
                            DA = weighted_l1(A, A, sigma) if use_local else None
                            feats = {}
                            for jj, j in enumerate(active):
                                bit = 1 << jj
                                S = idx[(idx & bit) == 0]
                                Pm = S | bit
                                w = w_all[pop[S]]
                                s = (q[Pm] + q[S]) / delta[j]
                                R = (D[Pm][:, :, None] + D[S][:, None, :]) if use_local else None
                                wn, sn = w.cpu().numpy(), s.cpu().numpy()
                                feats[j] = (w, wn, sn, R)
                                geo.append(dict(d=d, mode=mode, mass={t: float(wn[sn >= t].sum()) for t in S_THRESH}))
                            for model in cfg["models"]:
                                row = rows.get((d, rho, nA, model, xi, rj, mode))
                                if row is None:
                                    n_missing += 1
                                    continue
                                dF, lamR = None, {}
                                if use_local:
                                    fC, fX, fR = fv[model]
                                    fA = torch.cat([fC, fX[xi:xi + 1]]) if mode == "domain" else \
                                        torch.cat([fC, fX[xi:xi + 1], fR[rj:rj + 1]])
                                    dF = (fA[:, None] - fA[None, :]).abs()
                                    Lam = torch.where(DA > 1e-12, dF / DA.clamp_min(1e-12), torch.zeros_like(DA))
                                    for j in row["active"]:
                                        lamR[j] = Lam[None] * (feats[j][3] + delta[j])
                                for lev in LEVELS:
                                    e = row["levels"].get(lev)
                                    if not e:
                                        continue
                                    L = e["L"]
                                    k = L / row["L_min"]
                                    for j in row["active"]:
                                        w, wn, sn, R = feats[j]
                                        ri = float((wn * np.minimum(1.0, sn)).sum())
                                        regen_err = max(regen_err, abs(ri - row["rho_inf"][j]))
                                        rec = dict(d=d, model=model, mode=mode, lev=lev, k=k, rho=e["rho"][j],
                                                   low=rho_low(wn, sn, k), inf=ri, p=p_sat(wn, sn, k),
                                                   det=None, loc=None, box=None)
                                        if use_local:
                                            Ld = L * delta[j]
                                            LR = L * R

                                            def lb(T):
                                                return float((w * (T / Ld).clamp(min=0.0, max=1.0)).sum())

                                            rec["loc"] = lb((LR - lamR[j]).amin(dim=(1, 2)))
                                            rec["box"] = lb((LR - dF[None]).amin(dim=(1, 2)))
                                            rec["det"] = lb((LR - row["L_min"] * DA[None]).amin(dim=(1, 2)))
                                        recs.append(rec)
    return cfg, recs, geo, regen_err, n_missing


def table(recs, keyf, tol):
    G = defaultdict(list)
    for r in recs:
        G[keyf(r)].append(r)
    out = {}
    for key in sorted(G, key=lambda t: tuple(str(v) for v in t)):
        rs = G[key]
        a = {c: np.array([r[c] for r in rs]) for c in ("rho", "low", "inf", "p", "k")}
        rho, low = a["rho"], a["low"]
        sat = rho > 0.8                       # non-informative feature-instances
        expl = (lambda b: float(np.mean(b[sat] > 0.8)) if sat.any() else None)
        e = dict(n=len(rs),
                 viol_low=int(np.sum(rho < low - tol)), viol_inf=int(np.sum(rho > a["inf"] + tol)),
                 med_kappa=float(np.median(a["k"])), med_low=float(np.median(low)), med_rho=float(np.median(rho)),
                 med_inf=float(np.median(a["inf"])), med_p_sat=float(np.median(a["p"])),
                 explained_low=expl(low))
        if rs[0]["loc"] is not None:
            b = {c: np.array([r[c] for r in rs]) for c in LOCAL}
            det, loc, box = b["det"], b["loc"], b["box"]
            order = (low > det + tol) | (low > loc + tol) | (det > box + tol) | (loc > box + tol)
            e.update(viol_det=int(np.sum(rho < det - tol)), viol_loc=int(np.sum(rho < loc - tol)),
                     viol_box=int(np.sum(rho < box - tol)), viol_order=int(np.sum(order)),
                     med_det=float(np.median(det)), med_loc=float(np.median(loc)), med_box=float(np.median(box)),
                     med_gap_loc=float(np.median(rho - loc)), med_gap_box=float(np.median(rho - box)),
                     explained_det=expl(det), explained_loc=expl(loc), explained_box=expl(box))
        out["/".join(str(v) for v in key)] = e
    return out


def geo_table(geo):
    G = defaultdict(list)
    for g in geo:
        G[(g["d"], g["mode"])].append(g["mass"])
    return {f"d={d}/{m}": {f"mass_s_ge_{t:.2f}": float(np.mean([ms[t] for ms in v])) for t in S_THRESH}
            for (d, m), v in sorted(G.items())}


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (pair bounds are much faster on cuda)")
    ap.add_argument("--no-local", action="store_true", help="skip det/loc/box (global bound only)")
    args = ap.parse_args()
    for run in args.runs:
        cfg, recs, geo, regen_err, n_missing = process(run, args.device, not args.no_local)
        tol = cfg["tol"]
        res = dict(run=run, seed=cfg["seed"], regen_max_abs_err_rho_inf=regen_err, n_missing_rows=n_missing,
                   by_d_level=table(recs, lambda r: (r["d"], r["lev"]), tol),
                   by_d_mode_level=table(recs, lambda r: (r["d"], r["mode"], r["lev"]), tol),
                   by_d_model_level=table(recs, lambda r: (r["d"], r["model"], r["lev"]), tol),
                   geometry_by_d_mode=geo_table(geo))
        with open(os.path.join(run, "lowerbound_summary.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\n=== {run} (seed {cfg['seed']})  regen_err={regen_err:.2e}  missing={n_missing}")
        print("d/level | n | viol low/det/loc/box/inf/order | med k | med low/det/loc/box/rho | "
              "expl low/det/loc/box")
        for k, v in res["by_d_level"].items():
            viol = "/".join(str(v.get(c, "-")) for c in
                            ("viol_low", "viol_det", "viol_loc", "viol_box", "viol_inf", "viol_order"))
            med = "/".join(fmt(v.get(c)) for c in ("med_low", "med_det", "med_loc", "med_box", "med_rho"))
            ex = "/".join(fmt(v.get(c)) for c in ("explained_low", "explained_det", "explained_loc",
                                                  "explained_box"))
            print(f"{k} | {v['n']} | {viol} | {v['med_kappa']:.3f} | {med} | {ex}")
        print("by model (d=4,5): med k | med det/loc/box/rho | expl det/loc/box")
        for k, v in res["by_d_model_level"].items():
            if k.split("/")[0] in ("4", "5"):
                med = "/".join(fmt(v.get(c)) for c in ("med_det", "med_loc", "med_box", "med_rho"))
                ex = "/".join(fmt(v.get(c)) for c in ("explained_det", "explained_loc", "explained_box"))
                print(f"  {k} | {v['med_kappa']:.3f} | {med} | {ex}")
        print("geometry: Shapley-weight mass with s_S >= 1 / 3 / 5 / 12.1")
        for k, v in res["geometry_by_d_mode"].items():
            print(k, " ".join(f"{x:.3f}" for x in v.values()))


if __name__ == "__main__":
    main()