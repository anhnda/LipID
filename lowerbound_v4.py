#!/usr/bin/env python3
"""
lowerbound_v4.py — T6 re-analysis on an EXISTING run, using the directional decomposition
(see t6_directional.py). No LP, no hybrid evaluation; data regenerated exactly as lowerbound.py (v3).

What is new relative to v3
  * rho is recomputed from (c+, c-) and checked against optimistic.jsonl (regen_err_rho).
  * 'box' is computed via the identity T_box = min(B+, B-) (O(|A|) per hybrid); the O(|A|^2) version is
    kept as an oracle on every --oracle-every-th record (oracle_err).
  * Mechanical baseline: since rho_dmin >= 2 rho - 1, every case with rho > 0.9 is 'explained' automatically.
        base_mech = P(rho > 0.9 | rho > 0.8)     <= expl_dmin = P(rho_dmin > 0.8 | rho > 0.8)
        excess    = expl_dmin - base_mech         (the part NOT implied by rho itself)
        expl_band = P(rho_dmin > 0.8 | 0.8 < rho <= 0.9)
  * Directional asymmetry asym = rho - rho_dmin and negmass (edges with one direction forced).
  * Sign check (I3): certified sign => rho < 1/2;  frac(rho < 1/2) is an upper bound on the sign rate.
  * Slope-quantized ladder rho_K, K in --ks (K = 1 is det): how much slope resolution certifies saturation.
  * det diagnostic: on edges det misses but dmin catches, which anchor pairs block det
    (involving x / r, self pair, mean lambda/L_min at the blocking pair).

    python lowerbound_v4.py phase1_out phase1_seed2 phase1_seed3 phase1_seed4 [--device cuda]
Writes <run>/lowerbound_v4_summary.json and prints tables.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import phase1 as P
from bounds import all_masks, vertex_points
from envelope import weighted_l1
from t6_directional import (det_diagnostic, directional, edge_sets, envelopes_from_D, feature_stats,
                            lb_from_T, box_oracle_T, pair_tensor, quantized_T, slope_ratio)

LEVELS = ("Lmin", "LV", "Ltrue")
MODES = ("domain", "fixed")
SAT, BAND_HI = 0.8, 0.9


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def process(run, device, ks, oracle_every):
    cfg = json.load(open(os.path.join(run, "config.json")))
    tol = cfg["tol"]
    rows = {}
    with open(os.path.join(run, "optimistic.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("degenerate"):
                rows[(r["d"], r["rho"], r["n_anchor"], r["model"], r["x_idx"], r["r_idx"], r["mode"])] = r
    recs = []
    n_missing, n_rec = 0, 0
    for d in cfg["dims"]:
        for rho in cfg["rhos"]:
            for nA in cfg["n_anchor"]:
                gd = P.gen_of("data", cfg["seed"], d, rho, nA)          # same keys / draw order as phase1
                sample = P.gaussian_sampler(d, rho, device)
                sigma = P.robust_sigma(sample(cfg["n_train"], gd))
                C, Rc, Xev = sample(nA, gd), sample(cfg["n_candidate"], gd), sample(cfg["n_eval_x"], gd)
                plan = [(xi, torch.randperm(cfg["n_candidate"], generator=gd)[:cfg["refs_per_x"]].tolist())
                        for xi in range(cfg["n_eval_x"])]
                fv = {}
                for model in cfg["models"]:
                    f = P.make_model(model, cfg["a"])
                    fv[model] = (f(C), f(Xev), f(Rc))
                nC = C.shape[0]
                for xi, ridx in plan:
                    x = Xev[xi]
                    for rj in ridx:
                        r = Rc[rj]
                        delta = (x - r).abs() / sigma
                        active = torch.nonzero(delta > 0).flatten().tolist()
                        m = len(active)
                        if m == 0:
                            continue
                        Z = vertex_points(x, r, active, all_masks(m, x.device))
                        edges = edge_sets(m, x.dtype, x.device)
                        for mode in MODES:
                            A = torch.cat([C, x[None]]) if mode == "domain" else torch.cat([C, x[None], r[None]])
                            endpoints = [nC] if mode == "domain" else [nC, nC + 1]
                            D = weighted_l1(Z, A, sigma)                  # (2^m, |A|)
                            DA = weighted_l1(A, A, sigma)                 # (|A|, |A|)
                            Rj = {j: pair_tensor(D, edges[jj][0], edges[jj][1]) for jj, j in enumerate(active)}
                            for model in cfg["models"]:
                                row = rows.get((d, rho, nA, model, xi, rj, mode))
                                if row is None:
                                    n_missing += 1
                                    continue
                                fC, fX, fR = fv[model]
                                fA = torch.cat([fC, fX[xi:xi + 1]]) if mode == "domain" else \
                                    torch.cat([fC, fX[xi:xi + 1], fR[rj:rj + 1]])
                                dF = (fA[:, None] - fA[None, :]).abs()
                                L_min = row["L_min"]
                                ratio = slope_ratio(DA, dF, L_min)
                                for lev in LEVELS:
                                    e = row["levels"].get(lev)
                                    if not e:
                                        continue
                                    L = e["L"]
                                    ell, h = envelopes_from_D(D, fA, L)
                                    for jj, j in enumerate(active):
                                        if j not in row["active"]:
                                            continue
                                        S, Pm, w = edges[jj]
                                        Ld = float(L * delta[j])
                                        cp, cm = directional(ell, h, S, Pm, Ld)
                                        st = feature_stats(w, cp, cm)
                                        R = Rj[j]
                                        rec = dict(d=d, model=model, mode=mode, lev=lev, k=L / L_min,
                                                   rho_stored=e["rho"][j], rho=st.rho, dmin=st.dmin,
                                                   asym=st.asym, negmass=st.negmass,
                                                   sign_stored=int(e["sign"][j]), sign=st.sign(Ld, tol))
                                        for K in ks:
                                            rec[f"K{K}"] = lb_from_T(quantized_T(R, DA, ratio, L, L_min, K), Ld, w)
                                        rec.update({f"det_{k_}": v_ for k_, v_ in det_diagnostic(
                                            R, DA, ratio, cp, cm, w, L, L_min, Ld, endpoints, SAT).items()})
                                        rec["oracle_err"] = None
                                        if oracle_every > 0 and n_rec % oracle_every == 0:
                                            box = lb_from_T(box_oracle_T(R, dF, L), Ld, w)
                                            rec["oracle_err"] = abs(box - st.dmin)
                                        n_rec += 1
                                        recs.append(rec)
    return cfg, recs, n_missing


def _frac(b, cond):
    return float(np.mean(b[cond])) if cond.any() else None


def _med(a):
    a = np.asarray([v for v in a if v is not None], dtype=float)
    return float(np.median(a)) if a.size else None


def table(recs, keyf, tol, ks):
    G = defaultdict(list)
    for r in recs:
        G[keyf(r)].append(r)
    out = {}
    for key in sorted(G, key=lambda t: tuple(str(v) for v in t)):
        rs = G[key]
        col = lambda c: np.array([r[c] for r in rs], dtype=float)
        rho, dmin, asym, neg = col("rho"), col("dmin"), col("asym"), col("negmass")
        sign = np.array([r["sign"] for r in rs])
        sat = rho > SAT
        band = sat & (rho <= BAND_HI)
        expl = _frac(dmin > SAT, sat)
        base = _frac(rho > BAND_HI, sat)
        orc = [r["oracle_err"] for r in rs if r["oracle_err"] is not None]
        e = dict(
            n=len(rs), prev_sat=float(np.mean(sat)), med_rho=float(np.median(rho)),
            expl_dmin=expl, base_mech=base,
            excess=None if expl is None else expl - base,
            n_band=int(band.sum()), expl_band=_frac(dmin > SAT, band),
            med_asym=float(np.median(asym)), med_asym_sat=_med(asym[sat]) if sat.any() else None,
            med_negmass=float(np.median(neg)),
            sign_rate=float(np.mean(sign != 0)), frac_rho_lt_half=float(np.mean(rho < 0.5)),
            # theorem / consistency checks: any non-zero count is a bug
            viol_sign_half=int(np.sum((sign != 0) & (rho >= 0.5 - tol))),
            viol_dmin_lb=int(np.sum(dmin < 2 * rho - 1 - tol)),
            viol_dmin_ub=int(np.sum(dmin > rho + tol)),
            regen_err_rho=float(np.max(np.abs(rho - col("rho_stored")))),
            sign_mismatch=int(np.sum(sign != np.array([r["sign_stored"] for r in rs]))),
            oracle_max_err=max(orc) if orc else None,
        )
        prev = None
        for K in ks:
            v = col(f"K{K}")
            e[f"expl_K{K}"] = _frac(v > SAT, sat)
            bad = v > dmin + tol
            if prev is not None:
                bad |= v < prev - tol          # ladder must be non-decreasing along K = 1, 2, 4, ...
            e[f"viol_K{K}"] = int(np.sum(bad))
            prev = v
        satr = [r for r, s_ in zip(rs, sat) if s_]
        # per-record shares are nearly 0/1 (few masked edges) -> median is uninformative, report the mean
        sh = [r["det_share_endpoint"] for r in satr if r["det_share_endpoint"] is not None]
        e["n_det_diag"] = len(sh)
        e["det_share_endpoint"] = float(np.mean(sh)) if sh else None
        e["det_ratio_at_argmin"] = _med([r["det_ratio_at_argmin"] for r in satr])
        out["/".join(str(v) for v in key)] = e
    return out


def fmt(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--ks", default="1,2,4,8,16", help="slope levels, powers of two, must start at 1")
    # default 1: a stride like 10 aliases with the fixed record pattern per instance (levels x features),
    # so some (d, level) groups were never checked
    ap.add_argument("--oracle-every", type=int, default=1, help="0 disables the O(|A|^2) box oracle")
    args = ap.parse_args()
    ks = tuple(int(s) for s in args.ks.split(","))
    assert ks[0] == 1 and all(b == 2 * a for a, b in zip(ks, ks[1:])), "--ks must be 1,2,4,..."
    torch.set_default_dtype(torch.float64)
    for run in args.runs:
        cfg, recs, n_missing = process(run, args.device, ks, args.oracle_every)
        tol = cfg["tol"]
        res = dict(run=run, seed=cfg["seed"], n_missing_rows=n_missing, thresholds_in_config={
                       k: cfg.get(k) for k in ("sci_rho_target", "sci_min_frac", "sci_min_sign_rate")},
                   by_d_level=table(recs, lambda r: (r["d"], r["lev"]), tol, ks),
                   by_d_mode_level=table(recs, lambda r: (r["d"], r["mode"], r["lev"]), tol, ks),
                   by_d_model_level=table(recs, lambda r: (r["d"], r["model"], r["lev"]), tol, ks))
        with open(os.path.join(run, "lowerbound_v4_summary.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\n=== {run} (seed {cfg['seed']})  missing={n_missing}  thresholds={res['thresholds_in_config']}")
        cols = ["n", "prev_sat", "med_rho", "expl_dmin", "base_mech", "excess", "n_band", "expl_band",
                "med_asym_sat", "sign_rate", "frac_rho_lt_half"] + [f"expl_K{K}" for K in ks] + \
               ["n_det_diag", "det_share_endpoint", "det_ratio_at_argmin"]
        checks = ["viol_sign_half", "viol_dmin_lb", "viol_dmin_ub", "regen_err_rho", "sign_mismatch",
                  "oracle_max_err"] + [f"viol_K{K}" for K in ks]
        for name in ("by_d_level", "by_d_model_level"):
            print(f"-- {name}: " + " | ".join(cols))
            for k, v in res[name].items():
                print(k, " | ".join(fmt(v.get(c)) for c in cols))
        print("-- checks (must be 0 / ~1e-12): " + " | ".join(checks))
        sci = lambda x: "-" if x is None else (f"{x:.1e}" if isinstance(x, float) else str(x))
        for k, v in res["by_d_level"].items():
            print(k, " | ".join(sci(v.get(c)) for c in checks))


if __name__ == "__main__":
    main()