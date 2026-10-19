#!/usr/bin/env python3
"""
lowerbound_v4.py — T6 re-analysis on an EXISTING run, using the directional decomposition
(see t6_directional.py). No LP, no hybrid evaluation; data regenerated exactly as lowerbound.py (v3).

Reported per group (d x level, d x mode x level, d x model x level, d x model x mode x level):
  * rho recomputed from (c+, c-) vs optimistic.jsonl (regen_err_rho); sign recomputed (sign_mismatch).
  * directional-min (former box) via T_box = min(B+, B-); O(|A|^2) oracle on every record (oracle_max_err).
  * Mechanical baseline (I1): base_mech = P(rho > 0.9 | rho > 0.8) <= expl_dmin.
        excess    = expl_dmin - base_mech = P(dmin > 0.8, rho <= 0.9 | rho > 0.8)   (descriptive only)
        expl_band = P(dmin > 0.8 | 0.8 < rho <= 0.9)                                  (the informative one)
  * rho - dmin split (I2): asym_pos (both directions >= 0) + neg_part (one direction forced).
  * Sign: viol_sign_half (I3); eligibility = P(rho < 1/2); efficiency = P(sign | rho < 1/2).
  * Slope ladder rho_K (K = 1 is det). In fixed mode at L_min, det = 0 by (I5): viol_det_fixed counts breaks.
  * Tie-robust margin diagnostic (no argmin): straddlers (g = -1) on 'rescued' edges (det < 0.8 <= box) and the
    box near-minimizer set on 'blocked' edges (box < 0.8).
  * Fixed mode, efficiency cap (I6): violations of rho <= (1 + min(1, X))/2 and dmin <= clamp(X), forced-sign
    violations, and eff_sign_share = P(X < 0 | sign certified)  (share of certified signs explained by efficiency).

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
from t6_directional import (box_oracle_T, det_T, directional, edge_sets, efficiency_cap, envelopes_from_D,
                            feature_stats, lb_from_T, margin_diagnostic, pair_tensor, quantized_T, slope_ratio)

LEVELS = ("Lmin", "LV", "Ltrue")
MODES = ("domain", "fixed")
SAT, BAND_HI = 0.8, 0.9
MIN_SAT = 30          # minimum number of saturated feature-instances for the K4/box ratio
MD_KEYS = ("m_err", "cert_viol", "rescued_mass", "blocked_mass", "rescued_frac_strad", "rescued_frac_ep_strad",
           "rescued_med_log10_nstrad", "blocked_nm_share_strad", "blocked_nm_share_ep", "blocked_nm_t",
           "blocked_nm_h", "G_strad", "G_neg", "G_01", "G_gt1", "full_box_mass", "full_K4_mass",
           "cert_psi_box_share", "cert_unif_box_share", "cert_psi_K4_share")
MD_MEAN = MD_KEYS[2:]


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def process(run, device, ks, oracle_every, nm_eps=0.02):
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
                        D_xr = float(delta.sum())                         # d(x, r)
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
                                Delta = float(fX[xi] - fR[rj])
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
                                        dj = float(delta[j])
                                        Ld = L * dj
                                        cp, cm = directional(ell, h, S, Pm, Ld)
                                        st = feature_stats(w, cp, cm)
                                        R = Rj[j]
                                        rec = dict(d=d, model=model, mode=mode, lev=lev, k=L / L_min,
                                                   rho_stored=e["rho"][j], rho=st.rho, dmin=st.dmin,
                                                   asym=st.asym, asym_pos=st.asym_pos, neg_part=st.neg_part,
                                                   negmass=st.negmass,
                                                   sign_stored=int(e["sign"][j]), sign=st.sign(Ld, tol))
                                        rec["K1"] = lb_from_T(det_T(R, DA, L, L_min), Ld, w)
                                        for K in ks[1:]:
                                            rec[f"K{K}"] = lb_from_T(quantized_T(R, DA, ratio, L, L_min, K), Ld, w)
                                        md = margin_diagnostic(R, DA, ratio, cp, cm, w, L, L_min, dj, endpoints, SAT,
                                                               nm_eps=nm_eps)
                                        rec.update({f"md_{k_}": md[k_] for k_ in MD_KEYS})
                                        rec["eff"] = None
                                        if mode == "fixed":
                                            rec["eff"] = efficiency_cap(D_xr, Delta, L, dj, tol)
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


def _mean(a):
    a = np.asarray([v for v in a if v is not None], dtype=float)
    return float(np.mean(a)) if a.size else None


def table(recs, keyf, tol, ks):
    G = defaultdict(list)
    for r in recs:
        G[keyf(r)].append(r)
    out = {}
    for key in sorted(G, key=lambda t: tuple(str(v) for v in t)):
        rs = G[key]
        col = lambda c: np.array([r[c] for r in rs], dtype=float)
        rho, dmin = col("rho"), col("dmin")
        sign = np.array([r["sign"] for r in rs])
        sat = rho > SAT
        band = sat & (rho <= BAND_HI)
        elig = rho < 0.5
        expl = _frac(dmin > SAT, sat)
        base = _frac(rho > BAND_HI, sat)
        orc = [r["oracle_err"] for r in rs if r["oracle_err"] is not None]
        e = dict(
            n=len(rs), prev_sat=float(np.mean(sat)), med_rho=float(np.median(rho)),
            expl_dmin=expl, base_mech=base, excess=None if expl is None else expl - base,
            n_band=int(band.sum()), expl_band=_frac(dmin > SAT, band),
            med_asym_pos=float(np.median(col("asym_pos"))), med_neg_part=float(np.median(col("neg_part"))),
            med_negmass=float(np.median(col("negmass"))),
            sign_rate=float(np.mean(sign != 0)), eligibility=float(np.mean(elig)),
            efficiency=_frac(sign != 0, elig),
            # theorem / consistency checks: any non-zero count is a bug
            viol_sign_half=int(np.sum((sign != 0) & (rho >= 0.5 - tol))),
            viol_dmin_lb=int(np.sum(dmin < 2 * rho - 1 - tol)),
            viol_dmin_ub=int(np.sum(dmin > rho + tol)),
            viol_asym_split=int(np.sum(np.abs(col("asym") - col("asym_pos") - col("neg_part")) > tol)),
            regen_err_rho=float(np.max(np.abs(rho - col("rho_stored")))),
            sign_mismatch=int(np.sum(sign != np.array([r["sign_stored"] for r in rs]))),
            oracle_max_err=max(orc) if orc else None,
            m_err=max(r["md_m_err"] for r in rs),
        )
        fixed_lmin = np.array([r["mode"] == "fixed" and r["lev"] == "Lmin" for r in rs])
        e["viol_det_fixed_Lmin"] = int(np.sum(fixed_lmin & (col("K1") > tol)))
        prev = None
        for K in ks:
            v = col(f"K{K}")
            e[f"expl_K{K}"] = _frac(v > SAT, sat)
            bad = v > dmin + tol
            if prev is not None:
                bad |= v < prev - tol
            e[f"viol_K{K}"] = int(np.sum(bad))
            prev = v
        # ratio only with >= MIN_SAT saturated cases (pre-registered); absolute rates are always reported
        e["K4_over_box"] = (None if int(sat.sum()) < MIN_SAT or e.get("expl_K4") is None or not expl
                            else e["expl_K4"] / expl)
        for k_ in MD_MEAN:
            e[f"md_{k_}"] = _mean([r[f"md_{k_}"] for r in rs])
        e["cert_viol"] = int(sum(r["md_cert_viol"] for r in rs))
        e["n_sat"] = int(sat.sum())
        fx = [r for r in rs if r["eff"] is not None]
        if fx:
            X = np.array([r["eff"]["X"] for r in fx])
            rf, bf = np.array([r["rho"] for r in fx]), np.array([r["dmin"] for r in fx])
            sf = np.array([r["sign"] for r in fx])
            forced = np.array([r["eff"]["forced_sign"] for r in fx])
            e.update(viol_eff_rho=int(np.sum(rf > np.array([r["eff"]["rho_cap"] for r in fx]) + tol)),
                     viol_eff_box=int(np.sum(bf > np.array([r["eff"]["box_cap"] for r in fx]) + tol)),
                     viol_eff_sign=int(np.sum((forced != 0) & (sf != forced))),
                     frac_X_neg=float(np.mean(X < 0)),
                     eff_sign_share=_frac(X < 0, sf != 0))
        out["/".join(str(v) for v in key)] = e
    return out


def fmt(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def sci(x):
    return "-" if x is None else (f"{x:.1e}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--ks", default="1,2,4,8,16", help="slope levels, powers of two, must start at 1")
    # default 1: a stride aliases with the fixed record pattern per instance (levels x features)
    ap.add_argument("--oracle-every", type=int, default=1, help="0 disables the O(|A|^2) box oracle")
    # scientific near-minimizer margin (units of delta_i); distinct from the round-off tie tolerance eta = 1e-9.
    # Run e.g. 0.01 / 0.02 / 0.05 and report the sensitivity of the blocked_nm_* columns.
    ap.add_argument("--nm-eps", type=float, default=0.02)
    args = ap.parse_args()
    ks = tuple(int(s) for s in args.ks.split(","))
    assert ks[0] == 1 and all(b == 2 * a for a, b in zip(ks, ks[1:])), "--ks must be 1,2,4,..."
    torch.set_default_dtype(torch.float64)
    for run in args.runs:
        cfg, recs, n_missing = process(run, args.device, ks, args.oracle_every, args.nm_eps)
        tol = cfg["tol"]
        res = dict(run=run, seed=cfg["seed"], n_missing_rows=n_missing, thresholds_in_config={
                       k: cfg.get(k) for k in ("sci_rho_target", "sci_min_frac", "sci_min_sign_rate")},
                   by_d_level=table(recs, lambda r: (r["d"], r["lev"]), tol, ks),
                   by_d_mode_level=table(recs, lambda r: (r["d"], r["mode"], r["lev"]), tol, ks),
                   by_d_model_level=table(recs, lambda r: (r["d"], r["model"], r["lev"]), tol, ks),
                   by_d_model_mode_level=table(recs, lambda r: (r["d"], r["model"], r["mode"], r["lev"]), tol, ks))
        with open(os.path.join(run, "lowerbound_v4_summary.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\n=== {run} (seed {cfg['seed']})  missing={n_missing}  thresholds={res['thresholds_in_config']}")
        cols = ["n", "n_sat", "prev_sat", "med_rho", "expl_dmin", "base_mech", "n_band", "expl_band",
                "med_asym_pos", "med_neg_part", "sign_rate", "eligibility", "efficiency"] + \
               [f"expl_K{K}" for K in ks] + ["K4_over_box",
               "md_rescued_frac_strad", "md_rescued_frac_ep_strad", "md_rescued_med_log10_nstrad",
               "md_blocked_nm_share_strad", "md_blocked_nm_share_ep", "md_blocked_nm_t", "md_blocked_nm_h",
               "md_G_strad", "md_G_neg", "md_G_01", "md_G_gt1",
               "md_full_box_mass", "md_cert_psi_box_share", "md_cert_unif_box_share",
               "md_full_K4_mass", "md_cert_psi_K4_share", "frac_X_neg", "eff_sign_share"]
        checks = ["viol_sign_half", "viol_dmin_lb", "viol_dmin_ub", "viol_asym_split", "regen_err_rho",
                  "sign_mismatch", "oracle_max_err", "m_err", "cert_viol", "viol_det_fixed_Lmin",
                  "viol_eff_rho", "viol_eff_box", "viol_eff_sign"] + [f"viol_K{K}" for K in ks]
        for name in ("by_d_mode_level", "by_d_model_mode_level"):
            print(f"-- {name}: " + " | ".join(cols))
            for k, v in res[name].items():
                print(k, " | ".join(fmt(v.get(c)) for c in cols))
        print("-- checks (must be 0 / ~1e-12): " + " | ".join(checks))
        for k, v in res["by_d_mode_level"].items():
            print(k, " | ".join(sci(v.get(c)) for c in checks))


if __name__ == "__main__":
    main()
