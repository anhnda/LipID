#!/usr/bin/env python3
"""
nested_anchors.py — P0' nested-anchor experiment (sample complexity of Shapley identifiability, T-A).

Design (per seed, d, rho_corr): one draw of sigma (train set), background anchors C_max (n_max points),
references, eval points. Nested anchor sets  C_16 ⊂ C_64 ⊂ C_256 ⊂ C_1024  are PREFIXES of C_max, so x, r, sigma
and the model f are identical along the chain. Modes: domain A_n = C_n ∪ {x};  fixed A_n = C_n ∪ {x, r}.

Two Lipschitz levels per anchor set:
  opt    : L_n = L_A(A_n)                      (optimistic; moves with n -> NOT monotone in n, see N1 below)
  common : L* fixed along the chain and across modes:
           L* = max(L_true, L_A(C_max ∪ {x, r}))  for linear / unstable_id (L_true known)
           L* = L_A(C_max ∪ {x, r})                for fa / stable_ood
           L* >= L_A(A_n) for every n and mode, so every class is non-empty.
Curves (per d, mode, model, rho_corr):
  A  p_opt(n)    = P(rho(A_n; L_n) > 0.8)
  B  p_common(n) = P(rho(A_n; L*) > 0.8)          non-increasing in n by N1 (checked: viol_N1 must be 0)
  C  kappa_n     = L* / L_A(A_n)                  slack of the anchor pairs (T-A1 condition)
  plus  p_low(n) = P(rho_low(kappa_n) > 0.8) at L* (T3 bound, purely geometric + kappa: the T-A1 route),
        t_xr     = |f(x) - f(r)| / (L D(x, r)) (endpoint slack, fixed mode; t -> 1 forces identification, I6),
        frac_I6  = share of feature-instances whose sign is forced by I6 at L*.

Only envelopes are used (O(|A|) per hybrid): exact rho via the directional slacks, rho_low via distances.
No pairwise (|A|^2 per edge) diagnostics -> runs at d = 6, n = 1024 on one GPU.

Checks (any non-zero count is a bug):  viol_N1 (rho at L* increases along the chain), viol_signN1 (a sign
certified at L* is lost or flips for a larger set), viol_M (rho_opt > rho_common), viol_low (rho_low > rho),
viol_L (L* < L_A(A_n)).

    python nested_anchors.py --seeds 1 2 3 4 --out nested_explore --device cuda
    python nested_anchors.py --seeds 5 6 7  --out nested_confirm --device cuda      # only after the protocol is locked
Writes <out>/nested_summary.json (+ nested_records.jsonl with --raw) and prints tables.
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

import phase1 as P
from bounds import all_masks, vertex_points
from envelope import weighted_l1
from t6_directional import edge_sets

SAT = 0.8


def prefix_max(v: torch.Tensor, ns) -> list:
    return [float(v[:n].max()) if n > 0 else 0.0 for n in ns]


def slopes(fa, fb, D):
    return torch.where(D > 1e-12, (fa - fb).abs() / D.clamp_min(1e-12), torch.zeros_like(D))


def feature_stats(ell, h, edges, active, delta, L, q, kappa, tol):
    """Per active feature: exact rho, sign, rho_low(kappa) (T3)."""
    out = {}
    for jj, j in enumerate(active):
        S, Pm, w = edges[jj]
        dj = float(delta[j])
        Ld = L * dj
        cp, cm = (h[Pm] - ell[S]) / Ld, (h[S] - ell[Pm]) / Ld
        hp, hm = cp.clamp(max=1.0), cm.clamp(max=1.0)
        rho = float((w * (hp + hm)).sum()) / 2.0
        lo, hi = -Ld * float((w * hm).sum()), Ld * float((w * hp).sum())
        sign = 1 if lo > tol else (-1 if hi < -tol else 0)
        low = 0.0
        if kappa > 1.0:
            s = (q[Pm] + q[S]) / dj
            v = torch.clamp(torch.minimum(torch.ones_like(s), (1.0 - 1.0 / kappa) * s - 1.0 / kappa), min=0.0)
            low = float((w * v).sum())
        out[j] = (rho, sign, low)
    return out


def run_seed(seed, args, ns, device):
    cfg = P.Config()
    tol = cfg.tol
    nmax = max(ns)
    recs = []
    for d in args.dims:
        for rhoc in args.rhos:
            gd = P.gen_of("nested", seed, d, rhoc)                      # own key namespace: fresh data
            sample = P.gaussian_sampler(d, rhoc, device)
            sigma = P.robust_sigma(sample(cfg.n_train, gd))
            C = sample(nmax, gd)
            Rc = sample(cfg.n_candidate, gd)
            Xev = sample(args.n_x, gd)
            plan = [(xi, torch.randperm(cfg.n_candidate, generator=gd)[:args.refs].tolist())
                    for xi in range(args.n_x)]
            DCC = weighted_l1(C, C, sigma)
            models = {}
            for model in args.models:
                f = P.make_model(model, cfg.a)
                fC, fX, fR = f(C), f(Xev), f(Rc)
                lamCC = slopes(fC[:, None], fC[None, :], DCC)
                LCC = [float(lamCC[:n, :n].max()) for n in ns]          # L_A of the nested background sets
                models[model] = (fC, fX, fR, LCC, P.lipschitz_true(model, d, sigma, cfg.a))
            for xi, ridx in plan:
                x = Xev[xi]
                dxC = weighted_l1(x[None], C, sigma)[0]
                for rj in ridx:
                    r = Rc[rj]
                    delta = (x - r).abs() / sigma
                    active = torch.nonzero(delta > 0).flatten().tolist()
                    m = len(active)
                    if m == 0:
                        continue
                    D_xr = float(delta.sum())
                    Z = vertex_points(x, r, active, all_masks(m, x.device))
                    edges = edge_sets(m, x.dtype, x.device)
                    DZC = weighted_l1(Z, C, sigma)                       # (2^m, nmax)
                    dZx = weighted_l1(Z, x[None], sigma)                 # (2^m, 1)
                    dZr = weighted_l1(Z, r[None], sigma)
                    drC = weighted_l1(r[None], C, sigma)[0]
                    for model, (fC, fX, fR, LCC, Ltrue) in models.items():
                        fx, fr = fX[xi], fR[rj]
                        Lx = prefix_max(slopes(fx, fC, dxC), ns)
                        Lr = prefix_max(slopes(fr, fC, drC), ns)
                        lxr = float((fx - fr).abs()) / D_xr
                        LA = {"domain": [max(a, b) for a, b in zip(LCC, Lx)],
                              "fixed": [max(a, b, c, lxr) for a, b, c in zip(LCC, Lx, Lr)]}
                        Lstar = LA["fixed"][-1]
                        if Ltrue is not None:
                            Lstar = max(Lstar, float(Ltrue))
                        if not (math.isfinite(Lstar) and Lstar > 0):
                            continue
                        Delta = float(fx - fr)
                        for mode in ("domain", "fixed"):
                            for k_n, n in enumerate(ns):
                                L_n = LA[mode][k_n]
                                if not (math.isfinite(L_n) and L_n > 0):
                                    continue
                                if mode == "domain":
                                    Dsub = torch.cat([DZC[:, :n], dZx], 1)
                                    fA = torch.cat([fC[:n], fx.reshape(1)])
                                else:
                                    Dsub = torch.cat([DZC[:, :n], dZx, dZr], 1)
                                    fA = torch.cat([fC[:n], fx.reshape(1), fr.reshape(1)])
                                q = Dsub.min(1).values
                                kappa = Lstar / L_n
                                res = {}
                                for lev, L in (("opt", L_n), ("com", Lstar)):
                                    LD = L * Dsub
                                    ell = (fA[None, :] - LD).max(1).values
                                    h = (fA[None, :] + LD).min(1).values
                                    res[lev] = feature_stats(ell, h, edges, active, delta, L, q,
                                                             kappa if lev == "com" else 1.0, tol)
                                for j in active:
                                    ro, so, _ = res["opt"][j]
                                    rc, sc, lc = res["com"][j]
                                    X = (D_xr - abs(Delta) / Lstar) / float(delta[j]) - 1.0
                                    recs.append(dict(
                                        seed=seed, d=d, rhoc=rhoc, model=model, mode=mode, n=n,
                                        inst=(xi, rj), j=j, rho_opt=ro, sign_opt=so, rho_com=rc, sign_com=sc,
                                        low_com=lc, kappa=kappa, viol_L=int(Lstar < L_n * (1 - 1e-12)),
                                        t_com=abs(Delta) / (Lstar * D_xr) if mode == "fixed" else None,
                                        t_opt=abs(Delta) / (L_n * D_xr) if mode == "fixed" else None,
                                        I6=int(mode == "fixed" and X * Lstar * float(delta[j]) < -tol
                                               and Delta != 0),
                                        Ltrue_used=Ltrue is not None))
            print(f"[nested] seed={seed} d={d} rho={rhoc} records={len(recs)}", flush=True)
    return recs


def chain_checks(recs, tol):
    """N1 along the nested chain at L*: rho non-increasing; a certified sign persists (same sign)."""
    G = defaultdict(list)
    for r in recs:
        G[(r["seed"], r["d"], r["rhoc"], r["model"], r["mode"], r["inst"], r["j"])].append(r)
    v_rho = v_sign = 0
    for rs in G.values():
        rs.sort(key=lambda r: r["n"])
        for a, b in zip(rs, rs[1:]):
            v_rho += int(b["rho_com"] > a["rho_com"] + tol)
            v_sign += int(a["sign_com"] != 0 and b["sign_com"] != a["sign_com"])
    return v_rho, v_sign


def _med(a):
    a = [v for v in a if v is not None]
    return float(np.median(a)) if a else None


def table(recs, keyf, tol):
    G = defaultdict(list)
    for r in recs:
        G[keyf(r)].append(r)
    out = {}
    for key in sorted(G, key=lambda t: tuple(str(v) for v in t)):
        rs = G[key]
        col = lambda c: np.array([r[c] for r in rs], dtype=float)
        ro, rc, lc = col("rho_opt"), col("rho_com"), col("low_com")
        out["/".join(str(v) for v in key)] = dict(
            n=len(rs),
            p_opt=float(np.mean(ro > SAT)), p_common=float(np.mean(rc > SAT)), p_low=float(np.mean(lc > SAT)),
            med_rho_opt=float(np.median(ro)), med_rho_common=float(np.median(rc)),
            sign_opt=float(np.mean(col("sign_opt") != 0)), sign_common=float(np.mean(col("sign_com") != 0)),
            med_kappa=float(np.median(col("kappa"))), frac_kappa_lt_1_05=float(np.mean(col("kappa") < 1.05)),
            med_t_common=_med([r["t_com"] for r in rs]), med_t_opt=_med([r["t_opt"] for r in rs]),
            frac_I6=float(np.mean(col("I6"))) if rs[0]["mode"] == "fixed" else None,
            viol_M=int(np.sum(ro > rc + tol)), viol_low=int(np.sum(lc > rc + tol)),
            viol_L=int(np.sum(col("viol_L"))))
    return out


def fmt(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6, 0.9, 0.99])
    ap.add_argument("--nA", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--models", nargs="+", default=["linear", "fa", "stable_ood", "unstable_id"])
    ap.add_argument("--n-x", type=int, default=5)
    ap.add_argument("--refs", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="also write every feature-instance record")
    args = ap.parse_args()
    ns = sorted(args.nA)
    os.makedirs(args.out, exist_ok=True)
    tol = P.Config().tol
    summary = dict(args=vars(args), seeds={})
    cols = ["n", "p_opt", "p_common", "p_low", "med_rho_opt", "med_rho_common", "sign_opt", "sign_common",
            "med_kappa", "frac_kappa_lt_1_05", "med_t_common", "frac_I6"]
    for seed in args.seeds:
        recs = run_seed(seed, args, ns, args.device)
        v_rho, v_sign = chain_checks(recs, tol)
        res = dict(viol_N1=v_rho, viol_signN1=v_sign,
                   by_d_mode_n=table(recs, lambda r: (r["d"], r["mode"], r["n"]), tol),
                   by_d_mode_model_n=table(recs, lambda r: (r["d"], r["mode"], r["model"], r["n"]), tol),
                   by_d_mode_rhoc_n=table(recs, lambda r: (r["d"], r["mode"], r["rhoc"], r["n"]), tol))
        summary["seeds"][seed] = res
        if args.raw:
            with open(os.path.join(args.out, f"nested_records_seed{seed}.jsonl"), "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
        print(f"\n=== nested seed {seed}: viol_N1={v_rho} viol_signN1={v_sign}")
        for name in ("by_d_mode_n", "by_d_mode_model_n"):
            print(f"-- {name}: " + " | ".join(cols))
            for k, v in res[name].items():
                print(k, " | ".join(fmt(v.get(c)) for c in cols))
        tot = {c: sum(v[c] for v in res["by_d_mode_n"].values()) for c in ("viol_M", "viol_low", "viol_L")}
        print(f"-- checks (must be 0): viol_N1={v_rho} viol_signN1={v_sign} " +
              " ".join(f"{c}={x}" for c, x in tot.items()))
    with open(os.path.join(args.out, "nested_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
