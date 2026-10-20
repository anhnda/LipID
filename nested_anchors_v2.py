#!/usr/bin/env python3
"""
nested_anchors_v2.py — P0'' nested-anchor experiment, v2 (sample complexity of Shapley identifiability, T-A).

Changes vs v1 (round-9 review):
  * L* no longer depends on the anchor budget:
        L* = max(L_true, L_V),  L_V = Lipschitz constant of f on  C_data ∪ C_prod ∪ Z   (Z = all hybrids, incl. x, r)
    -> L* >= L_A(A_n) for every n, mode, arm;  the true f restricted to A ∪ Z is L*-Lipschitz, so the true phi_i
       MUST lie in [lo, hi] at L*  (checked: viol_cover);  kappa = 1 at n_max is no longer true by construction.
  * C has its own RNG stream (x, r, sigma do not move when n_max changes).
  * Two anchor arms with the SAME x, r, sigma, f and L*:
        data : C ~ N(0, Sigma_rho)               (passive anchors from the data distribution)
        prod : C ~ N(0, I)  (product of the same unit-variance marginals: covers the hybrids)
    rho_c = 0 is a NEGATIVE CONTROL (both arms have the same distribution).
  * Per feature-instance (all at L* unless "opt"):
        rho_com, rho_opt (L_n = L_A(A_n)), signs, kappa = L*/L_n
        low_com   : T3            sum_S w [min(1, (1-1/k) s_S - 1/k)]_+
        range_com : RANGE         sum_S w [min(1, s_S - Omega_A/(L delta_i))]_+
        uni_com   : unified edge  sum_S w [min(1, max(T3_S, RANGE_S))]_+        (both are valid per edge)
        range_opt : RANGE at L_n (kappa = 1, where T3 is void)
        ceil_t3 / ceil_uni : the same certificates with s_S replaced by the endpoint-only value sbar_S
                    (A = {x} domain, {x, r} fixed): anchor-free ceilings, checked low <= ceil_t3, uni <= ceil_uni
        s_mean, sbar_mean, w_s1/3/5/T5 : Shapley-weighted geometry (s_S = (q(z_S+i)+q(z_S))/delta_i),
                    T5 threshold s* = (k+1)/(k-1)
        omega_rel : Omega_A / (L* D(x,r))
        lowdens   : Shapley-weighted share of hybrid endpoints with Mahalanobis M(z) above the empirical
                    99% quantile of M under the data distribution
        cover_opt : true phi_i inside the optimistic interval at L_n (may fail: L_n can be below the true slope)
        dRho      : paired drop rho_com(n_prev) - rho_com(n)                 (>= 0 by N1)
        N2        : rho(A_n';L_n') - rho(A_n;L_n) = [rho(A_n';L_n') - rho(A_n;L_n')]  (info, <= 0)
                                                 + [rho(A_n;L_n') - rho(A_n;L_n)]    (calib, >= 0)
                    (the only feasible order: rho(A_n'; L_n) is a possibly empty class)
        dpair     : rho_com(data) - rho_com(prod), same instance / feature / n / mode
Checks (any non-zero count is a bug): viol_N1, viol_signN1, viol_N2, viol_M, viol_low, viol_ceil, viol_L, viol_cover.

Exploratory predictions written BEFORE the run (to be turned into locked thresholds for fresh seeds 8-11):
  E1  mean paired dRho at L* shrinks with d (d x log n interaction); the d5-6 curves start to bend by n = 4096.
  E2  rho_c = .99: p_common(data) > p_common(prod) at n >= 256, also within D/delta bins; rho_c = 0: ~ 0.
  E3  cert_share(uni) < 10% at n >= 1024, d >= 5; range_opt non-trivial mostly for unstable_id.
  E4  for d >= 5, N2_info and N2_calib are both clearly non-zero and largely cancel (|N2_total| << |N2_info|).

    python nested_anchors_v2.py --seeds 1 2 3 4 --out nested2_explore --device cuda
    python nested_anchors_v2.py --seeds 8 9 10 11 --out nested2_confirm --device cuda    # after the lock
Writes <out>/nested2_summary.json after every seed (+ nested2_records_seed<s>.npz with --raw) and prints tables.
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
DBINS = (2.0, 4.0, 8.0)
DLAB = ("lt2", "2-4", "4-8", "ge8")
NAN = float("nan")


# ----------------------------------------------------------------------------------------------- helpers
def slopes(fa, fb, D):
    return torch.where(D > 1e-12, (fa - fb).abs() / D.clamp_min(1e-12), torch.zeros_like(D))


def newpoint_max(C, fC, sigma, block):
    """m[k] = max_{j<k} slope(c_k, c_j);  L_A(C[:n]) = cummax(m)[n-1].  Row-blocked, O(block * n) memory."""
    n = C.shape[0]
    out = torch.zeros(n, dtype=C.dtype, device=C.device)
    ar = torch.arange(n, device=C.device)
    for s in range(0, n, block):
        e = min(n, s + block)
        S = slopes(fC[s:e, None], fC[None, :e], weighted_l1(C[s:e], C[:e], sigma))
        S = torch.where(ar[None, :e] < ar[s:e, None], S, torch.zeros_like(S))
        out[s:e] = S.max(1).values
    return out


def cross_max(Pp, fP, Q, fQ, sigma, block):
    best = 0.0
    for s in range(0, Pp.shape[0], block):
        S = slopes(fP[s:s + block, None], fQ[None, :], weighted_l1(Pp[s:s + block], Q, sigma))
        best = max(best, float(S.max()))
    return best


def maha(Z, Sinv):
    return ((Z @ Sinv) * Z).sum(1)


def edge_terms(D, fA, Sx, Px, W, dl, L):
    """Exact normalized AIW per active feature + normalized endpoints of the identified interval."""
    LD = L * D
    ell = (fA[None, :] - LD).max(1).values
    h = (fA[None, :] + LD).min(1).values
    Ld = L * dl
    cp = ((h[Px] - ell[Sx]) / Ld).clamp(max=1.0)
    cm = ((h[Sx] - ell[Px]) / Ld).clamp(max=1.0)
    return (W * (cp + cm)).sum(1) / 2.0, -(W * cm).sum(1), (W * cp).sum(1)


def wsum(W, t):
    return (W * t.clamp(min=0.0, max=1.0)).sum(1)


# ----------------------------------------------------------------------------------------------- one seed
def run_seed(seed, args, ns, device):
    cfg = P.Config()
    tol = cfg.tol
    nmax = max(ns)
    idx = [n - 1 for n in ns]
    K = len(ns)
    col = defaultdict(list)
    for d in args.dims:
        for rhoc in args.rhos:
            gd = P.gen_of("nested2", seed, d, rhoc)
            sample = P.gaussian_sampler(d, rhoc, device)
            sigma = P.robust_sigma(sample(cfg.n_train, gd))
            Rc = sample(cfg.n_candidate, gd)
            Xev = sample(args.n_x, gd)
            plan = [(xi, torch.randperm(cfg.n_candidate, generator=gd)[:args.refs].tolist())
                    for xi in range(args.n_x)]
            Carm = {}
            if "data" in args.arms:
                Carm["data"] = sample(nmax, P.gen_of("nested2-C", seed, d, rhoc))
            if "prod" in args.arms:
                Carm["prod"] = P.gaussian_sampler(d, 0.0, device)(nmax, P.gen_of("nested2-P", seed, d, rhoc))
            arms = list(Carm)
            pair = arms == ["data", "prod"]
            Call = torch.cat([Carm[a] for a in arms])
            cov = (1.0 - rhoc) * torch.eye(d) + rhoc * torch.ones(d, d)
            Sinv = torch.linalg.inv(cov).to(device)
            Mq = float(torch.quantile(maha(sample(args.m_ref, P.gen_of("nested2-M", seed, d, rhoc)), Sinv), 0.99))

            models = {}
            for model in args.models:
                f = P.make_model(model, cfg.a)
                per = {}
                for a in arms:
                    fC = f(Carm[a])
                    LCCv = torch.cummax(newpoint_max(Carm[a], fC, sigma, args.block), 0).values
                    per[a] = dict(fC=fC, LCC=LCCv[idx].tolist(),
                                  fmax=torch.cummax(fC, 0).values[idx].tolist(),
                                  fmin=torch.cummin(fC, 0).values[idx].tolist())
                Lcc_all = max(per[a]["LCC"][-1] for a in arms)
                for i1 in range(len(arms)):
                    for i2 in range(i1 + 1, len(arms)):
                        a1, a2 = arms[i1], arms[i2]
                        Lcc_all = max(Lcc_all, cross_max(Carm[a1], per[a1]["fC"], Carm[a2], per[a2]["fC"],
                                                         sigma, args.block))
                fCall = torch.cat([per[a]["fC"] for a in arms])
                models[model] = (f, f(Xev), f(Rc), per, Lcc_all, fCall, P.lipschitz_true(model, d, sigma, cfg.a))

            for xi, ridx in plan:
                x = Xev[xi]
                dxC = {a: weighted_l1(x[None], Carm[a], sigma)[0] for a in arms}
                for rj in ridx:
                    r = Rc[rj]
                    delta = (x - r).abs() / sigma
                    active = torch.nonzero(delta > 0).flatten().tolist()
                    m = len(active)
                    if m == 0:
                        continue
                    D_xr = float(delta.sum())
                    Z = vertex_points(x, r, active, all_masks(m, x.device))
                    ed = edge_sets(m, x.dtype, x.device)
                    Sx = torch.stack([e[0] for e in ed])
                    Px = torch.stack([e[1] for e in ed])
                    W = torch.stack([e[2] for e in ed])
                    dl = delta[active][:, None]
                    dbin = [DLAB[int(np.digitize(D_xr / float(delta[j]), DBINS))] for j in active]
                    dZx = weighted_l1(Z, x[None], sigma)[:, 0]
                    dZr = weighted_l1(Z, r[None], sigma)[:, 0]
                    qbar = {"domain": dZx, "fixed": torch.minimum(dZx, dZr)}
                    sbar = {md: (qb[Px] + qb[Sx]) / dl for md, qb in qbar.items()}
                    hd = (maha(Z, Sinv) > Mq).to(Z.dtype)
                    lowdens = ((W * (hd[Px] + hd[Sx])).sum(1) / 2.0).tolist()
                    DZ = {a: weighted_l1(Z, Carm[a], sigma) for a in arms}
                    DZm = {a: torch.cummin(DZ[a], 1).values for a in arms}
                    drC = {a: weighted_l1(r[None], Carm[a], sigma)[0] for a in arms}
                    DZZ = weighted_l1(Z, Z, sigma)
                    for model, (f, fX, fR, per, Lcc_all, fCall, Ltrue) in models.items():
                        fx, fr = fX[xi], fR[rj]
                        fxv, frv = float(fx), float(fr)
                        fZ = f(Z)
                        phi = (W * (fZ[Px] - fZ[Sx])).sum(1)
                        LV = max(Lcc_all, cross_max(Z, fZ, Call, fCall, sigma, args.block),
                                 float(slopes(fZ[:, None], fZ[None, :], DZZ).max()))
                        Lstar = max(LV, float(Ltrue)) if Ltrue is not None else LV
                        if not (math.isfinite(Lstar) and Lstar > 0):
                            continue
                        lxr = abs(fxv - frv) / D_xr
                        base = {}
                        for a in arms:
                            fC = per[a]["fC"]
                            Lx = torch.cummax(slopes(fx, fC, dxC[a]), 0).values[idx].tolist()
                            Lr = torch.cummax(slopes(fr, fC, drC[a]), 0).values[idx].tolist()
                            for mode in ("domain", "fixed"):
                                if mode == "domain":
                                    LA = [max(c, u) for c, u in zip(per[a]["LCC"], Lx)]
                                    Om = [max(M, fxv) - min(mn, fxv) for M, mn in zip(per[a]["fmax"], per[a]["fmin"])]
                                else:
                                    LA = [max(c, u, v, lxr) for c, u, v in zip(per[a]["LCC"], Lx, Lr)]
                                    Om = [max(M, fxv, frv) - min(mn, fxv, frv)
                                          for M, mn in zip(per[a]["fmax"], per[a]["fmin"])]
                                if not all(math.isfinite(L) and L > 0 for L in LA):
                                    continue
                                sb = sbar[mode]
                                prev = None
                                for k in range(K):
                                    n = ns[k]
                                    if mode == "domain":
                                        Dk = torch.cat([DZ[a][:, :n], dZx[:, None]], 1)
                                        fA = torch.cat([fC[:n], fx.reshape(1)])
                                        q = torch.minimum(DZm[a][:, n - 1], dZx)
                                    else:
                                        Dk = torch.cat([DZ[a][:, :n], dZx[:, None], dZr[:, None]], 1)
                                        fA = torch.cat([fC[:n], fx.reshape(1), fr.reshape(1)])
                                        q = torch.minimum(torch.minimum(DZm[a][:, n - 1], dZx), dZr)
                                    L_n = LA[k]
                                    kap_raw = Lstar / L_n
                                    kap = max(kap_raw, 1.0)
                                    rc, loc, hic = edge_terms(Dk, fA, Sx, Px, W, dl, Lstar)
                                    ro, loo, hio = edge_terms(Dk, fA, Sx, Px, W, dl, L_n)
                                    rx = (edge_terms(Dk, fA, Sx, Px, W, dl, LA[k + 1])[0] if k + 1 < K
                                          else torch.full_like(rc, NAN))
                                    s = (q[Px] + q[Sx]) / dl
                                    t3 = (1.0 - 1.0 / kap) * s - 1.0 / kap
                                    rg = s - Om[k] / (Lstar * dl)
                                    rgo = s - Om[k] / (L_n * dl)
                                    t3b = (1.0 - 1.0 / kap) * sb - 1.0 / kap
                                    rgb = sb - Om[k] / (Lstar * dl)
                                    sT5 = (kap + 1.0) / (kap - 1.0) if kap > 1.0 + 1e-12 else math.inf
                                    Lsd, Lnd = Lstar * dl[:, 0], L_n * dl[:, 0]
                                    cov_c = (phi >= (loc - 1e-9) * Lsd) & (phi <= (hic + 1e-9) * Lsd)
                                    cov_o = (phi >= (loo - 1e-9) * Lnd) & (phi <= (hio + 1e-9) * Lnd)
                                    V = torch.stack([
                                        rc, loc, hic, ro, loo, hio, rx,
                                        wsum(W, t3), wsum(W, rg), wsum(W, torch.maximum(t3, rg)), wsum(W, rgo),
                                        wsum(W, t3b), wsum(W, torch.maximum(t3b, rgb)),
                                        (W * s).sum(1), (W * sb).sum(1),
                                        (W * (s >= 1.0)).sum(1), (W * (s >= 3.0)).sum(1), (W * (s >= 5.0)).sum(1),
                                        (W * (s >= sT5)).sum(1),
                                        cov_c.to(rc.dtype), cov_o.to(rc.dtype)]).cpu().numpy()
                                    sg_c = [1 if V[1, jj] > tol else (-1 if V[2, jj] < -tol else 0) for jj in range(m)]
                                    sg_o = [1 if V[4, jj] > tol else (-1 if V[5, jj] < -tol else 0) for jj in range(m)]
                                    om_rel = Om[k] / (Lstar * D_xr)
                                    for jj, j in enumerate(active):
                                        r_c, r_o = float(V[0, jj]), float(V[3, jj])
                                        if prev is None:
                                            dR = info = calib = tot = NAN
                                            vN1 = vS = vN2 = 0
                                        else:
                                            dR = prev["rc"][jj] - r_c
                                            info = r_o - prev["rx"][jj]
                                            calib = prev["rx"][jj] - prev["ro"][jj]
                                            tot = r_o - prev["ro"][jj]
                                            vN1 = int(r_c > prev["rc"][jj] + tol)
                                            vS = int(prev["sg"][jj] != 0 and sg_c[jj] != prev["sg"][jj])
                                            vN2 = int(info > tol or calib < -tol)
                                        if pair and a == "data":
                                            base[(mode, k, jj)] = r_c
                                        dp = base.get((mode, k, jj), NAN) - r_c if (pair and a == "prod") else NAN
                                        row = dict(
                                            seed=seed, d=d, rhoc=rhoc, model=model, arm=a, mode=mode, n=n,
                                            inst=xi * 1000 + rj, j=j, Dbin=dbin[jj],
                                            rho_com=r_c, rho_opt=r_o, sign_com=sg_c[jj], sign_opt=sg_o[jj],
                                            kappa=kap_raw, low_com=V[7, jj], range_com=V[8, jj], uni_com=V[9, jj],
                                            range_opt=V[10, jj], ceil_t3=V[11, jj], ceil_uni=V[12, jj],
                                            s_mean=V[13, jj], sbar_mean=V[14, jj], w_s1=V[15, jj], w_s3=V[16, jj],
                                            w_s5=V[17, jj], w_sT5=V[18, jj], omega_rel=om_rel, lowdens=lowdens[jj],
                                            cover_com=V[19, jj], cover_opt=V[20, jj],
                                            dRho=dR, N2_info=info, N2_calib=calib, N2_total=tot, dpair=dp,
                                            v_N1=vN1, v_sign=vS, v_N2=vN2, v_L=int(Lstar < L_n * (1 - 1e-12)))
                                        for c_, v_ in row.items():
                                            col[c_].append(v_)
                                    prev = dict(rc=V[0].tolist(), ro=V[3].tolist(), rx=V[6].tolist(), sg=sg_c)
            print(f"[nested2] seed={seed} d={d} rho={rhoc} rows={len(col['rho_com'])}", flush=True)
    return {c: np.asarray(v) for c, v in col.items()}


# ----------------------------------------------------------------------------------------------- tables
def _nm(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else None


def _nq(a, qq):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.quantile(a, qq)) if a.size else None


def stats(C, ix, tol):
    g = lambda c: C[c][ix].astype(float)
    rc, ro, kap = g("rho_com"), g("rho_opt"), g("kappa")
    low, rng, uni, rgo = g("low_com"), g("range_com"), g("uni_com"), g("range_opt")
    pc = float(np.mean(rc > SAT))
    out = dict(
        n=int(len(ix)),
        p_opt=float(np.mean(ro > SAT)), p_common=pc,
        p_low=float(np.mean(low > SAT)), p_range_com=float(np.mean(rng > SAT)), p_uni_com=float(np.mean(uni > SAT)),
        cert_share=(float(np.mean(uni > SAT)) / pc if pc > 0 else None),
        p_range_opt=float(np.mean(rgo > SAT)),
        p_ceilT3=float(np.mean(g("ceil_t3") > SAT)), p_ceil_uni=float(np.mean(g("ceil_uni") > SAT)),
        mean_rho_com=float(rc.mean()), q10_rho_com=float(np.quantile(rc, .1)),
        q50_rho_com=float(np.quantile(rc, .5)), q90_rho_com=float(np.quantile(rc, .9)),
        mean_rho_opt=float(ro.mean()), mean_rho_opt_abs=float(np.mean(ro / kap)),
        sign_opt=float(np.mean(g("sign_opt") != 0)), sign_common=float(np.mean(g("sign_com") != 0)),
        med_kappa=float(np.median(kap)), frac_kappa_lt_1_05=float(np.mean(kap < 1.05)),
        mean_s=float(g("s_mean").mean()), mean_sbar=float(g("sbar_mean").mean()),
        w_s1=float(g("w_s1").mean()), w_s3=float(g("w_s3").mean()), w_s5=float(g("w_s5").mean()),
        w_sT5=float(g("w_sT5").mean()),
        med_omega_rel=float(np.median(g("omega_rel"))), lowdens=float(g("lowdens").mean()),
        mean_dRho=_nm(g("dRho")), q90_dRho=_nq(g("dRho"), .9),
        N2_info=_nm(g("N2_info")), N2_calib=_nm(g("N2_calib")), N2_total=_nm(g("N2_total")),
        cover_opt=float(g("cover_opt").mean()),
        mean_dpair=_nm(g("dpair")),
        frac_data_gt_prod=(None if np.all(np.isnan(g("dpair"))) else
                           float(np.mean(g("dpair")[~np.isnan(g("dpair"))] > tol))),
        viol_M=int(np.sum(ro > rc + tol)),
        viol_low=int(np.sum(np.maximum(np.maximum(low, rng), uni) > rc + tol) + np.sum(rgo > ro + tol)),
        viol_ceil=int(np.sum(low > g("ceil_t3") + tol) + np.sum(uni > g("ceil_uni") + tol)),
        viol_L=int(np.sum(g("v_L"))), viol_N1=int(np.sum(g("v_N1"))), viol_signN1=int(np.sum(g("v_sign"))),
        viol_N2=int(np.sum(g("v_N2"))), viol_cover=int(np.sum(g("cover_com") < 0.5)))
    return out


def table(C, keys, tol):
    G = defaultdict(list)
    karr = [C[k] for k in keys]
    for i in range(len(C["rho_com"])):
        G[tuple(ka[i].item() if hasattr(ka[i], "item") else ka[i] for ka in karr)].append(i)
    return {"/".join(str(v) for v in key): stats(C, np.asarray(G[key]), tol)
            for key in sorted(G, key=lambda t: tuple((0, v) if not isinstance(v, str) else (1, v) for v in t))}


CHECKS = ("viol_N1", "viol_signN1", "viol_N2", "viol_M", "viol_low", "viol_ceil", "viol_L", "viol_cover")


def fmt(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6, 0.9, 0.99])
    ap.add_argument("--nA", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    ap.add_argument("--models", nargs="+", default=["linear", "fa", "stable_ood", "unstable_id"])
    ap.add_argument("--arms", nargs="+", default=["data", "prod"], choices=["data", "prod"])
    ap.add_argument("--n-x", type=int, default=5)
    ap.add_argument("--refs", type=int, default=4)
    ap.add_argument("--m-ref", type=int, default=200000, help="reference sample for the Mahalanobis 99%% quantile")
    ap.add_argument("--block", type=int, default=2048, help="row block for the pairwise slope maxima")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="also write every feature-instance row (.npz)")
    args = ap.parse_args()
    args.arms = [a for a in ("data", "prod") if a in args.arms]
    ns = sorted(set(args.nA))
    os.makedirs(args.out, exist_ok=True)
    tol = P.Config().tol
    summary = dict(args=vars(args), seeds={})
    cols = ["n", "p_opt", "p_common", "p_uni_com", "cert_share", "p_ceilT3", "p_range_opt", "mean_rho_com",
            "mean_dRho", "N2_info", "N2_calib", "N2_total", "med_kappa", "mean_s", "lowdens", "cover_opt"]
    for seed in args.seeds:
        C = run_seed(seed, args, ns, args.device)
        res = dict(
            by_d_mode_arm_n=table(C, ("d", "mode", "arm", "n"), tol),
            by_d_mode_arm_model_n=table(C, ("d", "mode", "arm", "model", "n"), tol),
            by_d_mode_arm_rhoc_n=table(C, ("d", "mode", "arm", "rhoc", "n"), tol),
            by_d_mode_arm_rhoc_Dbin_n=table(C, ("d", "mode", "arm", "rhoc", "Dbin", "n"), tol))
        res["checks"] = {c: sum(v[c] for v in res["by_d_mode_arm_n"].values()) for c in CHECKS}
        summary["seeds"][seed] = res
        if args.raw:
            np.savez_compressed(os.path.join(args.out, f"nested2_records_seed{seed}.npz"), **C)
        with open(os.path.join(args.out, "nested2_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)

        print(f"\n=== nested2 seed {seed}")
        print("-- by_d_mode_arm_n: key | " + " | ".join(cols))
        for k, v in res["by_d_mode_arm_n"].items():
            print(k, "|", " | ".join(fmt(v.get(c)) for c in cols))
        if args.arms == ["data", "prod"]:
            print("-- data vs prod: d/mode/rhoc/n | p_common_data | p_common_prod | mean_dpair | frac_data_gt_prod")
            T = res["by_d_mode_arm_rhoc_n"]
            for k, v in T.items():
                d_, mode_, arm_, rc_, n_ = k.split("/")
                if arm_ != "prod":
                    continue
                vd = T.get(f"{d_}/{mode_}/data/{rc_}/{n_}", {})
                print(f"{d_}/{mode_}/{rc_}/{n_} | {fmt(vd.get('p_common'))} | {fmt(v['p_common'])} | "
                      f"{fmt(v['mean_dpair'])} | {fmt(v['frac_data_gt_prod'])}")
        print("-- checks (must be 0): " + " ".join(f"{c}={x}" for c, x in res["checks"].items()))


if __name__ == "__main__":
    main()
