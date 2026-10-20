"""
lipid/pipeline.py — P0'' nested-anchor experiment, protocol lipid-v3 (REP-03, CAL-01, REP-01, REP-02).

This is nested_anchors_v2.run_seed with
  * the SAME random-number streams, draw order and arithmetic (helpers are imported from nested_anchors_v2), so the
    archived P0'' quantities are reproduced row for row (tests/test_pipeline_regression.py);
  * row-level outputs instead of summaries only, and two additional Lipschitz levels.

Four Lipschitz levels per (feature, instance, anchor budget n):  column L_kind
    opt     L_n = L_A(A_n)                       optimistic: NO guarantee that f is in the class
    LV      L_V = Lip(f on C_data ∪ C_prod ∪ hybrids)          oracle (needs hybrid responses; validation only)
    oracle  L*  = max(L_true, L_V)               the level used in the draft ("oracle-valid L*")
    cert    L_cert (lipid.calibration)           certified globally from generator parameters and sigma only;
                                                 independent of hybrid responses, anchor labels and L_V
Hybrid responses f(z_S) are evaluated ONLY for (i) true_phi (evaluation of containment) and (ii) L_V (oracle).

Outputs per seed (written by scripts/run_nested.py):
  records  long format, one row per (feature-instance, n, L_kind)                — the per-feature schema of the TODO
  diag     one row per (feature-instance, n) with the exact nested_anchors_v2 columns (+ fid, fid_prev, fid_pair)
           — feeding nested_anchors_v2.table() reproduces nested2_summary.json
  skips    every instance / mode that was not evaluated, with the reason (nothing is dropped silently)
  blocks   runtime and memory per (seed, d, rho_corr) block

Independence structure (for REP-01): within one seed, each (d, rho_corr) has ONE sigma draw, ONE set of 20 (x, r)
pairs and ONE nested anchor chain per anchor law, shared by all generators and both modes. The independent unit is
therefore the seed; chain_id = s<seed>-d<d>-r<rho>-<law>.
"""
from __future__ import annotations

import math
import resource
import sys
import time
from collections import defaultdict

import numpy as np
import torch

import lipid
import nested_anchors_v2 as V2
import phase1 as P
from bounds import all_masks, vertex_points
from envelope import weighted_l1
from lipid.calibration import certify
from t6_directional import edge_sets

PROTOCOL = lipid.PROTOCOL
L_KINDS = ("opt", "LV", "oracle", "cert")
TOL_CONTAIN = 1e-9      # scale-aware: relative to L * delta_i (same as nested_anchors_v2 cover checks)
REL_L = 1e-12           # relative tolerance for ordering of Lipschitz levels
NAN = float("nan")

# diag columns, in the order nested_anchors_v2 writes them
V2_COLS = ("seed", "d", "rhoc", "model", "arm", "mode", "n", "inst", "j", "Dbin", "rho_com", "rho_opt", "sign_com",
           "sign_opt", "kappa", "low_com", "range_com", "uni_com", "range_opt", "ceil_t3", "ceil_uni", "s_mean",
           "sbar_mean", "w_s1", "w_s3", "w_s5", "w_sT5", "omega_rel", "lowdens", "cover_com", "cover_opt", "dRho",
           "N2_info", "N2_calib", "N2_total", "dpair", "v_N1", "v_sign", "v_N2", "v_L")


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0


class Cols:
    """Column store: every add() appends one chunk of `m` rows; a value is a scalar (repeated) or a length-m array."""

    def __init__(self):
        self.ch = defaultdict(list)
        self.len = []

    def add(self, m: int, **kv):
        if self.len and set(kv) != set(self.ch):
            raise KeyError(f"column set changed: {sorted(set(kv) ^ set(self.ch))}")
        for k, v in kv.items():
            self.ch[k].append(v)
        self.len.append(m)

    def __len__(self):
        return int(sum(self.len))

    def finalize(self) -> dict:
        L = np.asarray(self.len, dtype=np.int64)
        out = {}
        for k, chunks in self.ch.items():
            if all(np.ndim(c) == 0 for c in chunks):
                out[k] = np.repeat(np.asarray(chunks), L)
            else:
                out[k] = np.concatenate([np.broadcast_to(np.asarray(c), (l,)) for c, l in zip(chunks, L)]) \
                    if len(chunks) else np.zeros(0)
        return out


def _sign(lo_n: np.ndarray, hi_n: np.ndarray, tol: float) -> np.ndarray:
    return np.where(lo_n > tol, 1, np.where(hi_n < -tol, -1, 0)).astype(np.int8)


def run_seed(seed: int, args, ns, device: str, log=print):
    """Returns dict(records=..., diag=..., skips=list, blocks=list, meta=...)."""
    cfg = P.Config()
    tol = cfg.tol
    nmax = max(ns)
    idx = [n - 1 for n in ns]
    K = len(ns)
    rec, diag = Cols(), Cols()
    skips, blocks, cert_meta, rng_keys = [], [], [], []
    fid = 0
    for d in args.dims:
        for rhoc in args.rhos:
            t0 = time.perf_counter()
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            # ------------------------------------------------ identical to nested_anchors_v2.run_seed (RNG order!)
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
            cond = float(torch.linalg.cond(cov))
            Sinv = torch.linalg.inv(cov).to(device)
            Mq = float(torch.quantile(V2.maha(sample(args.m_ref, P.gen_of("nested2-M", seed, d, rhoc)), Sinv), 0.99))
            rng_keys.append(dict(seed=seed, d=d, rho_corr=rhoc,
                                 key_main=P.seed_of("nested2", seed, d, rhoc),
                                 key_C=P.seed_of("nested2-C", seed, d, rhoc),
                                 key_P=P.seed_of("nested2-P", seed, d, rhoc),
                                 key_M=P.seed_of("nested2-M", seed, d, rhoc),
                                 cov_condition_number=cond, mahalanobis_q99=Mq,
                                 sigma=sigma.detach().cpu().tolist()))

            models = {}
            for model in args.models:
                f = P.make_model(model, cfg.a)
                per = {}
                for a in arms:
                    fC = f(Carm[a])
                    LCCv = torch.cummax(V2.newpoint_max(Carm[a], fC, sigma, args.block), 0).values
                    per[a] = dict(fC=fC, LCC=LCCv[idx].tolist(),
                                  fmax=torch.cummax(fC, 0).values[idx].tolist(),
                                  fmin=torch.cummin(fC, 0).values[idx].tolist())
                Lcc_all = max(per[a]["LCC"][-1] for a in arms)
                for i1 in range(len(arms)):
                    for i2 in range(i1 + 1, len(arms)):
                        a1, a2 = arms[i1], arms[i2]
                        Lcc_all = max(Lcc_all, V2.cross_max(Carm[a1], per[a1]["fC"], Carm[a2], per[a2]["fC"],
                                                            sigma, args.block))
                fCall = torch.cat([per[a]["fC"] for a in arms])
                cert = certify(model, d, sigma, cfg.a)          # sigma + generator parameters only
                cert_meta.append(dict(seed=seed, d=d, rho_corr=rhoc, **cert.as_dict()))
                models[model] = (f, f(Xev), f(Rc), per, Lcc_all, fCall, P.lipschitz_true(model, d, sigma, cfg.a),
                                 cert.L_cert)

            n_rec0 = len(rec)
            for xi, ridx in plan:
                x = Xev[xi]
                dxC = {a: weighted_l1(x[None], Carm[a], sigma)[0] for a in arms}
                for rj in ridx:
                    r = Rc[rj]
                    delta = (x - r).abs() / sigma
                    active = torch.nonzero(delta > 0).flatten().tolist()
                    m = len(active)
                    if m == 0:
                        skips.append(dict(seed=seed, d=d, rho_corr=rhoc, x_id=xi, r_id=rj, model="*",
                                          anchor_law="*", mode="*", reason="no_active_feature"))
                        continue
                    D_xr = float(delta.sum())
                    Z = vertex_points(x, r, active, all_masks(m, x.device))
                    ed = edge_sets(m, x.dtype, x.device)
                    Sx = torch.stack([e[0] for e in ed])
                    Px = torch.stack([e[1] for e in ed])
                    W = torch.stack([e[2] for e in ed])
                    dl = delta[active][:, None]
                    dbin = [V2.DLAB[int(np.digitize(D_xr / float(delta[j]), V2.DBINS))] for j in active]
                    dbin_np = np.asarray(dbin)
                    dod = np.asarray([D_xr / float(delta[j]) for j in active])
                    delta_np = delta[active].detach().cpu().numpy()
                    sigma_np = sigma[active].detach().cpu().numpy()
                    act_np = np.asarray(active, dtype=np.int16)
                    dZx = weighted_l1(Z, x[None], sigma)[:, 0]
                    dZr = weighted_l1(Z, r[None], sigma)[:, 0]
                    qbar = {"domain": dZx, "fixed": torch.minimum(dZx, dZr)}
                    sbar = {md: (qb[Px] + qb[Sx]) / dl for md, qb in qbar.items()}
                    hd = (V2.maha(Z, Sinv) > Mq).to(Z.dtype)
                    lowdens = ((W * (hd[Px] + hd[Sx])).sum(1) / 2.0).tolist()
                    lowdens_np = np.asarray(lowdens)
                    DZ = {a: weighted_l1(Z, Carm[a], sigma) for a in arms}
                    DZm = {a: torch.cummin(DZ[a], 1).values for a in arms}
                    drC = {a: weighted_l1(r[None], Carm[a], sigma)[0] for a in arms}
                    DZZ = weighted_l1(Z, Z, sigma)
                    for model, (f, fX, fR, per, Lcc_all, fCall, Ltrue, Lc) in models.items():
                        fx, fr = fX[xi], fR[rj]
                        fxv, frv = float(fx), float(fr)
                        fZ = f(Z)                                       # evaluation / oracle only
                        phi = (W * (fZ[Px] - fZ[Sx])).sum(1)
                        phi_np = phi.detach().cpu().numpy()
                        LV = max(Lcc_all, V2.cross_max(Z, fZ, Call, fCall, sigma, args.block),
                                 float(V2.slopes(fZ[:, None], fZ[None, :], DZZ).max()))
                        Lstar = max(LV, float(Ltrue)) if Ltrue is not None else LV
                        if not (math.isfinite(Lstar) and Lstar > 0):
                            skips.append(dict(seed=seed, d=d, rho_corr=rhoc, x_id=xi, r_id=rj, model=model,
                                              anchor_law="*", mode="*", reason=f"L_star_invalid({Lstar})"))
                            continue
                        Lt = NAN if Ltrue is None else float(Ltrue)
                        lxr = abs(fxv - frv) / D_xr
                        base, base_fid = {}, {}
                        for a in arms:
                            fC = per[a]["fC"]
                            Lx = torch.cummax(V2.slopes(fx, fC, dxC[a]), 0).values[idx].tolist()
                            Lr = torch.cummax(V2.slopes(fr, fC, drC[a]), 0).values[idx].tolist()
                            chain_id = f"s{seed}-d{d}-r{rhoc}-{a}"
                            for mode in ("domain", "fixed"):
                                if mode == "domain":
                                    LA = [max(c, u) for c, u in zip(per[a]["LCC"], Lx)]
                                    Om = [max(M, fxv) - min(mn, fxv) for M, mn in zip(per[a]["fmax"], per[a]["fmin"])]
                                else:
                                    LA = [max(c, u, v, lxr) for c, u, v in zip(per[a]["LCC"], Lx, Lr)]
                                    Om = [max(M, fxv, frv) - min(mn, fxv, frv)
                                          for M, mn in zip(per[a]["fmax"], per[a]["fmin"])]
                                if not all(math.isfinite(L) and L > 0 for L in LA):
                                    skips.append(dict(seed=seed, d=d, rho_corr=rhoc, x_id=xi, r_id=rj, model=model,
                                                      anchor_law=a, mode=mode, reason=f"L_A_invalid({LA})"))
                                    continue
                                sb = sbar[mode]
                                prev = None
                                prevL = {}
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
                                    rc, loc, hic = V2.edge_terms(Dk, fA, Sx, Px, W, dl, Lstar)
                                    ro, loo, hio = V2.edge_terms(Dk, fA, Sx, Px, W, dl, L_n)
                                    rx = (V2.edge_terms(Dk, fA, Sx, Px, W, dl, LA[k + 1])[0] if k + 1 < K
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
                                        V2.wsum(W, t3), V2.wsum(W, rg), V2.wsum(W, torch.maximum(t3, rg)),
                                        V2.wsum(W, rgo),
                                        V2.wsum(W, t3b), V2.wsum(W, torch.maximum(t3b, rgb)),
                                        (W * s).sum(1), (W * sb).sum(1),
                                        (W * (s >= 1.0)).sum(1), (W * (s >= 3.0)).sum(1), (W * (s >= 5.0)).sum(1),
                                        (W * (s >= sT5)).sum(1),
                                        cov_c.to(rc.dtype), cov_o.to(rc.dtype)]).cpu().numpy()
                                    # ---- new in v3: L_V and L_cert levels (same Theorem-1 formula)
                                    rv, lov, hiv = V2.edge_terms(Dk, fA, Sx, Px, W, dl, LV)
                                    rce, loce, hice = V2.edge_terms(Dk, fA, Sx, Px, W, dl, Lc)
                                    E = torch.stack([rv, lov, hiv, rce, loce, hice]).cpu().numpy()

                                    # ---- diag (nested_anchors_v2 columns, vectorized over features)
                                    sg_c = _sign(V[1], V[2], tol)
                                    sg_o = _sign(V[4], V[5], tol)
                                    om_rel = Om[k] / (Lstar * D_xr)
                                    r_c, r_o = V[0], V[3]
                                    fids = np.arange(fid, fid + m, dtype=np.int64)
                                    if prev is None:
                                        dR = info = calib = tot = np.full(m, NAN)
                                        vN1 = vS = vN2 = np.zeros(m, dtype=np.int8)
                                        fid_prev = np.full(m, -1, dtype=np.int64)
                                    else:
                                        dR = prev["rc"] - r_c
                                        info = r_o - prev["rx"]
                                        calib = prev["rx"] - prev["ro"]
                                        tot = r_o - prev["ro"]
                                        vN1 = (r_c > prev["rc"] + tol).astype(np.int8)
                                        vS = ((prev["sg"] != 0) & (sg_c != prev["sg"])).astype(np.int8)
                                        vN2 = ((info > tol) | (calib < -tol)).astype(np.int8)
                                        fid_prev = prev["fid"]
                                    if pair and a == "data":
                                        base[(mode, k)] = r_c
                                        base_fid[(mode, k)] = fids
                                    if pair and a == "prod" and (mode, k) in base:
                                        dp = base[(mode, k)] - r_c
                                        fid_pair = base_fid[(mode, k)]
                                    else:
                                        dp = np.full(m, NAN)
                                        fid_pair = np.full(m, -1, dtype=np.int64)
                                    vL = int(Lstar < L_n * (1 - 1e-12))
                                    diag.add(m, seed=seed, d=d, rhoc=rhoc, model=model, arm=a, mode=mode, n=n,
                                             inst=xi * 1000 + rj, j=act_np, Dbin=dbin_np,
                                             rho_com=r_c, rho_opt=r_o, sign_com=sg_c, sign_opt=sg_o,
                                             kappa=kap_raw, low_com=V[7], range_com=V[8], uni_com=V[9],
                                             range_opt=V[10], ceil_t3=V[11], ceil_uni=V[12],
                                             s_mean=V[13], sbar_mean=V[14], w_s1=V[15], w_s3=V[16],
                                             w_s5=V[17], w_sT5=V[18], omega_rel=om_rel, lowdens=lowdens_np,
                                             cover_com=V[19], cover_opt=V[20],
                                             dRho=dR, N2_info=info, N2_calib=calib, N2_total=tot, dpair=dp,
                                             v_N1=vN1, v_sign=vS, v_N2=vN2, v_L=vL,
                                             fid=fids, fid_prev=fid_prev, fid_pair=fid_pair,
                                             chain_id=chain_id, x_id=xi, r_id=rj)

                                    # ---- records: long format, one chunk per Lipschitz level
                                    levels = dict(opt=(L_n, V[3], V[4], V[5]), LV=(LV, E[0], E[1], E[2]),
                                                  oracle=(Lstar, V[0], V[1], V[2]), cert=(Lc, E[3], E[4], E[5]))
                                    order_ok = (Lc >= LV * (1 - REL_L)) and (LV >= L_n * (1 - REL_L)) and \
                                               (Ltrue is None or Lc >= float(Ltrue) * (1 - REL_L))
                                    rho_mono = ((levels["opt"][1] <= levels["LV"][1] + tol)
                                                & (levels["LV"][1] <= levels["oracle"][1] + tol)
                                                & (levels["oracle"][1] <= levels["cert"][1] + tol))
                                    for kind in L_KINDS:
                                        Lu, rho_k, lo_n, hi_n = levels[kind]
                                        Ld = Lu * delta_np
                                        cont = ((phi_np >= (lo_n - TOL_CONTAIN) * Ld)
                                                & (phi_np <= (hi_n + TOL_CONTAIN) * Ld)).astype(np.int8)
                                        if kind == "opt":
                                            vnest = np.full(m, -1, dtype=np.int8)     # L moves along the chain
                                        elif k == 0:
                                            vnest = np.zeros(m, dtype=np.int8)
                                        else:
                                            plo, phi_prev = prevL[kind]
                                            vnest = ((lo_n < plo - tol) | (hi_n > phi_prev + tol)).astype(np.int8)
                                        prevL[kind] = (lo_n, hi_n)
                                        rec.add(m, seed=seed, model=model, d=d, rho_corr=rhoc, mode=mode,
                                                anchor_law=a, n_anchor=n, chain_id=chain_id,
                                                instance_id=xi * 1000 + rj, x_id=xi, r_id=rj, feature_i=act_np,
                                                fid=fids, delta_i=delta_np, sigma_i=sigma_np, D_over_delta_i=dod,
                                                Dbin=dbin_np, L_kind=kind, L_A=L_n, L_V=LV, L_true=Lt, L_star=Lstar,
                                                L_cert=Lc, L_used=Lu, kappa=Lu / L_n,
                                                lower_norm=lo_n, upper_norm=hi_n,
                                                lower_phi=lo_n * Ld, upper_phi=hi_n * Ld,
                                                Gamma=2.0 * rho_k * Ld, rho=rho_k, true_phi=phi_np,
                                                contained=cont, cert_sign=_sign(lo_n, hi_n, tol),
                                                saturation_gt_08=(rho_k > V2.SAT).astype(np.int8),
                                                v_nest=vnest, v_Lorder=int(not order_ok),
                                                v_rhoL=(~rho_mono).astype(np.int8) if order_ok
                                                else np.full(m, -1, dtype=np.int8))
                                    prev = dict(rc=V[0].copy(), ro=V[3].copy(), rx=V[6].copy(), sg=sg_c, fid=fids)
                                    fid += m
            dt = time.perf_counter() - t0
            blocks.append(dict(seed=seed, d=d, rho_corr=rhoc, runtime_sec=dt, max_rss_mb=peak_rss_mb(),
                               cuda_peak_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                                             if str(device).startswith("cuda") else NAN),
                               n_record_rows=len(rec) - n_rec0))
            log(f"[lipid-v3] seed={seed} d={d} rho={rhoc} records={len(rec)} diag={len(diag)} "
                f"skips={len(skips)} {dt:.1f}s", flush=True)
    return dict(records=rec.finalize(), diag=diag.finalize(), skips=skips, blocks=blocks, cert=cert_meta,
                rng=rng_keys)


def v2_summary_from_diag(C: dict, tol: float) -> dict:
    """Exactly the per-seed aggregation of nested_anchors_v2.main(), applied to v3 diag columns."""
    res = dict(by_d_mode_arm_n=V2.table(C, ("d", "mode", "arm", "n"), tol),
               by_d_mode_arm_model_n=V2.table(C, ("d", "mode", "arm", "model", "n"), tol),
               by_d_mode_arm_rhoc_n=V2.table(C, ("d", "mode", "arm", "rhoc", "n"), tol),
               by_d_mode_arm_rhoc_Dbin_n=V2.table(C, ("d", "mode", "arm", "rhoc", "Dbin", "n"), tol))
    res["checks"] = {c: sum(v[c] for v in res["by_d_mode_arm_n"].values()) for c in V2.CHECKS}
    return res
