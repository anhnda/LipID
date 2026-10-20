"""
lipid/instances.py — the Section-6 (P0'') nested-anchor instances, regenerated chain by chain with the SAME random
streams, draw order and arithmetic as nested_anchors_v2.run_seed and lipid.pipeline.run_seed.

Used by the v1.2 analyses that need more than the archived per-feature records (pairwise ranking JNT-02, absolute
ranking JNT-03, certificate diagnostics CERT-02). Everything is matched to `runs/<tag>/records_seed*` through the keys
(seed, model, d, rho_corr, mode, anchor_law, n_anchor, instance_id, feature_i, L_kind); tests/test_v12_instances.py
checks on a small grid that the four Lipschitz levels and the Theorem-1 endpoints coincide with lipid.pipeline.

IMPORTANT: the anchor arms used to build L_V (Lipschitz constant of f on C_data u C_prod u hybrids) must be the same as
in the records run (default: both arms). Selecting x's with `select_x` does not change any random draw (the plan of
(x, r) pairs is drawn before the loop), so subsets stay matched.

A chain is one (seed, d, rho_c, x, r, model, anchor law, mode); it holds the nested budgets k = 0..K-1 with
    Dk   = weighted-l1 distances (2^m, n_k + 1 or + 2) from the hybrid vertices to A_{n_k} (anchors C[:n], x, (r))
    fA   = anchor labels,  L_n = L_A(A_{n_k}) (optimistic),  Omega = max_A f - min_A f.
Hybrid responses fZ are used ONLY for the true attribution (evaluation) and the oracle level L_V, exactly as before.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import lipid  # noqa: F401
import nested_anchors_v2 as V2
import phase1 as P
from bounds import all_masks, vertex_points
from envelope import weighted_l1
from lipid.calibration import certify
from t6_directional import edge_sets

L_KINDS = ("opt", "LV", "oracle", "cert")


def iter_chains(seed: int, args, ns, device: str, *, select_x=None, skips: list | None = None, blocks=None):
    """Generator of chain dicts. `args` needs dims, rhos, models, arms, n_x, refs, block (same meaning as
    scripts/run_nested.py). `select_x(d, rhoc, xi) -> bool` skips explained points without touching the RNG.
    `skips` (list) receives every skipped unit with its reason. `blocks(d, rhoc, event)` is an optional callback
    ('start' / 'end') for timing."""
    cfg = P.Config()
    ns = sorted(ns)
    nmax = max(ns)
    idx = [n - 1 for n in ns]
    K = len(ns)
    arms_req = [a for a in ("data", "prod") if a in args.arms]
    for d in args.dims:
        for rhoc in args.rhos:
            if blocks:
                blocks(d, rhoc, "start")
            # ------------------------------------------- identical to nested_anchors_v2.run_seed (RNG order!)
            gd = P.gen_of("nested2", seed, d, rhoc)
            sample = P.gaussian_sampler(d, rhoc, device)
            sigma = P.robust_sigma(sample(cfg.n_train, gd))
            Rc = sample(cfg.n_candidate, gd)
            Xev = sample(args.n_x, gd)
            plan = [(xi, torch.randperm(cfg.n_candidate, generator=gd)[:args.refs].tolist())
                    for xi in range(args.n_x)]
            Carm = {}
            if "data" in arms_req:
                Carm["data"] = sample(nmax, P.gen_of("nested2-C", seed, d, rhoc))
            if "prod" in arms_req:
                Carm["prod"] = P.gaussian_sampler(d, 0.0, device)(nmax, P.gen_of("nested2-P", seed, d, rhoc))
            arms = list(Carm)
            Call = torch.cat([Carm[a] for a in arms])

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
                cert = certify(model, d, sigma, cfg.a)
                models[model] = (f, f(Xev), f(Rc), per, Lcc_all, fCall, P.lipschitz_true(model, d, sigma, cfg.a),
                                 cert.L_cert)

            for xi, ridx in plan:
                if select_x is not None and not select_x(d, rhoc, xi):
                    continue
                x = Xev[xi]
                dxC = {a: weighted_l1(x[None], Carm[a], sigma)[0] for a in arms}
                for rj in ridx:
                    r = Rc[rj]
                    delta = (x - r).abs() / sigma
                    active = torch.nonzero(delta > 0).flatten().tolist()
                    m = len(active)
                    if m == 0:
                        if skips is not None:
                            skips.append(dict(seed=seed, d=d, rho_corr=rhoc, x_id=xi, r_id=rj, model="*",
                                              anchor_law="*", mode="*", reason="no_active_feature"))
                        continue
                    D_xr = float(delta.sum())
                    Z = vertex_points(x, r, active, all_masks(m, x.device))
                    ed = edge_sets(m, x.dtype, x.device)
                    Sx = torch.stack([e[0] for e in ed])
                    Px = torch.stack([e[1] for e in ed])
                    W = torch.stack([e[2] for e in ed])
                    dl = delta[active]
                    dod = np.asarray([D_xr / float(delta[j]) for j in active])
                    dbin = np.asarray([V2.DLAB[int(np.digitize(v, V2.DBINS))] for v in dod])
                    dZx = weighted_l1(Z, x[None], sigma)[:, 0]
                    dZr = weighted_l1(Z, r[None], sigma)[:, 0]
                    DZ = {a: weighted_l1(Z, Carm[a], sigma) for a in arms}
                    drC = {a: weighted_l1(r[None], Carm[a], sigma)[0] for a in arms}
                    DZZ = weighted_l1(Z, Z, sigma)
                    for model, (f, fX, fR, per, Lcc_all, fCall, Ltrue, Lc) in models.items():
                        fx, fr = fX[xi], fR[rj]
                        fxv, frv = float(fx), float(fr)
                        fZ = f(Z)                                       # evaluation / oracle only
                        phi = (W * (fZ[Px] - fZ[Sx])).sum(1)
                        LV = max(Lcc_all, V2.cross_max(Z, fZ, Call, fCall, sigma, args.block),
                                 float(V2.slopes(fZ[:, None], fZ[None, :], DZZ).max()))
                        Lstar = max(LV, float(Ltrue)) if Ltrue is not None else LV
                        if not (math.isfinite(Lstar) and Lstar > 0):
                            if skips is not None:
                                skips.append(dict(seed=seed, d=d, rho_corr=rhoc, x_id=xi, r_id=rj, model=model,
                                                  anchor_law="*", mode="*", reason=f"L_star_invalid({Lstar})"))
                            continue
                        lxr = abs(fxv - frv) / D_xr
                        for a in arms:
                            fC = per[a]["fC"]
                            Lx = torch.cummax(V2.slopes(fx, fC, dxC[a]), 0).values[idx].tolist()
                            Lr = torch.cummax(V2.slopes(fr, fC, drC[a]), 0).values[idx].tolist()
                            for mode in ("domain", "fixed"):
                                if mode == "domain":
                                    LA = [max(c, u) for c, u in zip(per[a]["LCC"], Lx)]
                                    Om = [max(M, fxv) - min(mn, fxv) for M, mn in zip(per[a]["fmax"], per[a]["fmin"])]
                                else:
                                    LA = [max(c, u, v, lxr) for c, u, v in zip(per[a]["LCC"], Lx, Lr)]
                                    Om = [max(M, fxv, frv) - min(mn, fxv, frv)
                                          for M, mn in zip(per[a]["fmax"], per[a]["fmin"])]
                                if not all(math.isfinite(L) and L > 0 for L in LA):
                                    if skips is not None:
                                        skips.append(dict(seed=seed, d=d, rho_corr=rhoc, x_id=xi, r_id=rj,
                                                          model=model, anchor_law=a, mode=mode,
                                                          reason=f"L_A_invalid({LA})"))
                                    continue
                                levels = []
                                for k in range(K):
                                    n = ns[k]
                                    if mode == "domain":
                                        Dk = torch.cat([DZ[a][:, :n], dZx[:, None]], 1)
                                        fA = torch.cat([fC[:n], fx.reshape(1)])
                                        Ak = torch.cat([Carm[a][:n], x[None]])
                                    else:
                                        Dk = torch.cat([DZ[a][:, :n], dZx[:, None], dZr[:, None]], 1)
                                        fA = torch.cat([fC[:n], fx.reshape(1), fr.reshape(1)])
                                        Ak = torch.cat([Carm[a][:n], x[None], r[None]])
                                    levels.append(dict(k=k, n=n, Dk=Dk, fA=fA, A=Ak, L_n=LA[k], Omega=Om[k]))
                                yield dict(seed=seed, d=d, rhoc=rhoc, sigma=sigma, xi=xi, rj=rj,
                                           instance_id=xi * 1000 + rj, x=x, r=r, delta=delta, active=active, m=m,
                                           D_xr=D_xr, Z=Z, Sx=Sx, Px=Px, W=W, dl=dl, dod=dod, dbin=dbin,
                                           model=model, fZ=fZ, phi=phi, L_V=LV, L_star=Lstar,
                                           L_true=(math.nan if Ltrue is None else float(Ltrue)), L_cert=float(Lc),
                                           arm=a, mode=mode, chain_id=f"s{seed}-d{d}-r{rhoc}-{a}", levels=levels)
            if blocks:
                blocks(d, rhoc, "end")


def level_values(chain: dict, lev: dict) -> dict:
    """The four Lipschitz levels of one budget, keyed like lipid.pipeline.L_KINDS."""
    return dict(opt=float(lev["L_n"]), LV=float(chain["L_V"]), oracle=float(chain["L_star"]),
                cert=float(chain["L_cert"]))


def envelopes_from_D(Dk: torch.Tensor, fA: torch.Tensor, Lvec: torch.Tensor):
    """Batched McShane-Whitney envelopes for several L at once: Lvec (K,) -> ell, h (K, N)."""
    LD = Lvec[:, None, None] * Dk[None]
    return (fA[None, None, :] - LD).max(2).values, (fA[None, None, :] + LD).min(2).values


__all__ = ["L_KINDS", "iter_chains", "level_values", "envelopes_from_D"]
