#!/usr/bin/env python3
"""
met02_scale_sensitivity.py — MET-02: sensitivity of AIW, saturation and certificates to the feature scales sigma (Q2).

    python scripts/met02_scale_sensitivity.py --device cuda            # synthetic (4 seeds) + diabetes
    python scripts/met02_scale_sensitivity.py --quick                  # d in {3,4}, 1 seed, n = 256
    python scripts/met02_scale_sensitivity.py --analyze-only

Scale choices (computed on a separate training sample, then fixed):  mad = 1.4826 MAD (SD fallback where MAD = 0),
sd = standard deviation, unit = raw units. Each vector is rescaled to the geometric mean of the mad vector: this is
cosmetic, because sigma -> c sigma together with L -> c L leaves every interval unchanged (checked on every instance and
reported in met02_checks.json). For every choice the Lipschitz levels are recomputed in that metric: L_cert (appendix-H
certificate with that sigma), L* = max(L_true, L_V) (oracle, evaluation only) and L_A (optimistic). The model,
observations, (x, r) and the Shapley game are unchanged — so the true attribution is identical across choices.

Design (protocol MET-02): with the Section-6 Gaussian design (identical unit-variance marginals) the three choices
coincide up to estimation noise and a common factor, so the comparison would be vacuous. The synthetic design therefore
uses a Gaussian copula (rho_c in {0, .9}) with coordinate marginals cycling gauss / t3 / lognormal / binary(Z > .5),
multiplied by log-spaced scales from 0.25 to 4. Real data: the REAL-01 diabetes split and models (ridge, ReLU net).
Note: sklearn's diabetes features are pre-scaled to equal SD on the FULL 442 records; sigma is computed on the 242-record
training split, where the SDs differ slightly, so `unit` and `sd` are close but not identical.

Outputs: results_v12/met02_features.csv.gz, met02_instances.csv.gz, met02_summary.csv, met02_paired.csv,
met02_real.csv, met02_table.tex, met02_real.tex, met02_checks.json, numbers_met02.csv, figures_v12/fig_met02.pdf.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

import phase1 as P  # noqa: E402
from bounds import all_masks, vertex_points  # noqa: E402
from envelope import max_ratio, weighted_l1  # noqa: E402
from lipid import pairwise as PW  # noqa: E402
from lipid.calibration import certify  # noqa: E402
from lipid.core import edge_tensors  # noqa: E402
from lipid.instances import envelopes_from_D  # noqa: E402
from lipid.provenance import base_manifest, write_json  # noqa: E402
from lipid.report import Numbers, tex_table  # noqa: E402

SCALES = ("mad", "sd", "unit")
FAMS = ("gauss", "t3", "lognormal", "binary")
TOL = PW.SIGN_TOL
INV_C = 3.7


# ================================================================================================ data
def marg_scales(d):
    return torch.exp(torch.linspace(math.log(0.25), math.log(4.0), d))


def sample_mixed(n, d, rho, gen, device):
    from scipy.special import ndtr
    from scipy.stats import t as tdist
    cov = (1.0 - rho) * torch.eye(d) + rho * torch.ones(d, d)
    Z = torch.randn(n, d, generator=gen) @ torch.linalg.cholesky(cov).T
    X = torch.empty_like(Z)
    for j in range(d):
        fam = FAMS[j % len(FAMS)]
        z = Z[:, j]
        if fam == "gauss":
            X[:, j] = z
        elif fam == "t3":
            u = np.clip(ndtr(z.numpy()), 1e-12, 1 - 1e-12)
            X[:, j] = torch.as_tensor(tdist.ppf(u, 3) / math.sqrt(3.0))
        elif fam == "lognormal":
            X[:, j] = torch.exp(z)
        else:
            X[:, j] = (z > 0.5).to(z.dtype)
    return (X * marg_scales(d)).to(device)


def scale_vectors(Xtr):
    med = Xtr.median(0).values
    mad = 1.4826 * (Xtr - med).abs().median(0).values
    sd = Xtr.std(0)
    s_mad = torch.where(mad > 0, mad, sd)
    out = dict(mad=s_mad, sd=sd, unit=torch.ones_like(sd))
    gm = torch.exp(torch.log(s_mad).mean())
    return {k: v * gm / torch.exp(torch.log(v).mean()) for k, v in out.items()}


def slopes_to(p, fp, Q, fQ, sigma):
    D = weighted_l1(p[None], Q, sigma)[0]
    return torch.where(D > 1e-12, (fp - fQ).abs() / D.clamp_min(1e-12), torch.zeros_like(D))


# ================================================================================================ core
def intervals(Z, A, fA, Lvec, sigma, active, delta, ed, pt):
    D = weighted_l1(Z, A, sigma)
    ell, h = envelopes_from_D(D, fA, Lvec)
    dl = delta[active]
    Ld = Lvec[:, None] * dl[None, :]
    lo, hi = PW.marginal_from_envelopes(ell, h, ed[0], ed[1], ed[2], Ld)
    c = Lvec[:, None] * (dl[pt["ia"]] + dl[pt["ib"]])[None, :]
    plo, phi_ = PW.pairwise_from_envelopes(ell, h, pt, c)
    return lo, hi, Ld, plo, phi_, c


def run_synthetic(args):
    from lipid.pipeline import Cols
    a = P.Config().a
    feat_cols, inst_cols, inv = Cols(), Cols(), []
    for seed in args.seeds:
        for d in args.dims:
            for rho in args.rhos:
                t0 = time.perf_counter()
                g = P.gen_of("met02", seed, d, rho)
                Xtr = sample_mixed(2000, d, rho, g, "cpu")
                Rc = sample_mixed(200, d, rho, g, args.device)
                Xev = sample_mixed(args.n_x, d, rho, g, args.device)
                plan = [(xi, torch.randperm(200, generator=g)[:args.refs].tolist()) for xi in range(args.n_x)]
                C = sample_mixed(max(args.nA), d, rho, P.gen_of("met02-C", seed, d, rho), args.device)
                sig = {k: v.to(args.device) for k, v in scale_vectors(Xtr).items()}
                for model in args.models:
                    f = P.make_model(model, a)
                    fC = f(C)
                    LCC = {(s, n): P_anchor_L(C[:n], fC[:n], sig[s]) for s in SCALES for n in args.nA}
                    LCCmax = {s: max(LCC[(s, n)] for n in args.nA) for s in SCALES}
                    for xi, ridx in plan:
                        x = Xev[xi]
                        fx = f(x[None])[0]
                        for rj in ridx:
                            r = Rc[rj]
                            fr = f(r[None])[0]
                            act = torch.nonzero((x - r) != 0).flatten().tolist()
                            m = len(act)
                            if m < 1:
                                continue
                            Z = vertex_points(x, r, act, all_masks(m, x.device))
                            fZ = f(Z)
                            ed = edge_tensors(m, Z.dtype, Z.device)
                            pt = PW.pair_tensors(m, Z.dtype, Z.device)
                            phi = (ed[2] * (fZ[ed[1]] - fZ[ed[0]])).sum(1).cpu().numpy()
                            tdiff = phi[pt["ia"].cpu().numpy()] - phi[pt["ib"].cpu().numpy()]
                            for s in SCALES:
                                sigma = sig[s]
                                delta = (x - r).abs() / sigma
                                LV = max(LCCmax[s], max_ratio(Z, fZ, C, fC, sigma), max_ratio(Z, fZ, Z, fZ, sigma))
                                Lt = P.lipschitz_true(model, d, sigma, a)
                                Lstar = max(LV, Lt) if Lt is not None else LV
                                Lc = certify(model, d, sigma, a).L_cert
                                for mode in ("domain", "fixed"):
                                    for n in args.nA:
                                        LA = max(LCC[(s, n)], float(slopes_to(x, fx, C[:n], fC[:n], sigma).max()))
                                        if mode == "fixed":
                                            LA = max(LA, float(slopes_to(r, fr, C[:n], fC[:n], sigma).max()),
                                                     float((fx - fr).abs() / weighted_l1(x[None], r[None], sigma)[0, 0]))
                                        A = torch.cat([C[:n], x[None]] + ([r[None]] if mode == "fixed" else []))
                                        fA = torch.cat([fC[:n], fx[None]] + ([fr[None]] if mode == "fixed" else []))
                                        levels = dict(opt=LA, oracle=Lstar, cert=Lc)
                                        Lvec = torch.tensor(list(levels.values()), device=Z.device)
                                        lo, hi, Ld, plo, phi_, c = intervals(Z, A, fA, Lvec, sigma, act, delta, ed, pt)
                                        if s == "mad":     # invariance: sigma -> c sigma, L -> c L
                                            lo2, hi2, _, plo2, phi2, _ = intervals(Z, A, fA, INV_C * Lvec, INV_C * sigma,
                                                                                    act, delta / INV_C, ed, pt)
                                            inv.append(float(max(((lo2 - lo).abs() / Ld).max(), ((hi2 - hi).abs() / Ld).max(),
                                                                 ((plo2 - plo).abs() / c).max() if c.numel() else 0.0,
                                                                 ((phi2 - phi_).abs() / c).max() if c.numel() else 0.0)))
                                        mlo, mhi = PW.marginal_pair_diff(lo, hi, pt)
                                        cp = PW.signed_cert(plo, phi_, c, TOL).cpu().numpy()
                                        cm = PW.signed_cert(mlo, mhi, c, TOL).cpu().numpy()
                                        sg = PW.signed_cert(lo, hi, Ld, TOL).cpu().numpy()
                                        lo_n, hi_n, Ld_n = lo.cpu().numpy(), hi.cpu().numpy(), Ld.cpu().numpy()
                                        plo_n, phi_n = plo.cpu().numpy(), phi_.cpu().numpy()
                                        c_n = c.cpu().numpy()
                                        fam = np.asarray([FAMS[j % len(FAMS)] for j in act])
                                        act_np = np.asarray(act, dtype=np.int16)
                                        for kk, (lev, L) in enumerate(levels.items()):
                                            rho_i = (hi_n[kk] - lo_n[kk]) / (2 * Ld_n[kk])
                                            cont = (phi >= lo_n[kk] - 1e-9 * Ld_n[kk]) & (phi <= hi_n[kk] + 1e-9 * Ld_n[kk])
                                            base = dict(seed=seed, d=d, rho_corr=rho, model=model, mode=mode, n_anchor=n,
                                                        scale=s, level=lev, L_used=float(L), L_V=float(LV),
                                                        L_cert=float(Lc), L_A=float(LA), x_id=xi, r_id=rj, m_active=m)
                                            feat_cols.add(m, **base, feature=act_np, family=fam, rho=rho_i,
                                                          sign=sg[kk].astype(np.int8), contained=cont.astype(np.int8))
                                            pc = ((tdiff >= plo_n[kk] - 1e-9 * c_n[kk]) & (tdiff <= phi_n[kk] + 1e-9 * c_n[kk]))
                                            inst_cols.add(1, **base, n_pairs=len(pt["pairs"]),
                                                          n_cert_pair=int((cp[kk] != 0).sum()),
                                                          n_cert_marg=int((cm[kk] != 0).sum()),
                                                          n_pair_not_contained=int((~pc).sum()))
                print(f"[MET-02] seed={seed} d={d} rho={rho} rows={len(feat_cols)} {time.perf_counter() - t0:.1f}s",
                      flush=True)
    return pd.DataFrame(feat_cols.finalize()), pd.DataFrame(inst_cols.finalize()), inv


def P_anchor_L(C, fC, sigma):
    return max_ratio(C, fC, C, fC, sigma)


def run_real(args):
    import real_audit as RA
    X, y, names, tr, lg, au, _, _ = RA.load()
    X, y = torch.tensor(X), torch.tensor(y)
    mu, sdv = y[tr].mean(), y[tr].std()
    ys = (y - mu) / sdv
    Xtr = X[tr]
    r = Xtr.median(0).values
    models = RA.train_models(Xtr, ys[tr])
    sig = scale_vectors(Xtr)
    rows = []
    for fam, f in models.items():
        Clog, fC = X[lg], f(X[lg])
        for s in SCALES:
            sigma = sig[s]
            Lc = f.L_cert(sigma)
            for ai, idx in enumerate(au.tolist()):
                x = X[idx]
                fx, fr = f(x[None])[0], f(r[None])[0]
                act = torch.nonzero((x - r) != 0).flatten().tolist()
                m = len(act)
                if m < 2:
                    continue
                Z = vertex_points(x, r, act, all_masks(m))
                fZ = f(Z)
                ed = edge_tensors(m)
                pt = PW.pair_tensors(m)
                phi = (ed[2] * (fZ[ed[1]] - fZ[ed[0]])).sum(1).numpy()
                delta = (x - r).abs() / sigma
                for mode in ("domain", "fixed"):
                    for n in (16, 150):
                        A = torch.cat([Clog[:n], x[None]] + ([r[None]] if mode == "fixed" else []))
                        fA = torch.cat([fC[:n], fx[None]] + ([fr[None]] if mode == "fixed" else []))
                        LA = max_ratio(A, fA, A, fA, sigma)
                        LV = max(LA, max_ratio(Z, fZ, A, fA, sigma), max_ratio(Z, fZ, Z, fZ, sigma))
                        levels = {k: v for k, v in dict(opt=LA, LV_eval=LV, cert=Lc).items() if v > 0}
                        Lvec = torch.tensor(list(levels.values()))
                        lo, hi, Ld, plo, phi_, c = intervals(Z, A, fA, Lvec, sigma, act, delta, ed, pt)
                        mlo, mhi = PW.marginal_pair_diff(lo, hi, pt)
                        cp = PW.signed_cert(plo, phi_, c, TOL).numpy()
                        cm = PW.signed_cert(mlo, mhi, c, TOL).numpy()
                        sg = PW.signed_cert(lo, hi, Ld, TOL).numpy()
                        for kk, (lev, L) in enumerate(levels.items()):
                            rho_i = ((hi[kk] - lo[kk]) / (2 * Ld[kk])).numpy()
                            cont = ((phi >= (lo[kk] - 1e-9 * Ld[kk]).numpy()) & (phi <= (hi[kk] + 1e-9 * Ld[kk]).numpy()))
                            rows.append(dict(family=fam, scale=s, mode=mode, n_log=n, audit_id=ai, level=lev, L_used=L,
                                             L_cert=Lc, L_V=LV, m_active=m, median_rho=float(np.median(rho_i)),
                                             p_sat=float(np.mean(rho_i > 0.8)), n_sign=int((sg[kk] != 0).sum()),
                                             containment=float(np.mean(cont)), n_pairs=len(pt["pairs"]),
                                             n_cert_pair=int((cp[kk] != 0).sum()), n_cert_marg=int((cm[kk] != 0).sum())))
        print(f"[MET-02 real] {fam} done", flush=True)
    return pd.DataFrame(rows)


# ================================================================================================ analysis
def analyze(F, I, Rr, inv, out, figs):
    keys = ["scale", "level", "d", "n_anchor"]
    S = F.groupby(keys).agg(n=("rho", "size"), median_rho=("rho", "median"), p_sat=("rho", lambda s: float(np.mean(s > 0.8))),
                            sign_rate=("sign", lambda s: float(np.mean(s != 0))), containment=("contained", "mean")
                            ).reset_index()
    Ii = I.groupby(keys).agg(n_pairs=("n_pairs", "sum"), n_cert_pair=("n_cert_pair", "sum"),
                             n_cert_marg=("n_cert_marg", "sum"), pair_not_contained=("n_pair_not_contained", "sum"),
                             inflation=("L_cert", lambda s: float(np.median(s / I.loc[s.index, "L_V"])))).reset_index()
    Ii["rank_pair"] = Ii.n_cert_pair / Ii.n_pairs
    Ii["rank_marg"] = Ii.n_cert_marg / Ii.n_pairs
    S = S.merge(Ii, on=keys, how="left")
    S.to_csv(os.path.join(out, "met02_summary.csv"), index=False)
    # paired across scale choices (same feature-instance), relative to mad
    k2 = ["seed", "d", "rho_corr", "model", "mode", "n_anchor", "level", "x_id", "r_id", "feature"]
    base = F[F.scale == "mad"][k2 + ["rho", "sign"]]
    pr = []
    for s in ("sd", "unit"):
        M = base.merge(F[F.scale == s][k2 + ["rho", "sign"]], on=k2, suffixes=("_mad", "_alt"))
        g = M.groupby(["level", "d", "n_anchor"]).apply(lambda q: pd.Series(dict(
            median_drho=float(np.median(q.rho_alt - q.rho_mad)), q90_abs_drho=float(np.quantile((q.rho_alt - q.rho_mad).abs(), .9)),
            sign_gained=float(np.mean((q.sign_mad == 0) & (q.sign_alt != 0))),
            sign_lost=float(np.mean((q.sign_mad != 0) & (q.sign_alt == 0)))))).reset_index()
        g["scale"] = s
        pr.append(g)
    PR = pd.concat(pr, ignore_index=True)
    PR.to_csv(os.path.join(out, "met02_paired.csv"), index=False)
    checks = dict(max_invariance_violation=float(max(inv)) if inv else None,
                  containment_cert_min=float(S[S.level == "cert"].containment.min()),
                  containment_oracle_min=float(S[S.level == "oracle"].containment.min()),
                  pair_not_contained_valid=int(S[S.level.isin(["cert", "oracle"])].pair_not_contained.sum()))
    rows = []
    for _, r in S[S.n_anchor == S.n_anchor.max()].sort_values(["level", "d", "scale"]).iterrows():
        rows.append([r.level, int(r.d), r.scale, f"{r.median_rho:.2f}", f"{r.p_sat:.2f}", f"{r.sign_rate:.3f}",
                     f"{r.rank_marg:.3f}", f"{r.rank_pair:.3f}", f"{r.inflation:.2f}", f"{r.containment:.3f}"])
    cap = ("MET-02: sensitivity to the feature scales $\\sigma$ (synthetic mixed-marginal design, $n=" f"{int(S.n_anchor.max())}"
           "$, generators, modes, $\\rho_c$ pooled). For every choice $L_{\\mathrm{cert}}$, $L^\\star$ and $L_A$ are "
           "recomputed in that metric. Inflation = median $L_{\\mathrm{cert}}/L_V$. Invariance check ($\\sigma\\to c\\sigma$, "
           f"$L\\to cL$): max normalized change {(checks['max_invariance_violation'] or 0.0):.1e}.")
    with open(os.path.join(out, "met02_table.tex"), "w") as fh:
        fh.write(tex_table(["level", "$d$", "$\\sigma$", "med.\\ $\\rho$", "$p_{\\mathrm{sat}}$", "sign", "rank (marg.)",
                            "rank (pair)", "infl.", "contain"], rows, cap, "tab:met02", align="lrlrrrrrrr",
                           size="\\scriptsize"))
    if Rr is not None and len(Rr):
        RS = Rr.groupby(["family", "scale", "mode", "n_log", "level"]).agg(
            median_rho=("median_rho", "median"), p_sat=("p_sat", "mean"), sign_rate=("n_sign", "sum"),
            m=("m_active", "sum"), rank_pair=("n_cert_pair", "sum"), rank_marg=("n_cert_marg", "sum"),
            n_pairs=("n_pairs", "sum"), containment=("containment", "mean"), L_used=("L_used", "median")).reset_index()
        RS["sign_rate"] = RS.sign_rate / RS.m
        RS["rank_pair"] = RS.rank_pair / RS.n_pairs
        RS["rank_marg"] = RS.rank_marg / RS.n_pairs
        RS.to_csv(os.path.join(out, "met02_real.csv"), index=False)
        rows = [[r.family, r.scale, r["mode"], int(r.n_log), r.level.replace("_", "\\_"), f"{r.median_rho:.2f}",
                 f"{r.p_sat:.2f}", f"{r.sign_rate:.3f}", f"{r.rank_pair:.3f}", f"{r.containment:.2f}"]
                for _, r in RS[RS.level.isin(["cert", "opt"])].sort_values(["family", "level", "mode", "n_log", "scale"]).iterrows()]
        with open(os.path.join(out, "met02_real.tex"), "w") as fh:
            fh.write(tex_table(["model", "$\\sigma$", "mode", "$n_{\\log}$", "level", "med.\\ $\\rho$", "$p_{\\mathrm{sat}}$",
                                "sign", "rank (pair)", "contain"], rows,
                               "MET-02 on the diabetes audit: the scale choice with re-certified $L$ (\\emph{unit} and "
                               "\\emph{sd} are close because the bundled features have equal SD on the full dataset, not on the training split).", "tab:met02real",
                               align="lllrlrrrrr", size="\\scriptsize"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, lev in zip(axes, ("cert", "oracle")):
        for s, col in zip(SCALES, ("C0", "C1", "C2")):
            v = S[(S.level == lev) & (S.n_anchor == S.n_anchor.max()) & (S.scale == s)].sort_values("d")
            ax.plot(v.d, v.p_sat, "-o", color=col, label=f"$p_{{sat}}$ ({s})")
            ax.plot(v.d, v.median_rho, "--", color=col, label=f"median $\\rho$ ({s})")
        ax.set_title(f"level {lev}")
        ax.set_xlabel("d")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, "fig_met02.pdf"))
    N = Numbers("MET-02", out)
    N.add("max_invariance", f"{(checks['max_invariance_violation'] or 0.0):.1e}",
          "max normalized change under sigma -> c sigma, L -> c L",
          "met02_checks.json")
    for _, r in PR[(PR.level == "cert") & (PR.n_anchor == PR.n_anchor.max())].iterrows():
        N.add(f"{r.scale}.d{int(r.d)}.median_drho", r.median_drho, f"median rho({r.scale}) - rho(mad), cert", "met02_paired.csv",
              nd=3)
    N.write()
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203, 204])
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.9])
    ap.add_argument("--nA", type=int, nargs="+", default=[256, 4096])
    ap.add_argument("--models", nargs="+", default=["linear", "fa", "stable_ood", "unstable_id"])
    ap.add_argument("--n-x", type=int, default=5)
    ap.add_argument("--refs", type=int, default=4)
    ap.add_argument("--no-real", dest="real", action="store_false")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v12"))
    ap.add_argument("--figs", default=os.path.join(ROOT, "figures_v12"))
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.dims, args.nA = args.seeds[:1], [3, 4], [256]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    pf, pi, pr_ = (os.path.join(args.out, f) for f in ("met02_features.csv.gz", "met02_instances.csv.gz",
                                                       "met02_real_rows.csv.gz"))
    if args.analyze_only:
        F, I = pd.read_csv(pf), pd.read_csv(pi)
        Rr = pd.read_csv(pr_) if os.path.exists(pr_) else None
        with open(os.path.join(args.out, "met02_invariance.json")) as fh:
            inv = json.load(fh)
    else:
        F, I, inv = run_synthetic(args)
        F.to_csv(pf, index=False, compression="gzip")
        I.to_csv(pi, index=False, compression="gzip")
        with open(os.path.join(args.out, "met02_invariance.json"), "w") as fh:
            json.dump(inv, fh)
        Rr = run_real(args) if args.real else None
        if Rr is not None:
            Rr.to_csv(pr_, index=False, compression="gzip")
    checks = analyze(F, I, Rr, inv, args.out, args.figs)
    with open(os.path.join(args.out, "met02_checks.json"), "w") as fh:
        json.dump(checks, fh, indent=1)
    write_json(os.path.join(args.out, "met02_manifest.json"),
               base_manifest(experiment="MET-02", args=vars(args), checks=checks, seconds=time.perf_counter() - t0))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
