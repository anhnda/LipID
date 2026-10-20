#!/usr/bin/env python3
"""
scl02_nonsaturated.py — SCL-02: Monte Carlo informativeness on NON-saturated instances (reviewer Q5).

    python scripts/scl02_nonsaturated.py pilot   --device cuda      # once; writes results_v12/scl02_pilot.json
    python scripts/scl02_nonsaturated.py eval    --device cuda      # independent instances; uses the pilot choice
    python scripts/scl02_nonsaturated.py analyze                    # tables + figure from results_v12/scl02_rows.csv.gz

Why: in Table 6 (SCALE-01) the exact intervals are already saturated for m >= 12, so any valid bound "looks accurate"
and zero coverage failures say nothing about precision. SCL-02 separates approximation quality from saturation.

Designs (fixed rules; protocols/v12_preregistration.json):
  passive        1024 data anchors from N(0, Sigma_.6) + x (domain mode) — the Table-6 design, reference negative result
  hybrid_local   B anchors at hybrid vertices z_S, |S| ~ U{0..m}, S | |S| uniform, jittered by tau * sigma * N(0, I),
                 + x. Non-adaptive and f-independent: it never uses responses outside the queried anchors nor any
                 interval value. It is a benchmark instance generator, NOT a proposed query strategy (that is P2).
Pilot (separate RNG key, linear generator, 5 instances, m in {8, 12}): choose (tau, B) by the pre-specified rule
  "minimise |median exact rho at m=12 - 0.5| subject to median rho in [0.2, 0.8]; ties: smaller B, then smaller tau".
Eval (independent RNG key): m in {8, 12, 16, 20, 24} (exact up to --exact-max-m, default 20: 2^24 x 24 vertices do not
  fit the enumeration code), 10 instances x {linear, fa} at L = L_cert, methods of lipid/mc_v2.py at n_k in {16, 64, 256},
  20 coalition-draw replications per instance, alpha = .05, and the zero-radius negative control.
Metrics: excess normalized width (MC - exact)/(2 L delta_i), exact sign / marginal-ranking certificates retained by MC,
  time, CUDA peak memory per configuration, coalitions per feature, simultaneous containment failures (Clopper-Pearson).
"""
import argparse
import itertools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

import phase1 as P  # noqa: E402
from lipid.calibration import L_cert  # noqa: E402
from lipid.core import marginal_intervals  # noqa: E402
from lipid.mc_v2 import METHODS, stratified_bounds_v2  # noqa: E402
from lipid.provenance import base_manifest, write_json  # noqa: E402
from lipid.report import Numbers, tex_table  # noqa: E402

PROTO = os.path.join(ROOT, "protocols", "v12_preregistration.json")
TOL = 1e-6
ALPHA = 0.05


def proto():
    with open(PROTO) as fh:
        return json.load(fh)["experiments"]["SCL-02"]


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def make_instance(phase, m, inst, model, device):
    g = P.gen_of("scl02", phase, m, inst)
    smp = P.gaussian_sampler(m, 0.6, device)
    sigma = P.robust_sigma(smp(2000, g))
    x, r = smp(1, g)[0], smp(1, g)[0]
    Cpas = smp(1024, g)
    a = P.Config().a
    return dict(phase=phase, m=m, inst=inst, model=model, x=x, r=r, sigma=sigma, Cpas=Cpas,
                f=P.make_model(model, a), L=L_cert(model, m, sigma, a))


def make_anchors(I, design, B=None, tau=None):
    x, r, sigma, m = I["x"], I["r"], I["sigma"], I["m"]
    if design == "passive":
        A = torch.cat([I["Cpas"], x[None]])
    else:
        g = P.gen_of("scl02-anchors", I["phase"], m, I["inst"], int(B), float(tau))
        sizes = torch.randint(0, m + 1, (int(B),), generator=g)
        ranks = torch.rand(int(B), m, generator=g).argsort(1).argsort(1)
        masks = (ranks < sizes[:, None]).to(x.device)
        jit = torch.randn(int(B), m, generator=g).to(x.device)
        Z = torch.where(masks, x, r) + float(tau) * sigma * jit
        A = torch.cat([Z, x[None]])
    return A, I["f"](A)


def exact(I, A, fA):
    sync(A.device)
    t0 = time.perf_counter()
    ex = marginal_intervals(I["x"], I["r"], A, fA, I["L"], I["sigma"])
    sync(A.device)
    ex["seconds"] = time.perf_counter() - t0
    return ex


def certs(lo, hi, Ld):
    sg = np.where(lo > TOL * Ld, 1, np.where(hi < -TOL * Ld, -1, 0))
    m = len(lo)
    rk = np.zeros((m, m), dtype=bool)
    for i, j in itertools.permutations(range(m), 2):
        rk[i, j] = lo[i] - hi[j] > TOL * (Ld[i] + Ld[j])
    return sg, rk


# ================================================================================================ pilot
def pilot(args):
    pr = proto()["pilot"]
    path = os.path.join(args.out, "scl02_pilot.json")
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"{path} exists: the pilot is run once (use --force only to document a repeated pilot)")
    rows = []
    for tau, B in itertools.product(pr["candidates"]["tau"], pr["candidates"]["B"]):
        for m in pr["m"]:
            for inst in range(pr["instances"]):
                I = make_instance("pilot", m, inst, pr["model"], args.device)
                A, fA = make_anchors(I, "hybrid_local", B, tau)
                ex = exact(I, A, fA)
                rows.append(dict(tau=tau, B=B, m=m, inst=inst, mean_rho=float(ex["rho"].mean())))
                print(f"[SCL-02 pilot] tau={tau} B={B} m={m} inst={inst} mean rho={rows[-1]['mean_rho']:.3f}", flush=True)
    for inst in range(pr["instances"]):          # passive reference at the same m (not a candidate)
        for m in pr["m"]:
            I = make_instance("pilot", m, inst, pr["model"], args.device)
            A, fA = make_anchors(I, "passive")
            rows.append(dict(tau=None, B=1024, m=m, inst=inst, mean_rho=float(exact(I, A, fA)["rho"].mean()),
                             passive=True))
    df = pd.DataFrame(rows)
    cand = df[df.tau.notna()].groupby(["tau", "B", "m"]).mean_rho.median().reset_index()
    at12 = cand[cand.m == 12].copy()
    at12["score"] = (at12.mean_rho - 0.5).abs()
    ok = at12[(at12.mean_rho >= 0.2) & (at12.mean_rho <= 0.8)].sort_values(["score", "B", "tau"])
    chosen = None if ok.empty else dict(tau=float(ok.iloc[0].tau), B=int(ok.iloc[0].B),
                                        median_rho_m12=float(ok.iloc[0].mean_rho))
    out = dict(rule=pr["rule"], candidates=cand.to_dict("records"),
               passive_median_rho=df[df.tau.isna()].groupby("m").mean_rho.median().to_dict(),
               chosen=chosen, note=("no candidate met the rule: eval runs all candidates as a negative result"
                                    if chosen is None else "chosen by the pre-specified rule"))
    write_json(path, out)
    print(json.dumps(out, indent=1, default=str))


# ================================================================================================ eval
def evaluate(args):
    pr = proto()
    ev = pr["eval"]
    with open(os.path.join(args.out, "scl02_pilot.json")) as fh:
        pil = json.load(fh)
    designs = [("passive", None, None)]
    if pil["chosen"] is not None and not args.all_designs:
        designs.append(("hybrid_local", pil["chosen"]["B"], pil["chosen"]["tau"]))
    else:
        designs += [("hybrid_local", B, tau) for tau, B in itertools.product(pr["pilot"]["candidates"]["tau"],
                                                                             pr["pilot"]["candidates"]["B"])]
    ms = args.m or ev["m"]
    rows = []
    t_start = time.perf_counter()
    methods = list(METHODS) if not args.methods else args.methods
    for model in ev["models"]:
        for m in ms:
            for inst in range(args.instances or ev["instances"]):
                I = make_instance("eval", m, inst, model, args.device)
                Ld = (I["L"] * (I["x"] - I["r"]).abs() / I["sigma"]).cpu().numpy()
                for design, B, tau in designs:
                    dlab = design if design == "passive" else f"hybrid_local(B={B},tau={tau})"
                    A, fA = make_anchors(I, design, B, tau)
                    base = dict(model=model, m=m, inst=inst, design=dlab, n_anchor=int(A.shape[0]), L=I["L"])
                    ex = None
                    if m <= args.exact_max_m:
                        if str(args.device).startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats()
                        ex = exact(I, A, fA)
                        elo, ehi = ex["lo"].cpu().numpy(), ex["hi"].cpu().numpy()
                        esg, erk = certs(elo, ehi, Ld)
                        rows.append(dict(base, method="exact", n_k=0, rep=-1, seconds=ex["seconds"],
                                         mean_rho=float(ex["rho"].mean()), coalitions_per_feature=2 ** (m - 1),
                                         n_sign_exact=int((esg != 0).sum()), n_rank_exact=int(erk.sum()),
                                         cuda_peak_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                                                       if str(args.device).startswith("cuda") else np.nan)))
                    for method, nk in itertools.product(methods + ["negative_control"], ev["n_k"]):
                        if method == "negative_control" and nk != min(ev["n_k"]):
                            continue
                        meth = "hoeffding_tight" if method == "negative_control" else method
                        rs = 0.0 if method == "negative_control" else 1.0
                        if str(args.device).startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats()
                        R = args.reps or ev["reps_per_instance"]
                        if ex is None:
                            R = min(R, 2)      # no exact reference: timing / width only
                        for rep in range(R):
                            gen = torch.Generator().manual_seed(P.seed_of("scl02-mc", dlab, model, m, inst, method, nk, rep))
                            sync(args.device)
                            mc = stratified_bounds_v2(I["x"], I["r"], A, fA, I["L"], I["sigma"], n_per_stratum=nk,
                                                      alpha=ALPHA, gen=gen, method=meth, radius_scale=rs)
                            sync(args.device)
                            lo, hi = mc["lo"].cpu().numpy(), mc["hi"].cpu().numpy()
                            row = dict(base, method=method, n_k=nk, rep=rep, seconds=mc["seconds"],
                                       mc_mean_width=float(((hi - lo) / (2 * Ld)).mean()),
                                       coalitions_per_feature=mc["coalitions_per_feature"], n_events=mc["n_events"],
                                       n_exhaustive_strata=len(mc["exhaustive_strata"]))
                            if ex is not None:
                                sg, rk = certs(lo, hi, Ld)
                                row.update(mean_rho=float(ex["rho"].mean()),
                                           excess_width=float((((hi - lo) - (ehi - elo)) / (2 * Ld)).mean()),
                                           contained=int(bool((lo <= elo + 1e-12 * Ld).all() and (hi >= ehi - 1e-12 * Ld).all())),
                                           n_sign_exact=int((esg != 0).sum()),
                                           n_sign_retained=int(((esg != 0) & (sg == esg)).sum()),
                                           n_rank_exact=int(erk.sum()), n_rank_retained=int((erk & rk).sum()))
                            rows.append(row)
                        rows[-1]["cuda_peak_mb_config"] = (torch.cuda.max_memory_allocated() / 2 ** 20
                                                           if str(args.device).startswith("cuda") else np.nan)
                    print(f"[SCL-02] {model} m={m} inst={inst} {dlab}: exact rho="
                          f"{(float(ex['rho'].mean()) if ex is not None else float('nan')):.3f}  "
                          f"{time.perf_counter() - t_start:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(args.out, "scl02_rows.csv.gz"), index=False, compression="gzip")
    write_json(os.path.join(args.out, "scl02_manifest.json"),
               base_manifest(experiment="SCL-02", args=vars(args), pilot=pil, designs=designs,
                             seconds=time.perf_counter() - t_start))


# ================================================================================================ analyze
def analyze(args):
    from scipy.stats import beta
    df = pd.read_csv(os.path.join(args.out, "scl02_rows.csv.gz"))
    mc = df[df.method != "exact"].copy()
    keys = ["design", "model", "m", "method", "n_k"]
    g = mc.groupby(keys)
    S = g.agg(reps=("rep", "size"), mean_rho_exact=("mean_rho", "mean"), mc_width=("mc_mean_width", "mean"),
              excess=("excess_width", "mean"), excess_q90=("excess_width", lambda s: float(s.quantile(0.9))),
              failures=("contained", lambda s: int((s == 0).sum())), n_cov=("contained", lambda s: int(s.notna().sum())),
              sign_exact=("n_sign_exact", "sum"), sign_ret=("n_sign_retained", "sum"),
              rank_exact=("n_rank_exact", "sum"), rank_ret=("n_rank_retained", "sum"),
              seconds=("seconds", "median"), coalitions=("coalitions_per_feature", "mean"),
              cuda_peak_mb=("cuda_peak_mb_config", "max")).reset_index()
    S["cp_upper95"] = [float(beta.ppf(0.95, f + 1, n - f)) if 0 < n and f < n else (1.0 if n else np.nan)
                       for f, n in zip(S.failures, S.n_cov)]
    S["sign_retained_frac"] = S.sign_ret / S.sign_exact.replace(0, np.nan)
    S["rank_retained_frac"] = S.rank_ret / S.rank_exact.replace(0, np.nan)
    E = df[df.method == "exact"].groupby(["design", "model", "m"]).agg(
        exact_seconds=("seconds", "median"), exact_cuda_mb=("cuda_peak_mb", "max"), exact_rho=("mean_rho", "mean"),
        exact_coalitions=("coalitions_per_feature", "first")).reset_index()
    S = S.merge(E, on=["design", "model", "m"], how="left")
    S.to_csv(os.path.join(args.out, "scl02_summary.csv"), index=False)
    lin = S[(S.model == "linear") & (S.n_k == 64) & (S.method != "negative_control")]
    rows = []
    for (design, m), gg in lin.groupby(["design", "m"]):
        for _, r in gg.iterrows():
            rows.append([design.replace("_", "\\_"), int(m), f"{r.exact_rho:.2f}" if np.isfinite(r.exact_rho) else "--",
                         r.method.replace("_", "\\_"), f"{r.coalitions:.0f}",
                         f"{r.excess:.3f}" if np.isfinite(r.excess) else "n/a",
                         f"{r.sign_retained_frac:.2f}" if np.isfinite(r.sign_retained_frac) else "--",
                         f"{r.rank_retained_frac:.2f}" if np.isfinite(r.rank_retained_frac) else "--",
                         f"{int(r.failures)}/{int(r.n_cov)}" if r.n_cov else "--",
                         f"{r.seconds:.3g}"])
    neg = S[S.method == "negative_control"]
    negtxt = "; ".join(f"{r.design} m={int(r.m)} {r.model}: {int(r.failures)}/{int(r.n_cov)}" for _, r in neg.iterrows()
                       if r.n_cov)
    cap = ("SCL-02: valid Monte Carlo bounds on non-saturated instances (linear generator, $L=L_{\\mathrm{cert}}$, "
           "$n_k=64$, $\\alpha=.05$). Exact $\\bar\\rho$ shows how far the instance is from saturation; excess = mean "
           "(MC $-$ exact width)$/(2L\\delta_i)$; retained = fraction of exact sign / marginal-ranking certificates "
           "still certified by MC; failures = simultaneous containment failures over coalition draws. "
           f"Zero-radius negative control failures: {negtxt or 'n/a'}.")
    with open(os.path.join(args.out, "scl02_table.tex"), "w") as fh:
        fh.write(tex_table(["design", "$m$", "exact $\\bar\\rho$", "method", "coal./feat.", "excess", "sign ret.",
                            "rank ret.", "fail.", "time [s]"], rows, cap, "tab:scl02", align="lrrlrrrrrr",
                           size="\\scriptsize"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    designs = sorted(S.design.unique())
    fig, axes = plt.subplots(1, len(designs), figsize=(4.6 * len(designs), 3.3), sharey=True, squeeze=False)
    for ax, des in zip(axes[0], designs):
        for (meth, m), gg in S[(S.design == des) & (S.model == "linear") & (S.method != "negative_control")].groupby(
                ["method", "m"]):
            gg = gg.sort_values("coalitions")
            ax.plot(gg.coalitions, gg.excess, marker="o", label=f"{meth}, m={m}")
        ax.set_xscale("log")
        ax.set_xlabel("coalitions per feature")
        ax.set_title(des, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("excess normalized width")
    axes[0][-1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    os.makedirs(args.figs, exist_ok=True)
    fig.savefig(os.path.join(args.figs, "fig_scl02_excess.pdf"))
    N = Numbers("SCL-02", args.out)
    for _, r in lin.iterrows():
        if np.isfinite(r.excess):
            N.add(f"{r.design.split('(')[0]}.m{int(r.m)}.{r.method}.excess", r.excess, "mean excess normalized width, "
                  "n_k = 64, linear", "scl02_summary.csv", nd=3)
    N.write()
    print(S.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "eval", "analyze"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--m", type=int, nargs="+", default=None)
    ap.add_argument("--exact-max-m", type=int, default=20)
    ap.add_argument("--instances", type=int, default=0)
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=None, choices=list(METHODS))
    ap.add_argument("--all-designs", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v12"))
    ap.add_argument("--figs", default=os.path.join(ROOT, "figures_v12"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"pilot": pilot, "eval": evaluate, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
