#!/usr/bin/env python3
"""
benchmark_scaling.py — SCALE-01: exact 2^m envelope enumeration vs appendix-F stratified Monte Carlo.

    python scripts/benchmark_scaling.py --device cuda            # full benchmark
    python scripts/benchmark_scaling.py --device cpu --quick     # small

Design (fixed before running):
  * instance per m: d = m features, X ~ N(0, Sigma_.6), sigma = 1.4826 MAD on 2000 training draws, x and r from the
    data law, |A| = n_anchor (default 1024) background anchors from the data law plus x (domain mode).
  * generators: linear (L_cert = L_true exact) and fa (a = 4, global L_cert); L = L_cert (certified, fixed).
  * exact: m in {4, 8, 12, 16}; Monte Carlo: m in {8, 12, 16, 24}, n_k in {16, 64, 256} coalitions per stratum,
    methods 'paper' (eps with log(4m^2/alpha)) and 'paper+exact_singletons'; alpha = 0.05.
  * coverage: for m <= 16, R replications of the coalition draw (default 100) -> simultaneous containment of the exact
    endpoints of ALL features (both sides) in the MC bounds; one-sided binomial test of failure rate > alpha and the
    Clopper-Pearson upper limit. Negative control: radius_scale = 0 (point estimates) must fail far more often.
  * every configuration runs in a fresh subprocess, so peak RSS is per configuration (ru_maxrss of that process).
  * response-function calls are counted: |A| for the anchor labels; 0 during envelope / bound computation.
Outputs: results/scaling.csv, results/scaling.tex (Table 6), figures/fig_scaling.pdf, results/scaling_manifest.json.
"""
import argparse
import multiprocessing as mp
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

EXACT_M = (4, 8, 12, 16)
MC_M = (8, 12, 16, 24)
BUDGETS = (16, 64, 256)
METHODS = ("paper", "paper+exact_singletons")
ALPHA = 0.05


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0


def _instance(m, n_anchor, model, device):
    import torch
    import phase1 as P
    from lipid.calibration import L_cert
    torch.set_default_dtype(torch.float64)
    g = P.gen_of("scale01", m, n_anchor)
    smp = P.gaussian_sampler(m, 0.6, device)
    sigma = P.robust_sigma(smp(2000, g))
    x, r = smp(1, g)[0], smp(1, g)[0]
    C = smp(n_anchor, g)
    calls = {"n": 0}
    f0 = P.make_model(model, P.Config().a)

    def f(Z):
        calls["n"] += Z.shape[0]
        return f0(Z)
    A = torch.cat([C, x[None]])
    fA = f(A)
    L = L_cert(model, m, sigma, P.Config().a)
    return x, r, A, fA, L, sigma, calls


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def worker(task, q):
    try:
        import torch
        import lipid  # noqa: F401
        from lipid.core import marginal_intervals
        from lipid.mc import stratified_bounds
        from scipy.stats import beta, binomtest
        torch.set_default_dtype(torch.float64)
        dev = task["device"]
        if dev.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        x, r, A, fA, L, sigma, calls = _instance(task["m"], task["n_anchor"], task["model"], dev)
        rss0 = _rss_mb()
        calls_before = calls["n"]
        out = dict(task)
        Ld = (L * (x - r).abs() / sigma)
        cache = os.path.join(task["cache"], f"exact_{task['model']}_m{task['m']}_A{task['n_anchor']}.npz")
        if task["kind"] == "exact":
            times = []
            for _ in range(task["timing_reps"]):
                _sync(dev)
                t0 = time.perf_counter()
                ex = marginal_intervals(x, r, A, fA, L, sigma)
                _sync(dev)
                times.append(time.perf_counter() - t0)
            np.savez(cache, lo=ex["lo"].cpu().numpy(), hi=ex["hi"].cpu().numpy())
            out.update(time_median_s=float(np.median(times)), time_max_s=float(np.max(times)),
                       n_hybrid_points=2 ** task["m"], n_anchor_distances=2 ** task["m"] * A.shape[0],
                       mean_norm_width=float((ex["Gamma"] / (2 * Ld)).mean()), coalition_budget_per_feature=2 ** (task["m"] - 1))
        else:
            kw = dict(n_per_stratum=task["n_k"], alpha=ALPHA, exact_singletons=(task["method"] == "paper+exact_singletons"),
                      radius_scale=task.get("radius_scale", 1.0))
            exact = np.load(cache) if os.path.exists(cache) else None
            times, fails, widths, excess = [], 0, [], []
            for rep in range(task["reps"]):
                gen = torch.Generator().manual_seed(10_000 * task["m"] + 97 * task["n_k"] + rep)
                _sync(dev)
                t0 = time.perf_counter()
                mc = stratified_bounds(x, r, A, fA, L, sigma, gen=gen, **kw)
                _sync(dev)
                times.append(time.perf_counter() - t0)
                lo, hi = mc["lo"].cpu().numpy(), mc["hi"].cpu().numpy()
                widths.append(float(((hi - lo) / (2 * Ld.cpu().numpy())).mean()))
                if exact is not None:
                    ok = bool((lo <= exact["lo"] + 1e-12).all() and (hi >= exact["hi"] - 1e-12).all())
                    fails += not ok
                    excess.append(float((((hi - lo) - (exact["hi"] - exact["lo"])) / (2 * Ld.cpu().numpy())).mean()))
            out.update(time_median_s=float(np.median(times)), time_max_s=float(np.max(times)),
                       n_hybrid_points=mc["n_hybrid_points"], n_anchor_distances=mc["n_anchor_distances"],
                       coalition_budget_per_feature=mc["coalition_budget_per_feature"],
                       mean_norm_width=float(np.mean(widths)))
            if exact is not None:
                R = task["reps"]
                out.update(coverage_reps=R, coverage_failures=fails, failure_rate=fails / R,
                           binom_p_failrate_gt_alpha=float(binomtest(fails, R, ALPHA, alternative="greater").pvalue),
                           cp_upper95=float(beta.ppf(0.95, fails + 1, R - fails)) if fails < R else 1.0,
                           mean_excess_norm_width=float(np.mean(excess)))
        out.update(peak_rss_mb=_rss_mb(), rss_after_setup_mb=rss0,
                   cuda_peak_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if dev.startswith("cuda") else np.nan),
                   oracle_calls_anchor_labels=calls_before, oracle_calls_during_bounds=calls["n"] - calls_before,
                   n_anchor_total=int(A.shape[0]), L=float(L))
        q.put(out)
    except Exception as e:  # pragma: no cover
        import traceback
        q.put(dict(task, error=repr(e), traceback=traceback.format_exc()))


def run_task(task):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(task, q))
    p.start()
    res = q.get()
    p.join()
    if "error" in res:
        print(res["traceback"])
        raise RuntimeError(res["error"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-anchor", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=100, help="coverage replications for m <= 16")
    ap.add_argument("--timing-reps", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["linear", "fa"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--figs", default=os.path.join(ROOT, "figures"))
    args = ap.parse_args()
    import pandas as pd
    from lipid.provenance import base_manifest, write_json
    exact_m, mc_m, budgets = (EXACT_M, MC_M, BUDGETS) if not args.quick else ((4, 8), (8, 12), (16,))
    reps = args.reps if not args.quick else 10
    cache = os.path.join(args.out, "scaling_cache")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)
    tasks = []
    for model in args.models:
        for m in exact_m:
            tasks.append(dict(kind="exact", method="exact", model=model, m=m, n_k=0, n_anchor=args.n_anchor,
                              device=args.device, cache=cache, timing_reps=args.timing_reps, reps=0))
        for m in mc_m:
            for nk in budgets:
                for meth in METHODS:
                    tasks.append(dict(kind="mc", method=meth, model=model, m=m, n_k=nk, n_anchor=args.n_anchor,
                                      device=args.device, cache=cache, reps=(reps if m in exact_m else args.timing_reps)))
            if m in exact_m:     # negative control: zero radius must be detected
                tasks.append(dict(kind="mc", method="negative_control_radius0", model=model, m=m, n_k=min(budgets),
                                  n_anchor=args.n_anchor, device=args.device, cache=cache, reps=reps,
                                  radius_scale=0.0))
    rows = []
    for t in tasks:
        if t["method"] == "negative_control_radius0":
            t = dict(t, method="paper")
            t["radius_scale"] = 0.0
            t["label"] = "negative_control_radius0"
        res = run_task(t)
        res.setdefault("label", res["method"])
        rows.append(res)
        print(f"[SCALE-01] {res['label']:>24} {res['model']:>7} m={res['m']:>2} n_k={res['n_k']:>3} "
              f"t={res['time_median_s']:.4f}s rss={res['peak_rss_mb']:.0f}MB "
              f"fail={res.get('coverage_failures', '-')}/{res.get('coverage_reps', '-')}", flush=True)
    df = pd.DataFrame(rows).drop(columns=["cache"])
    df.to_csv(os.path.join(args.out, "scaling.csv"), index=False)
    bad = df[(df.oracle_calls_during_bounds != 0)]
    if len(bad):
        print("WARNING: response-function calls during bound computation:", bad[["label", "m"]].to_dict("records"))
    # Table 6 (linear generator, n_k = 64, method 'paper')
    lin = df[df.model == df.model.iloc[0]]
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{Computational scaling with feature count $m$ (generator {lin.model.iloc[0]}, $L=L_{{\\mathrm{{cert}}}}$, "
             f"device {args.device}). Exact = $2^m$ envelope enumeration; MC = appendix-F stratified Hoeffding bounds "
             f"($\\alpha={ALPHA}$, $n_k=64$ per stratum). Excess width = mean (MC width $-$ exact width)$/(2L\\delta_i)$. "
             "Peak RSS is the whole process (includes the PyTorch runtime). No response-function call occurs during "
             "envelope or bound computation.}", "\\label{tab:scaling}", "\\small",
             "\\begin{tabular}{rrlll}", "\\toprule",
             "$m$ & $|A|$ & Exact time / peak RSS & MC budget / time & MC excess width \\\\", "\\midrule"]
    for m in sorted(set(exact_m) | set(mc_m)):
        e = lin[(lin.label == "exact") & (lin.m == m)]
        c = lin[(lin.label == "paper") & (lin.m == m) & (lin.n_k == 64)]
        ex = (f"{e.time_median_s.iloc[0]:.3g}\\,s / {e.peak_rss_mb.iloc[0]:.0f}\\,MB" if len(e) else "not attempted")
        mc = (f"{int(c.coalition_budget_per_feature.iloc[0])} coal./feat. / {c.time_median_s.iloc[0]:.3g}\\,s"
              if len(c) else "--")
        xs = (f"{c.mean_excess_norm_width.iloc[0]:.3f}" if len(c) and "mean_excess_norm_width" in c
              and np.isfinite(c.mean_excess_norm_width.iloc[0]) else ("n/a (no exact)" if len(c) else "--"))
        lines.append(f"{m} & {args.n_anchor + 1} & {ex} & {mc} & {xs} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(args.out, "scaling.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for model, ls in zip(args.models, ("-", "--")):
        d = df[df.model == model]
        e = d[d.label == "exact"].sort_values("m")
        axes[0].plot(e.m, e.time_median_s, "k" + ls, marker="o", label=f"exact ({model})")
        for nk, col in zip(budgets, ("C0", "C1", "C2")):
            c = d[(d.label == "paper") & (d.n_k == nk)].sort_values("m")
            axes[0].plot(c.m, c.time_median_s, ls, color=col, marker=".", label=f"MC n_k={nk} ({model})")
            if "mean_excess_norm_width" in c:
                axes[1].plot(c.m, c.mean_excess_norm_width, ls, color=col, marker=".", label=f"n_k={nk} ({model})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("m")
    axes[0].set_ylabel("median time per instance [s]")
    axes[1].set_xlabel("m")
    axes[1].set_ylabel("MC excess normalized width")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figs, "fig_scaling.pdf"))
    write_json(os.path.join(args.out, "scaling_manifest.json"), base_manifest(n_anchor=args.n_anchor, reps=reps,
                                                                             device=args.device, quick=args.quick))
    print(df[["label", "model", "m", "n_k", "time_median_s", "peak_rss_mb", "coverage_failures"]
             if "coverage_failures" in df else ["label", "model", "m", "n_k", "time_median_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
