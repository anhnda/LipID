#!/usr/bin/env python3
"""
real_audit.py — REAL-01: one non-synthetic limited-query audit (public tabular data, fixed model, fixed baseline).

    python scripts/real_audit.py               # CPU, deterministic

PRE-SPECIFIED PROTOCOL (fixed in this file before any interval is computed; written to
results/real_audit_protocol.md at the start of the run):
  data      sklearn.datasets.load_diabetes (bundled with scikit-learn, no download; 442 patients x 10 standardized
            baseline features; target = disease progression after one year). Public, documented (Efron et al. 2004).
  split     numpy default_rng(20261020) permutation -> train 242 | logged pool 150 | audit 50.
  models    (i) MLP 10-32-1 ReLU, float64, Adam lr 1e-2, weight decay 1e-4, 3000 full-batch epochs, torch seed 20261020;
            (ii) ridge regression (lambda = 1e-2, closed form). Both predict the target standardized by the TRAIN mean/SD.
  scales    sigma_j = 1.4826 MAD_j on train; if MAD_j = 0 (binary 'sex'), sigma_j = SD_j on train (pre-specified fallback).
  baseline  r = coordinatewise median of the train features (one fixed reference for every explained point).
  queries   the auditor sees model outputs only at: the first n_log points of the logged pool (fixed random order),
            n_log in {16, 32, 64, 150} (nested), plus the explained point x (domain) or x and r (fixed).
            Query budget = n_log + 1 (domain) or n_log + 2 (fixed). Every audit point is explained (50 x's).
  levels    opt  = L_A(A)  (optimistic; no guarantee)
            cert = global certificate independent of any query: MLP: max_j sigma_j sum_k |w2_k| |W1_kj| (valid for
                   every ReLU net with one hidden layer: a.e. gradient = sum_k w2_k s_k W1_kj, s_k in {0,1});
                   ridge: max_j sigma_j |beta_j| (exact).
  evaluation only (we own the model): exact true phi from the 2^m hybrids, and the hybrid oracle L_V.
  metrics   Gamma, rho, containment of true phi, strict sign certificates, top-1 rank certificate (Proposition 2,
            |phi| ordering) and the share of certified ordered feature pairs.
Estimand: the baseline (reference-conditioned) Shapley game with this r. No comparison with other Shapley estimands.
Outputs: results/real_audit.csv, results/real_audit_summary.csv, results/real_audit.tex, figures/fig_real_audit.pdf,
         results/real_audit_protocol.md, results/real_audit_manifest.json.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import lipid  # noqa: E402,F401
from lipid.core import marginal_intervals, shapley_on_vertices, sign_from_norm  # noqa: E402
from lipid.provenance import base_manifest, write_json  # noqa: E402
from envelope import anchor_lipschitz, max_ratio  # noqa: E402

torch.set_default_dtype(torch.float64)
SEED = 20261020
N_TRAIN, N_LOG, N_AUDIT = 242, 150, 50
N_LOGS = (16, 32, 64, 150)
TOL = 1e-6
SAT = 0.8
OUT = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")


def protocol_text():
    return __doc__.split("PRE-SPECIFIED PROTOCOL")[1].split("Outputs:")[0]


def load():
    from sklearn.datasets import load_diabetes
    import sklearn
    X, y = load_diabetes(return_X_y=True)
    names = list(load_diabetes().feature_names)
    perm = np.random.default_rng(SEED).permutation(len(y))
    tr, lg, au = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_LOG], perm[N_TRAIN + N_LOG:N_TRAIN + N_LOG + N_AUDIT]
    h = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
    return X, y, names, tr, lg, au, h, sklearn.__version__


class MLP:
    def __init__(self, W1, b1, w2, b2):
        self.W1, self.b1, self.w2, self.b2 = W1, b1, w2, b2

    def __call__(self, Z):
        return torch.relu(Z @ self.W1.T + self.b1) @ self.w2 + self.b2

    def L_cert(self, sigma):
        return float((sigma * (self.w2.abs()[:, None] * self.W1.abs()).sum(0)).max())


class Linear:
    def __init__(self, beta, b0):
        self.beta, self.b0 = beta, b0

    def __call__(self, Z):
        return Z @ self.beta + self.b0

    def L_cert(self, sigma):
        return float((sigma * self.beta.abs()).max())


def train_models(Xtr, ytr):
    torch.manual_seed(SEED)
    d = Xtr.shape[1]
    W1 = torch.nn.Parameter(torch.randn(32, d) * (2.0 / d) ** 0.5)
    b1 = torch.nn.Parameter(torch.zeros(32))
    w2 = torch.nn.Parameter(torch.randn(32) * (1.0 / 32) ** 0.5)
    b2 = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([W1, b1, w2, b2], lr=1e-2, weight_decay=1e-4)
    for _ in range(3000):
        opt.zero_grad()
        loss = ((torch.relu(Xtr @ W1.T + b1) @ w2 + b2 - ytr) ** 2).mean()
        loss.backward()
        opt.step()
    mlp = MLP(W1.detach(), b1.detach(), w2.detach(), b2.detach())
    Xc = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1)
    lam = 1e-2 * torch.eye(d + 1)
    lam[-1, -1] = 0.0
    coef = torch.linalg.solve(Xc.T @ Xc + lam * len(Xtr), Xc.T @ ytr)
    lin = Linear(coef[:-1], coef[-1])
    return {"mlp": mlp, "ridge": lin}


def model_hash(m):
    t = [m.W1, m.b1, m.w2, m.b2.reshape(1)] if isinstance(m, MLP) else [m.beta, m.b0.reshape(1)]
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in t)).hexdigest()


def rank_certificates(lo, hi):
    """Proposition 2: |phi_i| > |phi_j| for every feasible g if min_{t in I_i}|t| > max(|l_j|, |u_j|)."""
    absmin = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    absmax = np.maximum(np.abs(lo), np.abs(hi))
    m = len(lo)
    pairs = sum(1 for i in range(m) for j in range(m) if i != j and absmin[i] > absmax[j] + TOL)
    top1 = any(all(absmin[i] > absmax[j] + TOL for j in range(m) if j != i) for i in range(m)) if m > 1 else True
    return top1, pairs / max(1, m * (m - 1) / 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(OUT, "real_audit_protocol.md"), "w") as fh:
        fh.write("# REAL-01 protocol (pre-specified; copied from scripts/real_audit.py before computing)\n\n```"
                 + protocol_text() + "```\n")
    X, y, names, tr, lg, au, data_hash, skv = load()
    X = torch.tensor(X)
    y = torch.tensor(y)
    mu, sd = y[tr].mean(), y[tr].std()
    ys = (y - mu) / sd
    Xtr = X[tr]
    med = Xtr.median(0).values
    mad = (Xtr - med).abs().median(0).values
    sigma = torch.where(mad > 0, 1.4826 * mad, Xtr.std(0))
    sigma_fallback = [names[j] for j in range(len(names)) if not mad[j] > 0]
    r = med.clone()
    models = train_models(Xtr, ys[tr])
    meta = dict(dataset="sklearn.load_diabetes", sklearn=skv, data_sha256=data_hash, n=len(y), features=names,
                sigma=sigma.tolist(), sigma_fallback_features=sigma_fallback, reference=r.tolist())
    rows, summ = [], []
    for fam, f in models.items():
        pred = f(X[au])
        r2 = 1 - float(((pred - ys[au]) ** 2).mean() / ((ys[au] - ys[au].mean()) ** 2).mean())
        Lc = f.L_cert(sigma)
        meta[f"{fam}_test_R2_on_audit"] = r2
        meta[f"{fam}_sha256"] = model_hash(f)
        meta[f"{fam}_L_cert"] = Lc
        Clog = X[lg]
        fC = f(Clog)                           # the logged outputs available to the auditor
        for ai, idx in enumerate(au.tolist()):
            x = X[idx]
            fx = f(x[None])[0]
            fr = f(r[None])[0]
            for mode in ("domain", "fixed"):
                for n in N_LOGS:
                    A = torch.cat([Clog[:n], x[None]] + ([r[None]] if mode == "fixed" else []))
                    fA = torch.cat([fC[:n], fx.reshape(1)] + ([fr.reshape(1)] if mode == "fixed" else []))
                    LA = anchor_lipschitz(A, fA, sigma)
                    res_c = marginal_intervals(x, r, A, fA, Lc, sigma)
                    if res_c["m"] == 0:
                        continue
                    # ---- evaluation only: true phi and the hybrid oracle L_V
                    fZ = f(res_c["Z"])
                    phi = shapley_on_vertices(fZ, res_c["Sx"], res_c["Px"], res_c["W"]).numpy()
                    LV = max(LA, max_ratio(res_c["Z"], fZ, A, fA, sigma), max_ratio(res_c["Z"], fZ, res_c["Z"], fZ, sigma))
                    levels = {"cert": (Lc, res_c), "LV_eval": (LV, marginal_intervals(x, r, A, fA, LV, sigma))}
                    if LA > 0:
                        levels["opt"] = (LA, marginal_intervals(x, r, A, fA, LA, sigma))
                    for lev, (L, res) in levels.items():
                        lo, hi = res["lo"].numpy(), res["hi"].numpy()
                        rho = res["rho"].numpy()
                        Ld = L * res["delta"][res["active"]].numpy()
                        cont = (phi >= lo - 1e-9 * Ld) & (phi <= hi + 1e-9 * Ld)
                        sg = sign_from_norm(res["lo_norm"], res["hi_norm"], TOL).numpy()
                        top1, pair_share = rank_certificates(lo, hi)
                        for jj, j in enumerate(res["active"]):
                            rows.append(dict(family=fam, mode=mode, n_log=n, query_budget=len(A), audit_id=ai,
                                             row_index=idx, feature=names[j], level=lev, L_used=L, L_A=LA, L_cert=Lc,
                                             L_V=LV, lower_phi=lo[jj], upper_phi=hi[jj], Gamma=hi[jj] - lo[jj],
                                             rho=rho[jj], true_phi=phi[jj], contained=int(cont[jj]),
                                             cert_sign=int(sg[jj]), top1_certified=int(top1),
                                             certified_pair_share=pair_share, m_active=res["m"]))
        print(f"[REAL-01] {fam}: audit R2={r2:.3f}  L_cert={Lc:.4g}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "real_audit.csv"), index=False)
    inst = df.drop_duplicates(["family", "mode", "n_log", "audit_id", "level"])
    g = df.groupby(["family", "mode", "n_log", "level"])
    S = g.agg(n_feature_instances=("rho", "size"), median_rho=("rho", "median"),
              p_sat=("rho", lambda s: float(np.mean(s > SAT))), sign_rate=("cert_sign", lambda s: float(np.mean(s != 0))),
              containment=("contained", "mean"), query_budget=("query_budget", "first"),
              median_L_used=("L_used", "median")).reset_index()
    gi = inst.groupby(["family", "mode", "n_log", "level"])
    S = S.merge(gi.agg(top1_rate=("top1_certified", "mean"), mean_pair_share=("certified_pair_share", "mean"),
                       median_Lcert_over_LA=("L_cert", lambda s: float(np.median(s / inst.loc[s.index, "L_A"]))),
                       median_LV_over_LA=("L_V", lambda s: float(np.median(s / inst.loc[s.index, "L_A"])))
                       ).reset_index(), on=["family", "mode", "n_log", "level"])
    S.to_csv(os.path.join(OUT, "real_audit_summary.csv"), index=False)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Limited-query audit on the public diabetes data (50 explained patients, fixed median baseline). "
             "Budget = logged outputs $n_{\\log}$ plus the explained point (and the reference in fixed mode). "
             "\\emph{opt}: $L_A$ (no guarantee); \\emph{cert}: global certificate from the model weights only. "
             "Containment at \\emph{cert} must be 1. The hybrid oracle $L_V$ (evaluation only) is in the CSV.}",
             "\\label{tab:realaudit}", "\\small", "\\begin{tabular}{llrlrrrrr}", "\\toprule",
             "Model & Mode & $n_{\\log}$ & Level & med.\\ $\\rho$ & $p_{\\mathrm{sat}}$ & sign & top-1 & contain \\\\",
             "\\midrule"]
    for _, rr in S[S.level.isin(["opt", "cert"])].iterrows():
        lines.append(f"{rr.family} & {rr['mode']} & {rr.n_log} & {rr.level} & {rr.median_rho:.2f} & {rr.p_sat:.2f} & "
                     f"{rr.sign_rate:.2f} & {rr.top1_rate:.2f} & {rr.containment:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT, "real_audit.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, fam in zip(axes, models):
        for mode, ls in (("domain", "-"), ("fixed", "--")):
            for lev, col in (("opt", "C0"), ("LV_eval", "C2"), ("cert", "C3")):
                v = S[(S.family == fam) & (S["mode"] == mode) & (S.level == lev)].sort_values("n_log")
                ax.plot(v.n_log, v.median_rho, ls, color=col, marker="o", label=f"{lev} / {mode}")
        ax.set_xscale("log", base=2)
        ax.set_title(fam)
        ax.set_xlabel(r"logged outputs $n_{\log}$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"median $\rho$ over feature-instances")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_real_audit.pdf"))
    write_json(os.path.join(OUT, "real_audit_manifest.json"), base_manifest(**meta))
    print(S.to_string(index=False))
    viol = df[(df.level == "cert") & (df.contained == 0)]
    print("containment violations at cert (must be 0):", len(viol))


if __name__ == "__main__":
    main()
