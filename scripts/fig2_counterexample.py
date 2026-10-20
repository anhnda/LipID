#!/usr/bin/env python3
"""
fig2_counterexample.py — Figure 2 (calibration counterexample), computed rather than drawn from typed numbers.

f(z) = z_1 + 0.3 z_2, x = (1,1), r = (0,0), A_0 = {r, x}, A_1 = A_0 ∪ {(1,0)}, unit scales.
Expected (draft caption): L_A(A_0) = 0.65, L_A(A_1) = 1; rho_1(A_0; L_A) = 0, rho_1(A_1; L_A) = 0.175;
rho_1(A_0; 1) = 0.35, rho_1(A_1; 1) = 0.175. The script asserts these values (tolerance 1e-12) and writes
results/fig2_counterexample.csv + figures/fig_counterexample.pdf.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import lipid  # noqa: E402,F401
from lipid.core import anchor_L, marginal_intervals  # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    f = lambda Z: Z[:, 0] + 0.3 * Z[:, 1]
    x, r = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    s = torch.ones(2)
    A0 = torch.stack([r, x])
    A1 = torch.cat([A0, torch.tensor([[1.0, 0.0]])])
    rows = []
    for name, A in (("A0", A0), ("A1", A1)):
        fA = f(A)
        LA = anchor_L(A, fA, s)
        for lev, L in (("L_min", LA), ("L_star_1", 1.0)):
            if L <= 0:
                rho = 0.0          # L_A = 0 never happens here; kept for completeness
            else:
                rho = float(marginal_intervals(x, r, A, fA, L, s)["rho"][0])
            rows.append(dict(anchor_set=name, level=lev, L=L, L_A=LA, rho_1=rho))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "fig2_counterexample.csv"), index=False)
    print(df.to_string(index=False))
    exp = {("A0", "L_min"): 0.0, ("A1", "L_min"): 0.175, ("A0", "L_star_1"): 0.35, ("A1", "L_star_1"): 0.175}
    for (a, lev), v in exp.items():
        got = float(df[(df.anchor_set == a) & (df.level == lev)].rho_1.iloc[0])
        assert abs(got - v) < 1e-12, (a, lev, got, v)
    assert abs(df[df.anchor_set == "A0"].L_A.iloc[0] - 0.65) < 1e-12
    assert abs(df[df.anchor_set == "A1"].L_A.iloc[0] - 1.0) < 1e-12

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for lev, mk, lab in (("L_min", "o", r"$\rho(A_n; L_{\min}(A_n))$"), ("L_star_1", "s", r"$\rho(A_n; L^\star=1)$")):
        v = df[df.level == lev]
        ax.plot([0, 1], v.rho_1, marker=mk, label=lab)
    ax.set_xticks([0, 1], [r"$A_0=\{r,x\}$", r"$A_1=A_0\cup\{(1,0)\}$"])
    ax.set_xlabel("Nested observations")
    ax.set_ylabel(r"Normalized width $\rho_1$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig_counterexample.pdf"))
    print("ok: values match the draft caption")


if __name__ == "__main__":
    main()
