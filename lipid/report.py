"""
lipid/report.py — shared reporting helpers for the v1.2 analyses (tables, seed summaries, quoted numbers).

Every analysis script writes
  results_v12/<exp>_*.csv     machine-readable outputs (row level or seed level)
  results_v12/<exp>_*.tex     LaTeX fragments (\\input in the manuscript)
  results_v12/numbers_<exp>.csv   key, value, definition, source  — every number quoted in the text
scripts/build_revision.py merges the number files into results_v12/numbers_v12.tex, which defines \\lipnum{<key>}.
Spread convention: with fewer than lipid.stats.MIN_SEEDS_FOR_CI seeds only seed min--max is shown (descriptive);
with more, the 95% seed-level t interval is shown and labelled as such.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from lipid.stats import MIN_SEEDS_FOR_CI, t_interval

OUT_DEFAULT = "results_v12"
FIG_DEFAULT = "figures_v12"


def fnum(x, nd=2, pct=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if pct:
        return f"{100.0 * x:.{max(0, nd - 1)}f}"
    s = f"{x:.{nd}f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def across_seeds(df: pd.DataFrame, keys, cols, seed_col="seed") -> pd.DataFrame:
    """One row per keys: n_seeds, and for each col: mean, min, max, t_lo, t_hi (t only if >= MIN_SEEDS_FOR_CI)."""
    rows = []
    for k, g in df.groupby(list(keys), observed=True, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        row["n_seeds"] = int(g[seed_col].nunique())
        for c in cols:
            v = g[c].to_numpy(dtype=float)
            t = t_interval(v)
            row.update({f"{c}_mean": t["mean"], f"{c}_min": t["min"], f"{c}_max": t["max"],
                        f"{c}_t_lo": t["t_lo"] if t["ci_admissible"] else math.nan,
                        f"{c}_t_hi": t["t_hi"] if t["ci_admissible"] else math.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def spread(row, c, nd=2, pct=False) -> str:
    """'mean [lo, hi]' with t interval when admissible, else 'mean (min--max)'."""
    m = fnum(row.get(f"{c}_mean"), nd, pct)
    lo, hi = row.get(f"{c}_t_lo"), row.get(f"{c}_t_hi")
    if lo is not None and np.isfinite(lo):
        return f"{m} [{fnum(lo, nd, pct)}, {fnum(hi, nd, pct)}]"
    a, b = fnum(row.get(f"{c}_min"), nd, pct), fnum(row.get(f"{c}_max"), nd, pct)
    return m if a == b else f"{m} ({a}--{b})"


def tex_table(header, rows, caption, label, align=None, size="\\small") -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
           f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MIDRULE" else " & ".join(str(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


class Numbers:
    """Collects every number quoted in the manuscript text, with its definition and source file."""

    def __init__(self, exp: str, out_dir: str):
        self.exp, self.out_dir, self.rows = exp, out_dir, []

    def add(self, key: str, value, definition: str, source: str, nd: int = 2, pct: bool = False):
        txt = value if isinstance(value, str) else fnum(float(value), nd, pct)
        self.rows.append(dict(key=f"{self.exp.lower()}.{key}", value=txt,
                              raw=(value if isinstance(value, str) else float(value)), definition=definition,
                              source=source))

    def write(self):
        os.makedirs(self.out_dir, exist_ok=True)
        p = os.path.join(self.out_dir, f"numbers_{self.exp.lower().replace('-', '')}.csv")
        pd.DataFrame(self.rows, columns=["key", "value", "raw", "definition", "source"]).to_csv(p, index=False)
        return p


def savefig(fig, figs_dir: str, name: str):
    os.makedirs(figs_dir, exist_ok=True)
    p = os.path.join(figs_dir, name)
    fig.savefig(p, bbox_inches="tight")
    return p


__all__ = ["fnum", "across_seeds", "spread", "tex_table", "Numbers", "savefig", "OUT_DEFAULT", "FIG_DEFAULT",
           "MIN_SEEDS_FOR_CI"]
