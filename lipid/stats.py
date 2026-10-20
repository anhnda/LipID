"""
lipid/stats.py — seed-level uncertainty for the v1.2 confirmatory batch (STAT-01; reviewer Q7).

Unit of analysis: the SEED. Within a seed, every (d, rho_c) block shares one sigma draw, one set of (x, r) pairs and
one nested anchor chain per anchor law, across generators and modes; feature-instances are therefore NOT independent
and are never resampled. Each estimand is first reduced to one number per seed (a matched aggregate), then:

  * t interval:   mean +- t_{S-1, 1-a/2} * sd / sqrt(S)        (S independent seeds; reported when S >= 10)
  * bootstrap:    percentile interval of the mean over seeds, B resamples of seeds with replacement (sensitivity)
  * p value:      two-sided one-sample t test of mean = 0; Holm adjustment within each pre-specified family
These intervals describe variability over the random-seed design of the synthetic study (x, r, sigma, anchor chains),
not population-level uncertainty about real models or data.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

MIN_SEEDS_FOR_CI = 10


def t_interval(values, level: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    S = v.size
    out = dict(n_seeds=int(S), mean=math.nan, sd=math.nan, se=math.nan, t_lo=math.nan, t_hi=math.nan,
               p_two_sided=math.nan, min=math.nan, max=math.nan, ci_admissible=bool(S >= MIN_SEEDS_FOR_CI))
    if S == 0:
        return out
    out.update(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    if S < 2:
        return out
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(S)
    tc = float(st.t.ppf(0.5 + level / 2.0, S - 1))
    out.update(sd=sd, se=se, t_lo=out["mean"] - tc * se, t_hi=out["mean"] + tc * se, t_crit=tc)
    if se > 0:
        out["p_two_sided"] = float(2.0 * st.t.sf(abs(out["mean"]) / se, S - 1))
    else:
        out["p_two_sided"] = 0.0 if out["mean"] != 0 else 1.0
    return out


def bootstrap_interval(values, level: float = 0.95, B: int = 20000, seed: int = 20261020) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size < 2:
        return dict(boot_lo=math.nan, boot_hi=math.nan, B=B)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(B, v.size))].mean(1)
    a = (1.0 - level) / 2.0
    return dict(boot_lo=float(np.quantile(means, a)), boot_hi=float(np.quantile(means, 1.0 - a)), B=B)


def holm(pvals) -> list:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.size, np.nan)
    ok = np.isfinite(p)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    k = idx.size
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        out[i] = running
    return out.tolist()


def bonferroni_t_interval(values, family_size: int, level: float = 0.95) -> dict:
    """Simultaneous (Bonferroni) t interval for one member of a family of `family_size` estimands."""
    r = t_interval(values, level=1.0 - (1.0 - level) / max(1, family_size))
    return dict(bonf_lo=r["t_lo"], bonf_hi=r["t_hi"])


def summarize(values, family_size: int = 1, level: float = 0.95, B: int = 20000) -> dict:
    out = t_interval(values, level)
    out.update(bootstrap_interval(values, level, B))
    out.update(bonferroni_t_interval(values, family_size, level))
    return out


__all__ = ["MIN_SEEDS_FOR_CI", "t_interval", "bootstrap_interval", "holm", "bonferroni_t_interval", "summarize"]
