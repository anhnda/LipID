"""
tests/test_v12_stats_lock.py — STAT-01 helpers (seed-level intervals, Holm) and protocol-lock utilities; scale
invariance of the identified intervals (MET-02 sanity check).
"""
import json

import numpy as np
import pytest
import torch
from scipy import stats as st

from envelope import anchor_lipschitz
from lipid import pairwise as PW
from lipid.lock import key_collisions
from lipid.stats import bootstrap_interval, holm, t_interval

torch.set_default_dtype(torch.float64)


def test_t_interval_matches_scipy():
    v = np.array([0.12, 0.08, 0.15, 0.11, 0.09, 0.13, 0.10, 0.14, 0.07, 0.12, 0.16])
    r = t_interval(v)
    lo, hi = st.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=st.sem(v))
    assert abs(r["t_lo"] - lo) < 1e-12 and abs(r["t_hi"] - hi) < 1e-12 and r["ci_admissible"]
    assert abs(r["p_two_sided"] - st.ttest_1samp(v, 0.0).pvalue) < 1e-12
    b = bootstrap_interval(v, B=4000)
    assert b["boot_lo"] < v.mean() < b["boot_hi"]


def test_holm():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: .005 (x4=.02), .01 (x3=.03), .03 (x2=.06), .04 (x1=.04 -> monotone .06)
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_new_seeds_do_not_collide_with_historical_streams():
    import os

    import lipid
    with open(os.path.join(lipid.ROOT, "protocols", "v12_preregistration.json")) as fh:
        seeds = json.load(fh)["experiments"]["STAT-01"]["seeds"]
    assert key_collisions(seeds) == []
    assert not set(seeds) & set(range(1, 12))


@pytest.mark.parametrize("fixed", [False, True])
def test_scale_invariance_of_intervals(fixed):
    g = torch.Generator().manual_seed(3)
    d = 4
    x, r = torch.randn(d, generator=g), torch.randn(d, generator=g)
    sigma = 0.3 + torch.rand(d, generator=g)
    A = torch.cat([torch.randn(20, d, generator=g), x[None]] + ([r[None]] if fixed else []))
    fA = torch.tanh(A).sum(1)
    L = 1.4 * anchor_lipschitz(A, fA, sigma)
    a = PW.pairwise_bounds(x, r, A, fA, L, sigma)
    b = PW.pairwise_bounds(x, r, A, fA, 3.7 * L, 3.7 * sigma)
    for k in ("lo", "hi", "pair_lo", "pair_hi"):
        torch.testing.assert_close(a[k], b[k], rtol=1e-12, atol=1e-12)
