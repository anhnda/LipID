"""
tests/test_pipeline_regression.py — REP-03 / CAL-01 acceptance on a small grid (CPU).

(1) lipid.pipeline reproduces nested_anchors_v2.run_seed ROW FOR ROW (same RNG streams and arithmetic).
(2) v3 invariants: containment of the true attribution at L_V, L* and L_cert; L_cert >= L_V >= L_A(A_n);
    same-L interval nesting along every chain; rho monotone in L (Proposition 3).
"""
import argparse

import numpy as np
import pytest

import nested_anchors_v2 as V2
import phase1 as P
from lipid import pipeline as PL


def _args():
    return argparse.Namespace(dims=[3, 4], rhos=[0.0, 0.99], nA=[16, 64, 256],
                              models=["linear", "fa", "stable_ood", "unstable_id"], arms=["data", "prod"],
                              n_x=2, refs=2, m_ref=5000, block=256, device="cpu")


@pytest.fixture(scope="module")
def runs():
    args = _args()
    ns = sorted(set(args.nA))
    legacy = V2.run_seed(3, args, ns, "cpu")
    new = PL.run_seed(3, args, ns, "cpu", log=lambda *a, **k: None)
    return legacy, new


def test_diag_reproduces_v2_row_for_row(runs):
    legacy, new = runs
    D = new["diag"]
    assert len(D["rho_com"]) == len(legacy["rho_com"]) > 0
    for c in PL.V2_COLS:
        a, b = np.asarray(legacy[c]), np.asarray(D[c])
        assert a.shape == b.shape, c
        if a.dtype.kind in "iufb":
            np.testing.assert_array_equal(a.astype(float), b.astype(float), err_msg=c)
        else:
            assert (a.astype(str) == b.astype(str)).all(), c


def test_v2_summary_identical(runs):
    legacy, new = runs
    tol = P.Config().tol
    Cn = {c: np.asarray(new["diag"][c]) for c in PL.V2_COLS}
    s_new = PL.v2_summary_from_diag(Cn, tol)
    s_old = PL.v2_summary_from_diag({c: np.asarray(legacy[c]) for c in PL.V2_COLS}, tol)
    for tab in ("by_d_mode_arm_n", "by_d_mode_arm_model_n", "by_d_mode_arm_rhoc_n", "by_d_mode_arm_rhoc_Dbin_n"):
        assert s_new[tab].keys() == s_old[tab].keys()
        for k in s_old[tab]:
            for st, v in s_old[tab][k].items():
                w = s_new[tab][k][st]
                assert (v is None and w is None) or v == w or (v != v and w != w), (tab, k, st, v, w)
    assert all(v == 0 for v in s_new["checks"].values())


def test_records_are_consistent_with_diag(runs):
    _, new = runs
    R, D = new["records"], new["diag"]
    n = len(D["rho_com"])
    assert len(R["rho"]) == len(PL.L_KINDS) * n
    for kind, col in (("oracle", "rho_com"), ("opt", "rho_opt")):
        sel = R["L_kind"] == kind
        order = np.argsort(R["fid"][sel], kind="stable")
        np.testing.assert_array_equal(R["rho"][sel][order], D[col][np.argsort(D["fid"], kind="stable")])


def test_v3_invariants(runs):
    _, new = runs
    R = new["records"]
    for kind in ("LV", "oracle", "cert"):
        sel = R["L_kind"] == kind
        assert R["contained"][sel].min() == 1, f"true phi outside the interval at {kind}"
    assert R["v_Lorder"].max() == 0
    assert (R["L_cert"] >= R["L_V"] * (1 - 1e-12)).all()
    assert (R["L_V"] >= R["L_A"] * (1 - 1e-12)).all()
    assert (R["v_nest"] != 1).all()
    assert (R["v_rhoL"] != 1).all()
    # Gamma = upper - lower
    np.testing.assert_allclose(R["Gamma"], R["upper_phi"] - R["lower_phi"], rtol=1e-10, atol=1e-12)
    # certified sign implies rho < 1/2
    assert (R["rho"][R["cert_sign"] != 0] < 0.5 + 1e-9).all()
    # L* equals L_cert for the two generators with an exact analytic constant (a = 4 > 0)
    for model in ("linear", "unstable_id"):
        sel = R["model"] == model
        np.testing.assert_allclose(R["L_star"][sel], R["L_cert"][sel], rtol=1e-12)


def test_no_silent_drops(runs):
    _, new = runs
    # every skipped unit is listed with a reason; with this grid no instance should be skipped
    assert all("reason" in s for s in new["skips"])
