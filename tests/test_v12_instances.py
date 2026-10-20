"""
tests/test_v12_instances.py — the v1.2 runners regenerate EXACTLY the Section-6 instances of lipid.pipeline.

(1) lipid.pairwise_pipeline (via lipid.instances) gives the same four Lipschitz levels and Theorem-1 endpoints as the
    records of lipid.pipeline.run_seed (row-matched on seed/model/d/rho/mode/law/n/instance/feature/L_kind);
(2) pairwise integrity: true differences contained at LV/oracle/cert, pairwise never weaker than marginal;
(3) lipid.certdiag at L* reproduces nested_anchors_v2's rho_com, low_com (T3), range_com and uni_com.
Small grid, CPU (same grid as tests/test_pipeline_regression.py).
"""
import argparse

import numpy as np
import pandas as pd
import pytest
import torch

from lipid import pipeline as PL
from lipid import pairwise_pipeline as PP
from lipid.certdiag import relaxation_chain
from lipid.instances import iter_chains

KEYS = ["model", "d", "rho_corr", "mode", "anchor_law", "n_anchor", "instance_id", "feature_i", "L_kind"]


def _args():
    return argparse.Namespace(dims=[3, 4], rhos=[0.0, 0.99], nA=[16, 64, 256],
                              models=["linear", "fa", "stable_ood", "unstable_id"], arms=["data", "prod"],
                              n_x=2, refs=2, m_ref=5000, block=256, device="cpu")


@pytest.fixture(scope="module")
def runs():
    args = _args()
    ns = sorted(set(args.nA))
    ref = PL.run_seed(3, args, ns, "cpu", log=lambda *a, **k: None)
    new = PP.run_seed(3, args, ns, "cpu", keep_features=True, log=lambda *a, **k: None)
    return ref, new


def test_levels_and_endpoints_match_records(runs):
    ref, new = runs
    R = pd.DataFrame(ref["records"])
    F = pd.DataFrame(new["feat"])
    for df in (R, F):
        for c in ("model", "mode", "anchor_law", "L_kind"):
            df[c] = df[c].astype(str)
    M = R.merge(F, on=KEYS, suffixes=("_ref", "_new"), validate="one_to_one")
    assert len(M) == len(R) == len(F) > 0
    np.testing.assert_allclose(M.L_used_ref, M.L_used_new, rtol=0, atol=0)
    np.testing.assert_allclose(M.lower_norm_ref, M.lower_norm_new, rtol=0, atol=1e-12)
    np.testing.assert_allclose(M.upper_norm_ref, M.upper_norm_new, rtol=0, atol=1e-12)
    np.testing.assert_allclose(M.rho_ref, M.rho_new, rtol=0, atol=1e-12)
    np.testing.assert_allclose(M.true_phi_ref, M.true_phi_new, rtol=0, atol=0)


def test_pairwise_integrity(runs):
    _, new = runs
    P = pd.DataFrame(new["pairs"])
    valid = P.L_kind.astype(str).isin(["LV", "oracle", "cert"])
    assert int(((P.contained == 0) & valid).sum()) == 0
    assert int(P.v_dom.sum()) == 0
    assert ((P.cert_marg == 0) | (P.cert_pair == P.cert_marg)).all()
    assert (P.pair_rho <= P.marg_rho + 1e-12).all()
    I = pd.DataFrame(new["inst"])
    assert int(((I.top1_pair_correct == 0) & I.L_kind.astype(str).isin(["LV", "oracle", "cert"])).sum()) == 0
    assert (I.n_cert_pair >= I.n_cert_marg).all()


def test_certdiag_reproduces_archived_certificates(runs):
    ref, _ = runs
    D = pd.DataFrame({c: ref["diag"][c] for c in ("seed", "d", "rhoc", "model", "arm", "mode", "n", "inst", "j",
                                                  "rho_com", "low_com", "range_com", "uni_com")})
    for c in ("model", "arm", "mode"):
        D[c] = D[c].astype(str)
    rows = []
    for ch in iter_chains(3, _args(), [16, 64, 256], "cpu"):
        for lev in ch["levels"]:
            Lvec = torch.tensor([ch["L_star"]])
            R = relaxation_chain(lev["Dk"], lev["fA"], Lvec, float(lev["L_n"]), float(lev["Omega"]), ch["Sx"], ch["Px"],
                                 ch["W"], ch["dl"], D_AA=None, n_background=lev["n"], geo="none")
            for jj, j in enumerate(ch["active"]):
                rows.append(dict(d=ch["d"], rhoc=ch["rhoc"], model=ch["model"], arm=ch["arm"], mode=ch["mode"],
                                 n=lev["n"], inst=ch["instance_id"], j=j, rho_exact=float(R["rho_exact"][0, jj]),
                                 t3=float(R["rho_t3"][0, jj]), rg=float(R["rho_range"][0, jj]),
                                 uni=float(R["rho_uni"][0, jj])))
    N = pd.DataFrame(rows)
    M = D.merge(N, on=["d", "rhoc", "model", "arm", "mode", "n", "inst", "j"], validate="one_to_one")
    assert len(M) == len(D) > 0
    np.testing.assert_allclose(M.rho_com, M.rho_exact, atol=1e-12)
    np.testing.assert_allclose(M.low_com, M.t3, atol=1e-12)
    np.testing.assert_allclose(M.range_com, M.rg, atol=1e-12)
    np.testing.assert_allclose(M.uni_com, M.uni, atol=1e-12)
