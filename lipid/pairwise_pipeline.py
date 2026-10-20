"""
lipid/pairwise_pipeline.py — JNT-02: sharp pairwise ranking certificates on the Section-6 instances (reviewer Q3).

For every chain of lipid.instances (same RNG streams as the archived P0'' run), every nested budget n and every
Lipschitz level L_kind in (opt, LV, oracle, cert) this computes, with ONE batched envelope evaluation per budget:

  * Theorem-1 marginal intervals [lo_i, hi_i] (raw units; identical to the records run up to round-off);
  * for every unordered active pair (i, j), i < j in feature order:
        pair_lo, pair_hi       sharp bounds on phi_i - phi_j (lipid.pairwise)
        mdiff_lo, mdiff_hi     the marginal outer bounds lo_i - hi_j, hi_i - lo_j (existing Proposition 2 comparison)
        cert_pair, cert_marg   +1 (phi_i > phi_j certified), -1 (phi_j > phi_i), 0 — same normalized tol, same c_ij
        cert_new               1 if certified pairwise but not by the marginal comparison
        abs_marg               Proposition-2 |phi| certificate (sufficient); the exact one needs the LP (JNT-03)
        contained              true phi_i - phi_j inside [pair_lo, pair_hi] (must be 1 at LV, oracle, cert)
        v_dom                  1 if the pairwise bound is weaker than the marginal one (must be 0: it is a theorem)
  * one instance row per (chain, n, L_kind): counts, signed top-1 (marginal vs pairwise vs truth), |phi| top-1.

Strict tolerance: SIGN_TOL = 1e-6 on bound / c_ij (c_ij = L (delta_i + delta_j)) for both methods.
Hybrid responses are used only for the true attribution (evaluation) and L_V (oracle), as in lipid.pipeline.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import lipid
from lipid import pairwise as PW
from lipid.instances import L_KINDS, envelopes_from_D, iter_chains, level_values
from lipid.pipeline import Cols, peak_rss_mb

PROTOCOL = lipid.PROTOCOL + "+v12-pairwise"
TOL_CONTAIN = 1e-9
TOL_DOM = 1e-12


def run_seed(seed: int, args, ns, device: str, *, kinds=L_KINDS, keep_features: bool = False, log=print) -> dict:
    """Returns dict(pairs=columns, inst=columns, feat=columns or None, skips=list, blocks=list)."""
    pairs, inst = Cols(), Cols()
    feat = Cols() if keep_features else None
    skips, blocks = [], []
    clock = {}

    def blk(d, rhoc, ev):
        if ev == "start":
            clock[(d, rhoc)] = (time.perf_counter(), len(pairs))
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
        else:
            t0, n0 = clock[(d, rhoc)]
            blocks.append(dict(seed=seed, d=d, rho_corr=rhoc, runtime_sec=time.perf_counter() - t0,
                               max_rss_mb=peak_rss_mb(), n_pair_rows=len(pairs) - n0,
                               cuda_peak_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                                             if str(device).startswith("cuda") else float("nan"))))
            log(f"[JNT-02] seed={seed} d={d} rho={rhoc} pair_rows={len(pairs)} inst_rows={len(inst)} "
                f"{blocks[-1]['runtime_sec']:.1f}s", flush=True)

    kinds = tuple(kinds)
    tol = PW.SIGN_TOL
    for ch in iter_chains(seed, args, ns, device, skips=skips, blocks=blk):
        m = ch["m"]
        act = np.asarray(ch["active"], dtype=np.int16)
        pt = PW.pair_tensors(m, ch["Z"].dtype, ch["Z"].device)
        npair = len(pt["pairs"])
        ia_np, ib_np = pt["ia"].cpu().numpy(), pt["ib"].cpu().numpy()
        dl = ch["dl"]
        dl_np = dl.detach().cpu().numpy()
        phi_np = ch["phi"].detach().cpu().numpy()
        tdiff = phi_np[ia_np] - phi_np[ib_np]
        top_true = int(np.argmax(phi_np))
        top_abs_true = int(np.argmax(np.abs(phi_np)))
        common = dict(seed=seed, model=ch["model"], d=ch["d"], rho_corr=ch["rhoc"], mode=ch["mode"],
                      anchor_law=ch["arm"], chain_id=ch["chain_id"], instance_id=ch["instance_id"], x_id=ch["xi"],
                      r_id=ch["rj"], m_active=m)
        for lev in ch["levels"]:
            Lval = level_values(ch, lev)
            Lvec = torch.tensor([Lval[k] for k in kinds], dtype=ch["Z"].dtype, device=ch["Z"].device)
            ell, h = envelopes_from_D(lev["Dk"], lev["fA"], Lvec)                 # (K, N)
            Ld = Lvec[:, None] * dl[None, :]                                       # (K, m)
            lo, hi = PW.marginal_from_envelopes(ell, h, ch["Sx"], ch["Px"], ch["W"], Ld)
            c = Lvec[:, None] * (dl[pt["ia"]] + dl[pt["ib"]])[None, :]            # (K, npair)
            plo, phi_ = PW.pairwise_from_envelopes(ell, h, pt, c)
            mlo, mhi = PW.marginal_pair_diff(lo, hi, pt)
            cp = PW.signed_cert(plo, phi_, c, tol)
            cm = PW.signed_cert(mlo, mhi, c, tol)
            ca = PW.abs_marginal_cert(lo, hi, pt, c, tol)
            sg = PW.signed_cert(lo, hi, Ld, tol)
            T = torch.stack([lo, hi]).cpu().numpy()                                # (2, K, m)
            U = torch.stack([plo, phi_, mlo, mhi, c]).cpu().numpy()                # (5, K, npair)
            Ci = torch.stack([cp, cm, ca]).cpu().numpy()                           # (3, K, npair)
            sg_np = sg.cpu().numpy()
            for kk, kind in enumerate(kinds):
                L = Lval[kind]
                lo_k, hi_k = T[0, kk], T[1, kk]
                pl, ph, ml, mh, cc = U[0, kk], U[1, kk], U[2, kk], U[3, kk], U[4, kk]
                cpk, cmk, cak = Ci[0, kk], Ci[1, kk], Ci[2, kk]
                cont = ((tdiff >= pl - TOL_CONTAIN * cc) & (tdiff <= ph + TOL_CONTAIN * cc)).astype(np.int8)
                vdom = ((pl < ml - TOL_DOM * cc) | (ph > mh + TOL_DOM * cc)).astype(np.int8)
                new = ((cpk != 0) & (cmk == 0)).astype(np.int8)
                if npair:
                    pairs.add(npair, **common, n_anchor=lev["n"], L_kind=kind, L_used=L,
                              feat_i=act[ia_np], feat_j=act[ib_np], delta_i=dl_np[ia_np], delta_j=dl_np[ib_np],
                              c_ij=cc, pair_lo=pl, pair_hi=ph, mdiff_lo=ml, mdiff_hi=mh, true_diff=tdiff,
                              pair_rho=(ph - pl) / (2.0 * cc), marg_rho=(mh - ml) / (2.0 * cc),
                              cert_pair=cpk, cert_marg=cmk, cert_new=new, abs_marg=cak, contained=cont, v_dom=vdom)
                Gp = PW.dominance_matrix(cpk, m)
                Gm = PW.dominance_matrix(cmk, m)
                Ga = PW.dominance_matrix(cak, m)
                t1p, t1m, t1a = PW.top1(Gp), PW.top1(Gm), PW.top1(Ga)
                inst.add(1, **common, n_anchor=lev["n"], L_kind=kind, L_used=L, n_pairs=npair,
                         n_cert_marg=int(np.sum(cmk != 0)), n_cert_pair=int(np.sum(cpk != 0)),
                         n_cert_new=int(np.sum(new)), n_abs_marg=int(np.sum(cak != 0)),
                         n_sign_marg=int(np.sum(sg_np[kk] != 0)),
                         top1_marg=int(act[t1m]) if t1m >= 0 else -1, top1_pair=int(act[t1p]) if t1p >= 0 else -1,
                         top1_true=int(act[top_true]), top1_abs_marg=int(act[t1a]) if t1a >= 0 else -1,
                         top1_abs_true=int(act[top_abs_true]), any_new=int(np.sum(new) > 0),
                         top1_pair_correct=(-1 if t1p < 0 else int(t1p == top_true)),
                         n_not_contained=int(np.sum(cont == 0)), n_v_dom=int(np.sum(vdom)))
                if feat is not None:
                    Ldk = L * dl_np
                    feat.add(m, **common, n_anchor=lev["n"], L_kind=kind, L_used=L, feature_i=act,
                             lower_phi=lo_k, upper_phi=hi_k, lower_norm=lo_k / Ldk, upper_norm=hi_k / Ldk,
                             rho=(hi_k - lo_k) / (2.0 * Ldk), true_phi=phi_np)
    return dict(pairs=pairs.finalize(), inst=inst.finalize(), feat=(feat.finalize() if feat is not None else None),
                skips=skips, blocks=blocks)


__all__ = ["PROTOCOL", "run_seed"]
