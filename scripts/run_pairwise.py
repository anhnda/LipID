#!/usr/bin/env python3
"""
run_pairwise.py — JNT-02 runner: sharp pairwise ranking certificates on the Section-6 instances (reviewer Q3).

    python scripts/run_pairwise.py --seeds 8 9 10 11 --tag confirm --out runs/pairs_confirm --device cuda
    python scripts/run_pairwise.py --seeds 1 2 3 4   --tag explore --out runs/pairs_explore --device cuda
    python scripts/run_pairwise.py --seeds $(seq 101 124) --tag stat01 --out runs/pairs_stat01 --device cuda \
        --require-lock protocols/locks/STAT-01.lock.json

Same default grid, RNG streams and anchor arms as scripts/run_nested.py, so every row is matched to the records run
of the same seeds (keys: seed, model, d, rho_corr, mode, anchor_law, n_anchor, instance_id, feature, L_kind).

Writes in --out:
  config.json / manifest.json         configuration (written BEFORE the first seed) and provenance
  pairs_seed<s>.parquet               one row per (chain, n, L_kind, unordered active pair)  — see lipid/pairwise_pipeline
  inst_seed<s>.parquet                one row per (chain, n, L_kind)
  features_seed<s>.parquet            (only with --write-features) Theorem-1 endpoints, for cross-checking records
  skips_seed<s>.csv, blocks_seed<s>.csv
  seed_summary.csv                    per seed x d x mode x law x n x L_kind: pair / marginal certification rates
Per-seed console checks (must be 0): pairs not contained at LV/oracle/cert, v_dom (pairwise weaker than marginal).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import lipid  # noqa: E402
from lipid import pairwise_pipeline as PP  # noqa: E402
from lipid.instances import L_KINDS  # noqa: E402
from lipid.io import to_frame, write_table  # noqa: E402
from lipid.lock import verify_lock  # noqa: E402
from lipid.pairwise import SIGN_TOL  # noqa: E402
from lipid.provenance import base_manifest, now, output_checksums, write_json  # noqa: E402

torch.set_default_dtype(torch.float64)


def seed_summary(P: pd.DataFrame, I: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "d", "mode", "anchor_law", "n_anchor", "L_kind"]
    g = P.groupby(keys, observed=True)
    a = g.agg(n_pairs=("cert_pair", "size"), rate_pair=("cert_pair", lambda s: float(np.mean(s != 0))),
              rate_marg=("cert_marg", lambda s: float(np.mean(s != 0))), rate_new=("cert_new", "mean"),
              n_not_contained=("contained", lambda s: int(np.sum(s == 0))), n_v_dom=("v_dom", "sum"),
              median_pair_rho=("pair_rho", "median"), median_marg_rho=("marg_rho", "median")).reset_index()
    I2 = I[I.m_active >= 2]
    gi = I2.groupby(keys, observed=True)
    b = gi.agg(n_inst=("n_pairs", "size"), frac_any_new=("any_new", "mean"),
               top1_pair=("top1_pair", lambda s: float(np.mean(s >= 0))),
               top1_marg=("top1_marg", lambda s: float(np.mean(s >= 0)))).reset_index()
    return a.merge(b, on=keys, how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6, 0.9, 0.99])
    ap.add_argument("--nA", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    ap.add_argument("--models", nargs="+", default=["linear", "fa", "stable_ood", "unstable_id"])
    ap.add_argument("--arms", nargs="+", default=["data", "prod"], choices=["data", "prod"])
    ap.add_argument("--kinds", nargs="+", default=list(L_KINDS), choices=list(L_KINDS))
    ap.add_argument("--n-x", type=int, default=5)
    ap.add_argument("--refs", type=int, default=4)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write-features", action="store_true")
    ap.add_argument("--require-lock", default=None, help="protocol lock that must verify before any seed runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.arms = [a for a in ("data", "prod") if a in args.arms]
    if args.arms != ["data", "prod"]:
        print("WARNING: L_V depends on the anchor arms; rows are matched to a records run only if it used the same "
              "arms", flush=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; use --device cpu")
    lock_status = None
    if args.require_lock:
        lock_status = verify_lock(args.require_lock)
        if not lock_status["ok"]:
            raise SystemExit(f"protocol lock does not verify: {lock_status}")
    ns = sorted(set(args.nA))
    os.makedirs(args.out, exist_ok=True)
    config = dict(protocol=PP.PROTOCOL, experiment="JNT-02", tag=args.tag, args=vars(args), ns=ns,
                  sign_tol=SIGN_TOL,
                  contain_tol_rel=PP.TOL_CONTAIN, dominance_tol_rel=PP.TOL_DOM, lock=lock_status)
    write_json(os.path.join(args.out, "config.json"), config)
    man = base_manifest(tag=args.tag, status="running", started=now(), seeds=args.seeds, experiment="JNT-02")
    write_json(os.path.join(args.out, "manifest.json"), man)
    outputs, sums = [], []
    for seed in args.seeds:
        t0 = time.perf_counter()
        R = PP.run_seed(seed, args, ns, args.device, kinds=args.kinds, keep_features=args.write_features)
        Pdf = to_frame(R["pairs"])
        Idf = to_frame(R["inst"])
        Pdf["protocol"] = PP.PROTOCOL
        Idf["protocol"] = PP.PROTOCOL
        outputs.append(write_table(Pdf, os.path.join(args.out, f"pairs_seed{seed}")))
        outputs.append(write_table(Idf, os.path.join(args.out, f"inst_seed{seed}")))
        if R["feat"] is not None:
            outputs.append(write_table(to_frame(R["feat"]), os.path.join(args.out, f"features_seed{seed}")))
        p = os.path.join(args.out, f"skips_seed{seed}.csv")
        pd.DataFrame(R["skips"], columns=["seed", "d", "rho_corr", "x_id", "r_id", "model", "anchor_law", "mode",
                                          "reason"]).to_csv(p, index=False)
        outputs.append(p)
        p = os.path.join(args.out, f"blocks_seed{seed}.csv")
        pd.DataFrame(R["blocks"]).to_csv(p, index=False)
        outputs.append(p)
        sums.append(seed_summary(Pdf, Idf))
        pd.concat(sums).to_csv(os.path.join(args.out, "seed_summary.csv"), index=False)
        valid = Pdf.L_kind.isin(["LV", "oracle", "cert"])
        chk = dict(not_contained_valid_L=int(((Pdf.contained == 0) & valid).sum()),
                   not_contained_opt=int(((Pdf.contained == 0) & ~valid).sum()),
                   v_dom=int(Pdf.v_dom.sum()), top1_pair_wrong_valid_L=int(
                       ((Idf.top1_pair_correct == 0) & Idf.L_kind.isin(["LV", "oracle", "cert"])).sum()))
        print(f"\n=== seed {seed}: {len(Pdf)} pair rows, {len(Idf)} instance rows, {time.perf_counter() - t0:.0f}s")
        print("checks (must be 0 except not_contained_opt): " + " ".join(f"{k}={v}" for k, v in chk.items()),
              flush=True)
    outputs.append(os.path.join(args.out, "seed_summary.csv"))
    outputs.append(os.path.join(args.out, "config.json"))
    man.update(status="finished", finished=now(), outputs=output_checksums(outputs))
    write_json(os.path.join(args.out, "manifest.json"), man)
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
