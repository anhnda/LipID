#!/usr/bin/env python3
"""
run_nested.py — P0'' nested-anchor experiment, protocol lipid-v3, with row-level outputs.

    python scripts/run_nested.py --seeds 1 2 3 4   --tag explore --out runs/explore --device cuda
    python scripts/run_nested.py --seeds 8 9 10 11 --tag confirm --out runs/confirm --device cuda

Default grid = the archived P0'' grid (nested_anchors_v2 defaults): d in {3,4,5,6}, rho_corr in {0,.6,.9,.99},
n in {16,64,256,1024,4096}, 4 generators, data/prod anchors, 5 x 4 query/reference pairs, m_ref = 200000.

Writes in --out:
  manifest.json          provenance (git, environment, source sha256, argv, RNG keys, output sha256)
  config.json/.yaml      the exact configuration (written BEFORE the first seed runs)
  records_seed<s>.*      long per-feature table (L_kind in opt | LV | oracle | cert)
  diag_seed<s>.*         nested_anchors_v2 columns (+ fid links) — legacy-compatible
  skips_seed<s>.csv      instances not evaluated + reason;   blocks_seed<s>.csv  runtime / memory per block
  certificates.csv       L_cert per (seed, d, rho_corr, model) with the formula used
  rng_keys.csv           RNG stream keys, covariance condition numbers, sigma
  seed_summary.csv       quick per-seed aggregates (all L kinds)
  nested2_summary_v3.json  nested_anchors_v2-style summary computed from the v3 diag rows
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
import phase1 as P  # noqa: E402
from lipid import pipeline as PL  # noqa: E402
from lipid.calibration import constants  # noqa: E402
from lipid.io import to_frame, write_table  # noqa: E402
from lipid.provenance import base_manifest, output_checksums, write_json, write_yaml, now  # noqa: E402


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "d", "mode", "anchor_law", "n_anchor", "L_kind"]
    g = df.groupby(keys, observed=True)
    out = g.agg(n_rows=("rho", "size"), p_sat=("saturation_gt_08", "mean"), mean_rho=("rho", "mean"),
                median_rho=("rho", "median"), sign_rate=("cert_sign", lambda s: float(np.mean(s != 0))),
                containment=("contained", "mean"), n_not_contained=("contained", lambda s: int(np.sum(s == 0))),
                n_v_nest=("v_nest", lambda s: int(np.sum(s == 1))), n_v_Lorder=("v_Lorder", "sum"),
                n_v_rhoL=("v_rhoL", lambda s: int(np.sum(s == 1))),
                median_L_used=("L_used", "median")).reset_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True, help="explore | confirm | <other>; recorded in the manifest")
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6, 0.9, 0.99])
    ap.add_argument("--nA", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    ap.add_argument("--models", nargs="+", default=["linear", "fa", "stable_ood", "unstable_id"])
    ap.add_argument("--arms", nargs="+", default=["data", "prod"], choices=["data", "prod"])
    ap.add_argument("--n-x", type=int, default=5)
    ap.add_argument("--refs", type=int, default=4)
    ap.add_argument("--m-ref", type=int, default=200000)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.arms = [a for a in ("data", "prod") if a in args.arms]
    ns = sorted(set(args.nA))
    os.makedirs(args.out, exist_ok=True)
    cfg = P.Config()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; use --device cpu")

    config = dict(protocol=lipid.PROTOCOL, tag=args.tag, args=vars(args), ns=ns, L_kinds=list(PL.L_KINDS),
                  generator_a=cfg.a, n_train=cfg.n_train, n_candidate=cfg.n_candidate, sign_tol=cfg.tol,
                  contain_tol_rel=PL.TOL_CONTAIN, saturation_threshold=0.8, Dbins=list(PL.V2.DBINS),
                  Dbin_labels=list(PL.V2.DLAB), calibration_constants=constants(),
                  seed_partition=dict(explore=[1, 2, 3, 4], confirm=[8, 9, 10, 11],
                                      note="seeds 8-11 are the previously held-out P0'' confirmation seeds"))
    write_json(os.path.join(args.out, "config.json"), config)
    write_yaml(os.path.join(args.out, "config.yaml"), config)
    man = base_manifest(tag=args.tag, status="running", started=now(), seeds=args.seeds)
    write_json(os.path.join(args.out, "manifest.json"), man)

    summary = dict(args=vars(args), protocol=lipid.PROTOCOL, seeds={})
    certs, rngs, sums, outputs = [], [], [], []
    for seed in args.seeds:
        t0 = time.perf_counter()
        R = PL.run_seed(seed, args, ns, args.device)
        rec = to_frame(R["records"])
        blk = pd.DataFrame(R["blocks"])
        rec = rec.merge(blk[["seed", "d", "rho_corr", "runtime_sec", "max_rss_mb"]], on=["seed", "d", "rho_corr"],
                        how="left")
        rec["protocol"] = pd.Categorical([lipid.PROTOCOL] * len(rec))
        diag = to_frame(R["diag"])
        outputs.append(write_table(rec, os.path.join(args.out, f"records_seed{seed}")))
        outputs.append(write_table(diag, os.path.join(args.out, f"diag_seed{seed}")))
        sk = pd.DataFrame(R["skips"], columns=["seed", "d", "rho_corr", "x_id", "r_id", "model", "anchor_law",
                                                "mode", "reason"])
        p = os.path.join(args.out, f"skips_seed{seed}.csv")
        sk.to_csv(p, index=False)
        outputs.append(p)
        p = os.path.join(args.out, f"blocks_seed{seed}.csv")
        blk.to_csv(p, index=False)
        outputs.append(p)
        certs += R["cert"]
        rngs += R["rng"]
        sums.append(seed_summary(rec))

        C = {c: diag[c].to_numpy().astype(str) if c in ("model", "arm", "mode", "Dbin") else diag[c].to_numpy()
             for c in PL.V2_COLS}
        summary["seeds"][seed] = PL.v2_summary_from_diag(C, cfg.tol)
        write_json(os.path.join(args.out, "nested2_summary_v3.json"), summary)
        pd.concat(sums).to_csv(os.path.join(args.out, "seed_summary.csv"), index=False)
        ck = summary["seeds"][seed]["checks"]
        new = dict(not_contained_LV=int(((rec.L_kind == "LV") & (rec.contained == 0)).sum()),
                   not_contained_oracle=int(((rec.L_kind == "oracle") & (rec.contained == 0)).sum()),
                   not_contained_cert=int(((rec.L_kind == "cert") & (rec.contained == 0)).sum()),
                   v_nest=int((rec.v_nest == 1).sum()), v_Lorder=int(rec.v_Lorder.sum()),
                   v_rhoL=int((rec.v_rhoL == 1).sum()), skipped=len(sk))
        print(f"\n=== seed {seed}: {len(rec)} record rows, {len(diag)} diag rows, {time.perf_counter() - t0:.0f}s")
        print("legacy checks (must be 0): " + " ".join(f"{c}={v}" for c, v in ck.items()))
        print("v3 checks (must be 0; skipped is informational): " + " ".join(f"{c}={v}" for c, v in new.items()),
              flush=True)

    p = os.path.join(args.out, "certificates.csv")
    pd.DataFrame(certs).to_csv(p, index=False)
    outputs.append(p)
    p = os.path.join(args.out, "rng_keys.csv")
    pd.DataFrame(rngs).to_csv(p, index=False)
    outputs.append(p)
    outputs += [os.path.join(args.out, f) for f in ("seed_summary.csv", "nested2_summary_v3.json", "config.json",
                                                    "config.yaml")]
    man.update(status="finished", finished=now(), outputs=output_checksums(outputs))
    write_json(os.path.join(args.out, "manifest.json"), man)
    print(f"\nwrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
