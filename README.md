# LipID v1.1 — reproducibility package (protocol `lipid-v3.0`)

> **v1.2 revision:** new experiments, theory code, protocol locks and paper drafts are described in `README_v12.md`
> (`scripts/reproduce_revision.sh`). Nothing below changes.

Code for *LipID: Sharp Identification Bounds for Shapley Attributions from Limited Model Observations*.
This package regenerates every table, figure and quoted number in `lipid_v1_1.tex` from raw, row-level outputs.

**Historical code is unchanged.** The files at the repository root (`phase1.py`, `nested_anchors_v2.py`,
`envelope.py`, `bounds.py`, `t6_directional.py`, `hypercube_lp.py`, …) are the archived Phase-1 / P0'' code.
`nested_anchors_v2.py` produced the P0'' confirmation summary `legacy/nested2_summary_confirm.json`
(seeds 8–11; it matches Table 3 of the draft cell for cell). New code lives in `lipid/` and `scripts/`, and it
only *imports* the historical modules. The new nested runner reuses the same RNG streams and arithmetic, and
`tests/test_pipeline_regression.py` checks that its output matches `nested_anchors_v2.run_seed` row for row.

---

## 0. Setup

```bash
python -m pip install -r requirements.txt   # torch with CUDA optional; pyarrow recommended
git init && git add -A && git commit -m "lipid v1.1 code"   # skip if already a repo
```
Commit **before** running: every manifest records `git rev-parse HEAD` and the dirty flag.

## 1. One command

```bash
bash scripts/reproduce_main.sh                 # DEVICE = cuda if available, else cpu
DEVICE=cpu bash scripts/reproduce_main.sh      # CPU-only reproduction (identical inputs)
```
Steps run in order: `env tests explore confirm fig2 metric scaling real build checksums`. Pick a subset with
`STEPS="build checksums" bash scripts/reproduce_main.sh`. Each step writes `logs/<step>.log`, and
`logs/timeline.log` records when each step starts. The script stops at the first failing step.

## 2. The same, step by step

| # | Command | TODO item | Main outputs |
|---|---|---|---|
| 1 | `python -m pytest -q -rfEs tests.py test_t6.py tests/ --junitxml=logs/pytest_junit.xml` | all | test counts |
| 2 | `python scripts/run_nested.py --seeds 1 2 3 4 --tag explore --out runs/explore --device cuda` | REP-03, CAL-01 | `runs/explore/*` |
| 3 | `python scripts/run_nested.py --seeds 8 9 10 11 --tag confirm --out runs/confirm --device cuda` | REP-03, CAL-01 | `runs/confirm/*` |
| 4 | `python scripts/fig2_counterexample.py` | Fig. 2 | `figures/fig_counterexample.pdf` |
| 5 | `python scripts/metric_sensitivity.py` | MET-01 | `results/metric_*.{csv,tex}` |
| 6 | `python scripts/benchmark_scaling.py --device cuda` | SCALE-01 | `results/scaling.{csv,tex}`, `figures/fig_scaling.pdf` |
| 7 | `python scripts/real_audit.py` | REAL-01 | `results/real_audit*.{csv,tex,md}`, `figures/fig_real_audit.pdf` |
| 8 | `python scripts/build_results.py --explore runs/explore --confirm runs/confirm --legacy-confirm legacy/nested2_summary_confirm.json` | CAL-02, REP-01/02, PRE-01 | `results/*`, `figures/*` |

Optional step 2b, a second crosswalk that runs the *archived* program on the exploratory seeds:
`python nested_anchors_v2.py --seeds 1 2 3 4 --out legacy/nested2_explore --device cuda && cp legacy/nested2_explore/nested2_summary.json legacy/nested2_summary_explore.json`
(or `STEPS="legacy_explore" bash scripts/reproduce_main.sh`). Run it before step 8.

Steps 2–3 are the heavy part: the full P0'' grid, 4 seeds each. Every other step reads only their outputs or
uses small instances. The real audit and the metric LPs run on CPU.

## 3. What each run writes (`runs/<tag>/`)

| File | Content |
|---|---|
| `config.json`, `config.yaml` | exact configuration, written **before** the first seed |
| `manifest.json` | git commit + dirty files, Python/torch/numpy/scipy/pandas/pyarrow versions, CUDA devices, host, argv, sha256 of all source files and of every output |
| `records_seed<s>.parquet` | **long per-feature table**, one row per (feature-instance, n, `L_kind`) — schema below |
| `diag_seed<s>.parquet` | the exact `nested_anchors_v2` columns plus `fid`, `fid_prev` (same chain, previous n) and `fid_pair` (matched data-anchor row) |
| `skips_seed<s>.csv` | every instance/mode not evaluated, with the reason (nothing is dropped silently) |
| `blocks_seed<s>.csv` | runtime, peak RSS and CUDA peak memory per (seed, d, ρ_c) block |
| `certificates.csv` | L_cert per (seed, d, ρ_c, generator), with its formula and per-coordinate bounds |
| `rng_keys.csv` | RNG stream keys, covariance condition numbers, Mahalanobis 99% quantile, σ |
| `seed_summary.csv` | quick per-seed aggregates for every `L_kind` |
| `nested2_summary_v3.json` | `nested_anchors_v2`-style summary computed from the v3 diag rows |

### Lipschitz levels (`L_kind`)

| `L_kind` | Level | Regime |
|---|---|---|
| `opt` | L_A(A_n) | optimistic; does **not** guarantee f ∈ F_L(A) |
| `LV` | L_V = Lip(f on C_data ∪ C_prod ∪ hybrids) | oracle; uses hybrid responses, validation only |
| `oracle` | L\* = max(L_true, L_V) | the oracle-valid level used throughout the draft |
| `cert` | L_cert (`lipid/calibration.py`, appendix H) | certified globally from generator parameters and σ only; never touches hybrid responses, anchor labels or L_V |

For `linear` and `unstable_id` (a = 4 > 0), L_true equals the analytic certificate, so L\* = L_cert and the
draft's L\* results for these two generators are already certified. Hybrid responses are evaluated only to
compute `true_phi` (containment checks) and L_V.

### Record schema (`records_seed<s>`)
`seed, model, d, rho_corr, mode, anchor_law (data|prod), n_anchor, chain_id, instance_id, x_id, r_id, feature_i,
fid, delta_i, sigma_i, D_over_delta_i, Dbin, L_kind, L_A, L_V, L_true, L_star, L_cert, L_used, kappa (=L_used/L_A),
lower_norm, upper_norm, lower_phi, upper_phi, Gamma, rho, true_phi, contained, cert_sign, saturation_gt_08,
v_nest, v_Lorder, v_rhoL, runtime_sec, max_rss_mb, protocol`.
N2 terms, geometry/range certificates (`low_com`, `range_com`, `uni_com`, …), the Mahalanobis tail share
(`lowdens`) and `dRho` are in `diag_seed<s>`, under their `nested_anchors_v2` names.

Per-row check flags must all be 0: `contained` = 1 at `LV`, `oracle` and `cert`; `v_nest` = 0 (same-L
nesting); `v_Lorder` = 0 (L_cert ≥ L_V ≥ L_A); `v_rhoL` = 0 (ρ monotone in L). `run_nested.py` prints these
counts per seed, next to the legacy checks.

## 4. What `build_results.py` writes (`results/`, `figures/`)

| Draft object | File |
|---|---|
| Table 2 `tab:calcert` | `calibration_table.tex` (+ `calibration_by_seed.csv`, stratified by generator × d × mode × n × seed × law) |
| Table 3 `tab:prevalence` | `tab_prevalence.tex` (+ `prevalence_by_seed.csv` with denominators) |
| Table 4 `tab:dist` | `tab_dist.tex` (8 pooled seeds, as in the draft) and `tab_dist_confirm.tex` (seeds 8–11) |
| Table 5 `tab:criteria` | `tab_criteria.tex`, `criteria_status.csv`, `criteria_reconstructed_cells.csv`, `preregistration_ledger.md` |
| Table 6 (scaling) | `scaling.tex` (from step 6) |
| Metric sensitivity | `metric_sensitivity.tex` (step 5) |
| Real audit | `real_audit.tex` (step 7) |
| Figs. 2 / 3 / 4 | `fig_counterexample.pdf`, `fig_saturation_prevalence.pdf`, `fig_anchor_law_gap.pdf` |
| New figures | `fig_certified_calibration.pdf`, `fig_anchor_law_strata.pdf`, `fig_scaling.pdf`, `fig_real_audit.pdf` |
| Numbers quoted in §6.3–6.6 and §7 | `text_numbers.csv` (key, value range, definition, source) |
| Paired effects (REP-01) | `paired_effects_by_seed.csv`, `paired_effects_summary.csv`, `matched/<tag>_{nesting,anchor_law,N2}.parquet` (with `id_left`, `id_right`, `metric`, `difference`, `seed`, `chain_id`) |
| Anchor law, subgroups (REP-02) | `anchor_law_by_seed.csv`, `anchor_law_dpair_by_seed.csv`, `subgroups_d_over_delta.csv`, `anchor_law_reversal_counts.csv`, `mahalanobis_tail.csv` |
| Reproducibility | `legacy_crosswalk_confirm.json` (+ mismatches CSV), `provenance.md`, `uncertainty_methods.md`, `crosswalk_paper.md` (with figure captions), `build_report.json` (`issues` must be empty), `main_tables.tex` (all fragments) |

**Legacy crosswalk.** `build_results.py` recomputes the `nested_anchors_v2` summary from the new raw diag rows
and compares every statistic with `legacy/nested2_summary_confirm.json` (tolerance 1e-9). The verdict is
`REPRODUCED` or a list of deviations. CUDA reductions can differ in the last ulp; a deviation that crosses a
threshold (for example ρ > 0.8) shows up in the mismatches CSV and has to be reported.

**Uncertainty.** The independent unit is the seed: within a seed, one anchor chain is shared by all 20 (x, r)
pairs, all generators and both modes. Spreads are therefore descriptive seed min–max over 4 seeds. No
confidence intervals and no bootstrap over features (see `results/uncertainty_methods.md`).

**Pre-registration.** C1–C5 are evaluated mechanically from the reconstructed rules
(`lipid/analysis.py::OPERATIONALIZATION`). The historical labels (C2b not met, C4b partly met, C5 invalid
specification) are kept verbatim. The original locked protocol text is **not in this archive** and should be
attached to `results/preregistration_ledger.md`.

## 5. New modules

| File | Content |
|---|---|
| `lipid/calibration.py` | global weighted-ℓ1 certificates for the four generators; c_q closed form; c_t certified by a bracketed root argument. Uses \|a\| (the legacy `phase1.lipschitz_true` uses a signed a, which is invalid for a < 0; this does not affect the archived runs, which use a = 4) |
| `lipid/pipeline.py` | v3 nested runner (row-level outputs, four L levels) |
| `lipid/metric_lp.py` | exact finite-hybrid LP for any weighted ℓ_p (HiGHS) and the edgewise aggregate; appendix-B counterexample |
| `lipid/mc.py` | appendix-F stratified Hoeffding bounds (`log_constant='paper'` = log(4m²/α); `'tight'` = log(2m²/α) also valid) |
| `lipid/core.py` | Theorem-1 intervals by enumeration, exact Shapley from vertices, sign/containment helpers |
| `lipid/analysis.py` | criteria reconstruction, seed-level summaries |
| `lipid/provenance.py`, `lipid/io.py` | manifests, checksums, parquet/CSV I/O |

## 6. What to send back after running
`logs/` (all), `results/` (without `matched/` and `scaling_cache/` unless asked), `figures/`,
`runs/*/{manifest.json,config.yaml,seed_summary.csv,certificates.csv,skips_seed*.csv,blocks_seed*.csv}`.
The large `records_*/diag_*` parquet files only if a deviation has to be investigated.

The original Phase-1 README is kept as `legacy/README_phase1.md`.
