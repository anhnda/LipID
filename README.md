# AIW — Phase 1 (mathematical validation) + amendment v3 (P0)

Attribution Identifiability Width for reference-conditioned Shapley values over the
anchor-constrained Lipschitz class F_L(A), weighted-l1 metric.

**Status.** The v2 code was run by the team (confirmatory grid, seed 1). The v3 additions in this
revision (optimistic ceiling, L_V, rho_inf, MC negative control, GPU device, new tests) have **not been
executed by their author**. Run the unit tests first.

## Requirements
Python ≥ 3.9, `torch`, `numpy`, `scipy` (≥ 1.9, HiGHS), `pytest`. CUDA optional.

## Files
| File | Content |
|---|---|
| `envelope.py` | weighted-l1 metric, exact McShane–Whitney envelopes, L_A, member generator `min(h, max(ℓ, ψ))`, V_L |
| `bounds.py` | exact pairwise bounds B_pair + Shapley-weighted distance D_i; **v3:** `hybrid_vertices`, `shapley_full`, `geometric_rho_inf` (ρ^∞); stratified Monte Carlo certificates with `radius_scale` (negative control only) |
| `hypercube_lp.py` | exact LP on hypercube edges, intervals, sign certificates, exact \|φ\| pair margins (≤ 4 LPs), top-k membership, ordered top-k, verification of LP optima |
| `phase1.py` | experiments, technical GO and the separate scientific verdict |
| `tests.py` | deterministic unit tests (100), incl. references that share no code with what they test |

## Run
```bash
python -m pytest -q tests.py
python phase1.py --out phase1_quick --quick            # pilot (seed 0)

# confirmatory: register the scientific thresholds BEFORE the run (they go to config.json)
python phase1.py --out phase1_out \
    --sci-rho-target 0.8 --sci-min-frac 0.25 --sci-min-sign-rate 0.10

# replication on fresh seeds (seed 0 is reserved for the pilot)
for s in 2 3 4; do python phase1.py --out phase1_seed$s --seed $s --sci-rho-target 0.8 \
    --sci-min-frac 0.25 --sci-min-sign-rate 0.10; done

python phase1.py --out phase1_out --device cpu         # default --device auto = cuda if available
```
The threshold values above are placeholders: choose them **before** looking at any v3 output.
Data are drawn on CPU with CPU generators and then moved to the device, so the numbers are identical on
CPU and GPU. The LPs always run on CPU (scipy HiGHS). For d ≤ 5 the GPU gives little speed-up.

## Theory used by v3 (fixed A, x, r, weighted ℓ1)
1. **Exactness.** Pairwise bounds give the exact class interval (also for disjoint feature blocks as players).
2. **Monotonicity / nesting.** ρ_i(L) = Γ_i / (2Lδ_i) is non-decreasing in L; intervals are nested.
3. **Geometric limit.** ρ_i(L) → ρ_i^∞ = Σ_S w_S min(1, [d(z⁺,A) + d(z⁻,A)] / δ_i), and ρ_i(L) ≤ ρ_i^∞ ≤ min(1, D_i/δ_i).
   ρ^∞ is an **upper bound only**: f = u + v with anchors {(0,0), (1,1)} has ρ(L) = 1 − 1/L while ρ^∞ = 1.
4. **Optimistic ceiling.** For every valid L ≥ L_min = L_A(A): ρ(L) ≥ ρ(L_min), and the signs / top-k
   certified at L are a subset of those certified at L_min. No knowledge of L_true is needed.
5. **Local validity.** At L_V = Lip(f on hybrids ∪ A) the interval provably contains φ(f)
   (McShane extension of f restricted to hybrids ∪ A). L_min ≤ L_V ≤ L_true.

Chain reported per instance: ρ(L_min) ≤ ρ(L_V) ≤ ρ(L_true) ≤ ρ^∞.

## Outputs
| File | Content |
|---|---|
| `config.json` | the exact configuration, written before anything runs (pre-registration record) |
| `records.jsonl` | one line per (d, ρ, n_A, model, κ, x, r, mode); `LA_C` = L_A(C), `LA_used` = L_A(A) |
| `optimistic.jsonl` | **v3.** one line per (d, ρ, n_A, model, x, r, mode), **not** filtered by feasibility: ρ, signs, top-k at L_min, L_V (and L_true for `linear`, `unstable_id`), ρ^∞, κ_V = L_V / L_min, theorem checks |
| `stress.json` | saturated anchors, one-feature references, 2-D counterexample (+ fa at L_V, gated), u + v counterexample (gated), stable-OOD |
| `mc_coverage.json` | coverage on the least saturated feasible references (chosen by exact quantities before sampling), vacuity, negative control |
| `runtime.json` | pairwise vs LP vs Monte Carlo |
| `summary.json` | everything aggregated; `go` (technical) and `scientific` (usefulness) |

## Technical GO (`PHASE1_GO`)
All of the following must be `true`; a skipped part is `"not_evaluated"` and blocks GO:
- `stress_checks`: saturated, sparse reference, 2-D counterexample incl. fa inside at L_V, u + v closed form;
- `grid_correctness`: no violation above `tol` in `records.jsonl`;
- `optimistic_theorem_checks`: φ(f) inside at L_V (and at L_true), ρ monotone, ρ ≤ ρ^∞, nested signs / top-k, L_V ≤ L_true;
- `mc_coverage`: binomial test of failure rate > η not rejected at 0.01;
- `mc_non_vacuous`: median exact ρ < `mc_rho_max` (0.9) in both modes — otherwise the coverage test proves nothing;
- `mc_negative_control_detected`: with the radius multiplied by `mc_neg_required` (0.0), the test **must** reject.

## Scientific verdict (`scientific`, per d)
Needs `--sci-rho-target`, `--sci-min-frac`, `--sci-min-sign-rate` registered before the run.
- **S3 valid_and_useful:** frac(ρ(L_V) ≤ target) ≥ min_frac and certified-sign rate at L_V ≥ min_sign_rate;
- **S2 lost_to_model_roughness:** not S3, but frac(ρ(L_min) ≤ target) ≥ min_frac;
- **S1 not_identifiable:** otherwise — even the optimistic ceiling fails.

## Failure reporting
`failure_by_kappa`, `failure_by_d_kappa`, `failure_by_model_kappa`: infeasible rate, conditional violation
rate (among feasible) and total failure rate. Compare κ values with the total rate: per instance it is
non-increasing in κ, the conditional rate is not.

## Validity conditions for the Monte Carlo certificates
Candidate references fixed before sampling; `n_per_stratum` fixed in advance (no optional stopping);
L ≥ L_A of every anchor set used (instances violating this are recorded as `infeasible`).
`radius_scale < 1` gives **invalid** bounds by design and is used only as a negative control.

## Known limitations (not addressed in v3)
- Changing n_A redraws train (σ), C, R_candidate and x (`gen_of("data", seed, d, rho, nA)`): comparisons
  across n_A are population-level, not paired. A nested-anchor ablation with fixed x, r, σ is P1.
- Feature grouping and anchor design are P1.
