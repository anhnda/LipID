# AIW — Phase 1 (mathematical validation)

Attribution Identifiability Width for reference-conditioned Shapley values over the
anchor-constrained Lipschitz class F_L(A), weighted-l1 metric.

**This code has not been executed by its author.** Run the unit tests first.

## Requirements
Python ≥ 3.9, `torch`, `numpy`, `scipy` (≥ 1.9, HiGHS), `pytest`.

## Files
| File | Content |
|---|---|
| `envelope.py` | weighted-l1 metric, exact McShane–Whitney envelopes, L_A, member generator `min(h, max(ℓ, ψ))`, V_L |
| `bounds.py` | exact pairwise bounds B_pair + Shapley-weighted distance D_i; stratified Monte Carlo certificates (Hoeffding / empirical Bernstein, union bound over K references × d features) with no model evaluation at hybrids |
| `hypercube_lp.py` | exact LP on hypercube edges (Theorem 2), intervals, sign certificates, exact \|φ\| pair margins (≤ 4 LPs), top-k membership (k(d−k) pairs), ordered top-k, independent verification of LP optima |
| `phase1.py` | experiments + GO / NO-GO summary |
| `tests.py` | deterministic unit tests |

## Run
```bash
python -m pytest -q tests.py
python phase1.py --out phase1_quick --quick     # smoke run
python phase1.py --out phase1_out               # full pre-registered grid
python phase1.py --out phase1_out --skip runtime
```

## What Phase 1 checks
1. Γ_LP ≤ B_pair ≤ 2Lδ_i and LP interval ⊆ pairwise interval.
2. Exact members of F_L(A) (ℓ, h, clamped random Lipschitz ψ): 100 % coverage of intervals, sign and top-k certificates; members are re-verified to be L-Lipschitz on hybrids ∪ anchors.
3. LP optima satisfy Lipschitz on **all** vertex pairs and to every anchor directly (Theorem 2 check).
4. Monte Carlo family-wise failure rate vs η (binomial test), per mode and per method.
5. Stress tests: saturated anchors (closed form), one-feature references, the 2-D counterexample, stable-OOD vs local variance.
6. Runtime: pairwise vs LP vs Monte Carlo.

Empirical transfer (frozen-model φ outside the LP interval, V_L > 0) is reported but is **not** a theorem check.

## GO criteria (pre-registered in `Config`)
- deterministic correctness: no violation above `tol`, all stress checks pass;
- MC coverage: binomial test of failure rate > η not rejected at 0.01;
- LP tightness: median Γ_LP / B_pair ≤ 0.8 **or** certified-sign gain ≥ 10 points.

## Validity conditions for the Monte Carlo certificates
Candidate references fixed before sampling; `n_per_stratum` fixed in advance (no optional stopping);
L ≥ L_A of every anchor set used (instances violating this are recorded as `infeasible`).
