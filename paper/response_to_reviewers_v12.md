# LipID v1.2 — Response to reviewers (skeleton)

Fill every `[...]` only from `results_v12/numbers_v12.csv` / `results_v12/crosswalk_v12.md` after the runs. Keep the
v1.1 history (seeds 1–4 exploration, 8–11 confirmation, C2b/C4b failures, C5 specification error) exactly as reported.

---

**Q1 — Practical Lipschitz calibration without an oracle.**
We kept the operational answer of v1.1 (Sec. 5.1, App. H: certificates from generator parameters / model weights,
independent of anchor labels and hybrid responses) and made the access model explicit (new paragraph in Sec. 5.1): the
auditor receives anchor outputs and one certified scalar L from the model owner or an independent verifier; L_V is a
validation device only; local certificates are valid only on a certified domain containing all anchors and hybrids. We do
not claim a general oracle-free calibration procedure. *Changes:* Sec. 5.1 paragraph "Information access for calibration".

**Q2 — Metric / scale sensitivity.**
New MET-02 (Sec. 7.2, Table [met02]): MAD, SD and raw-unit scales with L re-certified per metric, on a mixed-marginal
design where the choice matters, and on the diabetes audit. We also state and verify the invariance under a common
rescaling of σ and L (max deviation [met-02.max_invariance]). [Summary of the observed changes.]

**Q3 — Joint identification.**
(i) Proposition 4.2: the joint identified set is exactly a linear image of an explicit polytope (exact LP formulation;
not polynomial in m). (ii) Corollary 4.3: closed-form sharp bounds for every pairwise difference φ_i − φ_j, attained.
(iii) Proposition 4.4: exact signed-ranking and top-k certificates; exact absolute-ranking certificates from Δ and Σ.
Validation (JNT-01): closed form = finite-hybrid LP (all vertex pairs) within 1e-8 on [N] comparisons, plus a
solver-free attainment check. Empirical value (JNT-02, Sec. 6.x): certified signed pairs rise from [..] to [..];
seed-level gain [stat-01.P4.mean] [CI]. [If small near saturation: say so; the result is then mainly a mathematical
clarification.] Absolute rankings (JNT-03): [..].

**Q4 — Conditional / manifold Shapley.**
New discussion (Sec. 2 and Sec. 8): these games define different estimands; we identify the three properties of the
baseline game our proofs use and explain which fail for conditional and manifold games. No numerical head-to-head
comparison is claimed. We list the extension as open.

**Q5 — Scalability of approximations.**
New SCL-02 (Sec. 7.1): the v1.1 benchmark is kept as a negative result; a non-saturated benchmark (fixed non-adaptive
hybrid-local anchor rule chosen on a pilot, evaluated on independent instances) separates approximation error from
saturation. Valid variants (exhaustive small strata, empirical Bernstein, sampling without replacement) are compared at
matched budgets with coverage, excess width, retained certificates and cost. [Results, including a negative outcome
if all remain near-vacuous.]

**Q6 — Saturation certificates.**
New Prop. E.2 (relaxation chain) and CERT-02: the gap between exact AIW and the geometry/range certificate splits into
direction asymmetry, slope magnitude and decoupling; [dominant class and shares]. No new certificate is proposed because
the informative pairwise geometry bound is costlier than the exact envelope.

**Q7 — Inferential uncertainty and reproducibility.**
STAT-01: [S] new independent seeds under a protocol locked before running (lock file, commit and timestamp in the
artifact; RNG stream keys checked to be disjoint from seeds 1–11); four pre-specified comparisons with seed-level 95% t
intervals, seed bootstrap and Holm adjustment (Table [stat01]). REP-01: public artifact at commit [..], DOI [..];
clean-environment rerun of [scope] reproduced [n/n] tables within rtol 1e-9 (verification report); the scope of the
independent rerun is stated explicitly.
