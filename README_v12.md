# LipID v1.2 — Kế hoạch coding cho bản revision (targeted revision)

> **Trạng thái:** toàn bộ code dưới đây được viết theo yêu cầu *không chạy* — chưa có lệnh nào được thực thi, kể cả test.
> Bước đầu tiên khi bạn chạy là `STEPS="tests_v12" bash scripts/reproduce_revision.sh`. Nếu một test fail, sửa trước
> khi chạy bất kỳ thí nghiệm nào (các test được thiết kế để bắt lỗi toán/RNG trước khi tốn GPU).

Code v1.1 (`lipid/*.py` cũ, `scripts/*` cũ, các module lịch sử ở root) **không bị sửa**. Mọi thứ mới chỉ *import*
chúng. `tests/test_pipeline_regression.py` vẫn là khóa an toàn cho pipeline cũ.

---

## 1. Phân tích bản kế hoạch revision — những điểm cần chỉnh khi chuyển sang code

| # | Điểm trong kế hoạch | Nhận xét khi hiện thực hóa | Quyết định trong code |
|---|---|---|---|
| 1 | A2: sharp pairwise bounds | Chứng minh đi được: hai họ đỉnh {P_S}, {Q_S} là hai mặt rời nhau; trong cùng mặt khoảng cách bất biến tịnh tiến, chéo mặt thêm đúng δ_i+δ_j. Đúng là *corollary*. | `paper/v12_theory.tex` (chứng minh đầy đủ), `lipid/pairwise.py` (closed form + nhãn cực trị dùng trong chứng minh) |
| 2 | A3: absolute ranking từ Δ và Σ | Điều kiện `(minΔ>0 ∧ minΣ>0) ∨ (maxΔ<0 ∧ maxΣ<0)` là **điều kiện cần và đủ** (tập (Δ,Σ) khả thi là đa diện lồi compact ⇒ tránh hai trục thì nằm trong một góc phần tư). Σ không có closed form (trọng số đổi dấu trên 4 mặt) ⇒ cần 2 LP/cặp. | `Prop. 4.4(c)`, `lipid/pairwise.abs_exact_cert`, `lipid/joint.JointPolytope.pair_sum` |
| 3 | Top-k | ∀g∀(i,j) giao hoán ⇒ top-k qua pairwise là **chính xác**; tập top-k (nếu có) duy nhất = {i: wins ≥ m−k}. Bản marginal chỉ là điều kiện đủ. | `certified_topk`, `top1` |
| 4 | JNT-01 tolerance 1e-8 | HiGHS mặc định chỉ ~1e-7 ⇒ phải scale bài toán và siết tolerance (1e-10). Quan trọng hơn: kiểm chứng *không phụ thuộc solver* — dựng nhãn cực trị, kiểm tra Lipschitz trên tập hữu hạn, McShane extension, rồi tính Δ bằng **công thức Shapley đầy đủ** (không qua identity). LP dùng **tất cả cặp đỉnh** (không dùng phép rút gọn cạnh) để tránh vòng lặp logic với Prop. J. | `scripts/jnt01_validate_pairwise.py` |
| 5 | JNT-02 "matched comparison" | Hai script khám phá đính kèm dùng RNG và L khác Section 6 (L = 1.15·max\|w\|σ, `default_rng`) ⇒ không matched; chỉ giữ vai trò exploratory. Runner mới tái tạo đúng RNG stream của `nested_anchors_v2` để join từng dòng với `records`. Marginal và pairwise dùng **cùng** tolerance chuẩn hóa theo **cùng** c_ij ⇒ "pairwise ⊇ marginal" đúng theo cấu trúc, không bị nhiễu do tol. | `lipid/instances.py`, `lipid/pairwise_pipeline.py`, `scripts/run_pairwise.py` |
| 6 | STAT-01 | Seed là đơn vị; 24 seed mới (101–124) — script lock kiểm tra key RNG (crc32) không trùng seed 1–11. P3 cho `linear`/`unstable_id` là **zero cấu trúc** (L_cert = L*) ⇒ báo cáo nhưng không test. P4 ≥ 0 theo định lý ⇒ CI đo độ lớn. Holm trong từng family. | `protocols/v12_preregistration.json`, `scripts/lock_protocol.py`, `lipid/stats.py`, `scripts/stat01_analyze.py` |
| 7 | MET-02 | **Vấn đề lớn:** với thiết kế Section 6 (biên Gaussian phương sai 1 giống nhau) thì MAD, SD và unit gần như trùng nhau sai khác một hằng số chung ⇒ theo tính bất biến (σ→cσ, L→cL) các khoảng **giống hệt** ⇒ thí nghiệm vô nghĩa. Tương tự với diabetes của sklearn (feature đã chuẩn hóa cùng SD ⇒ unit ≡ SD). Cần thiết kế có biên dị chủng (Gaussian copula: gauss / t3 / lognormal / binary × thang log 0.25–4). | `scripts/met02_scale_sensitivity.py` |
| 8 | SCL-02 hybrid-local | Phải là quy tắc **không thích nghi, không dùng f** ngoài anchor đã hỏi ⇒ không lấn sang P2 (active). Exact m=24 không khả thi với code enumeration hiện tại (Z: 2^24×24 double ≈ 3.2 GB + index cạnh) ⇒ exact tới m=20, m=24 chỉ đo MC. | `scripts/scl02_nonsaturated.py`, `lipid/mc_v2.py` |
| 9 | Variance-sensitive MC | Empirical Bernstein (Maurer–Pontil 2009, Thm 4) hợp lệ đồng thời qua union bound; sampling without replacement dùng bound Serfling; strata nhỏ liệt kê hết (eps=0); số event union đếm trên strata được *sample* — xác định trước theo (m, n_k). | `lipid/mc_v2.py` |
| 10 | CERT-02 | Phát hiện một đồng nhất thức sạch: relaxation pairwise-\|slope\| **bằng đúng** min(c⁺, c⁻) trên từng cạnh (vì −\|t\| = min(t,−t) tách được). Từ đó có chuỗi exact ≥ ABS ≥ max(GEO, RANGE) ≥ max(T3, RANGE) = cert v1.1 ⇒ khoảng cách tách thành 3 phần không âm: bất đối xứng hướng, độ lớn slope, decoupling. GEO tốn O(\|A\|²) — đắt hơn exact ⇒ đúng tinh thần "không đề xuất certificate mới". | `lipid/certdiag.py`, `scripts/cert02_diagnostics.py`, `Prop. E.2` |
| 11 | Q1 local certificate | Không code (tùy chọn trong kế hoạch); chỉ làm rõ access model trong text. | `paper/v12_discussion.tex` |

---

## 2. Bản đồ: reviewer → hạng mục → code → output → vị trí trong paper

| Reviewer | Hạng mục | Loại | Code | Output chính (`results_v12/`, `figures_v12/`) | Vị trí paper |
|---|---|---|---|---|---|
| Q3 | A1 Prop. J (joint polytope) | theory | `lipid/joint.py` | — | Sec. 4 + App. A.2 |
| Q3 | A2 Cor. (pairwise sharp) | theory | `lipid/pairwise.py` | — | Sec. 4 + App. A.3 |
| Q3 | A3 Prop. (ranking exact) | theory | `lipid/pairwise.py` | — | Sec. 4 + App. A.4 |
| Q3 | JNT-01 | verification | `scripts/jnt01_validate_pairwise.py` | `jnt01_table.tex`, `jnt01_summary.csv`, `jnt01_edgecases.csv`, `jnt01_rows.csv.gz` | App. A (bảng) |
| Q3 | JNT-02 | descriptive (+P4 trong STAT-01) | `run_pairwise.py`, `jnt02_analyze.py` | `jnt02_main.tex`, `jnt02_subgroups.tex`, `fig_jnt02_certification.pdf` | Sec. 6 mới "Joint ranking information" |
| Q3 | JNT-03 | descriptive | `jnt03_absolute_lp.py` | `jnt03_table.tex` | Supplement |
| Q7 | STAT-01 | **confirmatory** | `lock_protocol.py`, `stat01_analyze.py` | `stat01_primary.tex`, `stat01_effects.csv`, `uncertainty_methods_v12.md` | Sec. 6.1 + bảng mới |
| Q7 | REP-01 | verification | `verify_release.py`, `clean_env_reproduce.sh`, `make_release.sh`, `Dockerfile` | `verification_report.md`, `release/*.tar.gz` | Reproducibility |
| Q2 | MET-02 | descriptive | `met02_scale_sensitivity.py` | `met02_table.tex`, `met02_real.tex`, `fig_met02.pdf` | Sec. 7.2 |
| Q5 | SCL-02 | descriptive | `scl02_nonsaturated.py`, `lipid/mc_v2.py` | `scl02_table.tex`, `fig_scl02_excess.pdf` | Sec. 7.1 |
| Q6 | CERT-02 | diagnostic | `cert02_diagnostics.py`, `lipid/certdiag.py` | `cert02_table.tex`, `fig_cert02_classes.pdf` | Sec. 7 + App. E.2 |
| Q4 | discussion | text | — | — | Sec. 2, 8 (`paper/v12_discussion.tex`) |
| Q1 | discussion | text | — | — | Sec. 5.1 |

Tích hợp manuscript: `scripts/build_revision.py` sinh `results_v12/numbers_v12.tex` (macro `\lipnum{key}` cho mọi con
số trong text, kèm định nghĩa + file nguồn trong `numbers_v12.csv`), `main_tables_v12.tex`, `crosswalk_v12.md` (trạng
thái complete / partial / NOT RUN cho từng hạng mục) và `build_report_v12.json` (`issues` phải rỗng trước khi submit).

---

## 3. Thứ tự chạy (theo Phase của kế hoạch)

```bash
# Phase 1 — theory validation (CPU đủ)
STEPS="tests_v12 jnt01" bash scripts/reproduce_revision.sh
#   điều kiện qua: pytest pass; results_v12/jnt01_manifest.json -> verdict.accepted = true

# Phase 2 — main empirical evaluation (GPU), dùng lại runs/explore, runs/confirm của v1.1
STEPS="pairs_explore pairs_confirm jnt02 jnt03" bash scripts/reproduce_revision.sh
#   điều kiện qua: jnt02_integrity.json toàn 0; jnt03 checks.exact_weaker_than_prop2 = 0

# Phase 3 — confirmatory + reproducibility
#   (a) sửa protocols/v12_preregistration.json nếu cần -> git commit -> lock (cây sạch!) -> commit file lock
STEPS="lock_stat01" bash scripts/reproduce_revision.sh
git add protocols/locks/STAT-01.lock.json && git commit -m "lock STAT-01"
#   (b) chạy batch độc lập (lock được verify trước khi chạy; mọi thay đổi sau lock bị ghi là deviation)
STEPS="stat01_nested stat01_pairs stat01_analyze jnt02_final" bash scripts/reproduce_revision.sh
#   (c) tái lập trong môi trường sạch + đóng gói
STEPS="clean_env verify release" bash scripts/reproduce_revision.sh

# Phase 4 — targeted robustness (chỉ những gì trả lời reviewer)
STEPS="met02 scl02_pilot scl02_eval scl02_analyze cert02" bash scripts/reproduce_revision.sh

# Phase 5 — manuscript integration
STEPS="build_v12" bash scripts/reproduce_revision.sh     # đọc results_v12/crosswalk_v12.md
```

Mỗi bước ghi `logs/v12_<step>.log`; `logs/v12_timeline.log` lưu thời điểm bắt đầu. Script dừng ở bước fail đầu tiên.
Thời gian chạy **chưa được đo** (không chạy gì). Ước lượng tương đối: `run_pairwise` ≈ chi phí một seed của
`run_nested` (phải tái tạo L_V) — xem `runs/confirm/blocks_seed*.csv` của v1.1 để ước lượng; STAT-01 = 24 seed ×
(run_nested + run_pairwise). Nếu ngân sách không đủ cho 24 seed, giảm số seed **trong protocol trước khi lock**;
dưới 10 seed thì `stat01_analyze.py` tự động chỉ báo cáo min–max (không tạo CI).

---

## 4. Mô tả từng file mới

### Thư viện (`lipid/`)
| File | Mục tiêu | Ghi chú kiểm chứng |
|---|---|---|
| `pairwise.py` | Closed-form sharp bounds cho φ_i−φ_j; marginal Theorem 1 dạng raw; chứng chỉ signed / top-k / absolute (Prop. 2 và bản exact); nhãn cực trị của chứng minh | batched theo nhiều mức L; chạy trên GPU |
| `joint.py` | `JointPolytope`: LP chính xác của Prop. J (`edges`) và bản tổng quát mọi cặp đỉnh (`all_pairs`); kiểm tra vi phạm độc lập; McShane | HiGHS, scale theo L·D, tol 1e-10 |
| `instances.py` | Tái tạo chain Section 6 với **đúng** RNG stream/thứ tự rút của `nested_anchors_v2` | `tests/test_v12_instances.py` so từng dòng với `lipid.pipeline` |
| `pairwise_pipeline.py` | JNT-02: một lần tính envelope cho 4 mức L, ghi bảng cặp + bảng instance | cờ `contained`, `v_dom` phải = 0 |
| `mc_v2.py` | Hoeffding (bản paper + bản đếm event chặt), empirical Bernstein, Serfling WOR, liệt kê strata nhỏ | `hoeffding_paper` tái tạo `lipid.mc` từng draw |
| `certdiag.py` | Chuỗi relaxation exact ≥ ABS ≥ max(GEO,RANGE) ≥ UNI; phân loại failure mode; binding-pair diagnostics | tái tạo `rho_com/low_com/range_com/uni_com` của v1.1 |
| `stats.py` | t-interval theo seed, bootstrap theo seed, Holm, Bonferroni | so với scipy |
| `lock.py` | Lock protocol: sha256 protocol + code, git commit, timestamp, kiểm tra trùng key RNG | không bao giờ ghi đè lock |
| `report.py` | Bảng LaTeX, tóm tắt theo seed (min–max hoặc CI), registry con số trích dẫn | |

### Scripts (`scripts/`)
| Script | Hạng mục | Lệnh |
|---|---|---|
| `jnt01_validate_pairwise.py` | JNT-01 | `--workers 8` (CPU, LP song song); exit 1 nếu vượt tolerance |
| `run_pairwise.py` | JNT-02 runner | `--seeds 8 9 10 11 --tag confirm --out runs/pairs_confirm` |
| `jnt02_analyze.py` | JNT-02 bảng/hình | `--run confirm=runs/pairs_confirm ...` |
| `jnt03_absolute_lp.py` | JNT-03 | tập con cố định theo index (seed 8–11, x_id 0–1) |
| `lock_protocol.py` | STAT-01 lock | `lock` / `verify` |
| `stat01_analyze.py` | STAT-01 | chỉ partition `stat01` được suy diễn; 1–4, 8–11 mô tả |
| `met02_scale_sensitivity.py` | MET-02 | synthetic dị chủng + diabetes; kiểm tra bất biến |
| `scl02_nonsaturated.py` | SCL-02 | `pilot` (chạy 1 lần) → `eval` → `analyze` |
| `cert02_diagnostics.py` | CERT-02 | d = 5,6; n = 1024, 4096 |
| `verify_release.py` | REP-01 | manifest checksums + so sánh bảng tái lập (rtol 1e-9) + phạm vi |
| `clean_env_reproduce.sh` | REP-01 | `git archive HEAD` → venv/Docker mới → chạy các bước khai báo → so sánh |
| `make_release.sh` | REP-01 | tarball + SHA256SUMS (từ chối khi cây bẩn) |
| `reproduce_revision.sh` | điều phối | biến `STEPS`, `DEVICE`, `WORKERS` |
| `build_revision.py` | tích hợp paper | numbers_v12.tex, crosswalk, build report |

### Tests (`tests/test_v12_*.py`)
`pairwise` (identity, closed form = LP all-pairs và edges, attainment không cần solver, dominance, null players, point
identification, top-k, absolute exact vs phương pháp 4-LP cũ), `joint` (Shapley matrix, marginal LP = Theorem 1, edge
reduction = all-pairs theo hướng ngẫu nhiên), `mc_certdiag` (tái tạo `lipid.mc`, liệt kê đầy đủ = exact, WOR, độ phủ,
đồng nhất thức ABS = min(c⁺,c⁻) bằng brute force, thứ tự chuỗi), `instances` (khớp từng dòng với pipeline v1.1),
`stats_lock` (t/Holm/bất biến thang đo/không trùng key RNG).

### Paper (`paper/`)
`v12_theory.tex` (Prop. J, Cor. pairwise, Prop. ranking, Remark Σ, Prop. E.2 + chứng minh), `v12_experiments.tex`
(khung text dùng `\lipnum{...}`; chỗ `[...]` chỉ viết sau khi có kết quả), `v12_discussion.tex` (Q4, Q1, bốn thuộc
tính), `response_to_reviewers_v12.md` (khung trả lời Q1–Q7).
**Cần map label:** các `\ref` trong file tex mới giả định label `thm:sharp`, `prop:certs` (Prop. 2), `eq:envelopes`
(eq. 7), `eq:directional` (eq. 10), `prop:geomrange` (Prop. 7), `prop:optimistic` (Prop. 6), `tab:calcert` — đổi theo
label thật trong `lipid_v1_1.tex`.

---

## 5. Tiêu chí hoàn thành (definition of done, kiểm được bằng máy)
1. `pytest tests/` pass (cũ + mới).
2. `jnt01_manifest.json: verdict.accepted = true` (0 so sánh vượt 1e-8; 0 vi phạm attainment).
3. `jnt02_integrity.json`, `jnt03` checks, `cert02_checks.json`, `met02_checks.json`: mọi bộ đếm vi phạm = 0.
4. `stat01_manifest.json: lock.ok = true` (nếu false → kết quả bị gắn nhãn deviation trong bảng, không được giấu).
5. `verification_report.md: PASS`, với phạm vi rerun ghi rõ.
6. `build_report_v12.json: issues = []`, `crosswalk_v12.md` không còn hạng mục "NOT RUN" cho P0.

## 6. Ngoài phạm vi (P2) — không code
Active query acquisition; mở rộng định lý sang conditional/manifold Shapley (chỉ thảo luận); local Lipschitz
certificates cho `fa`/neural net (tùy chọn Q1, chưa làm).
