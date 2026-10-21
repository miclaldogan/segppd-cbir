# Results

Hardware: Arch Linux, 20-core CPU, RTX 4050 Laptop 6 GB (CUDA torch 2.14.1+cu130 for training; CPU torch for the baseline), seeds fixed. All numbers are read from CSVs under `artifacts/metrics/` (listed per table).

## 1. Mandated training-free baseline - Protocol A (primary cohort: 75 classes, 225 queries, gallery 1,731; `macro_ci.csv`)
| Descriptor + exact L2 | Hit@1 = P@1 | P@10 | Hit@10 | mAP | random P@K |
|---|---|---|---|---|---|
| A: 1000-D logits (raw) | 0.293 [0.23,0.36] | 0.170 [0.13,0.21] | 0.676 | 0.118 | 0.013 |
| B: 2048-D pooled (raw) | 0.391 [0.32,0.46] | 0.212 [0.18,0.25] | 0.756 | 0.145 | 0.013 |
B > A at every K (paired differences in `paired_A_minus_B.csv`). L2-normalised sensitivity: no consistent change (`sensitivity_l2norm.csv`). FAISS vs NumPy: identical top-100 for 222-224 of 225 queries (rest = near-ties, relative distance diff ~1e-6).

## 2. Primary scientific question - target-region occupancy (non-healthy queries; masks used only to stratify)
Frozen ResNet50 (A, 150 queries): small / medium / large P@1 = 0.368 / 0.353 / 0.348 (B desc.), CIs overlap; within-class Spearman(occupancy, P@10) = -0.003, +0.222, -0.203 over three seeds (not consistent).
Locked Protocol B, DINOv2-B ensemble (`protocolB_locked/lesion_by_bin.csv`): Hit@1 small 0.815 [0.73,0.89] (n=73, 36 classes), medium 0.827 [0.70,0.95] (38/28), large 0.806 [0.64,0.96] (39/24); within-class Spearman -0.03 [-0.21,0.16]. **The hypothesis "small targets retrieve worse" is not supported by this data** (grouped, observational; occupancy is confounded with disease type; absence of evidence at this sample size, not proof of no effect).
Oracle ROI (bbox of GT mask +10% margin, query and gallery; privileged information): paired change in P@10 for 2048-D = +0.016 [-0.004,0.034]; essentially no gain, i.e. little support for "background dilution" as the main failure cause (`oracle_roi_minus_whole_paired.csv`).

## 3. Protocol B locked test (225 never-used queries, 75 classes, gallery 809; `protocolB_locked/`)
| System | Hit@1 [95% class-CI] | P@10 | Hit@10 | mAP |
|---|---|---|---|---|
| ResNet50 frozen 1000-D (mandated, raw) | 0.262 [0.20,0.33] | 0.140 | 0.662 | 0.136 |
| ResNet50 frozen 2048-D (mandated, raw) | 0.320 [0.25,0.39] | 0.164 | 0.711 | 0.165 |
| ResNet50 fine-tuned + ArcFace (existing system, 1 seed) | 0.511 [0.44,0.59] | 0.367 | 0.827 | 0.387 |
| DINOv2-S frozen CLS | 0.538 [0.46,0.61] | 0.368 | 0.938 | 0.388 |
| DINOv2-B frozen CLS | 0.627 [0.55,0.69] | 0.400 | 0.951 | 0.452 |
| DINOv2-S last-4 + ArcFace (1 seed) | 0.760 [0.70,0.82] | 0.547 | 0.964 | 0.603 |
| DINOv2-S last-4 + Proxy-Anchor (3 seeds: .764/.742/.738) | mean 0.748 | ~0.58 | ~0.96 | ~0.645 |
| DINOv2-S last-4 + SupCon (3 seeds: .693/.720/.738) | mean 0.717 | ~0.59 | ~0.95 | ~0.659 |
| **DINOv2-B last-4 + SupCon (3 seeds: .813/.796/.813) - declared headline** | **mean 0.807, class-CI [0.756,0.856]** | 0.651 | 0.967 | 0.742 |
| DINOv2-B + SupCon, 3-seed ensemble | 0.822 [0.77,0.88] | 0.662 | 0.969 | 0.754 |
| ... + 0.2 DINOv2-S ensemble (dev-selected w) | 0.822 [0.76,0.88] | 0.662 | 0.969 | 0.756 |
| ... + 0.1 ResNet50-ft (dev-selected w) | 0.840 [0.78,0.89] | 0.664 | 0.964 | 0.757 |

**Does it reach Hit@1 >= 0.80? The point estimate does (0.807 headline; 0.813/0.796/0.813 per seed), but the 95% interval [0.756, 0.856] includes values below 0.80, so >=0.80 is not established with statistical confidence on 225 queries.**
Paired (same queries, class-bootstrap): DINOv2-B+SupCon vs existing ResNet50 system P@1 +0.28..+0.30 (CIs [0.20,0.39]), P@10 +0.28..+0.29 (CIs ~[0.24,0.33]); vs frozen DINOv2-B P@1 +0.196 [0.12,0.27] (ensemble); B vs S (SupCon s0) P@1 +0.120 [0.062,0.178]. Fusions vs the B-ensemble: S +0.000 [-0.027,0.027]; ResNet +0.018 [-0.004,0.044] -> **no significant hybrid gain**. SupCon vs ArcFace (S): P@10 +0.035 [0.015,0.052] but P@1 -0.067 [-0.129,-0.004]; vs Proxy-Anchor P@1 -0.071 [-0.138,-0.009] (loss was chosen on dev mAP; on the test set SupCon is better for deep ranks, worse at rank 1 for DINOv2-S; ArcFace/Proxy-Anchor were not run on DINOv2-B).
Class coverage (3 queries/class): DINOv2-B ensemble has 0 classes with Hit@1=0 and 45/75 perfect (3/3); the existing ResNet50 system has 14 classes at 0 and 18 perfect. Weakest classes (3-seed mean Hit@1): lemon_healthy 0.11, cotton_fusarium wilt 0.22, hops_nutrient deficiency, cassava_bacterial blight, lemon_sooty mold, tomato_mold (0.33). Dominant confusions are same-genus/healthy pairs (apple_healthy vs plum_healthy, lemon_sooty mold vs orange_canker, eggplant vs tomato blights).
Dev vs locked test agree (DINOv2-B dev Hit@1 0.80-0.83 -> test 0.80-0.81), so dev-based early stopping did not visibly overfit.

## 4. Complementary errors (dev, 150 queries, ResNet50-ft vs DINOv2-S-ft)
both correct 70, only ResNet 9, only DINOv2 35, both wrong 36. Oracle union (NOT a system) 0.76 vs 0.70 for DINOv2-S alone -> little headroom from ResNet50. Dev fusion curves: `protocolB_dev_fusion_*.csv`.

## 5. Cost (`cost_table.csv`, `retrieval_latency.csv`)
| Model | trainable params | train time (to early stop) | peak VRAM | embed (ms/img, incl. flip) | exact-L2 query latency p50 (809 gallery) | descriptor |
|---|---|---|---|---|---|---|
| ResNet50 frozen | 0 | 0 | - | ~40 (CPU, 92 s/2,263) | 2.35 ms | 2048-D fp32 (8 KB/img) |
| ResNet50 ft | 22.6 M | 660 s | 1.95 GB | 10.0 | 0.26 ms | 256-D (1 KB) |
| DINOv2-S last-4 | 7.2 M | 230-355 s | 1.34 GB | 7-8 | 0.26 ms | 256-D |
| DINOv2-B last-4 | 28.6 M | 600-770 s | 2.98 GB | 11.2 | 0.26 ms | 256-D (ensemble 768-D: 0.75 ms) |
Embedding-space diagnostics (intra/inter distances, nearest-neighbour margins): `embedding_diagnostics.csv`.

## 6. Negative / null results (kept on purpose)
Hand-crafted vectors (HSV/LBP/HOG/GLCM/Gabor): best block mAP 0.05 vs 0.154 deep; fusion hurt. Texture/colour *maps* through ResNet: CLAHE ~ RGB alone (mAP 0.160 vs 0.154); adding any map to RGB through a trained head: <=+0.01 (noise) or negative; attention fusion negative (`handimg_dev.csv`). Oracle ROI: ~0. L2 normalisation: ~0. ResNet50 in the hybrid: ~0.
Note: an earlier shell bug launched three DINOv2-S runs with default args; they are kept as `MISLABELED_rep1-3` replicates (same config: mAP 0.597-0.606, Hit@1 0.70-0.72 on dev) and are **not** used as the ArcFace/Proxy-Anchor/head-only results, which were rerun with correct arguments.

## 7. Exploratory Protocol A results for trained systems (A's test queries were already seen; not independent)
ResNet50 layers 3-4 fine-tuned + ArcFace on the full gallery: Hit@1 0.667-0.693, P@10 0.642-0.658 (3 seeds, `ext_S2_*.csv`); training-free multi-block + query expansion: Hit@1 0.453, P@10 0.260; ArcFace head on frozen multi-block features: Hit@1 0.56-0.60, P@10 0.54-0.55. DINOv2 was not evaluated on A's test queries.


## Protocol C: second locked query set (added after Protocol B was consumed)

**Construction** (`src/segppd_cbir/protocol_c.py`, seed 777, audit in `artifacts/audits/protocolC_audit.json`): 142 queries (2 per class for 67 classes, 1 for 8 classes with a small gallery) are taken from Protocol B's `gallery_B` and removed from it, leaving a 667-image `gallery_C`. No model was trained on these images (training used `train_B` only), so the frozen Protocol B models are valid with no retraining. Image-, hash- and group-level overlap with gallery and train is 0; nearest pHash distance 14 (outside the duplicate rule). Candidates (`artifacts/metrics/locked_candidates_C.json`) are identical to Protocol B's and were frozen before the run; the declared headline did not change. C was evaluated once.

**Result (Hit@1, macro, class-bootstrap 95% CI)** - `artifacts/metrics/protocolBC_hit1_summary.csv`, figure `artifacts/figures/protocolBC_hit1.png`

| System | B (225 q) | C (142 q) | B+C pooled (367 q) |
|---|---|---|---|
| ResNet50 frozen 2048-D | 0.320 | 0.333 | 0.329 [0.27, 0.39] |
| ResNet50 fine-tuned + ArcFace | 0.511 | 0.500 | 0.511 [0.44, 0.58] |
| DINOv2-B frozen | 0.627 | 0.567 | 0.607 [0.54, 0.67] |
| **DINOv2-B last-4 + SupCon, mean of 3 seeds (headline)** | **0.807** [0.756, 0.856] | **0.767** [0.689, 0.840] | **0.794** [0.741, 0.844] |

Reading: the ranking of systems reproduces on fresh queries and the gap to the ResNet50 systems is large and stable. The headline Hit@1 is 0.77 on C (single seeds 0.760-0.773), lower than on B, and the pooled estimate is 0.794 with a CI that includes 0.80 but also 0.74. **The >= 0.80 target is therefore not confirmed**; the honest statement is Hit@1 approximately 0.79-0.80. C is small (142 queries), so its CI is wide; the sampling difference between B and C is within noise.

**Precision-Recall curves** (full ranking, 11-point interpolated precision, macro over classes): `artifacts/figures/protocolB_PR_curve.png`, `protocolC_PR_curve.png`; numbers in `artifacts/metrics/protocol{B,C}_locked/pr_curves_11pt.csv`. Mean 11-point interpolated precision on B: ResNet50 frozen 0.18, ResNet50 fine-tuned 0.40, DINOv2-B frozen 0.48, DINOv2-B + SupCon 0.76.


## External validation: PlantSeg v2 (one shot, protocol pre-registered)

Source: PlantSeg v2 (Zenodo 13958858, CC BY-NC-ND 4.0, md5 verified; used for evaluation only, not redistributed). Registered before any model saw these images: `artifacts/external/external_protocol_registered.json`.

**Leakage finding.** PlantSeg and SegPPD-101 share web-photo sources: 867 of 11,458 PlantSeg images (7.6%) are copies or crops of a SegPPD image (229 by hash rule; the rest by frozen DINOv2-B cosine >= 0.85; threshold chosen by visual review, 0.92 missed cropped copies). All flagged images were removed (`artifacts/external/plantseg_leak_screen.{csv,json}`). Undetected, more heavily edited copies cannot be excluded.

**Set.** 11 SegPPD classes map 1:1 to PlantSeg diseases (strict tier: apple rust/scab, banana cordana leaf spot, bean rust, coffee leaf rust, cucumber powdery mildew, grape black rot, strawberry leaf scorch, tomato bacterial leaf spot/leaf mold/late blight); 1,485 screened queries. Loose-tier classes (4, name mismatch such as orange vs citrus) are secondary. Retrieval is against SegPPD `gallery_B` (809 images, all 75 classes, not restricted to the 11). Models: the declared headline config retrained with checkpoints (seeds 0-2; dev mAP 0.748/0.751/0.752, matching the original runs).

| System | External Hit@1, strict 11 classes (class-CI / image-CI) | Same 11 classes in-domain (Protocol B) |
|---|---|---|
| ResNet50 frozen 2048-D | 0.149 [0.09, 0.21] / [0.13, 0.17] | 0.303 |
| DINOv2-B frozen | 0.484 [0.31, 0.68] / [0.46, 0.51] | 0.576 |
| **DINOv2-B last-4 + SupCon, 3-seed mean** | **0.608 [0.47, 0.76] / [0.59, 0.63]** (seeds 0.609/0.617/0.598) | 0.778 (C: 0.758) |

Loose tier (4 classes, 559 images): headline 0.633 [0.48, 0.82]. Files: `artifacts/metrics/external_plantseg/`, figure `artifacts/figures/external_plantseg_hit1.png`.

Reading: the system ranking reproduces (ResNet50 << frozen DINOv2-B << trained DINOv2-B), but absolute accuracy drops about 17 points versus the same classes in-domain, and the 0.80 level is **not** reached externally (0.61). Part of the drop is the removal of easy near-copies, part is domain shift (full-resolution web photos squashed to 256x256, different framing); the two are not separable here. Per-class results vary widely (grape black rot 1.00 in-domain vs 0.31 external; strawberry leaf scorch 1.00 vs 1.00). Only 11 closed-set classes, no healthy classes: this is a robustness check, not a benchmark of the full 101-class task.
