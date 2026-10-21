# Paper-matched CBIR benchmark vs ChatLeafDisease: protocol and first results (Phases 2-5)

Protocol registered **before** any model output on these queries: `artifacts/chatleaf/benchmark_registered.json`. Code: `src/segppd_cbir/chatleaf_benchmark.py`, `chatleaf_report.py`; results: `artifacts/metrics/chatleaf_matched/` (`trackS_*`, `trackT_*`, `summary_all.json`); figure `artifacts/figures/chatleaf_matched.png`. Fixed models only: DINOv2-B last-4-blocks + SupCon, 3 retrained seeds (`D5ck_*`, trained on SegPPD `train_B` = 772 labelled images, 75 classes) and frozen DINOv2-B CLS as a control. No PlantSeg label was used to train, tune or select anything in Track S; nothing was tuned on the earlier 1,485-query evaluation. Retrieval = exact L2, prediction = class of the nearest reference image (Hit@1 = accuracy).

## Tracks
* **Track S - source-only transfer (headline).** Queries: deduplicated, SegPPD-screened PlantSeg images of the 5 categories with a SegPPD equivalent (410 images: tomato bacterial spot 91, tomato late blight 147, grape black rot 103, strawberry leaf scorch 24, bell pepper bacterial spot 45). Reference gallery = all usable SegPPD images **of the query's crop** (crop-conditional like the paper's per-crop descriptions): tomato 5 classes / 142 images, grape 4 / 91, strawberry 3 / 73, pepper 4 / 102; reference images per matched class 18-50. Candidate labels = every SegPPD class of that crop (healthy and non-matching diseases remain wrong options). Secondary variant: crop-agnostic gallery of all 101 classes (2,220 images). **The label universe is NOT the paper's:** 4 of 9 categories (265 of 507 paper images) cannot be predicted from SegPPD labels and are not evaluated, and tomato has 5 candidate labels instead of the paper's 6.
* **Track T - target-domain k-shot (separate, uses PlantSeg labels).** All 9 paper categories (grape leaf blight := PlantSeg grape leaf spot, **assumed**); per-crop label universe as in the paper's C4 example (tomato: 5 diseases + healthy; grape: black rot, leaf blight, healthy; strawberry: leaf scorch, healthy; pepper: bacterial spot, healthy; healthy references come from SegPPD because PlantSeg has none). k in {1,5,10} random PlantSeg references per disease category, 20 draws; queries = remaining deduplicated images, group-disjoint from the references, sampled up to the Table S2 counts (about 483-504 queries per draw). Not comparable with a zero-shot method.

## Track S results (410 queries, crop-conditional gallery; 3 seeds averaged for SupCon)
| Method (same labelled SegPPD references) | Accuracy / Hit@1 [95 % CI, bootstrap over images] | Macro-F1 (top-1) | P@5 / P@10 / mAP | Macro per-category recall |
|---|---|---|---|---|
| **CBIR 1-NN, DINOv2-B + SupCon (mean of 3 seeds)** | **0.637** [0.597, 0.677]; seeds 0.678 / 0.639 / 0.593 | 0.816 (0.844 / 0.816 / 0.789) | 0.632 / 0.638 / 0.706 | 0.733 |
| Linear classifier (logistic regression, C=1) on the SupCon embedding | 0.632 (0.688 / 0.607 / 0.600) | 0.814 | - | 0.729 |
| CBIR 1-NN on frozen DINOv2-B CLS | 0.576 [0.527, 0.622] | 0.719 | 0.534 / 0.522 / 0.479 | 0.632 |
| Linear classifier on frozen DINOv2-B CLS | 0.649 [0.602, 0.695] | 0.778 | - | 0.689 |
| Crop-agnostic gallery (101 classes): CBIR / linear (SupCon) | 0.543 / 0.500 | - | - | - |

Per-category recall of CBIR (SupCon mean) vs ChatLD's recall in its own Table S5 (different images): tomato bacterial spot 0.736 vs 0.610; tomato late blight 0.603 vs 0.844; grape black rot 0.369 vs 0.692; strawberry leaf scorch 0.986 vs 0.962; pepper bacterial spot 0.970 vs 1.000.
Paired exact McNemar tests on identical queries: CBIR vs linear classifier on the same SupCon features p = 0.70 / 0.12 / 0.79 (seeds 0/1/2): **no significant difference**; on frozen features the linear classifier is better (p = 0.002). SupCon-CBIR vs frozen-CBIR: better for seed 0 (p = 0.001) and seed 1 (p = 0.034), no difference for seed 2 (p = 0.61).
Count-matched view (random 59/100/13/24/44 = 240 queries, 500 draws): accuracy 0.729 +/- 0.013 (the headline is sensitive to the grape black rot share). Post-hoc sensitivity, not part of the registered result: without grape black rot (PlantSeg images are mostly berries, SegPPD references are leaves) accuracy 0.726 [0.683, 0.768] (n = 307).

## Track T results (accuracy, mean over 20 draws; 9 categories; target labels used)
| k refs/category | CBIR 1-NN (SupCon, 3-seed mean) | Linear (SupCon) | CBIR 1-NN (frozen) | Linear (frozen) |
|---|---|---|---|---|
| 1 | 0.546 | 0.510 | 0.287 | 0.087 |
| 5 | 0.650 | 0.674 | 0.498 | 0.519 |
| 10 | 0.674 | 0.718 | 0.563 | 0.631 |
Draw-to-draw SD 0.02-0.06 (`trackT_summary.csv`).

## The single comparison table (do not read it as a head-to-head)
| # | Result | Images | Label set | Supervision | Value |
|---|---|---|---|---|---|
| 1 | ChatLD, paper's reported figure | paper's own 507 PlantSeg images | 9 categories, per-crop sets incl. healthy | none (zero-shot GPT-4o + text) | **0.773** (S5-implied 0.819; on the 5 covered categories 0.820, macro recall 0.822) |
| 2 | Reproduced classification baseline (linear classifier, **not** ChatLD) | our 410 (Track S) / ~490 (Track T) | 5 covered categories, crop-conditional / 9 categories | SegPPD labels / + k PlantSeg labels | 0.632 (SupCon) and 0.649 (frozen) in Track S; 0.718 / 0.631 at k=10 in Track T |
| 3 | **Our matched CBIR** | same as row 2 | same as row 2 | same as row 2 | **0.637** [0.597, 0.677] (Track S); 0.674 at k=10 (Track T) |
| 4 | Original SegPPD 75-class result | 225 locked queries (B), 367 (B+C) | 75 classes | SegPPD train/gallery | 0.807 (B), 0.794 (B+C) |
| 5 | Original external PlantSeg result | 1,485 screened queries | 11 mapped classes, search over 75 | SegPPD train/gallery | 0.608 |

## Interpretation (honest)
* **We have not matched or surpassed 77.3 %.** Our matched numbers are 0.637 (Track S, 5 categories, source-only) and 0.674 (Track T, 9 categories, 10 PlantSeg labels per category); even on the easier 5-category subset the paper's own recall is higher (0.82). Differences in images, label sets and supervision make any ranking against ChatLD unreliable.
* **CBIR shows no accuracy advantage over a linear classifier on the same embedding** (no significant paired difference on SupCon features; the linear classifier is better on frozen features and in Track T). The claimed benefit of retrieval (no retraining for new classes, inspectable neighbours, exact search, ranked outputs) is not an accuracy benefit and was not tested here.
* **Error analysis (Track S, SupCon CBIR, seed mean):** the dominant errors are tomato late blight -> tomato mold (47 images), grape black rot -> grape healthy (36) and powdery mildew (23), tomato bacterial spot -> tomato mold (13). They concentrate on PlantSeg **fruit/stem/berry images** versus SegPPD **leaf** references and on visually similar leaf diseases; the SupCon embedding is worse than the frozen features for tomato late blight (0.60 vs 0.73) - consistent with source-domain overfitting.
* **Cost / requirements:** CBIR needs a trained embedding (772 labelled SegPPD images, ~10-15 min on one RTX 4050), 18-50 labelled reference images per class (408 references for the 4 crops), 11 ms/image feature extraction (GPU, with flip) and sub-millisecond exact search; ChatLD needs no labelled images but ~7 s and a GPT-4o call per image (paper) and an unreleased description database.

## Phase 5: next decision
* **Can we make a genuinely fair comparison with ChatLD?** Not exactly (unknown images, unreleased label sets, unreproducible model). A *category-matched approximation* is possible: 5 of 9 categories source-only, or all 9 with target labels (non-zero-shot).
* **How many classes/images qualify?** Track S: 5 categories, 410 queries (of 509 local images in those categories after removing SegPPD copies, label-conflict groups and within-PlantSeg duplicates); Track T: 9 categories, ~490 queries per draw (943 in-scope images in total).
* **Initial matched CBIR accuracy:** 0.637 (Track S) / 0.674 (Track T, k=10); gap to 0.773: -13.6 / -9.9 points (indicative only).
* **Most promising next method (single):** a *target-domain supervised track* that fine-tunes the SupCon embedding on SegPPD plus the PlantSeg **training split** (official `Training`/`Validation` labels) and is evaluated once on the PlantSeg `Test` split with this benchmark's label universe. Rationale: classification vs retrieval is not the bottleneck (equal accuracy), the errors are domain-driven (fruit/stem/berry vs leaf, different framing, missing categories), and the gallery lacks 4 of the 9 categories; in-the-wild labelled data addresses all three. It must be reported as a separate, supervised track; it should not be run as a model sweep.
