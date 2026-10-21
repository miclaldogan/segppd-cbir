# Track B - PlantSeg: direct classification vs nearest-neighbour CBIR under equal training information

Registered before any model touched the protocol: `artifacts/research_v2/trackB/registered.json` (protocol file `protocol_P.csv`, sha256[:12] `7c545a881b97`). Code: `src/segppd_cbir/rv2/plantseg_audit.py`, `plantseg_protocol.py`, `plantseg_train.py`, `plantseg_eval.py`, `trackB_report.py`. Results: `artifacts/research_v2/trackB/` (`results_by_seed.csv`, `results_seed_mean_sd.csv`, `paired_by_seed.csv`, `paired_mcnemar.csv`, `pred_M*.csv`, `per_class_*.csv`, `top_confusions_seed0.json`, `sensitivity_url_disjoint.json`). Figures: `artifacts/research_v2/figures/B1_classification_vs_cbir.*`, `B2_per_class_cbir_vs_classifier.*`.

## 1. Data audit (B1)
* PlantSeg v2 (Zenodo 13958858), local `Metadatav2.csv`: 11,458 images; 114 distinct disease names and 35 plant labels; the COCO file lists 117 categories / 11,480 images (3 categories only there: eggplant phomopsis, soybean rust, zucchini yellow mosaic). The arXiv paper reports 115 categories / 34 plants. The origin of the difference (version/labelling) could not be verified (journal version not accessible); we use the 114 metadata names.
* Official split field: Training 7,916 / Validation 1,247 / Test 2,295 (integrity: every image exists in its split folder).
* **The official split leaks near-duplicates.** Visual near-duplicate rule inside PlantSeg = pHash<=12 & dHash<=18 **or DINOv2-B cosine >= 0.95** (the 0.85 used against SegPPD chained unrelated look-alike photos inside PlantSeg: visual review of random pairs in 0.85-0.95 showed different photos): 1,773 duplicate pairs, 2,561 images in multi-image groups, **575 groups span more than one official split** (1,383 images), and **256 groups carry more than one disease label** (601 images: the same photo under conflicting labels). 728 of the 2,295 official test images share their source URL with a training image.
* Overlap with SegPPD-101: 867 images (hash or cosine >= 0.85, see earlier audit) - removed everywhere.
* **Registered eligibility (before any model):** drop SegPPD copies and label-conflict groups; keep one image per duplicate group in dev/test; dev/test images whose duplicate group contains a training image are dropped (training keeps the group; nothing is moved between splits); test images whose group has a validation image are dropped; a class is eligible iff train >= 10, validation >= 2, test >= 5 after these filters. **98 of 114 classes qualify; 16 do not for lack of images** (banana cordana leaf spot, banana panama disease, bell pepper frogeye leaf spot, bell pepper powdery mildew, broccoli alternaria leaf spot, broccoli ring spot, carrot cercospora leaf blight, coffee black rot, coffee brown eye spot, ginger leaf spot, peach anthracnose, peach rust, plum bacterial spot, plum rust, raspberry leaf spot, tobacco frogeye leaf spot) - none was dropped for being hard.
* Resulting protocol P: **train 6,744 / dev 899 / test 1,646 images, 98 classes, 35 crops**; 1,267 test images are also URL-disjoint from training (sensitivity subset).
* **Status of the test split:** never used for any selection here, but PlantSeg images of 11 mapped classes (all splits) were evaluated before in the external and ChatLeaf-matched experiments (inspected, never tuned on). This track is therefore a **controlled exploratory comparison**, not a confirmation; no untouched independent PlantSeg set exists.

## 2. Methods (identical train/dev/test images, labels and 224-px preprocessing from the 256-px cache)
| id | method | supervision used |
|---|---|---|
| M1 | frozen DINOv2-B CLS + L2-regularised logistic regression (C chosen on dev from {0.01,...,100}) | PlantSeg train labels |
| M2 | frozen DINOv2-B CLS + exact-L2 1-NN, gallery = the 6,744 training images | same labelled images as reference gallery |
| M3 | DINOv2-B last 4 blocks metric-learned with **Proxy-Anchor (alpha 32, delta 0.3, the Track A dev-selected loss)** on PlantSeg train + exact-L2 1-NN over the training images | PlantSeg train labels |
| M4 | logistic regression on the same M3 embedding (C chosen on dev) | same |
| M5 | supervised fine-tuning of the same last 4 blocks with a linear head + cross-entropy (separately identified classification baseline), same optimiser/sampler/augmentation/budget | same |
Training budget M3/M5: 60 steps x 16 classes x 4 images per epoch, max 30 epochs, patience 6, dev-accuracy selection (tie = one dev image, then secondary), seeds 0/1/2 (M1/M2 deterministic). Retrieval relevance = same disease class; Hit@1 = nearest-neighbour label accuracy. **Hit@1/P@K/mAP are retrieval metrics; kNN majority-vote accuracy is not reported.** Retrieval needs the labelled gallery; classifiers need only the trained head (this is a difference in deployment cost, not in training information).

## 3. Results (1,646 identical test images; mean +/- SD over 3 seeds where trained)
| method | accuracy = Hit@1 [95% image-bootstrap CI, seed 0] | macro-F1 | macro recall | mAP | P@5 | P@10 | Hit@10 |
|---|---|---|---|---|---|---|---|
| M1 frozen + linear | 0.794 [0.775, 0.813] | 0.769 | 0.767 | - | - | - | - |
| M2 frozen + CBIR 1-NN | 0.688 [0.667, 0.710] | 0.686 | 0.683 | 0.379 | 0.615 | 0.572 | 0.933 |
| **M3 metric-learned + CBIR 1-NN** | **0.847 +/- 0.005** (0.852, 0.847, 0.842) [0.836, 0.869] | 0.819 +/- 0.007 | 0.817 | 0.834 +/- 0.002 | 0.839 | 0.836 | 0.921 |
| M4 linear on M3 embedding | 0.852 +/- 0.007 (0.860, 0.851, 0.846) | 0.822 +/- 0.004 | 0.819 | - | - | - | - |
| M5 CE fine-tuning | 0.847 +/- 0.008 (0.842, 0.843, 0.857) | 0.822 +/- 0.010 | 0.822 | - | - | - | - |

Exact McNemar on identical images (per seed): **M3 vs M4** p = 0.25 / 0.62 / 0.55 (no difference); **M3 vs M5** p = 0.19 / 0.64 / 0.043 (seed 2 favours the classifier, seeds 0-1 no difference); M4 vs M5 p = 0.016 / 0.31 / 0.15; **M1 vs M2** p ~ 3e-23 (linear >> frozen 1-NN: +0.106); M3 vs M1 p < 1e-6 (metric learning on PlantSeg labels: +0.053 over the frozen classifier, +0.159 over frozen retrieval). URL-disjoint subset (1,267 images, seed 0): M1 0.784, M2 0.687, M3 0.848, M4 0.857, M5 0.838 - same ordering (`sensitivity_url_disjoint.json`).
Top seed-0 confusions of M3: squash <-> zucchini powdery mildew (13 / 8), cucumber powdery mildew -> squash (6), cucumber vs zucchini bacterial wilt (5), grape downy mildew -> grape leaf spot (5), potato early -> late blight (5): visually near-identical symptoms on closely related crops. Per-class recall of CBIR vs classifier is in `B2_per_class_cbir_vs_classifier.png` / `per_class_M3_s0.csv`, `per_class_M5_s0.csv` (precision, recall, F1, support for all 98 classes).

## 4. Cost
| | M2 | M3 | M1 / M4 / M5 |
|---|---|---|---|
| stored at inference | 6,744 x 768 floats (20.7 MB) | 6,744 x 256 floats (6.9 MB) | 98 x 768 (M1), 98 x 256 (M4) weights; M5 = fine-tuned network |
| search / classify per query | 0.53 ms (exact L2) | 0.47 ms (exact L2) | 0.009 ms (M1), 0.010 ms (M4) on the cached feature |
| training | none (+114 s LR fit for M1; 209 s for M4) | 1,065 s (30 epochs, 3.0 GB VRAM) | M5 640-868 s (3.0 GB VRAM) |
Embedding extraction costs ~10.6 ms/image (M3) and 16-18 ms/image (M5 logits + features) on the RTX 4050.

## 5. Answer to the research question
**Under equal training information CBIR does not beat direct classification on PlantSeg; it ties it.** On frozen features the classifier is clearly better (+10.6 points). After metric learning, 1-NN CBIR (0.847) equals both the linear head on the same embedding (0.852) and a CE-fine-tuned classifier (0.847); no pairwise Hit@1 difference is consistent across seeds. Retrieval's measurable advantages here are ranking quality (mAP 0.834, P@10 0.836) and a lower-dimensional gallery; classification is ~45x cheaper per query on the cached feature (the embedding step is common to both). **Reference-image traceability and incremental gallery updates were not tested** and are therefore not claimed.
Limitations: one backbone, one loss (from Track A), 3 seeds, official-test caveat above, no independent confirmation set, class-label relevance only.
