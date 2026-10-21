# ChatLeafDisease benchmark reconstruction and audit (Phase 1)

Target paper: Pan, Zhong, Xia, Huang, Zhu, Yang, Lin (2025), *ChatLeafDisease: a chain-of-thought prompting approach for crop disease classification using large language models*, Plant Phenomics 7(3):100094, DOI 10.1016/j.plaphe.2025.100094 (https://pmc.ncbi.nlm.nih.gov/articles/PMC12709970/). Main text read via the PMC page; supplementary `mmc1.docx` (Tables S1-S5, Figures S1-S18, prompts C1-C4) obtained through the Europe PMC REST API (`/PMC12709970/supplementaryFiles`; the PMC download endpoint is behind a proof-of-work bot check, which was not bypassed) and stored locally in `data/external/chatleaf_supp/` (not redistributed). All numbers below were measured/read in this session; code: `src/segppd_cbir/chatleaf_audit.py` -> `artifacts/chatleaf/audit_*`.

## 1. What the published PlantSeg experiment is

| Item | What the paper/supplement states |
|---|---|
| Task | Zero-shot (training-free) crop-disease **classification**: GPT-4o with a chain-of-thought scoring prompt compares an image with per-disease text descriptions (condensed from University of Minnesota Extension manuals) and picks the disease with the highest score (prompt C2). |
| Images | "We selected samples of tomato, grape, strawberry, and pepper diseases" from PlantSeg (Sec. 2.4). **Table S2: 507 images, 9 categories**: tomato bacterial spot 59, yellow leaf curl virus 64, late blight 100, mosaic virus 43, early blight 112 (tomato 378); grape black rot 13, leaf blight 46; strawberry leaf scorch 26; pepper bacterial spot 44. No healthy images ("PlantSeg does not include healthy samples"). |
| Label space shown to the model | **Not stated for PlantSeg in the main text.** The supplementary example (C4) shows six scores for a PlantSeg tomato image: bacterial spot, early blight, late blight, mosaic virus, yellow leaf curl virus and **Healthy**, i.e. a per-crop closed set that includes a healthy option. Candidate sets for grape / strawberry / pepper are not given (Table S1 lists the PlantVillage sets: grape black rot, black measles, leaf blight, healthy; strawberry healthy, leaf scorch, powdery mildew, angular leaf spot; pepper bacterial spot, healthy). The crop is therefore presumably known. |
| Metric | Accuracy "computed on the pooled confusion matrix across all classes"; precision/recall/F1 macro-averaged per crop (C3). Headline 77.3 % (text only). |
| Selection / test identifiers | **Not released**: no image list, seed or sampling rule; no code availability statement. |
| Runs / repeats | Not stated for PlantSeg (the five-repeat design is described for tomato/PlantVillage). |
| Cost | ChatLD 7.06 s per image (paper), GPT-4o API. |

**Internal inconsistency:** Table S5 (per-class recall on PlantSeg) combined with the Table S2 counts implies a pooled accuracy of **81.9 %** (macro recall 80.1 %, macro F1 82.7 %), not 77.3 %. We cannot tell which quantity the 77.3 % is (different run, different counts, rounding); we therefore quote 77.3 % as the paper's headline and report the S5-derived values as context.

## 2. Is the original evaluation set recoverable? (local PlantSeg v2, Zenodo 13958858, 11,458 images)

| Paper category (S2 count) | PlantSeg v2 category | local images | local test split | paper/local |
|---|---|---|---|---|
| tomato bacterial spot (59) | tomato bacterial leaf spot | 109 | 22 | 0.54 |
| tomato yellow leaf curl virus (64) | tomato yellow leaf curl virus | 93 | 19 | 0.69 |
| tomato late blight (100) | tomato late blight | 163 | 33 | 0.61 |
| tomato mosaic virus (43) | tomato mosaic virus | 63 | 13 | 0.68 |
| tomato early blight (112) | tomato early blight | 187 | 37 | 0.60 |
| grape black rot (13) | grape black rot | 122 | 24 | 0.11 |
| grape leaf blight (46) | **no category of this name**; `grape leaf spot` assumed | 91 | 18 | 0.51 |
| strawberry leaf scorch (26) | strawberry leaf scorch | 39 | 8 | 0.67 |
| pepper bacterial spot (44) | bell pepper bacterial spot | 76 | 15 | 0.58 |

**No category count matches** (paper counts are 11-69 % of the local counts, also not equal to the test-split counts), so the 507 images are an unknown subsample; neither the image identities nor the sampling rule can be reconstructed. The grape "leaf blight" correspondence is an unverified assumption (name differs; plausible leaf-lesion category but not confirmed).

## 3. Manual verification of label mappings (contact sheet of 6 random images per category vs SegPPD classes, viewed in this session)

* tomato bacterial leaf spot <-> SegPPD `tomato_bacterial spot` (92): same disease; PlantSeg additionally contains fruit lesions, SegPPD is leaf-level.
* tomato late blight <-> `tomato_late blight` (94): same disease; PlantSeg has many fruit/stem images.
* grape black rot <-> `grape_black rot` (42): **same name but different organ**: PlantSeg images are mostly berries, SegPPD images are leaves. Label equal, visual modality not.
* strawberry leaf scorch <-> `strawberry_leaf scorch` (81): good match (leaves).
* bell pepper bacterial spot <-> `pepper_bacterial spot` (71): plausible (bell pepper is a pepper); loose tier.
* grape leaf spot vs SegPPD `grape_yellow blight` (44): visually different; **not mapped**.
* No SegPPD equivalent exists for tomato early blight, tomato mosaic virus, tomato yellow leaf curl virus (SegPPD tomato classes: healthy, bacterial spot, mold, late blight, powdery mildew) nor for grape leaf blight.

Coverage of the paper's categories by SegPPD: **5 of 9 categories; 242 of 507 paper images (47.7 %); 265 images (tomato EB/MV/YLCV, grape LB) cannot be predicted by a source-only retrieval.**

## 4. Duplicate / leakage audit

* PlantSeg vs **all 2,263 SegPPD images** (hash rule pHash<=12 & dHash<=18, or frozen DINOv2-B cosine >= 0.85; same screen as the external evaluation): flagged images per category - bacterial spot 7, early blight 8, late blight 8, grape black rot 6, grape leaf spot 12, strawberry leaf scorch 6, bell pepper bacterial spot 21, YLCV 0, mosaic virus 0 (68 of 943 in-scope images). Flagged images are removed from every query set.
* **Within PlantSeg** (in-scope 943 images): 196 near-duplicate pairs, 229 images in multi-image groups, and **10 groups that span more than one category** (the same photo under different disease labels = label conflicts, as in SegPPD). Such groups are dropped; inside a group only one image is used as a query (`artifacts/chatleaf/audit_inscope_groups.csv`).
* The earlier 1,485-query external evaluation contained images of the same five covered categories; that result is historical evidence only and nothing in the new benchmark was tuned on it.

## 5. Can ChatLD itself be reproduced? **No.**
Needs the paid GPT-4o API (model version 2024-08-06) - no key and not permitted by the project rules; the condensed disease-description database and the preprocessing of descriptions with exemplar images (Fig. S1, prompt C1) are not released; no code; test identifiers unreleased; candidate sets for three crops unspecified. An approximate re-implementation would not be "ChatLD", so none was built. Instead a **linear classifier on the same DINOv2 features with the same labelled references** is the controlled classification baseline.

## 6. Decision gate
* **Exact published-protocol comparison: NOT possible** (unknown images, unreleased ids, unknown label sets for three crops, unreproducible model).
* **Approximate reproduction: possible only partially.** (a) *Track S* (source-only): reproduces the *category-matched, crop-conditional* setting for 5 of 9 categories; (b) *Track T* (k-shot with target labels): covers all 9 categories with count-matched random samples but uses PlantSeg labels, so it is not a zero-shot comparison.
* **Evidence for a head-to-head statement against 77.3 %: insufficient.** Any number we report is on different images, a different (smaller or supervised) label setting, or both, and is compared with 77.3 % only as context.
