# Plan: confirmatory test of "single DINOv2-B + Proxy-Anchor + ResNet50 fusion" on untouched data

Status: **PLAN ONLY. Not run. Not locked** (dataset and screening below are still to be done). Written 2026-10-21 at the user's request ("işlerimiz bitince"). Nothing here uses a test set.

## Why
Exploratory result (Protocol B/C tests already consumed; `docs/DISTANCE_METRIC_STUDY.md` + `artifacts/research_v3/metrics/single_fusion_*.csv`): adding ResNet50-ft embeddings (sqrt-weighted concat of unit vectors, exact L2) to a single Proxy-Anchor DINOv2-B gave Hit@1 +0.013 [+0.003, +0.025] on B and +0.018 [-0.004, +0.040] on C at w=0.05 (dev +0.007 [-0.004, +0.018]). Cost vs the 3-seed-ensemble fusion: ~2.6x fewer MACs, 2x faster embedding. Because B/C were inspected, this needs a fresh test.

## Blocker found while planning
The ResNet50-ft ArcFace model used in the fusion (`R1_resnet_arcface_s0`) **has no saved checkpoint**: only its embeddings of the SegPPD images exist (`artifacts/extensions/B/R1_resnet_arcface_s0_embeddings.npz`), so it cannot embed new images. Needed first: retrain R1 with the same recipe (args in `R1_resnet_arcface_s0_log.json`: resnet50, last 4 blocks, ArcFace, d=256, 60 epochs, patience 10, 20 steps/epoch, lr_bb 1e-4, lr_head 1e-3, ~660 s, 1.95 GB) with `--save_ckpt`, seeds 0-2 (~33 min). The retrained model is a new model: the fusion weight must be re-selected on dev (grid below) and the exploratory numbers above are NOT a prediction for it. The PA checkpoints already exist and are hashed (`artifacts/research_v3/PRESERVATION_MANIFEST.json`).

## Candidates (frozen before the fresh data is opened)
- C0: single Proxy-Anchor DINOv2-B (3 existing seeds, a32 d0.3), exact L2 on 256-d. Reference.
- C1: C0 + retrained ResNet50-ft ArcFace (matching seed), w in {0.05, 0.1, 0.2} chosen on dev only by the registered rule (dev Hit@1, tie band 1/150, then dev mAP); one w goes to the test.
- Nothing else (no ensembles, whitening, L1, re-ranking) is tested, to keep a single primary comparison C1 vs C0.

## Fresh test set requirements (decision still open)
- Labelled plant-disease images never inspected, never used for training/selection/tuning, from a source other than SegPPD-101 and PlantSeg (ChatLeaf images are already inspected).
- Duplicate screen against SegPPD (all 2,263) and PlantSeg (all) before any model runs: pHash<=12 & dHash<=18 or DINOv2-B cosine >= 0.85 (rule used so far); screening reported, flagged images removed, nothing moved between splits.
- Strict crop-disease class mapping only (as `external.MAPPING` strict tier); report the number of usable classes/queries. Gallery = SegPPD gallery (source-only setting) and, separately, a target-supervised arm only if the new source offers its own training split (never mixed in one claim).
- Minimum size: enough classes/queries for a class-bootstrap CI half-width <= ~0.04; if the usable set is smaller, the result is labelled underpowered up front.
- Candidate sources to evaluate (not yet fetched): other public leaf-disease sets with disease labels and web/field photographs. A data download needs the user's go-ahead.

## Analysis (fixed in advance)
Primary: class-macro Hit@1 of C1 minus C0, paired on identical queries, class-bootstrap 95% CI, evaluated once. Secondary: mAP, P@10, per-class changes, cost. Decision rule: report as an improvement only if the CI excludes 0; otherwise report "no confirmed gain" (a negative result is acceptable). No repeated looks; no re-tuning after opening the data.
