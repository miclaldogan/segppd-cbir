# Experiment A: Incremental novel-class retrieval (PlantSeg, few-shot)

Status: **exploratory class-level cross-validation inside PlantSeg. Not an independent external test.** Registered before any training: `artifacts/research_v3/incr/registered.json`. Research V2 artefacts are untouched (hash manifest `artifacts/research_v3/PRESERVATION_MANIFEST.json`, files set read-only).

## Question
Can a CBIR index learn a **new disease class from 1–10 labelled images, without retraining the encoder**, and how does it compare with equally cheap few-shot alternatives that use the same encoder and the same support images?

## Design (as registered)
- Data: PlantSeg protocol P (98 classes, Track B; sha256[:12] 7c545a881b97).
- Class split: classes ordered by sha256("segppd-cbir/expA/v1" + name), cut into 5 folds (20/20/19/20/19). In fold *f* that fold's classes are **novel**, the other ~78 are **base**. Every class is novel exactly once. The ordering is unrelated to difficulty or any observed result. `class_folds.csv`.
- Encoder: DINOv2-B (no plant labels) + last 4 blocks, Proxy-Anchor (α32, δ0.3), exactly the Track B M3 config, **no tuning**, seed 0, trained on BASE-class train images only. Epochs selected on BASE-class official-validation images only. Novel classes' images and labels are not seen in training, selection or hyper-parameter choice (asserted in `incr_train.py`; `runs/foldN_s0/manifest.csv` lists every image's role). The encoder is frozen after `best.pt`; all embeddings are computed once.
- Support: k = 1, 3, 5, 10 train images per novel class (nested, one image per duplicate group first), 5 seeded draws (`support_draws.csv`). Queries: the novel classes' official Test images (group-disjoint from train by protocol P), 1,646 images over all folds.
- Scenarios: **S1** gallery = supports only (~20-way, chance 0.051). **S2** gallery = all base-class train images (~5.4k) + supports, 98-way (chance 0.010): the open, non-trivial setting (headline).
- Methods, same encoder, same supports: **CBIR** exact squared-L2 1-NN; **NCP** nearest class prototype (class-mean embedding; base prototypes from all base train); **LR** multinomial logistic regression on the 256-d embeddings, C=10 fixed in advance, balanced class weights, refit after every insertion. Plus the same three on raw frozen DINOv2-B CLS (never fine-tuned).
- Statistics: class-macro Hit@1 over the 98 novel-class evaluations averaged over 5 draws; class-level bootstrap (2,000) CIs; paired differences on identical classes, queries and supports.
- Files: `per_query_predictions.csv.gz` (395,040 rows), `ranking_top10_draw0.csv.gz` (top-10 references per query), `summary_by_shot.csv`, `paired_differences.csv`, `per_fold_cbir_S2.csv`, `per_class_novel.csv`, `cost_summary.csv`, `base_interference.csv`. Code: `src/segppd_cbir/rv2/incr_{protocol,train,eval,stats,diag}.py`. Figures: `artifacts/research_v3/figures/A1_support_size.*`, `A2_update_cost.*`.

## Results (base-trained encoder, class-macro Hit@1 [95 % class-bootstrap CI])

| Scenario | Method | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| S1 (novel only) | CBIR | 0.625 [0.582, 0.667] | 0.728 [0.691, 0.763] | 0.753 [0.717, 0.788] | 0.791 [0.758, 0.822] |
| S1 | NCP | 0.625 [0.582, 0.667] | 0.740 [0.700, 0.778] | 0.771 [0.733, 0.806] | 0.792 [0.755, 0.827] |
| S1 | LR | 0.627 [0.582, 0.669] | 0.751 [0.713, 0.788] | 0.788 [0.753, 0.820] | 0.821 [0.791, 0.850] |
| **S2 (open gallery)** | **CBIR** | 0.179 [0.146, 0.212] | 0.327 [0.283, 0.370] | 0.399 [0.354, 0.445] | 0.493 [0.448, 0.539] |
| S2 | NCP | 0.255 [0.223, 0.289] | 0.424 [0.381, 0.468] | 0.466 [0.423, 0.511] | 0.504 [0.461, 0.547] |
| S2 | LR | 0.236 [0.203, 0.272] | 0.482 [0.439, 0.526] | 0.565 [0.522, 0.609] | 0.654 [0.616, 0.693] |
| S2, frozen DINOv2-B | CBIR | 0.152 | 0.264 | 0.328 | 0.428 |

Reference (violates the novel-class rule, **not a competitor**): the Track B M3 model, trained with these classes labelled, gets 0.825 [0.788, 0.859] on the same classes (seed 0).

CBIR in S2 additionally: Hit@5 0.242 / 0.436 / 0.527 / 0.640, mAP 0.218 / 0.230 / 0.238 / 0.258, P@5 0.048 / 0.141 / 0.218 / 0.336 (P@5 is capped at min(k,5)/5), Recall@10 0.269 / 0.265 / 0.257 / 0.236 for k = 1/3/5/10. mAP is nearly flat because the number of relevant items (k) grows with k while the many base look-alikes stay in front.

Replication across folds (CBIR, S2, k=1 / 10): fold 0 0.157 / 0.445, fold 1 0.236 / 0.544, fold 2 0.174 / 0.508, fold 3 0.196 / 0.527, fold 4 0.129 / 0.441. The ordering over k is identical in every fold; absolute level varies by ~10 points between folds.

### What the numbers say
1. **Learning a new class without retraining works, but it is far from solved in an open gallery.** Every method is well above chance (0.010) and climbs with k. With 10 supports CBIR reaches 0.49 in S2, versus 0.825 for a model trained with those classes labelled.
2. **CBIR is the weakest of the three methods in the open gallery** at k ≤ 5. Paired differences (S2): CBIR − NCP −0.076 [−0.087, −0.066] (k=1), −0.097 (k=3), −0.067 (k=5), −0.011 [−0.034, +0.013] at k=10 (tie); CBIR − LR −0.058, −0.155, −0.166, −0.161, all CIs excluding 0. In the novel-only gallery S1 the three are close: CBIR − LR is −0.002 (k=1, tie), −0.023, −0.034 and −0.030 (k=3/5/10; CIs exclude 0 but the gaps are 2–3 points), CBIR − NCP is 0 to −0.018 (see `paired_differences.csv`). **Hypothesis H2 (NCP ≥ CBIR) is supported at k ≤ 5; H3 (LR competitive) is supported in S1 and exceeded in S2, where LR is best at k ≥ 3.** We did not predict that LR would be the best; it is also the only method that costs a refit.
3. **Why CBIR loses in S2 (diagnostic, post hoc).** 83–95 % of CBIR's S2 errors land on a *base* class (95 % at k=1, 83 % at k=10): a novel image is closer to one of the ~55 images of a similar base class than to its own 1–10 supports. When the base gallery is thinned to k images per base class (same k as the novel support) CBIR rises from 0.179/0.327/0.399/0.493 to 0.310/0.453/0.504/0.565 (this diagnostic averages over queries, not class-macro, so it is only indicative). Gallery imbalance is therefore a main cause; the gap is not a defect of the encoder. This diagnostic was run after seeing the main results and is **not** a registered method and was not used to choose anything. The hardest novel classes are siblings of base classes (squash, cucumber and zucchini powdery mildew 0.04–0.07 at k=5 vs. 0.60–0.71 for the labelled-trained model).
4. **What base-class training contributes.** Compared with the same method on frozen DINOv2-B, base-class metric learning helps CBIR in S2 at k = 3/5/10 (+0.063, +0.071, +0.065; CIs exclude 0), and LR by +0.09 to +0.16. In S1 the differences are small (≤ +0.03). For NCP at k=10 base training is worse than frozen (−0.067 [−0.102, −0.032]); we have no tested explanation for it.
5. **No interference with the old classes.** Base-class Hit@1 with 10 supports per novel class inserted: 0.861 → 0.850 (base-trained encoder); frozen 0.711 → 0.705. A loss of ~1 point is the cost of adding ~20 look-alike classes.

## Cost, memory, retraining (measured, CPU, mean over folds/draws)
| | CBIR | NCP | LR (head) |
|---|---|---|---|
| Encoder retraining needed | no | no | no |
| Extra step after inserting k·20 images | append rows to the index: ~1 ms | recompute prototypes: ~7 ms | **refit: 9.5–11.9 s** (S2) |
| Memory | gallery 5.5–5.7 MB (float32, 256-d × ~5.4k) | 0.10 MB (98 prototypes) | 0.10 MB (weights) |
| Embedding the support images | ~10.6 ms/image on the RTX 4050 (same for all three) | | |

CBIR insertion is the cheapest and needs no fitting at all, but it must keep every reference image (55× the storage of prototypes). Encoder retraining was not run as a comparison; its measured cost per fold was 660–1,150 s. A supervised head fine-tuned on the supports was **not** tested.

## What CBIR provides here that a fixed-label classifier does not (and what is not shown)
- **Shown:** a class can be added by appending embeddings (ms, no fit, no weight change), and it does not disturb the old classes (−1 point). A fixed-label classifier's output layer cannot represent a new class without being re-fit; the cheapest classifier alternative (LR head) needs ~10 s per update and in this benchmark is more accurate than CBIR at k ≥ 3.
- **Available but not scored:** every answer comes with the nearest labelled reference images (`ranking_top10_draw0.csv.gz`), so a prediction can be audited against concrete examples. Whether that helps a human was **not** measured.
- **Not shown:** that CBIR is more accurate than classification on novel classes (it is not, in S2), and any user-facing benefit.

## Limits
- Exploratory: PlantSeg official Test images were partly inspected in earlier experiments and every class is evaluated as novel once; CIs resample 98 classes, not independent fresh datasets.
- One encoder seed per fold (training variance not measured); between-fold spread is ~10 points. The 5 support draws capture support-sampling variance only.
- Duplicate control relies on protocol P's groups (hash/embedding). PlantSeg is web-scraped, so near-duplicates across sources may remain.
- LR C=10 and the NCP/LR definitions were fixed in advance; other few-shot methods (cosine classifiers, fine-tuning on supports, ProtoNet training) were not run.
- No state-of-the-art claim is made; no matched published benchmark was used.
