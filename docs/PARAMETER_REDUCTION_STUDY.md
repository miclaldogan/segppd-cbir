# Parameter-reduction study (research_v4): fewer parameters at the same Hit@1

Registered before any run: `artifacts/research_v4/registered.json` (goal, selection rule, grids; revisions time-stamped inside). Code `src/segppd_cbir/rv4/`. All numbers: class-macro Hit@1 (mAP), exact squared L2, flip-averaged 256-d embeddings. **Dev is the only selection set; Protocol B and C tests are consumed, so every B/C number is exploratory** (no new independent claim). Hardware RTX 4050 6 GB, one GPU job at a time.

**Declared target:** Hit@1 >= 0.85 on dev AND B AND C jointly, with fewer inference parameters than the previous best system (SupCon 3-seed ensemble + ResNet50-ft, 281.8 M, B 0.840 / C 0.773; with PCA whitening B 0.858 / C 0.760). The B-only 0.858 was explicitly not accepted as a target because it did not replicate on C.

Reference: single DINOv2-B + Proxy-Anchor (a32 d0.3), 85.9 M params, 21.9 GMAC: dev 0.820 (0.757), B 0.825 (0.753), C 0.782 (0.751), 3-seed means.

## Stage Z: zero-parameter post-processing (CPU, existing embeddings) - negative
`rv4/zero_param.py`, outputs `artifacts/research_v4/zero/`. Families: DBA (gallery-side augmentation, k in {1,2,3,5}), alpha-QE (k in {1,2,3,5}, alpha in {1,3}), gallery-indexed k-reciprocal re-ranking (k1 in {10,20}, k2 in {3,6}, lambda in {0.3,0.5}; each query alone against the gallery), NCA (sklearn, fit on train_B only, 256/128/64-d), shrinkage LDA (74/64/32-d), plus a greedy dev-selected stack. Seeds averaged.

| Model | Variant (dev-selected per family) | dev | B | C |
|---|---|---|---|---|
| PA single | L2 baseline | 0.820 | 0.825 | 0.782 |
| | DBA k=1 | 0.809 | 0.828 | 0.769 |
| | aQE k=1 a=1 | 0.820 | 0.825 | 0.782 |
| | k-reciprocal (10, 3, 0.5) | 0.800 | 0.816 | 0.760 |
| | LDA 74-d | 0.809 | 0.836 | 0.780 |
| | NCA 64/128/256-d | 0.780 / 0.778 / 0.778 | 0.779-0.785 | 0.722 |
| PA single + ResNet50 (w .05) | L2 baseline | 0.827 | 0.839 | 0.800 |
| | best family (aQE k=1) | 0.827 | 0.839 | 0.800 |
| | LDA 74-d | 0.816 | 0.837 | 0.771 |

No family improves dev Hit@1 for either model; aQE with k=1 is a no-op in effect, NCA is clearly worse (the 772-image training set is too small for a learned metric), LDA gains +0.011 on B but loses on dev and C. The stack adopted nothing that changed any number. The professor's NCA-style feature selection therefore does not transfer to these learned embeddings. Nothing adopted.

## Stage T: test-time multi-view embedding (zero parameters) - no Hit@1 gain
`rv4/tta.py`, `artifacts/research_v4/tta/`. Reference PA checkpoints, 3 seeds. Views: flip | flip + 1.25x scale | flip + 5-crop | all.

| Views | dev | B | C | embed ms/img |
|---|---|---|---|---|
| flip (reference) | 0.820 (0.758) | 0.825 (0.753) | 0.782 (0.751) | 6.5 |
| flip + ms | 0.818 (0.767) | 0.834 (0.765) | 0.800 (0.758) | 16.8 |
| flip + 5crop | 0.816 (0.764) | 0.822 (0.762) | 0.813 (0.757) | 39.1 |
| all | 0.818 (0.765) | 0.824 (0.763) | 0.811 (0.759) | 49.3 |

Dev Hit@1 is unchanged (all within the one-query band; the rule picked "all" on dev mAP +0.007). The C-test gain (+0.03) is exploratory and costs 7.5x embedding time. Not adopted into the reference system; reported as an option.

## Stage S: backbone sweep (seed 0, dev-selected; same Proxy-Anchor recipe as research_v2)
`rv4/make_configs.py`, `rv4/train.py`, runs in `artifacts/research_v4/sweep/` and `followup/`; table `artifacts/research_v4/results_all.csv`; figure `artifacts/research_v4/figures/P1_params_vs_hit1.*`.

| Backbone (recipe) | params M | dev | B (expl.) | C (expl.) | train s | VRAM GB |
|---|---|---|---|---|---|---|
| DINOv2-B last-4 (reference, 3-seed mean) | 85.9 | 0.820 (0.757) | 0.825 | 0.782 | 617 | 2.98 |
| DINOv3 ViT-B/16 last-4 | 85.8 | 0.793 (0.665) | 0.800 | 0.760 | 724 | 2.75 |
| DINOv2-B truncated to 9 blocks | 64.7 | 0.713 (0.587) | 0.671 | 0.647 | 654 | 2.89 |
| DINOv2-B truncated to 8 blocks | 57.6 | 0.687 (0.548) | 0.631 | 0.613 | 607 | 2.87 |
| DINOv3 ConvNeXt-T last stage, 224 px | 28.0 | 0.773 (0.684) | 0.689 | 0.713 | 225 | 0.98 |
| **DINOv3 ConvNeXt-T last stage, 336 px** | 28.0 | **0.793 (0.686)** | 0.711 | 0.693 | ~750 | ~2.5 |
| DINOv3 ConvNeXt-T, lr 1e-4 | 28.0 | 0.753 | 0.667 | 0.687 | 220 | 0.98 |
| DINOv3 ConvNeXt-T, full fine-tune (3e-5 / 1e-4) | 28.0 | 0.687 / 0.147 (diverged) | | | | 4.47 |
| DINOv3 ViT-S+/16 | 28.8 | 0.733 (0.570) | 0.747 | 0.693 | 234 | 1.64 |
| DINOv3 ViT-S/16 | 21.7 | 0.747 (0.523) | 0.698 | 0.640 | 129 | 1.22 |
| DINOv2-S (no registers) | 21.7 | 0.713 (0.535) | 0.649 | 0.620 | 83 | 1.34 |
| DINOv2-S reg4, 224 px | 21.7 | 0.760 (0.636) | 0.756 | 0.660 | 216 | 1.35 |
| DINOv2-S reg4, lr 1e-4 | 21.7 | 0.773 (0.643) | 0.698 | 0.700 | 134 | 1.35 |
| DINOv2-S reg4, full fine-tune (3e-5 / 1e-4) | 21.7 | 0.773 (0.656) / 0.767 (0.669) | 0.729 / 0.729 | 0.687 / 0.720 | 472 / 300 | 3.59 |
| **DINOv2-S reg4, 336 px** | 21.9 | **0.780 (0.674)** | 0.747 | 0.733 | 747 | 2.77 |

Reading: (i) no backbone below 86 M reaches the reference; the best small recipes sit 3-4 points below on dev and 8-11 points below on B; (ii) depth truncation of DINOv2-B is the worst option (-11 to -13 points for -25 % parameters); (iii) resolution (336 px) is the only follow-up that helped both small models (+2 points), full fine-tuning never helped; (iv) DINOv3 ViT-B is not better than DINOv2-B on this data with this recipe.

## Fusion of light models (zero training; `rv4/fusion_study.py`, `artifacts/research_v4/fusion/`)
Sqrt-weighted concat of unit embeddings, w dev-selected. ConvNeXt-T 336 + DINOv2-S reg4 336 (49.9 M): dev 0.793 -> the rule keeps w=0 (no gain). ConvNeXt-T 336 + ViT-S+ (56.8 M): 0.800 (0.703) at w=0.2, B 0.756, C 0.733. ConvNeXt-T 336 + ResNet50-ft (52 M): no gain (w=0). Two light models do not replace one DINOv2-B.

## Stage D: relational knowledge distillation (teacher = PA 3-seed ensemble, or ensemble + ResNet50; loss = Proxy-Anchor + lambda * MSE of batch cosine-similarity matrices; teacher similarities from train_B rows only)
Student DINOv2-B (86 M), seed 0: dev 0.813 / 0.813 / 0.813 (ensemble teacher, lambda 30/100/300) and 0.820 / 0.793 / 0.787 (ensemble + ResNet teacher); reference seed 0 is 0.820. The KD term converges (MSE 0.03 -> 0.0014) but Hit@1 does not move: the single model already matches the ensemble's similarity structure as far as this loss can transfer it. No gain.

Small students (ConvNeXt-T 336, DINOv2-S reg4 336; lambda {30,100}, both teachers): see the table below (filled when the queue finished).

| Student (seed 0) | teacher | lambda | dev | B | C |
|---|---|---|---|---|---|
| ConvNeXt-T 336, no KD | - | - | 0.793 (0.686) | 0.711 | 0.693 |
| ConvNeXt-T 336 | ensemble | 30 | 0.780 (0.692) | 0.722 | 0.713 |
| ConvNeXt-T 336 | ensemble | 100 | 0.767 (0.679) | 0.698 | 0.673 |
| ConvNeXt-T 336 | ensemble + ResNet | 30 | 0.767 (0.650) | 0.676 | 0.700 |
| ConvNeXt-T 336 | ensemble + ResNet | 100 | 0.780 (0.686) | 0.729 | 0.713 |

All four KD runs are at or below the undistilled student on dev. The queue was cut after these four (registered revision 3): the DINOv2-S reg4 KD runs were not executed, since 0 of 10 KD runs had helped. Relational KD from a 3-seed ensemble of the same family does not transfer anything the Proxy-Anchor loss does not already provide on 772 training images.

## Stage F: final light candidate, 3 seeds (`artifacts/research_v4/final/`)
DINOv3 ConvNeXt-T, last stage + head trained, 336 px, Proxy-Anchor a32 d0.3, 28.0 M params. Seeds 0/1/2 dev Hit@1 0.793 / 0.727 / 0.800 (**mean 0.773, SD 0.041**, mAP 0.674); B 0.711 / 0.698 / 0.698 (mean 0.702, mAP 0.634); C 0.693 / 0.680 / 0.700 (mean 0.691, mAP 0.646). The seed-0 search result (0.793) was optimistic by 2 points, and the seed spread is 4x that of DINOv2-B (SD 0.010).
Paired difference vs the reference single DINOv2-B (identical queries, seed-averaged, class bootstrap; `final/paired_small_minus_reference.csv`): dev -0.047 [-0.082, -0.009], B -0.123 [-0.163, -0.083], C -0.091 [-0.149, -0.031]; mAP -0.08 to -0.12 everywhere. The 3x parameter saving costs a reliable 5-12 points of Hit@1.

## Conclusion
1. **The target (>= 0.85 on dev, B and C jointly) was not reached by any system, light or heavy.** The best point estimates remain the single DINOv2-B (+ ResNet50) family at 0.82-0.84, and the earlier 0.858 was a B-only, non-replicating number.
2. **Parameter reduction that keeps accuracy: 281.8 M -> 85.9 M (single DINOv2-B + Proxy-Anchor, 3.3x fewer, 3.2x fewer MACs) or 109.9 M with the ResNet50 fusion (2.6x).** This is the usable result: the ensemble was never needed (research_v3 cost study, confirmed here by the KD runs, which show that a single model already carries the ensemble's similarity structure).
3. **Below ~86 M the accuracy drops reliably.** The best 28 M model (DINOv3 ConvNeXt-T at 336 px) loses 5 points on dev and 9-12 on B/C with wide seed variance; depth-truncated DINOv2-B is worse still; fusing two light models does not recover the gap; distillation, resolution, full fine-tuning, learning-rate changes and the DINOv3 ViT family do not close it.
4. **Zero-parameter tricks are exhausted:** metric variants (research_v3), PCA/NCA/LDA projections, DBA, query expansion, k-reciprocal re-ranking and multi-view TTA do not move dev Hit@1; only TTA shows an exploratory +0.03 on C at 7.5x embedding cost.
5. **What would move Hit@1 is data, not architecture:** 772 training images cap every backbone at ~0.79-0.82 on dev. The cross-source experiment (research_v3) already showed mixed SegPPD + PlantSeg training keeps SegPPD accuracy while adding a second domain; a larger labelled training pool, and a fresh untouched test set (docs/FRESH_TEST_PLAN_PA_RESNET.md) for confirmation, are the next steps. Quantum circuits were considered and rejected without experiment: at 256-d / 772 images a quantum-kernel or QSVM retrieval offers no plausible advantage over exact L2 and would only be simulated classically.

Recommended system for the report: **single DINOv2-B + Proxy-Anchor, exact L2, 256-d (85.9 M)**; optional +ResNet50-ft fusion (w = 0.05) for +0.013 on B (exploratory). The 28 M ConvNeXt-T is the documented low-resource alternative with its measured cost in accuracy.
