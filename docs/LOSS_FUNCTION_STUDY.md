# Track A - DINOv2-B loss-function study (ArcFace vs Proxy-Anchor vs SupCon)

Registered before any run: `artifacts/research_v2/trackA/registered.json`. Code: `src/segppd_cbir/rv2/` (`train.py` runner, `make_configs.py`, `trackA_select.py`, `trackA_report.py`). Every run directory (`artifacts/research_v2/trackA/runs/*`) holds `config`, per-epoch curves (`result.json: history`), `best.pt` (sha256 recorded), embeddings of all 2,263 images and per-query CSVs. Historical results/checkpoints were not touched.

## 1. Provenance of the earlier "ArcFace 0.760 / Proxy-Anchor 0.748 / SupCon 0.717" table
Verified from `artifacts/metrics/protocolB_locked/`, `artifacts/extensions/B/*_log.json`, `docs/EXPERIMENT_RESULTS.md`: backbone **DINOv2-SMALL** (`vit_small_patch14_dinov2`), last 4 blocks, d_emb 256; split = **Protocol B locked test** (225 queries, 75 classes, gallery 809); ArcFace **1 seed** (m=0.3, s=30), Proxy-Anchor and SupCon = **3-seed means**; mAP values 0.603 / 0.645 / 0.658; the loss had been chosen on **dev mAP** (dev mAP 0.597 / 0.636 / 0.663). DINOv2-B had only ever been trained with SupCon. The Protocol B/C test sets are consumed, so everything below that touches them is **exploratory**.

## 2. Controlled protocol (identical across losses)
Official LVD-142M DINOv2-B init; last 4 blocks + final norm trainable; layer-wise lr decay 0.8, lr_bb 3e-5, lr_head 1e-3; d_emb 256; 224 px bicubic; flip-averaged L2-normalised embedding; 16x4 class-balanced batches, 20 steps/epoch, max 60 epochs, patience 10, AdamW, cosine lr, bf16; the same augmentation; seeds 0/1/2; `train_B` only (772 images, leakage and duplicate-group checks asserted at start of every run); exact squared-L2 retrieval; relevance = same class.
**Selection rule (fixed beforehand):** dev class-macro Hit@1; improvement needs more than one dev query (1/150 = 0.0067); inside that tie band the higher dev mAP wins - for epochs, for configurations and for the loss choice. Each loss gets its own justified hyper-parameter grid (seed 0, dev only): ArcFace margin {0.2,0.3,0.5}; Proxy-Anchor (alpha,delta) {(32,.1),(16,.1),(32,.3)} + proxy lr x10; SupCon tau {0.05,0.1,0.2}.

### Search stage (seed 0, dev only)
| config (seed 0) | loss hyper-parameters | best epoch | dev Hit@1 | dev mAP | dev P@10 | epochs run | train s |
|---|---|---|---|---|---|---|---|
| A_arc_s30_m0.5_s0 | {"s": 30.0, "m": 0.5} | 9 | 0.773 | 0.694 | 0.620 | 19 | 270 |
| A_arc_s30_m0.3_s0 | {"s": 30.0, "m": 0.3} | 24 | 0.760 | 0.684 | 0.617 | 34 | 483 |
| A_arc_s30_m0.2_s0 | {"s": 30.0, "m": 0.2} | 13 | 0.740 | 0.665 | 0.603 | 23 | 331 |
| A_pro_alpha16_delta0.1_s0 | {"alpha": 16.0, "delta": 0.1} | 45 | 0.820 | 0.755 | 0.663 | 55 | 783 |
| A_pro_alpha32_delta0.3_s0 | {"alpha": 32.0, "delta": 0.3} | 46 | 0.820 | 0.764 | 0.667 | 56 | 797 |
| A_pro_alpha32_delta0.1_plr10_s0 | {"alpha": 32.0, "delta": 0.1} + proxy lr x10 | 50 | 0.807 | 0.736 | 0.643 | 60 | 854 |
| A_pro_alpha32_delta0.1_s0 | {"alpha": 32.0, "delta": 0.1} | 53 | 0.807 | 0.739 | 0.650 | 60 | 853 |
| A_sup_tau0.1_s0 | {"tau": 0.1} | 36 | 0.800 | 0.748 | 0.663 | 46 | 654 |
| A_sup_tau0.05_s0 | {"tau": 0.05} | 28 | 0.787 | 0.729 | 0.645 | 38 | 541 |
| A_sup_tau0.2_s0 | {"tau": 0.2} | 12 | 0.773 | 0.709 | 0.625 | 22 | 313 |

Selected by the rule (`loss_selection_dev_only.json`, computed from dev results before any test number was looked at): ArcFace m=0.5; Proxy-Anchor alpha=32, delta=0.3 (tied with alpha=16 at 0.820, higher mAP); SupCon tau=0.1. Across losses (3-seed dev means) Proxy-Anchor and SupCon tie on dev Hit@1 (0.820 vs 0.820); the tie-break by dev mAP (0.757 vs 0.747) picks **Proxy-Anchor** as the loss carried into Tracks B and C.

## 3. Final stage: 3 matched seeds per loss
### Dev (selection set; optimistic because epochs/configs were chosen on it)
| Model | Hit@1 (macro) mean +/- SD | per-seed Hit@1 | class-bootstrap 95% CI (seed-mean) | P@5 | P@10 | Hit@5 | Hit@10 | R@10 | mAP +/- SD | query-weighted Hit@1 / mAP | intra/inter dist. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arcface (Hit@1-selected) | 0.776 +/- 0.014 | [0.78, 0.76, 0.7867] | [0.709, 0.842] | 0.732 | 0.628 | 0.909 | 0.938 | 0.610 | 0.703 +/- 0.009 | 0.776 / 0.703 | 0.40 |
| proxyanchor (Hit@1-selected) | 0.820 +/- 0.007 | [0.82, 0.8133, 0.8267] | [0.762, 0.876] | 0.762 | 0.663 | 0.911 | 0.929 | 0.653 | 0.757 +/- 0.007 | 0.820 / 0.757 | 0.29 |
| supcon (Hit@1-selected) | 0.820 +/- 0.024 | [0.8, 0.8467, 0.8133] | [0.760, 0.878] | 0.748 | 0.664 | 0.922 | 0.938 | 0.650 | 0.747 +/- 0.001 | 0.820 / 0.747 | 0.21 |
| supcon tau0.1 (HISTORICAL, mAP-selected) | 0.813 +/- 0.013 | [0.8, 0.8267, 0.8133] | - | 0.755 | 0.667 | 0.924 | 0.936 | 0.653 | 0.750 +/- 0.002 | 0.813 / 0.750 | 0.21 |

### Protocol B test - EXPLORATORY (consumed set)
| Model | Hit@1 (macro) mean +/- SD | per-seed Hit@1 | class-bootstrap 95% CI (seed-mean) | P@5 | P@10 | Hit@5 | Hit@10 | R@10 | mAP +/- SD | query-weighted Hit@1 / mAP | intra/inter dist. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arcface (Hit@1-selected) | 0.793 +/- 0.011 | [0.8044, 0.7911, 0.7822] | [0.738, 0.843] | 0.737 | 0.631 | 0.942 | 0.963 | 0.616 | 0.708 +/- 0.011 | 0.793 / 0.708 | 0.39 |
| proxyanchor (Hit@1-selected) | 0.825 +/- 0.010 | [0.8311, 0.8133, 0.8311] | [0.776, 0.871] | 0.757 | 0.656 | 0.933 | 0.972 | 0.646 | 0.753 +/- 0.007 | 0.825 / 0.753 | 0.29 |
| supcon (Hit@1-selected) | 0.806 +/- 0.009 | [0.8133, 0.8089, 0.7956] | [0.751, 0.856] | 0.753 | 0.649 | 0.941 | 0.970 | 0.639 | 0.740 +/- 0.003 | 0.806 / 0.740 | 0.21 |
| supcon tau0.1 (HISTORICAL, mAP-selected) | 0.807 +/- 0.010 | [0.8133, 0.7956, 0.8133] | - | 0.752 | 0.651 | 0.942 | 0.967 | 0.640 | 0.742 +/- 0.005 | 0.807 / 0.742 | 0.21 |

### Protocol C test - EXPLORATORY (consumed set)
| Model | Hit@1 (macro) mean +/- SD | per-seed Hit@1 | class-bootstrap 95% CI (seed-mean) | P@5 | P@10 | Hit@5 | Hit@10 | R@10 | mAP +/- SD | query-weighted Hit@1 / mAP | intra/inter dist. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arcface (Hit@1-selected) | 0.744 +/- 0.028 | [0.7533, 0.7667, 0.7133] | [0.664, 0.820] | 0.697 | 0.560 | 0.918 | 0.964 | 0.660 | 0.696 +/- 0.005 | 0.765 / 0.710 | 0.39 |
| proxyanchor (Hit@1-selected) | 0.782 +/- 0.025 | [0.7533, 0.7933, 0.8] | [0.713, 0.847] | 0.733 | 0.598 | 0.920 | 0.958 | 0.716 | 0.751 +/- 0.002 | 0.786 / 0.759 | 0.28 |
| supcon (Hit@1-selected) | 0.771 +/- 0.004 | [0.7667, 0.7733, 0.7733] | [0.696, 0.842] | 0.716 | 0.583 | 0.911 | 0.958 | 0.688 | 0.731 +/- 0.010 | 0.786 / 0.747 | 0.21 |
| supcon tau0.1 (HISTORICAL, mAP-selected) | 0.767 +/- 0.007 | [0.7667, 0.7733, 0.76] | - | 0.719 | 0.583 | 0.913 | 0.958 | 0.688 | 0.733 +/- 0.012 | 0.782 / 0.749 | 0.20 |

The "HISTORICAL" row is the original mAP-selected SupCon (tau 0.1, DINOv2-B; embeddings of the original D5 runs), kept separate from the new Hit@1-selected checkpoints; re-evaluated with the same code it reproduces the earlier numbers.

## 4. Paired comparisons on identical queries (seed-averaged per-query differences, class bootstrap; per-seed McNemar)
| A vs B | set | Hit@1 diff [95% class-bootstrap CI] | mAP diff [95% CI] | per-seed exact McNemar p (Hit@1) |
|---|---|---|---|---|
| arcface - supcon | dev | -0.044 [-0.091, -0.002] | -0.044 [-0.062, -0.026] | [0.6291, 0.0044, 0.5235] |
| arcface - supcon | testB_exploratory | -0.013 [-0.050, +0.025] | -0.033 [-0.051, -0.014] | [0.8679, 0.6358, 0.7493] |
| arcface - supcon | testC_exploratory | -0.027 [-0.073, +0.016] | -0.035 [-0.057, -0.014] | [0.8145, 1.0, 0.2101] |
| proxyanchor - supcon | dev | +0.000 [-0.033, +0.029] | +0.010 [-0.008, +0.028] | [0.6072, 0.2266, 0.8036] |
| proxyanchor - supcon | testB_exploratory | +0.019 [-0.013, +0.053] | +0.012 [-0.006, +0.031] | [0.5413, 1.0, 0.1849] |
| proxyanchor - supcon | testC_exploratory | +0.011 [-0.038, +0.064] | +0.021 [-0.004, +0.049] | [0.7905, 1.0, 0.8036] |
| arcface - proxyanchor | dev | -0.044 [-0.082, -0.009] | -0.054 [-0.072, -0.035] | [0.2379, 0.0768, 0.146] |
| arcface - proxyanchor | testB_exploratory | -0.033 [-0.068, +0.004] | -0.045 [-0.062, -0.030] | [0.4177, 0.4869, 0.0801] |
| arcface - proxyanchor | testC_exploratory | -0.038 [-0.082, +0.004] | -0.055 [-0.076, -0.036] | [1.0, 1.0, 0.1338] |

Figures: `artifacts/research_v2/figures/A1_hit1_vs_map.png` (Hit@1 vs mAP), `A2_pk_rk.png` (Precision@K and Recall@K per loss; PDF versions alongside). Per-query predictions: `runs/*/per_query_*.csv`. Most frequent dev confusions per loss: `dev_top_confusions_seed0.json` (e.g. tomato mold -> tomato late blight, lemon sooty mold -> orange canker, apple rust -> apple scab, banana cordana -> sigatoka, bean rust -> peanut rust: biologically/visually similar pairs, mostly the same for every loss).

## 5. Cost
| loss | mean train s | mean epochs | peak VRAM GB | embed ms/image (GPU, incl. flip) | best epochs | checkpoint sha256[:12] |
|---|---|---|---|---|---|---|
| arcface | 435 | 30.7 | 2.98 | 9.9 | [9, 30, 23] | ['468e286a2e49', '3d9b56e3d10e', 'ef542eaf7ac1'] |
| proxyanchor | 617 | 43.3 | 2.98 | 10.7 | [46, 33, 21] | ['e8a01b981e7e', '83df0565c9f0', '3f6f73d4089c'] |
| supcon | 630 | 44.3 | 2.98 | 9.6 | [36, 40, 27] | ['45e61ff25d98', '7fafbc6d507b', '0f65c1114b49'] |
Exact retrieval over the 809-image gallery: 0.26 ms/query for any of the three 256-D embeddings (`artifacts/metrics/protocolB_locked/retrieval_latency.csv`).

## 6. Answers
1. **Best DINOv2-B Hit@1:** Proxy-Anchor on every set (dev 0.820, B-test 0.825, C-test 0.782), but its lead over SupCon (dev 0.820 / B 0.806 / C 0.771) is within noise (paired differences +0.000, +0.019, +0.011; every CI includes 0). ArcFace is lowest (0.776 / 0.793 / 0.744).
2. **Best mAP:** Proxy-Anchor (0.757 / 0.753 / 0.751) > SupCon (0.747 / 0.740 / 0.731) > ArcFace (0.703 / 0.708 / 0.696).
3. **Meaningful?** Only the ArcFace deficit is reliable in mAP (-0.033 to -0.044 vs SupCon, CIs exclude 0; ArcFace vs Proxy-Anchor -0.045 to -0.055). For Hit@1 no pairwise difference is statistically established on B/C (ArcFace vs Proxy-Anchor: dev -0.044 [-0.082, -0.009]); practically Proxy-Anchor and SupCon are equivalent (<= 2 points) and both are 3-5 points above ArcFace in mAP. The DINOv2-S pattern (ArcFace best at Hit@1, SupCon best at mAP) did **not** replicate on DINOv2-B: here a higher mAP went together with an equal-or-higher Hit@1.
4. Selecting by Hit@1 instead of mAP changed nothing material for SupCon (new vs historical: dev 0.820 vs 0.813; B-test 0.806 vs 0.807).
5. **Recommendation:** for Hit@1 and for overall retrieval quality the dev-selected choice is **Proxy-Anchor (alpha 32, delta 0.3)** - best or tied-best on every metric with the same cost as SupCon (~10 min of training). Caveats: tie with SupCon on dev Hit@1 (decided by dev mAP), three seeds only, test sets consumed, a single backbone/split/hyper-parameter budget. No new independent test result is claimed.
