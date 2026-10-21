# Distance-metric study (reference model: DINOv2-B + Proxy-Anchor, 3 seeds)

No training. Existing embeddings (`artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s{0,1,2}`), code `src/segppd_cbir/rv2/metric_study.py`, outputs `artifacts/research_v3/metrics/`. Registered metric stays exact squared L2; the others are comparisons. Protocol B/C tests are consumed, so those columns are **exploratory**; selection used the dev set only (Hit@1, tie band 1/150, then dev mAP).

Variants: L2; cosine; inner product (on L2-normalised embeddings these rank identically to L2: checked, identical in every cell); L1; PCA-whitening fitted on train_B embeddings only, power alpha in {0.25, 0.5, 1}, dims in {256, 128, 64}, then L2-normalise + L2.

Hit@1 (class-macro, mean of 3 seeds) / mAP:

| Variant | dev | B test (expl.) | C test (expl.) |
|---|---|---|---|
| L2 = cosine = dot (reference) | 0.820 / 0.757 | 0.825 / 0.753 | 0.782 / 0.751 |
| L1 | 0.818 / 0.755 | 0.827 / 0.751 | 0.791 / 0.751 |
| whiten a=0.25, 128-d (dev-selected) | 0.816 / 0.760 | 0.827 / 0.757 | 0.796 / 0.756 |
| whiten a=0.5, 128-d | 0.820 / 0.755 | 0.830 / 0.752 | 0.787 / 0.752 |
| whiten a=1.0 (full), 256-d | 0.631 / 0.324 | 0.646 / 0.320 | 0.640 / 0.334 |

Paired differences vs L2 (class bootstrap), L1: dev -0.002 [-0.013, 0.007], B +0.002 [-0.009, 0.012], C +0.009 [-0.002, 0.022]. Whitening a=0.25/128: dev -0.004 [-0.013, 0.002], B +0.002 [-0.009, 0.012], C +0.013 [0.004, 0.027].

Reading: no variant beats L2 on dev. The dev rule picked a whitening variant only because six variants sit inside the one-query tie band and it has the highest dev mAP (0.760 vs 0.757). Its gain on C test (+0.013) is exploratory, uncorrected for the 13 variants tried, and absent on dev and B. Full whitening (alpha=1, 256-d) amplifies noise directions and is much worse. Conclusion: the metric is not a bottleneck; exact L2/cosine stays the reference. Not tested: other backbones' metrics, learned metrics, non-normalised embeddings.

## Added: 3-seed ensembles (same protocol, `metric_study.py PA_single PA_ensemble SUPCON_ensemble`)
Ensemble = concatenation of sqrt(1/3)·each unit-norm seed embedding (768-d; squared L2 = mean of the three squared L2). Hit@1 (mAP); selection on dev only.

| Model | Metric | dev | B test (expl.) | C test (expl.) |
|---|---|---|---|---|
| PA ensemble | L2 = cosine = dot | 0.813 (0.768) | 0.836 (0.766) | 0.813 (0.766) |
| | L1 | 0.800 (0.767) | 0.831 (0.765) | 0.813 (0.767) |
| | whiten a=0.25, 128-d (dev-selected) | 0.820 (0.771) | 0.836 (0.768) | 0.833 (0.772) |
| SupCon ensemble (the report's 0.822) | L2 = cosine = dot | 0.807 (0.761) | 0.822 (0.754) | 0.747 (0.739) |
| | L1 | 0.800 (0.761) | 0.822 (0.754) | 0.747 (0.739) |
| | whiten a=0.5, 64-d (dev-selected) | 0.827 (0.770) | 0.836 (0.767) | 0.747 (0.746) |

Paired vs L2 (class bootstrap), selected whitening: PA ensemble dev +0.007 [-0.020, 0.033], B 0.000 [-0.018, 0.018], C +0.020 [0.000, 0.047]; SupCon ensemble dev +0.020 [-0.007, 0.053], B +0.013 [-0.013, 0.040], C 0.000 [-0.053, 0.053]. L1: no gain (dev -0.013 / -0.007, tests 0). Conclusion unchanged: no metric is reliably better than exact L2; whitening gains are inconsistent across sets/models, uncorrected for 13 variants, and exploratory. Files: `artifacts/research_v3/metrics/metric_study_{mean,by_seed}_{PA_single,PA_ensemble,SUPCON_ensemble}.csv`, `per_query_*.csv`, `metric_selection_dev_only_*.json`. The report (Ek, Tablo 8) carries the same numbers, plus the Chamfer re-ranking summary from Track C.

Table-4-format metrics (Hit@1, P@10, Hit@10, mAP, macro-F1, pairwise ROC-AUC; Protocol B test, exploratory): `artifacts/research_v3/metrics/metric_study_tableB_format.csv` (code `rv2/metric_table_b.py`); this is report Tablo 8. The SupCon-ensemble L2 row reproduces report Tablo 4 exactly (0.822 / 0.662 / 0.969 / 0.754 / 0.812 / 0.991).
