# Retrieval protocols and metric definitions

Relevance = exact dataset category equality (a **label-agreement proxy**, not verified visual or agronomic similarity). Labels are used only to score results, never to filter candidates. Final stage everywhere: **exact Euclidean (L2) ranking**, ties broken by (distance, image_id); FAISS `IndexFlatL2` cross-check matches (`faiss_crosscheck.csv`; squared L2 and L2 give the same order). No self-match, no query in gallery.

## Protocol A (original, train-as-gallery)
Primary cohort: class eligible iff >=3+10 usable images -> 75 classes; 3 queries/class (225), gallery = remaining images of those classes (1,731). Secondary cohort: 26 classes, 1 query each (reported separately, `artifacts/metrics/summary_macro_micro.csv`). Seeds 42/1337/2024.
Trained systems under A used the gallery images (labels) for training, so the gallery overlaps training by design. **A's 225 test queries were evaluated during exploration, so every new result on A is exploratory.** A was audited (`artifacts/audits/protocol_audit.json`): dev queries disjoint from dev-run training; test queries absent from every training set; no duplicate group straddles; min pHash distance test-vs-training = 14 (> duplicate rule); one loose candidate pair was a different photo. Final A runs retrain on the full gallery (which contains the dev queries) *after* selection - by design.

## Protocol B (leakage-controlled; headline)
Per primary class (usable pool U_c, >=13):
* `test_query_B`: 3 images drawn from U_c minus A-test and A-dev queries (**locked**: not seen by any selection or evaluation until the single final run);
* `dev_query_B`: the same 2/class dev queries as A (selection, early stopping);
* the rest split ~50/50 into `train_B` (772 imgs) and `gallery_B` (809).
Training uses `train_B` only; retrieval = queries vs `gallery_B`. Sets are mutually disjoint at image **and** duplicate-group level (tests: `tests/test_protocols.py`). Classes are known (closed set). Min 4 train / 5 gallery positives per class. Gallery is smaller than A: **A and B metrics are not comparable.**
Splits: `artifacts/splits/protocolB_seed4242.csv`.

## Locked-test rules followed
Candidate list + headline definition frozen in `artifacts/metrics/locked_candidates.json` before the run; run once (`python -m segppd_cbir.final_locked ...`). Headline (declared in advance): mean over 3 seeds of DINOv2-B+SupCon single models, Hit@1. Ensembles and fusions are secondary. **The locked set is now consumed: further model selection needs a new locked set.**

## Metrics (per query q; R_q = relevant gallery items; rel_i in {0,1})
P@K = sum_{i<=K} rel_i / K; Recall@K = sum_{i<=K} rel_i / R_q; Hit@K = 1[any relevant in top K]; **Hit@1 = P@1**;
AP@K = sum_{i<=K} P@i*rel_i / min(R_q,K); full AP = same over the whole ranking / R_q; mAP = mean of full AP. Ceiling = min(R_q,K)/K.
Macro = mean of per-class means (headline); micro = query-weighted (equal here: 3 queries/class). CIs: 2,000-sample class-level bootstrap (seed 42); paired differences on identical queries with class-level bootstrap. Random baseline on the identical gallery: P@K = R_q/|G| (analytic), Hit/AP via simulation.
K in {1,3,5,10,20,50,100}; K=20/50/100 are capped by R_q (>=5 in B). kNN-vote accuracy is **not** reported as Hit@1.
