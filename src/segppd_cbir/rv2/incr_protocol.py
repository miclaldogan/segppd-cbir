"""Experiment A (incremental novel-class retrieval): class folds, support draws and registration. No model is involved; written before any training.
Class ordering = sha256(salt + class name) (not related to difficulty, size or any observed result). 98 eligible classes of PlantSeg protocol P are cut into 5 consecutive
folds of ~20 by that order; in fold f the 20 classes are NOVEL and the rest BASE. Every class is novel exactly once (exploratory class-level cross-validation).
Fold 0 is the pre-declared primary split, folds 1-4 are replications. Support draws: for each novel class a seeded permutation of its TRAIN images (one image per duplicate group first),
nested so the k=1 support is contained in k=3, 5, 10. 5 draws."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("artifacts/research_v3/incr"); SALT = "segppd-cbir/expA/v1"; N_FOLD, N_DRAW, SHOTS = 5, 5, (1, 3, 5, 10)
P = pd.read_csv("artifacts/research_v2/trackB/protocol_P.csv"); cls = P.drop_duplicates("label").sort_values("label")[["label", "Disease"]]
cls["key"] = [hashlib.sha256((SALT + d).encode()).hexdigest() for d in cls.Disease]; cls = cls.sort_values("key").reset_index(drop=True); cls["rank"] = np.arange(len(cls))
cls["fold"] = np.minimum(cls["rank"] * N_FOLD // len(cls), N_FOLD - 1); cls.to_csv(OUT / "class_folds.csv", index=False)
rows = []
for f in range(N_FOLD):
    nov = set(cls[cls.fold == f].label)
    for d in range(N_DRAW):
        for c in sorted(nov):
            t = P[(P.role == "train") & (P.label == c)]; r = np.random.default_rng([ord("A"), f, d, c])
            first = t.drop_duplicates("group"); rest = t[~t.Name.isin(first.Name)]
            order = list(first.sample(frac=1, random_state=int(r.integers(1 << 30))).Name) + list(rest.sample(frac=1, random_state=int(r.integers(1 << 30))).Name)
            for i, n in enumerate(order[:10]): rows.append(dict(fold=f, draw=d, label=c, rank_in_draw=i + 1, Name=n))
S = pd.DataFrame(rows); S.to_csv(OUT / "support_draws.csv", index=False)
assert S.groupby(["fold", "draw", "label"]).size().min() == 10
nov_sizes = cls.groupby("fold").size().to_dict()
reg = dict(registered_before_any_training=True, experiment="A incremental novel-class retrieval", salt=SALT, folds=N_FOLD, fold_sizes=nov_sizes, primary_fold=0, support_draws=N_DRAW, shots=list(SHOTS),
 data="PlantSeg protocol P (artifacts/research_v2/trackB/protocol_P.csv, sha256[:12] 7c545a881b97): 98 classes. Novel classes: only their TRAIN images as support (1/3/5/10) and their official TEST images as queries. Novel dev images are never used.",
 encoder_training=dict(init="DINOv2-B lvd142m (no plant labels)", data="TRAIN images of BASE classes only", selection="epoch chosen on BASE-class official-validation images only (Hit@1, tie = 1 query, then mAP); config = Track B M3 config with NO tuning (Proxy-Anchor a32 d0.3, last 4 blocks, 30 epochs max, patience 6, seed 0)",
                  frozen_after="best.pt saved; embeddings of every image computed once; no further training of any kind"),
 scenarios=dict(S1_novel_only="gallery = k supports per novel class; queries = novel test; chance = 1/n_novel",
                S2_open_gallery="gallery = ALL base-class train images + k supports per novel class; queries = novel test (the headline setting); 98-way",
                S2_base_check="same gallery, queries = base test: measures interference/forgetting vs the same base queries against the base-only gallery"),
 methods=dict(CBIR="exact squared-L2 ranking, 1-NN label; ties by image id", NCP="nearest class prototype: mean embedding of the class's reference images (base: all base train, novel: its k supports); exact L2 to prototypes; no learned parameters",
              LR="multinomial logistic regression on the same L2-normalised 256-d embeddings, trained on gallery images (base train + supports), C=10 FIXED in advance (not tuned), balanced class weights; this is the 'retrain the classifier head' alternative",
              frozen_reference="same three methods on raw frozen DINOv2-B CLS (never fine-tuned) to isolate what base-class metric learning contributes",
              oracle_reference="Track B M3 (trained on ALL 98 classes incl. these as labelled) restricted to the same classes: an upper bound that violates the novel-class rule; NOT a competitor"),
 metrics=["Hit@1 (class-macro over novel classes, headline)", "Hit@5", "P@5 (upper bound min(k,5)/5)", "Recall@10", "mAP (relevant = reference images of the class in the gallery)", "base Hit@1 before/after insertion", "gallery insertion seconds", "memory bytes", "retraining required (yes/no)"],
 statistics="class-level bootstrap (2000) over novel classes pooled across folds, averaged over support draws; paired differences between methods on identical queries and identical supports; exact McNemar on pooled queries as a secondary check. No multiplicity claims beyond the paired CIs.",
 status="Exploratory class-level cross-validation inside PlantSeg. The official Test images were not used for any selection, but part of PlantSeg was inspected in earlier experiments (see registered.json of Track B), so this is NOT an independent external test.",
 hypotheses="H1: with k>=1, base-trained CBIR retrieves novel classes far above chance without retraining. H2: NCP >= CBIR at k>=5 (prototype averaging suppresses noisy supports); we make no prediction at k=1. H3: LR on 256-d embeddings is competitive with CBIR at k>=5 but costs a refit per insertion. No hypothesis about superiority is assumed.")
json.dump(reg, open(OUT / "registered.json", "w"), indent=1); print(nov_sizes); print(cls.head(8))
