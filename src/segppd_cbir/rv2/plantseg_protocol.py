"""Builds and registers the Track-B data protocol P from audit outputs (no model involved).
Eligibility (fixed before any model is evaluated): after removing SegPPD copies and label-conflict duplicate groups, enforcing group-disjointness
(dev/test images whose duplicate group has a training image are dropped; test images whose group has a validation image are dropped) and keeping one image per
duplicate group in dev/test, a class is eligible iff train>=10, validation>=2, test>=5. Ineligible classes are dropped from ALL methods; no class is dropped for being hard."""
import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path("artifacts/research_v2/trackB")
ok = pd.read_csv(OUT / "audit_filtered_images.csv"); cnt = pd.read_csv(OUT / "audit_class_split_counts_after_filters.csv", index_col=0)
elig = cnt[(cnt.Training >= 10) & (cnt.Validation >= 2) & (cnt.Test >= 5)].index
P = ok[ok.Disease.isin(elig)].copy(); P["role"] = P.Split.map({"Training": "train", "Validation": "dev", "Test": "test"})
cls = {d: i for i, d in enumerate(sorted(elig))}; P["label"] = P.Disease.map(cls)
P[["Name", "Plant", "Disease", "label", "role", "Split", "group", "URL", "url_host", "w", "h", "Resolution"]].to_csv(OUT / "protocol_P.csv", index=False)
tr_url = set(P[P.role == "train"].URL); P["url_disjoint"] = ~P.URL.isin(tr_url)
summary = dict(n_classes=len(elig), dropped_classes=sorted(set(cnt.index) - set(elig)), n_train=int((P.role == "train").sum()), n_dev=int((P.role == "dev").sum()), n_test=int((P.role == "test").sum()),
               test_url_disjoint=int(((P.role == "test") & P.url_disjoint).sum()), crops=int(P.Plant.nunique()), min_train_per_class=int(cnt.loc[elig].Training.min()), min_test_per_class=int(cnt.loc[elig].Test.min()))
reg = dict(registered_before_any_model_on_plantseg_protocol_P=True, question="Given equal training information and a common PlantSeg evaluation set, does nearest-neighbour CBIR beat a direct classification head?",
           data=summary, protocol_file=str(OUT / "protocol_P.csv"), protocol_sha256_12=hashlib.sha256(open(OUT / "protocol_P.csv", "rb").read()).hexdigest()[:12],
           split_policy="official PlantSeg Training/Validation/Test fields; cross-split duplicate groups resolved by dropping the later-split members (never by moving images); SegPPD copies removed everywhere; label-conflict groups removed; dev = official Validation (selection of epochs and of regularisation), test = official Test.",
           status_of_test="The official Test split was NOT used for any model selection, but PlantSeg images of 11 mapped classes (all splits) were already evaluated in the earlier external and ChatLeaf-matched experiments (inspected, never tuned on). This track is therefore a CONTROLLED EXPLORATORY comparison; an untouched external confirmation set does not exist.",
           methods={"M1": "frozen DINOv2-B CLS + L2-regularised logistic regression (C chosen on dev from {0.01,0.1,1,10,100})", "M2": "frozen DINOv2-B CLS + exact-L2 1-NN over the training images (gallery = training set)",
                    "M3": "DINOv2-B (last 4 blocks) metric-learned on PlantSeg train with the loss selected in Track A (dev Hit@1) + exact-L2 1-NN over the training images",
                    "M4": "logistic regression on the same M3 embedding (same training images)", "M5": "supervised fine-tuning of the same last-4 blocks with a linear head + cross-entropy, same budget (classification baseline)",
                    "M3-src": "extra, separate label: embedding trained on SegPPD only (D5ck) with PlantSeg training images as gallery"},
           identical_for_all="eligible classes, training/dev/test images, labels, 224 px preprocessing from the 256x256 cache (data/external/plantseg_256), flip-averaged embeddings",
           metrics=["accuracy/Hit@1", "macro-F1", "per-class precision/recall", "confusion matrix", "exact McNemar on identical test images", "image-level bootstrap 95% CI", "P@5, P@10, Recall@K, mAP (retrieval methods only)", "latency and storage"],
           sensitivity="URL-disjoint test subset (test images whose source URL is not shared with a training image).")
json.dump(reg, open(OUT / "registered.json", "w"), indent=1, default=str); print(json.dumps(summary, indent=1)); print(reg["protocol_sha256_12"])
