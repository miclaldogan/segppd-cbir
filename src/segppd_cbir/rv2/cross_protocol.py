"""Experiment B (cross-source representation learning): label alignment, manifests, feasibility audit and registration. Written before any mixed model is trained.
Unified label space: SegPPD classes of Protocol B train (75) + PlantSeg protocol-P classes (98); the 'strict' crop-disease pairs from external.MAPPING (reviewed before any model, never changed) share one id.
Loose pairs and same-name-but-unmapped pairs are NOT merged (conservative)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..external import MAPPING
from . import common as C

OUT = Path("artifacts/research_v3/cross"); P = pd.read_csv("artifacts/research_v2/trackB/protocol_P.csv"); S = C.segppd_sets()
tr = S["train"]; seg_cls = sorted(tr.class_id.unique()); ps_cls = sorted(P.label.unique()); ps_name = P.drop_duplicates("label").set_index("label").Disease
seg_name = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str}).drop_duplicates("class_id").set_index("class_id").raw_class_label.str.replace("_", " ")
strict = {v[0]: k for k, v in MAPPING.items() if v[1] == "strict"}  # PlantSeg disease -> SegPPD class id
ps_by_name = {n: l for l, n in ps_name.items()}
shared = [(d, cid, ps_by_name[d]) for d, cid in strict.items() if d in ps_by_name and cid in seg_cls]
uni, nxt = {}, 0
for c in seg_cls: uni[("seg", c)] = nxt; nxt += 1
for d, cid, pl in shared: uni[("ps", pl)] = uni[("seg", cid)]
for l in ps_cls:
    if ("ps", l) not in uni: uni[("ps", l)] = nxt; nxt += 1
rows = [dict(unified=u, source=s, source_label=l, name=(seg_name[l] if s == "seg" else ps_name[l])) for (s, l), u in uni.items()]
M = pd.DataFrame(rows).sort_values(["unified", "source"]); M["shared"] = M.groupby("unified").unified.transform("size") > 1; M.to_csv(OUT / "label_alignment.csv", index=False)
# conservative-alignment audit: same normalised name in both sources but NOT in the strict map; and strict pairs missing from one side
sn = {n: c for c, n in seg_name.items() if c in seg_cls}; unmapped_same = [n for n in ps_name.values if n in sn and (n not in strict)]
miss = [(d, "not in PlantSeg protocol P" if d not in ps_by_name else "not in SegPPD train_B") for d, cid in strict.items() if d not in ps_by_name or cid not in seg_cls]
# manifests
mk = lambda df, src, lab, name, role: pd.DataFrame(dict(source=src, Name=df[name].to_numpy(), role=role, unified=[uni[(src, l)] for l in df[lab]]))
man = pd.concat([mk(tr, "seg", "class_id", "image_id", "train"), mk(S["dev_q"], "seg", "class_id", "image_id", "dev"), mk(S["gallery_B"], "seg", "class_id", "image_id", "gallery"), mk(S["testB_q"], "seg", "class_id", "image_id", "test"),
                 mk(P[P.role == "train"], "ps", "label", "Name", "train"), mk(P[P.role == "dev"], "ps", "label", "Name", "dev"), mk(P[P.role == "test"], "ps", "label", "Name", "test")])
man.to_csv(OUT / "manifest_all.csv", index=False)
sh_ids = set(M[M.shared].unified); cnt = man[man.unified.isin(sh_ids)].groupby(["unified", "source", "role"]).size().unstack([1, 2]).fillna(0).astype(int); cnt.insert(0, "name", [M[M.unified == u].name.iloc[0] for u in cnt.index]); cnt.to_csv(OUT / "shared_class_counts.csv")
n_tr = man[man.role == "train"].groupby("source").size().to_dict(); n_cls = int(man[man.role == "train"].unified.nunique())
dup = dict(cross_source_pairs_cos_ge_0p85="0 (frozen DINOv2-B CLS; all 9,289 protocol-P images vs all 2,263 SegPPD images; PlantSeg copies of SegPPD images were already removed by the Track B audit, rule hash OR cos>=0.85)", pairs_cos_ge_0p80=744)
reg = dict(registered_before_any_mixed_training=True, experiment="B cross-source representation learning",
 labels=dict(unified_classes_in_training=n_cls, seg_train_classes=len(seg_cls), plantseg_classes=len(ps_cls), shared_strict=[dict(plantseg=d, seg_class_id=int(c), plantseg_label=int(p)) for d, c, p in shared], strict_pairs_unusable=miss, same_name_unmapped=unmapped_same,
             policy="only strict pairs are merged; loose pairs, and any same-name pair not in the reviewed strict list, stay separate classes"),
 splits=dict(seg="Protocol B (artifacts/splits/protocolB_seed4242.csv): train_B only for training; dev_query_B vs gallery_B for selection; test_query_B exploratory (consumed)", plantseg="protocol P (official Training/Validation/Test, duplicate-group disjoint; sha256[:12] 7c545a881b97)",
             train_images=n_tr, note="no image is moved between splits; no SegPPD or PlantSeg dev/test image is used for training"), duplicates=dup,
 conditions={"S0": "SegPPD-only (existing Track A Proxy-Anchor a32 d0.3, seeds 0-2) [source-only encoder]", "T0": "PlantSeg-only (existing Track B M3 Proxy-Anchor, seeds 0-2) [target-supervised encoder]",
             "X1": "mixed: SegPPD train_B + PlantSeg train, unified labels, Proxy-Anchor a32 d0.3, 60 steps/epoch, class-balanced PK batches (same budget as T0), 3 seeds",
             "X2": "X1 + ONE change: domain-balanced batches (8 classes drawn from the SegPPD pool + 8 from the PlantSeg pool, images from that source), 3 seeds",
             "X3": "X1 + ONE change: colour/lighting augmentation ('color', the only augmentation within the dev tie band in Track C), 3 seeds"},
 selection="epoch: mean of (SegPPD dev class-macro Hit@1 [dev_query_B vs gallery_B], PlantSeg dev accuracy [official Validation vs <=10 train images per class]); tie band 1/300; inside the band higher mean mAP; patience 6, 30 epochs max. Nothing else is tuned.",
 evaluation={"E1_plantseg_target": "queries = protocol-P Test (1,646; 10 shared-class subset reported separately), gallery = protocol-P train (98 classes), relevance = PlantSeg label",
             "E2_segppd_target": "queries = Protocol B test (225, 75 classes), gallery = gallery_B (809)", "E3_cross_domain_gallery": "queries = protocol-P Test images of the 10 shared classes, gallery = SegPPD gallery_B (75 classes), relevance = unified id (needs no PlantSeg gallery: tests transfer into the SegPPD gallery)"},
 regimes="S0 is source-only for E1 and E3; T0 is target-supervised for E1 and source-only for E2/E3; X* are mixed (supervised on both sources) and are never mixed with S0/T0 in one claim. Reported separately per evaluation.",
 comparisons="X1 vs T0 (budget-matched: same steps/epoch/classes sampler, PlantSeg-only) on E1/E2/E3; X2 vs X1; X3 vs X1; paired on identical queries, class-bootstrap CIs, 3 seeds averaged per query.",
 status="EXPLORATORY. No untouched labelled data exists: Protocol B/C tests and the PlantSeg test were inspected earlier. Selection uses only dev sets. Not a confirmatory test.",
 organ_analysis="Only verified organ labels may be used. None exist (the 121 provisional labels have a single AI annotator, unverified). No organ result is produced; a blank verification worksheet is exported instead.")
json.dump(reg, open(OUT / "registered.json", "w"), indent=1, default=int); print(json.dumps(dict(n_tr=n_tr, n_cls=n_cls, shared=len(shared), miss=miss, unmapped_same_name=unmapped_same), default=int)); print(cnt.to_string())
