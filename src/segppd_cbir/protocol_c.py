"""Protocol C: a second locked query set carved from Protocol B's gallery (images no model was ever trained on).

Per primary class c with gallery_B size n_c:  n_c >= 7 -> 2 queries, 5 <= n_c <= 6 -> 1 query, taken from gallery_B (seeded).
Those images leave the gallery (gallery_C = gallery_B minus queries). train_B, dev and test_B are untouched, so the frozen
Protocol B models are valid for C with no retraining. Group-level disjointness from gallery_C is inherited from B.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .protocols import nearest_dups

C_SEED = 777


def build(b: pd.DataFrame, seed: int = C_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    role = pd.Series("not_in_protocol_C", index=b.index, dtype=object)
    role[b.role_B == "train_B"] = "train_B"
    role[b.role_B == "dev_query_B"] = "dev_query_B"
    role[b.role_B == "test_query_B"] = "test_query_B_consumed"
    for cid, sub in b[b.role_B == "gallery_B"].groupby("class_id"):
        sub = sub.sort_values("image_id")
        nq = 2 if len(sub) >= 7 else 1
        q = sub.index[rng.choice(len(sub), nq, replace=False)]
        role[sub.index] = "gallery_C"
        role[q] = "test_query_C"
    out = b.copy()
    out["role_C"] = role
    return out


def audit(c: pd.DataFrame) -> dict:
    q, g = c[c.role_C == "test_query_C"], c[c.role_C == "gallery_C"]
    tr = c[c.role_C == "train_B"]
    ids = lambda d: set(d.image_id)
    grp = lambda d: set(d.group_id)
    res = dict(n_query=len(q), n_gallery=len(g), n_classes=int(q.class_id.nunique()),
               queries_per_class=q.groupby("class_id").size().value_counts().to_dict(),
               image_overlap_q_g=len(ids(q) & ids(g)), image_overlap_q_train=len(ids(q) & ids(tr)),
               group_overlap_q_g=len(grp(q) & grp(g)), group_overlap_q_train=len(grp(q) & grp(tr)),
               sha_overlap_q_gallery=len(set(q.image_sha256) & set(g.image_sha256)),
               every_query_class_has_gallery=bool(set(q.class_id) <= set(g.class_id)),
               min_gallery_per_class=int(g.groupby("class_id").size().min()))
    inv = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str})
    if "phash" in inv.columns:
        m = inv.set_index("image_id")
        for nm, ref in (("gallery", g), ("train", tr)):
            nd, dmin = nearest_dups(m.loc[q.image_id].reset_index(), m.loc[ref.image_id].reset_index())
            res[f"min_phash_dist_vs_{nm}"] = dmin
            res[f"near_dup_pairs_vs_{nm}"] = int(len(nd))
            res[f"near_dup_same_class_vs_{nm}"] = int((nd.q_class == nd.ref_class).sum()) if len(nd) else 0
    return res


if __name__ == "__main__":
    b = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str})
    c = build(b)
    c.to_csv("artifacts/splits/protocolC_seed777.csv", index=False)
    a = audit(c)
    Path("artifacts/audits/protocolC_audit.json").write_text(json.dumps(a, indent=1, default=str))
    print(json.dumps(a, indent=1, default=str))
