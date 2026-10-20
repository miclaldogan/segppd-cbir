"""Deterministic query/gallery split (no training split exists: the backbone is frozen).

Rules, fixed before any retrieval score was computed:
  usable pool   = images with dup_status in {unique, kept_representative}
  PRIMARY       class eligible iff usable count >= N_QUERY + MIN_GALLERY_POS (3 + 10 = 13);
                3 queries per eligible class sampled with a seeded RNG; PRIMARY gallery = all other
                usable images of eligible classes (the other classes' images are NOT in this gallery).
  SECONDARY     classes with 2 <= usable count < 13: 1 query per class; SECONDARY gallery = PRIMARY gallery
                + the remaining usable images of those classes (primary queries stay out). Reported separately.
  Group safety  every duplicate group is reduced to <=1 usable member, so no group can straddle the split;
                this is asserted anyway (check_split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_QUERY = 3
MIN_GALLERY_POS = 10
SEEDS = (42, 1337, 2024)  # 42 = headline; others = sensitivity, fixed in advance


def make_split(inv: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    inv = inv.copy()
    usable = inv["dup_status"].isin(["unique", "kept_representative"])
    n_use = inv[usable].groupby("class_id").size()
    elig = set(n_use[n_use >= N_QUERY + MIN_GALLERY_POS].index)
    sec = set(n_use[(n_use >= 2) & (n_use < N_QUERY + MIN_GALLERY_POS)].index)
    rng = np.random.default_rng(seed)
    role = pd.Series("excluded_duplicate_or_conflict", index=inv.index, dtype=object)
    for cid in sorted(n_use.index):
        idx = inv.index[usable & (inv["class_id"] == cid)]
        order = inv.loc[idx].sort_values("image_id").index  # canonical order -> reproducible given seed
        if cid in elig:
            q = set(rng.choice(len(order), N_QUERY, replace=False))
            for k, ix in enumerate(order):
                role[ix] = "query_primary" if k in q else "gallery_primary"
        elif cid in sec:
            q = int(rng.integers(len(order)))
            for k, ix in enumerate(order):
                role[ix] = "query_secondary" if k == q else "gallery_secondary_extra"
        else:
            for ix in order:
                role[ix] = "unscorable_class"
    inv["role"] = role
    inv["cohort_class"] = inv["class_id"].map(lambda c: "primary" if c in elig else ("secondary" if c in sec else "unscorable"))
    inv["seed"] = seed
    return inv


def check_split(m: pd.DataFrame) -> dict:
    q = m[m["role"].isin(["query_primary", "query_secondary"])]
    g = m[m["role"].isin(["gallery_primary", "gallery_secondary_extra"])]
    assert set(q["image_id"]).isdisjoint(g["image_id"]), "query/gallery image overlap"
    assert set(q["image_path"]).isdisjoint(g["image_path"]), "path overlap"
    assert set(q["image_sha256"]).isdisjoint(g["image_sha256"]), "byte-identical image on both sides"
    assert set(q["group_id"]).isdisjoint(g["group_id"]), "duplicate group straddles split"
    assert q["image_id"].is_unique and g["image_id"].is_unique
    return {"n_query": len(q), "n_gallery": len(g)}
