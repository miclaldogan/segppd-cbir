"""Conservative duplicate/provenance groups (union-find over exact-SHA or pHash-near pairs).

Policy (declared before any retrieval score existed):
  * pair = identical image SHA-256  OR  (pHash Hamming <= PHASH_MAX AND dHash Hamming <= DHASH_MAX)
  * a group whose members carry >1 distinct class_id is LABEL-CONFLICTED: every member is excluded
  * a same-class group keeps its lowest image_id as representative; the rest are excluded as duplicates
Nothing is deleted from disk; exclusion is a manifest column.
"""
from __future__ import annotations

import imagehash
import numpy as np
import pandas as pd

PHASH_MAX = 12
DHASH_MAX = 18  # chosen by visual review of all pairs with pHash<=14 (true dups: dHash<=17; false positives: >=20)


def _find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def build_groups(df: pd.DataFrame, phash_max: int = PHASH_MAX, dhash_max: int = DHASH_MAX) -> pd.DataFrame:
    """df needs: image_id, class_id, image_sha256, phash. Returns df + group_id, dup_status, dup_reason."""
    df = df.reset_index(drop=True).copy()
    n = len(df)
    ph = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in df["phash"]])
    P = (ph[:, None, :] != ph[None, :, :]).sum(2)
    dh = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in df["dhash"]])
    D = (dh[:, None, :] != dh[None, :, :]).sum(2)
    sha = df["image_sha256"].to_numpy(dtype=object)
    same_sha = sha[:, None] == sha[None, :]
    near = (same_sha | ((P <= phash_max) & (D <= dhash_max))) & ~np.eye(n, dtype=bool)
    parent = list(range(n))
    for i, j in np.argwhere(np.triu(near)):
        ri, rj = _find(parent, i), _find(parent, j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [_find(parent, i) for i in range(n)]
    df["group_id"] = [f"g{df['image_id'][r]}" for r in roots]
    df["group_size"] = df.groupby("group_id")["image_id"].transform("size")
    df["group_n_classes"] = df.groupby("group_id")["class_id"].transform("nunique")
    status, reason = [], []
    rep = df.groupby("group_id")["image_id"].transform("min")
    for r in df.itertuples():
        if r.group_size == 1:
            status.append("unique"); reason.append("")
        elif r.group_n_classes > 1:
            status.append("excluded"); reason.append("label_conflict_duplicate_group")
        elif r.image_id == rep[r.Index]:
            status.append("kept_representative"); reason.append("")
        else:
            status.append("excluded"); reason.append(f"duplicate_of:{rep[r.Index]}")
    df["dup_status"], df["dup_reason"] = status, reason
    return df
