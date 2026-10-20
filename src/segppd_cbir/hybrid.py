"""Phase 3: complementary-error analysis and development-selected embedding fusion (Protocol B).
Fusion = concat( sqrt(w) * Za , sqrt(1-w) * Zb ) with Za, Zb L2-normalised => squared L2 = w*d_a^2 + (1-w)*d_b^2 (still exact Euclidean)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ext_dev as D
from .index import rank


def load_protocol_b():
    b = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str})
    return (b[b.role_B == "dev_query_B"].sort_values("image_id"), b[b.role_B == "gallery_B"].sort_values("image_id"),
            b[b.role_B == "test_query_B"].sort_values("image_id"), b)


def get(ids, Z, df):
    pos = {i: k for k, i in enumerate(ids)}
    return Z[[pos[i] for i in df["image_id"]]]


def top1(Q, G, ql, gl, gid):
    o, _ = rank(Q, G, gid, k=1)
    return (gl[o[:, 0]] == ql).astype(int), gl[o[:, 0]]


def complementary(Qa, Ga, Qb, Gb, dq, ga, na="A", nb="B"):
    ql, gl, gid = dq.class_id.to_numpy(), ga.class_id.to_numpy(), ga.image_id.to_numpy(dtype=str)
    ca, pa = top1(Qa, Ga, ql, gl, gid)
    cb, pb = top1(Qb, Gb, ql, gl, gid)
    tab = {f"both_correct": int(((ca == 1) & (cb == 1)).sum()), f"only_{na}": int(((ca == 1) & (cb == 0)).sum()),
           f"only_{nb}": int(((ca == 0) & (cb == 1)).sum()), "both_wrong": int(((ca == 0) & (cb == 0)).sum()), "n": len(ql)}
    # oracle ceiling (NOT a valid system; shows headroom only)
    tab["oracle_union_hit1_NOT_A_SYSTEM"] = float(((ca == 1) | (cb == 1)).mean())
    cls = pd.DataFrame({"class_id": ql, "a": ca, "b": cb}).groupby("class_id").mean()
    cls["diff_a_minus_b"] = cls["a"] - cls["b"]
    names = dict(zip(dq.class_id, dq.raw_class_label))
    cls["label"] = cls.index.map(names)

    def confusions(c, p):
        e = pd.DataFrame({"true": ql, "pred": p})[c == 0]
        return e.value_counts().rename("n").reset_index()
    return tab, cls.sort_values("diff_a_minus_b"), confusions(ca, pa), confusions(cb, pb), ca, cb


def fuse(Za, Zb, w):
    return np.concatenate([np.sqrt(w) * D.l2n(Za), np.sqrt(1 - w) * D.l2n(Zb)], axis=1)


def select_weight(Qa, Ga, Qb, Gb, dq, ga, grid=np.round(np.arange(0, 1.01, 0.1), 2)):
    ql, gl, gid = dq.class_id.to_numpy(), ga.class_id.to_numpy(), ga.image_id.to_numpy(dtype=str)
    rows = []
    for w in grid:
        s = D.quick_scores(fuse(Qa, Qb, w), fuse(Ga, Gb, w), ql, gl, gid)
        rows.append(dict(w_A=float(w), **{k: float(v) for k, v in s.items()}))
    return pd.DataFrame(rows)
