"""DEV protocol for extension tracks: selection happens ONLY here; test queries are never used for design choices."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import metrics as M
from .index import rank

DEV_PER_CLASS = 2
DEV_SEED = 7


def make_dev(split: pd.DataFrame):
    """From the (primary) test GALLERY, hold out 2 images/class as dev queries; dev gallery = the rest of that gallery."""
    g = split[split["role"] == "gallery_primary"].sort_values("image_id")
    rng = np.random.default_rng(DEV_SEED)
    dq = []
    for cid, sub in g.groupby("class_id"):
        dq += list(sub.index[rng.choice(len(sub), DEV_PER_CLASS, replace=False)])
    dev_q = g.loc[sorted(dq)]
    dev_g = g.drop(index=dev_q.index)
    return dev_q, dev_g


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build(blocks: dict, names):
    return np.concatenate([l2n(blocks[n]) for n in names], axis=1)


def expand_queries(Q, G, gid, m, alpha=1.0):
    """alpha-QE: new query = L2n(q + sum_i w_i g_i) over the top-m gallery items (w_i = 1/(1+rank)^alpha); exact L2 re-query."""
    if m == 0:
        return Q
    o, _ = rank(Q, G, gid, k=m)
    w = 1.0 / (1.0 + np.arange(m)) ** alpha
    Qn = Q + (G[o] * w[None, :, None]).sum(1)
    return Qn


def quick_scores(Q, G, qlab, glab, gid):
    o, _ = rank(Q, G, gid)
    ap, p1, p10 = [], [], []
    for i in range(len(Q)):
        rel = (glab[o[i]] == qlab[i]).astype(float)
        R = int((glab == qlab[i]).sum())
        ap.append(M.full_ap(rel, R)); p1.append(rel[0]); p10.append(rel[:10].mean())
    df = pd.DataFrame(dict(c=qlab, ap=ap, p1=p1, p10=p10))
    m = df.groupby("c").mean().mean()
    return dict(mAP=m.ap, P1=m.p1, P10=m.p10)


def search(blocks_by_scale, pos, dev_q, dev_g, max_comb=3, qe_grid=((0, 1.0), (2, 1.0), (4, 1.0))):
    names = [f"s{s}:{k}" for s, b in blocks_by_scale.items() for k in b]
    flat = {f"s{s}:{k}": v for s, b in blocks_by_scale.items() for k, v in b.items()}
    qi = [pos[i] for i in dev_q["image_id"]]; gi = [pos[i] for i in dev_g["image_id"]]
    ql, gl = dev_q["class_id"].to_numpy(), dev_g["class_id"].to_numpy()
    gid = dev_g["image_id"].to_numpy(dtype=str)
    rows = []
    for r in range(1, max_comb + 1):
        for combo in itertools.combinations(names, r):
            X = build(flat, combo)
            for m, a in qe_grid:
                Q, G = X[qi], X[gi]
                s = quick_scores(expand_queries(Q, G, gid, m, a), G, ql, gl, gid)
                rows.append(dict(recipe="+".join(combo), qe_m=m, qe_alpha=a, **s))
    return pd.DataFrame(rows).sort_values("mAP", ascending=False)
