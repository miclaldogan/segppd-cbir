"""ONE-SHOT test evaluation of extension configs chosen on DEV. Same primary queries/gallery/relevance/K as the baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ext_dev as D
from . import ext_trained as TR
from . import evaluate as E
from .index import rank

OUT = Path("artifacts/metrics")


def _orders(Q, G, gid, qe_m=0, qe_alpha=1.0):
    Q = D.expand_queries(Q, G, gid, qe_m, qe_alpha)
    o, _ = rank(Q, G, gid)
    return o


def run(configs: dict, which: list[str]):
    """configs: {'T_recipe': [...blocks], 'T_qe_m': int, 'S1': {...}, 'S2_emb': path|None}"""
    sp = pd.read_csv("artifacts/splits/split_seed42.csv", dtype={"image_id": str})
    q, g = E.cohort_sets(sp, "primary")
    gid = g["image_id"].to_numpy(dtype=str)
    B = {s: {k: v for k, v in np.load(f"artifacts/features/ext_blocks_s{s}.npz").items() if k != "image_id"} for s in (256, 320)}
    ids = np.load("artifacts/features/ext_blocks_s256.npz")["image_id"]
    pos = {i: k for k, i in enumerate(ids)}
    flat = {f"s{s}:{k}": v for s, b in B.items() for k, v in b.items()}
    qi, gi = [pos[i] for i in q["image_id"]], [pos[i] for i in g["image_id"]]
    res = []
    if "T" in which:
        X = D.build(flat, configs["T_recipe"])
        o = _orders(X[qi], X[gi], gid, configs["T_qe_m"])
        res.append(E.per_query_metrics(o, q, g, rep="T_trainingfree_multiblock", cohort="primary", seed=42))
        o0 = _orders(X[qi], X[gi], gid, 0)
        res.append(E.per_query_metrics(o0, q, g, rep="T_trainingfree_multiblock_noQE", cohort="primary", seed=42))
    if "S1" in which:
        X = D.build(flat, configs["T_recipe"])
        cls = sorted(sp["class_id"].unique()); cm = {c: i for i, c in enumerate(cls)}
        ytr = np.array([cm[c] for c in g["class_id"]])
        for s in (0, 1, 2):
            emb = TR.fit(X[gi], ytr, seed=s, **configs["S1"])
            o = _orders(emb(X[qi]), emb(X[gi]), gid, 0)
            res.append(E.per_query_metrics(o, q, g, rep=f"S1_trained_head_seed{s}", cohort="primary", seed=42))
    if "S2" in which:
        z = np.load(configs["S2_emb"]); p2 = {i: k for k, i in enumerate(z["image_id"])}
        Z = z["emb"]
        o = _orders(Z[[p2[i] for i in q["image_id"]]], Z[[p2[i] for i in g["image_id"]]], gid, 0)
        res.append(E.per_query_metrics(o, q, g, rep="S2_finetuned_layer34", cohort="primary", seed=42))
    pq = pd.concat(res, ignore_index=True)
    pq["query_id"] = pq["query_id"].astype(str)
    return pq


def summarize(pq_ext: pd.DataFrame):
    base = pd.read_csv(OUT / "per_query_all.csv", dtype={"query_id": str})
    base = base[(base.seed == 42) & (base.cohort == "primary")]
    allp = pd.concat([base, pq_ext], ignore_index=True)
    summ = E.macro_micro(allp, by=("rep", "cohort", "K"))
    ci = []
    for (rep, K), s in allp.groupby(["rep", "K"]):
        for m in ["P", "Hit", "AP_K", "full_AP"]:
            mean, lo, hi = E.class_bootstrap_ci(s, m)
            ci.append(dict(rep=rep, K=K, metric=m, macro_mean=mean, ci_lo=lo, ci_hi=hi))
    pdiff = []
    for rep in pq_ext["rep"].unique():
        for ref in ("A_logits1000", "B_pooled2048"):
            for m in ["P", "Hit", "AP_K"]:
                d = E.paired_diff_ci(pq_ext[pq_ext.rep == rep], base[base.rep == ref], m)
                d.insert(0, "ref", ref); d.insert(0, "rep", rep); pdiff.append(d)
    return summ, pd.DataFrame(ci), pd.concat(pdiff)
