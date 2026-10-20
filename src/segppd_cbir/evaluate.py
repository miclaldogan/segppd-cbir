"""Exact-L2 retrieval evaluation. Labels are used ONLY to score results, never to filter candidates."""
from __future__ import annotations

import time
from math import comb

import numpy as np
import pandas as pd

from . import metrics as M
from .index import rank

KS = (1, 3, 5, 10, 20, 50, 100)
REPS = {"A_logits1000": "logits", "B_pooled2048": "pooled"}


def load_feats(npz_path):
    z = np.load(npz_path)
    return {"ids": z["image_id"], "logits": z["logits"], "pooled": z["pooled"]}


def cohort_sets(split: pd.DataFrame, cohort: str):
    if cohort == "primary":
        q = split[split["role"] == "query_primary"]
        g = split[split["role"] == "gallery_primary"]
    elif cohort == "secondary":
        q = split[split["role"] == "query_secondary"]
        g = split[split["role"].isin(["gallery_primary", "gallery_secondary_extra"])]
    else:
        raise ValueError(cohort)
    return q.sort_values("image_id"), g.sort_values("image_id")


def retrieve(feats, rep, q, g):
    """Return (orders[nq, ng], sq_dist[nq, ng], per-query latency seconds) with exact L2, ties by image_id."""
    pos = {i: k for k, i in enumerate(feats["ids"])}
    X = feats[REPS[rep]]
    Q = X[[pos[i] for i in q["image_id"]]]
    G = X[[pos[i] for i in g["image_id"]]]
    gid = g["image_id"].to_numpy(dtype=str)
    rank(Q[:2], G, gid)  # warm-up
    lat = []
    orders, d2 = [], []
    for i in range(len(Q)):
        t = time.perf_counter()
        o, d = rank(Q[i:i + 1], G, gid)
        lat.append(time.perf_counter() - t)
        orders.append(o[0]); d2.append(d[0])
    return np.array(orders), np.array(d2), np.array(lat)


def per_query_metrics(orders, q, g, ks=KS, rep="", cohort="", seed=0, n_perm=50):
    gl = g["class_id"].to_numpy()
    ql = q["class_id"].to_numpy()
    ng = len(g)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(q)):
        rel = (gl[orders[i]] == ql[i]).astype(float)
        R = int((gl == ql[i]).sum())
        fap = M.full_ap(rel, R)
        # random-ranking baseline on the identical gallery/relevance: analytic P/R/Hit, Monte-Carlo AP
        allrel = (gl == ql[i]).astype(float)
        perms = [rng.permutation(allrel) for _ in range(n_perm)]
        rand_fap = float(np.nanmean([M.full_ap(p, R) for p in perms])) if R else np.nan
        for k in ks:
            if k > ng:
                continue
            hit_rand = 1.0 - comb(ng - R, k) / comb(ng, k) if R else np.nan
            rows.append(dict(rep=rep, cohort=cohort, seed=seed, query_id=q["image_id"].iloc[i], class_id=int(ql[i]),
                             K=k, R=R, P=M.precision_at_k(rel, k), Recall=M.recall_at_k(rel, k, R), Hit=M.hit_at_k(rel, k),
                             AP_K=M.ap_at_k(rel, k, R), full_AP=fap, ceiling=M.precision_ceiling(R, k),
                             rand_P=R / ng, rand_Recall=k / ng, rand_Hit=hit_rand,
                             rand_AP_K=float(np.nanmean([M.ap_at_k(p, k, R) for p in perms])) if R else np.nan,
                             rand_full_AP=rand_fap))
    return pd.DataFrame(rows)


METRICS = ["P", "Recall", "Hit", "AP_K", "full_AP", "ceiling", "rand_P", "rand_Recall", "rand_Hit", "rand_AP_K", "rand_full_AP"]


def macro_micro(pq: pd.DataFrame, by=("rep", "cohort", "K")) -> pd.DataFrame:
    by = list(by)
    cls = pq.groupby(by + ["class_id"])[METRICS].mean().reset_index()
    macro = cls.groupby(by)[METRICS].mean().add_prefix("macro_")
    micro = pq.groupby(by)[METRICS].mean().add_prefix("micro_")
    n = pq.groupby(by).agg(n_queries=("query_id", "nunique"), n_classes=("class_id", "nunique"))
    return pd.concat([n, macro, micro], axis=1).reset_index()


def class_bootstrap_ci(pq: pd.DataFrame, metric: str, n_boot=2000, seed=42, alpha=0.05):
    """Resample CLASSES with replacement; statistic = macro mean of per-class means."""
    cm = pq.groupby("class_id")[metric].mean().dropna().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cm), (n_boot, len(cm)))
    b = cm[idx].mean(1)
    return float(cm.mean()), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def paired_diff_ci(pa: pd.DataFrame, pb: pd.DataFrame, metric: str, n_boot=2000, seed=42):
    """Paired A-B difference on identical queries; class-level bootstrap of per-class mean differences."""
    m = pa[["query_id", "class_id", "K", metric]].merge(pb[["query_id", "K", metric]], on=["query_id", "K"], suffixes=("_a", "_b"))
    m["d"] = m[metric + "_a"] - m[metric + "_b"]
    out = []
    for k, sub in m.groupby("K"):
        cm = sub.groupby("class_id")["d"].mean().to_numpy()
        rng = np.random.default_rng(seed + int(k))
        b = cm[rng.integers(0, len(cm), (n_boot, len(cm)))].mean(1)
        out.append(dict(K=k, metric=metric, mean_diff_A_minus_B=float(cm.mean()), ci_lo=float(np.quantile(b, .025)),
                        ci_hi=float(np.quantile(b, .975)), n_queries=len(sub), n_classes=len(cm),
                        frac_queries_A_better=float((sub["d"] > 0).mean()), frac_queries_B_better=float((sub["d"] < 0).mean())))
    return pd.DataFrame(out)
