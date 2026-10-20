"""Occupancy-stratified retrieval analysis. Masks are used ONLY to stratify results after ranking.
'occupancy' = annotated TARGET-region area / image area (spot diseases: lesions; diffuse diseases and healthy: whole leaf)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E
from .masks import occupancy_bin

BIN_ORDER = ["small", "medium", "large"]


def attach(pq: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    s = split[["image_id", "occupancy", "healthy_by_name", "crop"]].rename(columns={"image_id": "query_id"})
    s["query_id"] = s["query_id"].astype(str)
    out = pq.merge(s, on="query_id", how="left")
    out["bin"] = out["occupancy"].map(occupancy_bin)
    return out


def by_bin(pq: pd.DataFrame, include_healthy=False, ks=(1, 5, 10, 20)) -> pd.DataFrame:
    d = pq if include_healthy else pq[~pq["healthy_by_name"]]
    d = d[d["K"].isin(ks)]
    rows = []
    for (rep, K, b), sub in d.groupby(["rep", "K", "bin"]):
        r = dict(rep=rep, K=K, bin=b, n_queries=sub["query_id"].nunique(), n_classes=sub["class_id"].nunique(),
                 mean_R=sub["R"].mean(), mean_ceiling=sub["ceiling"].mean())
        for m in ["P", "Recall", "Hit", "AP_K"]:
            mean, lo, hi = E.class_bootstrap_ci(sub, m)
            r.update({f"{m}_macro": mean, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_micro": sub[m].mean()})
        rows.append(r)
    return pd.DataFrame(rows)


def within_class(pq: pd.DataFrame, metric="P", K=10, n_boot=2000, seed=42) -> dict:
    """Class fixed-effects association: centre occupancy and metric within class, pool, correlate (Pearson+Spearman)."""
    d = pq[(pq["K"] == K) & ~pq["healthy_by_name"]].copy()
    d["occ_c"] = d["occupancy"] - d.groupby("class_id")["occupancy"].transform("mean")
    d["m_c"] = d[metric] - d.groupby("class_id")[metric].transform("mean")
    d = d[d.groupby("class_id")["query_id"].transform("nunique") >= 2]

    def corr(x):
        from scipy.stats import spearmanr
        return spearmanr(x["occ_c"], x["m_c"])[0]

    r = corr(d)
    classes = d["class_id"].unique()
    rng = np.random.default_rng(seed)
    g = {c: d[d["class_id"] == c] for c in classes}
    boots = []
    for _ in range(n_boot):
        s = pd.concat([g[c] for c in rng.choice(classes, len(classes))])
        v = corr(s)
        if np.isfinite(v):
            boots.append(v)
    return dict(metric=metric, K=K, n_queries=len(d), n_classes=len(classes), spearman_within_class=float(r),
                ci_lo=float(np.quantile(boots, .025)), ci_hi=float(np.quantile(boots, .975)))
