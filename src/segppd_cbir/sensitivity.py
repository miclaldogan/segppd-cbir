"""Pre-declared SENSITIVITY check (not the mandated baseline): L2-normalised descriptors + Euclidean.
Raw A/B remain the headline; this never replaces them. No fitting of any kind."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E


def run(feat_path, split_dir, out_csv, seed=42):
    feats = E.load_feats(feat_path)
    split = pd.read_csv(Path(split_dir) / f"split_seed{seed}.csv", dtype={"image_id": str})
    rows = []
    for rep, key in E.REPS.items():
        X = feats[key]
        nf = dict(feats)
        nf[key] = X / np.linalg.norm(X, axis=1, keepdims=True)
        for cohort in ("primary", "secondary"):
            q, g = E.cohort_sets(split, cohort)
            o, _, _ = E.retrieve(nf, rep, q, g)
            pq = E.per_query_metrics(o, q, g, rep=rep + "_L2NORM", cohort=cohort, seed=seed)
            for K, ks in pq.groupby("K"):
                for m in ["P", "Hit", "AP_K"]:
                    mean, lo, hi = E.class_bootstrap_ci(ks, m)
                    rows.append(dict(rep=rep + "_L2NORM", cohort=cohort, K=K, metric=m, macro_mean=mean, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
