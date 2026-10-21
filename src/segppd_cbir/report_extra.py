"""Extra evaluation metrics for the written report (computed on the SAME locked queries/galleries; nothing is selected with them):
1-NN accuracy, macro-F1 of the 1-NN class decision, pooled pairwise ROC-AUC (score = -squared L2, positive = same class), mean AP.
Also writes ROC curves. Reads only saved embeddings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from .final_locked import build_matrix
from .hybrid import get, load_protocol_b
from .index import rank, squared_l2


def split(tag):
    if tag == "C":
        c = pd.read_csv("artifacts/splits/protocolC_seed777.csv", dtype={"image_id": str})
        return c[c.role_C == "gallery_C"].sort_values("image_id"), c[c.role_C == "test_query_C"].sort_values("image_id")
    _, ga, tq, _ = load_protocol_b()
    return ga, tq


def run(cfg_path, tag):
    cfg = json.load(open(cfg_path))
    ga, tq = split(tag)
    gid, ql, gl = ga.image_id.to_numpy(dtype=str), tq.class_id.to_numpy(), ga.class_id.to_numpy()
    cache, rows, curves = {}, [], {}
    for name, spec in cfg["candidates"].items():
        cache[name] = build_matrix(spec, cache)
        ids, Z = cache[name]
        Q, G = get(ids, Z, tq), get(ids, Z, ga)
        order, _ = rank(Q, G, gid, k=1)
        pred = gl[order[:, 0]]
        d2 = squared_l2(Q, G)
        y = (ql[:, None] == gl[None, :]).ravel()
        s = -d2.ravel()
        fpr, tpr, _ = roc_curve(y, s)
        curves[name] = (fpr, tpr)
        rows.append(dict(rep=name, accuracy_1nn=float((pred == ql).mean()),
                         macro_f1_1nn=float(f1_score(ql, pred, labels=np.unique(gl), average="macro", zero_division=0)),
                         roc_auc_pairwise=float(roc_auc_score(y, s))))
    out = Path(f"artifacts/metrics/protocol{tag}_locked"); pd.DataFrame(rows).to_csv(out / "classification_roc_metrics.csv", index=False)
    pd.to_pickle(curves, out / "roc_curves.pkl")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
