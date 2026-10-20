"""ONE-SHOT locked evaluation on Protocol B test queries. Candidate list comes from a JSON file frozen BEFORE this runs.
candidates: {name: {"kind":"emb","path":..,"key":"emb"} | {"kind":"fuse","parts":[[name,weight],...]}}  (fusion weights fixed in advance)"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E
from . import ext_dev as D
from .hybrid import get, load_protocol_b
from .index import rank

OUT = Path("artifacts/metrics/protocolB_locked")
OUT_C = Path("artifacts/metrics/protocolC_locked")


def build_matrix(spec, cache):
    if spec["kind"] == "emb":
        z = np.load(spec["path"])
        X = z[spec.get("key", "emb")].astype(np.float32)
        return z["image_id"], (D.l2n(X) if spec.get("l2n", True) else X)
    ids = None
    parts = []
    for nm, w in spec["parts"]:
        i, Z = cache[nm]
        ids = i if ids is None else ids
        assert (ids == i).all()
        parts.append(np.sqrt(w) * D.l2n(Z))  # fusion parts are always L2-normalised
    return ids, np.concatenate(parts, axis=1)


def diagnostics(Q, G, ql, gl):
    """Embedding-space diagnostics on query->gallery distances (squared L2 on L2-normalised vectors)."""
    d = ((Q[:, None, :] - G[None, :, :]) ** 2).sum(2)
    same = ql[:, None] == gl[None, :]
    intra = np.where(same, d, np.nan)
    inter_nn = np.where(~same, d, np.inf).min(1)
    return dict(mean_intra_sqdist=float(np.nanmean(intra)), mean_inter_sqdist=float(d[~same].mean()),
                intra_over_inter=float(np.nanmean(intra) / d[~same].mean()),
                mean_nearest_intra_sqdist=float(np.nanmin(np.where(same, d, np.nan), axis=1).mean()),
                mean_nearest_inter_sqdist=float(inter_nn.mean()),
                frac_nn_intra_closer_than_nn_inter=float((np.nanmin(np.where(same, d, np.nan), axis=1) < inter_nn).mean()))


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    global OUT
    if cfg.get("protocol") == "C":  # second locked query set carved from gallery_B; same frozen models
        OUT = OUT_C
        c = pd.read_csv("artifacts/splits/protocolC_seed777.csv", dtype={"image_id": str})
        ga, tq = c[c.role_C == "gallery_C"].sort_values("image_id"), c[c.role_C == "test_query_C"].sort_values("image_id")
    else:
        dq, ga, tq, _ = load_protocol_b()
    OUT.mkdir(parents=True, exist_ok=True)
    gid = ga["image_id"].to_numpy(dtype=str)
    ql, gl = tq["class_id"].to_numpy(), ga["class_id"].to_numpy()
    cache, per_q, diag, lat = {}, [], [], []
    for name, spec in cfg["candidates"].items():
        cache[name] = build_matrix(spec, cache)
        ids, Z = cache[name]
        Q, G = get(ids, Z, tq), get(ids, Z, ga)
        # latency of exact L2 retrieval (excluding embedding), per query, after warm-up
        rank(Q[:2], G, gid)
        ts = []
        for i in range(len(Q)):
            t = time.perf_counter(); rank(Q[i:i + 1], G, gid); ts.append(time.perf_counter() - t)
        o, _ = rank(Q, G, gid)
        pq = E.per_query_metrics(o, tq, ga, rep=name, cohort=f"protocol{cfg.get('protocol', 'B')}_test", seed=cfg["seed"])
        per_q.append(pq)
        diag.append(dict(rep=name, dim=Z.shape[1], **diagnostics(Q, G, ql, gl)))
        lat.append(dict(rep=name, dim=Z.shape[1], p50_ms=1e3 * np.median(ts), p95_ms=1e3 * np.quantile(ts, .95)))
    pq = pd.concat(per_q, ignore_index=True)
    pq["query_id"] = pq["query_id"].astype(str)
    pq.to_csv(OUT / "per_query.csv", index=False)
    E.macro_micro(pq, by=("rep", "cohort", "K")).to_csv(OUT / "summary_macro_micro.csv", index=False)
    ci = []
    for (rep, K), s in pq.groupby(["rep", "K"]):
        for m in ["P", "Recall", "Hit", "AP_K", "full_AP"]:
            mean, lo, hi = E.class_bootstrap_ci(s, m)
            ci.append(dict(rep=rep, K=K, metric=m, macro_mean=mean, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(ci).to_csv(OUT / "macro_ci.csv", index=False)
    pdiff = []
    for a, b in cfg["paired"]:
        for m in ["P", "Hit", "AP_K"]:
            d = E.paired_diff_ci(pq[pq.rep == a], pq[pq.rep == b], m)
            d.insert(0, "B", b); d.insert(0, "A", a); pdiff.append(d)
    pd.concat(pdiff).to_csv(OUT / "paired_differences.csv", index=False)
    pc = pq[pq.K == 1].groupby(["rep", "class_id"]).P.mean().unstack(0)
    pc.to_csv(OUT / "per_class_hit1.csv")
    pd.DataFrame(diag).to_csv(OUT / "embedding_diagnostics.csv", index=False)
    pd.DataFrame(lat).to_csv(OUT / "retrieval_latency.csv", index=False)
    print("locked evaluation written to", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
