"""Shared evaluation utilities for research_v2: protocol sets, per-query metrics, summaries, bootstraps, diagnostics. No training code here."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..index import rank

KS = (1, 3, 5, 10, 20, 50)
TIE_EPS = 1 / 150  # one dev query (dev has 2 queries/class, class-macro Hit@1 moves in steps of 1/150)
RV2 = Path("artifacts/research_v2")


def gpu_status():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(",")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return dict(used_mb=int(r[0]), total_mb=int(r[1]), free_mb=int(r[1]) - int(r[0]), compute_apps=apps)


def segppd_sets():
    """Protocol B/C sets (dev = selection; B test and C test were consumed earlier -> EXPLORATORY here)."""
    b = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str})
    c = pd.read_csv("artifacts/splits/protocolC_seed777.csv", dtype={"image_id": str})
    s = lambda d, col, v: d[d[col] == v].sort_values("image_id")
    return dict(train=s(b, "role_B", "train_B"), dev_q=s(b, "role_B", "dev_query_B"), gallery_B=s(b, "role_B", "gallery_B"),
                testB_q=s(b, "role_B", "test_query_B"), testC_q=s(c, "role_C", "test_query_C"), gallery_C=s(c, "role_C", "gallery_C"))


def per_query(Q, G, ql, gl, gid, qid, extra_order=False):
    """Exact-L2 retrieval; returns a per-query DataFrame (relevance = same class label) and the rank order."""
    o, d2 = rank(Q, G, gid)
    rel = (gl[o] == ql[:, None]).astype(float)
    R = (gl[None, :] == ql[:, None]).sum(1)
    cs = np.cumsum(rel, 1); prec = cs / np.arange(1, rel.shape[1] + 1)
    d = dict(query_id=qid, class_id=ql, n_rel=R, ap=(prec * rel).sum(1) / np.maximum(R, 1), pred=gl[o[:, 0]], top1_id=gid[o[:, 0]])
    for k in KS:
        d[f"hit{k}"] = rel[:, :k].max(1); d[f"p{k}"] = rel[:, :k].mean(1); d[f"r{k}"] = rel[:, :k].sum(1) / np.maximum(R, 1)
    return pd.DataFrame(d), o


def summarize(pq):
    """class-macro (headline) and query-weighted means."""
    cols = [c for c in pq.columns if c.startswith(("hit", "p", "r")) and c[1:].isdigit() or c in ("ap",) or (c[:3] == "hit" and c[3:].isdigit())]
    cols = [c for c in pq.columns if c == "ap" or (c[0] in "pr" and c[1:].isdigit()) or (c.startswith("hit") and c[3:].isdigit())]
    mac = pq.groupby("class_id")[cols].mean().mean(); mic = pq[cols].mean()
    out = {f"macro_{c}": float(mac[c]) for c in cols}; out.update({f"micro_{c}": float(mic[c]) for c in cols})
    out["macro_mAP"], out["micro_mAP"] = out.pop("macro_ap"), out.pop("micro_ap"); out["n_queries"] = len(pq); out["n_classes"] = int(pq.class_id.nunique())
    return out


def class_boot(pq, col="hit1", n=2000, seed=42):
    cm = pq.groupby("class_id")[col].mean().to_numpy(); r = np.random.default_rng(seed)
    b = cm[r.integers(0, len(cm), (n, len(cm)))].mean(1); return float(cm.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def embedding_diagnostics(Q, G, ql, gl):
    d2 = ((Q[:, None, :] - G[None, :, :]) ** 2).sum(2); same = ql[:, None] == gl[None, :]
    intra, inter = d2[same].mean(), d2[~same].mean()
    nn_intra = np.where(same, d2, np.inf).min(1); nn_inter = np.where(~same, d2, np.inf).min(1)
    return dict(mean_intra_sqdist=float(intra), mean_inter_sqdist=float(inter), intra_over_inter=float(intra / inter), nn_intra_closer_frac=float((nn_intra < nn_inter).mean()),
                margin_mean=float((nn_inter - nn_intra).mean()))


def confusions(pq, names, top=10):
    e = pq[pq.pred != pq.class_id]; t = e.groupby(["class_id", "pred"]).size().sort_values(ascending=False).head(top)
    return [dict(true=names.get(int(a), a), pred=names.get(int(b), b), n=int(n)) for (a, b), n in t.items()]


def mcnemar_exact(a, b):
    from scipy.stats import binomtest
    x = int(((a == 1) & (b == 0)).sum()); y = int(((a == 0) & (b == 1)).sum())
    return dict(a_only=x, b_only=y, p=float(binomtest(x, x + y, .5).pvalue) if x + y else 1.0)


def evaluate_npz(npz_path, sets=None):
    """Evaluate one embedding file on dev (selection), B-test and C-test (exploratory). Returns dict of summaries + per-query frames."""
    S = sets or segppd_sets(); z = np.load(npz_path); pos = {i: k for k, i in enumerate(z["image_id"])}; E = z["emb"]
    f = lambda df: E[[pos[i] for i in df.image_id]]
    names = dict(zip(S["train"].class_id, S["train"].raw_class_label))
    out, pqs = {}, {}
    for nm, q, g in (("dev", S["dev_q"], S["gallery_B"]), ("testB_exploratory", S["testB_q"], S["gallery_B"]), ("testC_exploratory", S["testC_q"], S["gallery_C"])):
        gid = g.image_id.to_numpy(dtype=str); pq, _ = per_query(f(q), f(g), q.class_id.to_numpy(), g.class_id.to_numpy(), gid, q.image_id.to_numpy(dtype=str))
        out[nm] = summarize(pq); out[nm]["hit1_class_ci"] = class_boot(pq); out[nm]["diag"] = embedding_diagnostics(f(q), f(g), q.class_id.to_numpy(), g.class_id.to_numpy())
        out[nm]["top_confusions"] = confusions(pq, names); pqs[nm] = pq
    return out, pqs
