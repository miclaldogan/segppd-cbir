"""Track B common evaluation: identical test images for every method (classification and retrieval), paired tests, CIs, per-class tables."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from ..index import rank

OUT = Path("artifacts/research_v2/trackB"); L2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def load_protocol():
    P = pd.read_csv(OUT / "protocol_P.csv"); return {r: P[P.role == r].reset_index(drop=True) for r in ("train", "dev", "test")}


def frozen_features(P):
    fx = np.load("artifacts/features/plantseg_dinov2b_frozen_s224.npz"); pos = {n: i for i, n in enumerate(fx["name"])}
    return {r: fx["cls"][[pos[n] for n in df.Name]].astype(np.float32) for r, df in P.items()}


def retrieval(Q, G, ql, gl, gid, k=50):
    """Exact squared-L2 ranking; returns predictions (label of nearest reference), nearest id and retrieval metrics."""
    o, _ = rank(Q, G, gid, k=len(G)); rel = (gl[o] == ql[:, None]).astype(float); R = (gl[None, :] == ql[:, None]).sum(1)
    cs = np.cumsum(rel, 1); ap = ((cs / np.arange(1, rel.shape[1] + 1)) * rel).sum(1) / np.maximum(R, 1)
    d = dict(pred=gl[o[:, 0]], top1_ref=gid[o[:, 0]], ap=ap)
    for kk in (1, 5, 10): d[f"p{kk}"] = rel[:, :kk].mean(1); d[f"hit{kk}"] = rel[:, :kk].max(1); d[f"r{kk}"] = rel[:, :kk].sum(1) / np.maximum(R, 1)
    return d


def fit_lr(Xtr, ytr, Xdv, ydv, seed=0):
    """Regularisation C chosen on the dev images by accuracy (grid fixed in the registration); returns (predict_fn, best_C, seconds)."""
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xdv); best = None; t0 = time.time()
    for c in C_GRID:
        lr = LogisticRegression(C=c, max_iter=300, random_state=seed).fit(A, ytr); acc = (lr.predict(B) == ydv).mean()
        if best is None or acc > best[0] + 1e-9: best = (acc, c, lr)
    return (lambda X: best[2].predict(sc.transform(X))), best[1], time.time() - t0, float(best[0])


def mcnemar(a, b):
    x = int(((a == 1) & (b == 0)).sum()); y = int(((a == 0) & (b == 1)).sum()); return dict(a_only=x, b_only=y, p=float(binomtest(x, x + y, .5).pvalue) if x + y else 1.0)


def img_boot(correct, n=2000, seed=42):
    r = np.random.default_rng(seed); b = correct[r.integers(0, len(correct), (n, len(correct)))].mean(1); return float(np.quantile(b, .025)), float(np.quantile(b, .975))


def summarize(name, truth, pred, extra=None, labels=None):
    correct = (truth == pred).astype(int); labs = labels if labels is not None else np.unique(truth)
    p, r, f, s = precision_recall_fscore_support(truth, pred, labels=labs, zero_division=0)
    row = dict(method=name, n=len(truth), accuracy=float(correct.mean()), acc_ci_lo=img_boot(correct)[0], acc_ci_hi=img_boot(correct)[1], macro_f1=float(f1_score(truth, pred, labels=labs, average="macro", zero_division=0)),
               macro_recall=float(r.mean()), macro_precision=float(p.mean()))
    if extra is not None:
        for k in ("ap", "p5", "p10", "r5", "r10", "hit5", "hit10"): row[k if k != "ap" else "mAP"] = float(np.mean(extra[k]))
    return row, pd.DataFrame(dict(label=labs, precision=p, recall=r, f1=f, support=s)), correct


def run_frozen():
    P = load_protocol(); F = frozen_features(P); ytr, ydv, yte = (P[r].label.to_numpy() for r in ("train", "dev", "test"))
    gid = P["train"].Name.to_numpy(dtype=str); res, per_class, correct = [], {}, {}
    # M2: frozen CBIR (raw CLS, exact L2) - same representation as earlier frozen baselines
    t0 = time.time(); ex = retrieval(F["test"], F["train"], yte, ytr, gid); lat = (time.time() - t0) / len(yte)
    row, pc, c = summarize("M2 frozen DINOv2-B + CBIR (1-NN)", yte, ex["pred"], ex); row.update(infer_ms_per_query_search=1e3 * lat, gallery_floats=int(F["train"].size)); res.append(row); per_class["M2"] = pc; correct["M2"] = c
    pd.DataFrame(dict(Name=P["test"].Name, truth=yte, pred=ex["pred"], top1_ref=ex["top1_ref"], correct=c)).to_csv(OUT / "pred_M2.csv", index=False)
    # M1: frozen linear classifier
    fn, C, sec, dev_acc = fit_lr(F["train"], ytr, F["dev"], ydv); t0 = time.time(); pr = fn(F["test"]); lat = (time.time() - t0) / len(yte)
    row, pc, c = summarize("M1 frozen DINOv2-B + linear classifier", yte, pr); row.update(best_C=C, dev_accuracy=dev_acc, fit_seconds=sec, infer_ms_per_query=1e3 * lat, model_floats=int(768 * len(np.unique(ytr)))); res.append(row); per_class["M1"] = pc; correct["M1"] = c
    pd.DataFrame(dict(Name=P["test"].Name, truth=yte, pred=pr, correct=c)).to_csv(OUT / "pred_M1.csv", index=False)
    pd.DataFrame(res).to_csv(OUT / "results_frozen.csv", index=False); print(pd.DataFrame(res).round(4).to_string(index=False))
    print("McNemar M1 vs M2:", mcnemar(correct["M1"], correct["M2"]))


if __name__ == "__main__":
    run_frozen()


def run_trained(seedtag="s0"):
    """M3/M4 (metric embedding) and M5 (CE fine-tune) on the identical test images; paired tests against M1/M2; writes results_all.csv and per-class tables."""
    P = load_protocol(); ytr, ydv, yte = (P[r].label.to_numpy() for r in ("train", "dev", "test")); gid = P["train"].Name.to_numpy(dtype=str)
    res, correct, per_class = [], {}, {}
    prev = pd.read_csv(OUT / "results_frozen.csv"); res += prev.to_dict("records")
    for k in ("M1", "M2"): correct[k] = pd.read_csv(OUT / f"pred_{k}.csv").correct.to_numpy()
    d3 = OUT / "runs" / f"B_M3_metric_proxyanchor_{seedtag}"
    if (d3 / "embeddings.npz").exists():
        Z = np.load(d3 / "embeddings.npz"); t0 = time.time(); ex = retrieval(Z["test"], Z["train"], yte, ytr, gid); lat = (time.time() - t0) / len(yte)
        row, pc, c = summarize("M3 metric-learned DINOv2-B (Proxy-Anchor, PlantSeg train) + CBIR", yte, ex["pred"], ex); row.update(infer_ms_per_query_search=1e3 * lat, gallery_floats=int(Z["train"].size)); res.append(row); correct["M3"] = c; per_class["M3"] = pc
        pd.DataFrame(dict(Name=P["test"].Name, truth=yte, pred=ex["pred"], top1_ref=ex["top1_ref"], correct=c)).to_csv(OUT / "pred_M3.csv", index=False)
        fn, C, sec, dev_acc = fit_lr(Z["train"], ytr, Z["dev"], ydv); t0 = time.time(); pr = fn(Z["test"]); lat = (time.time() - t0) / len(yte)
        row, pc, c = summarize("M4 linear classifier on the same M3 embedding", yte, pr); row.update(best_C=C, dev_accuracy=dev_acc, fit_seconds=sec, infer_ms_per_query=1e3 * lat, model_floats=int(256 * len(np.unique(ytr)))); res.append(row); correct["M4"] = c; per_class["M4"] = pc
        pd.DataFrame(dict(Name=P["test"].Name, truth=yte, pred=pr, correct=c)).to_csv(OUT / "pred_M4.csv", index=False)
    d5 = OUT / "runs" / f"B_M5_ce_finetune_{seedtag}"
    if (d5 / "outputs.npz").exists():
        o = np.load(d5 / "outputs.npz"); pr = o["logits_test"].argmax(1); row, pc, c = summarize("M5 supervised fine-tuning (CE head, same last-4 blocks)", yte, pr); res.append(row); correct["M5"] = c; per_class["M5"] = pc
        pd.DataFrame(dict(Name=P["test"].Name, truth=yte, pred=pr, correct=c)).to_csv(OUT / "pred_M5.csv", index=False)
    R = pd.DataFrame(res); R.to_csv(OUT / "results_all.csv", index=False); pd.set_option("display.width", 250)
    print(R[["method", "n", "accuracy", "acc_ci_lo", "acc_ci_hi", "macro_f1", "macro_recall", "mAP", "p5", "p10", "hit10"]].round(4).to_string(index=False))
    keys = list(correct); pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)): pairs.append(dict(a=keys[i], b=keys[j], acc_a=float(correct[keys[i]].mean()), acc_b=float(correct[keys[j]].mean()), **mcnemar(correct[keys[i]], correct[keys[j]])))
    pd.DataFrame(pairs).to_csv(OUT / "paired_mcnemar.csv", index=False); print(pd.DataFrame(pairs).round(4).to_string(index=False))
    # URL-disjoint sensitivity
    te = P["test"]; tr_url = set(P["train"].URL); mask = ~te.URL.isin(tr_url).to_numpy()
    sens = {k: dict(n=int(mask.sum()), accuracy=float(v[mask].mean())) for k, v in correct.items()}; json.dump(sens, open(OUT / "sensitivity_url_disjoint.json", "w"), indent=1); print("URL-disjoint test subset:", sens)
    for k, v in per_class.items(): v.to_csv(OUT / f"per_class_{k}.csv", index=False)
