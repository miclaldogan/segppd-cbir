"""uv run python -m segppd_cbir.cli <command>   (steps are resumable: outputs are skipped if present)"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import feature_extractor as fe
from .dataset_audit import audit
from .duplicate_groups import build_groups
from .reproducibility import dumps, environment_report
from .split import SEEDS, check_split, make_split

ROOT = Path("data/extracted/SegPPD-101")
AUD = Path("artifacts/audits")
SPL = Path("artifacts/splits")
FEAT = Path("artifacts/features")


def cmd_audit(a):
    if (AUD / "file_inventory.csv").exists() and not a.force:
        print("audit exists; skipping")
        return
    audit(ROOT, AUD)


def cmd_split(a):
    inv = pd.read_csv(AUD / "file_inventory.csv", dtype={"image_id": str})
    inv = build_groups(inv)
    SPL.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "image_path", "mask_path", "class_id", "raw_class_label", "crop", "condition", "healthy_by_name",
            "img_w", "img_h", "image_sha256", "occupancy", "group_id", "group_size", "group_n_classes",
            "dup_status", "dup_reason", "role", "cohort_class", "seed"]
    for s in SEEDS:
        m = make_split(inv, s)
        info = check_split(m)
        m[cols].to_csv(SPL / f"split_seed{s}.csv", index=False)
        print(s, info)
    inv.to_csv(AUD / "dataset_manifest.csv", index=False)


def cmd_extract(a):
    out = FEAT / "resnet50_v2_full.npz"
    if out.exists() and not a.force:
        print("features exist; skipping")
        return
    FEAT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    inv = pd.read_csv(AUD / "dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    model, tf = fe.load_model("cpu")
    fe.assert_frozen(model)
    w0 = fe.checkpoint_fingerprint(model)
    t0 = time.perf_counter()
    from tqdm import tqdm
    imgs = (fe.open_rgb(ROOT / p) for p in tqdm(inv["image_path"], desc="extract", mininterval=30))
    logits, pooled = fe.extract(model, tf, imgs, "cpu", a.batch)
    dt = time.perf_counter() - t0
    w1 = fe.checkpoint_fingerprint(model)
    assert w0 == w1, "weights changed during extraction"
    assert logits.shape == (len(inv), 1000) and pooled.shape == (len(inv), 2048)
    assert np.isfinite(logits).all() and np.isfinite(pooled).all()
    assert (np.abs(pooled).sum(1) > 0).all() and (np.abs(logits).sum(1) > 0).all()
    np.savez(out, image_id=inv["image_id"].to_numpy(dtype=str), image_sha256=inv["image_sha256"].to_numpy(dtype=str),
             logits=logits.astype(np.float32), pooled=pooled.astype(np.float32))
    rep = {"weights_enum": str(fe.WEIGHTS_ENUM), "weights_url": fe.WEIGHTS_ENUM.url, "weights_file_sha256": fe.weights_file_sha256(),
           "state_dict_fingerprint_sha256": w0, "weights_unchanged_after_extraction": True,
           "transform": fe.transform_description(tf), "device": "cpu", "batch_size": a.batch, "threads": a.threads,
           "n_images": len(inv), "logits_shape": list(logits.shape), "pooled_shape": list(pooled.shape),
           "dtype": "float32", "extract_seconds_total": dt, "extract_seconds_per_image": dt / len(inv),
           "npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "environment": environment_report(),
           "descriptors": {"A_primary": "1000-D raw fc logits (pre-softmax, ImageNet class-evidence scores)",
                           "B_comparison": "2048-D global-average-pooled features (input of fc)"}}
    (FEAT / "feature_report.json").write_text(dumps(rep))
    print(f"done {dt:.0f}s")


def cmd_roi(a):
    from . import roi_experiment as R
    out = FEAT / "resnet50_v2_oracle_roi.npz"
    if out.exists() and not a.force:
        print("roi features exist; skipping")
        return
    torch.set_num_threads(a.threads)
    inv = pd.read_csv(AUD / "dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    model, tf = fe.load_model("cpu")
    w0 = fe.checkpoint_fingerprint(model)
    t0 = time.perf_counter()
    logits, pooled = fe.extract(model, tf, R.crops(ROOT, inv), "cpu", a.batch)
    assert fe.checkpoint_fingerprint(model) == w0 and np.isfinite(logits).all() and np.isfinite(pooled).all()
    np.savez(out, image_id=inv["image_id"].to_numpy(dtype=str), image_sha256=inv["image_sha256"].to_numpy(dtype=str),
             logits=logits.astype(np.float32), pooled=pooled.astype(np.float32))
    (FEAT / "roi_report.json").write_text(dumps({"margin_frac": R.MARGIN, "n": len(inv), "seconds": time.perf_counter() - t0,
                                                 "note": "ORACLE: uses ground-truth masks for query AND gallery; not deployable"}))
    print("roi done")


def cmd_extblocks(a):
    from . import ext_features as X
    torch.set_num_threads(a.threads)
    inv = pd.read_csv(AUD / "dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    model, _ = fe.load_model("cpu")
    for size in (256, 320):
        out = FEAT / f"ext_blocks_s{size}.npz"
        if out.exists() and not a.force:
            continue
        t0 = time.perf_counter()
        b = X.extract_blocks(model, (fe.open_rgb(ROOT / p) for p in inv["image_path"]), size, a.batch, flip=True)
        np.savez(out, image_id=inv["image_id"].to_numpy(dtype=str), **b)
        print(size, {k: v.shape for k, v in b.items()}, f"{time.perf_counter()-t0:.0f}s")


def cmd_handimg(a):
    from PIL import Image
    from . import ext_features as X
    from . import hand_images as HI
    torch.set_num_threads(a.threads)
    inv = pd.read_csv(AUD / "dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    model, _ = fe.load_model("cpu")
    for name, fn in HI.VARIANTS.items():
        out = FEAT / f"handimg_{name}.npz"
        if out.exists() and not a.force:
            continue
        t0 = time.perf_counter()
        imgs = (Image.fromarray(fn(np.asarray(fe.open_rgb(ROOT / p)))) for p in inv["image_path"])
        b = X.extract_blocks(model, imgs, 256, a.batch, flip=False)
        np.savez(out, image_id=inv["image_id"].to_numpy(dtype=str), **b)
        print(name, f"{time.perf_counter()-t0:.0f}s", flush=True)


def cmd_evaluate(a):
    import faiss
    from . import evaluate as E
    MET = Path("artifacts/metrics")
    MET.mkdir(parents=True, exist_ok=True)
    feats = E.load_feats(FEAT / "resnet50_v2_full.npz")
    allq, lat_rows, faiss_rows = [], [], []
    for seed in SEEDS:
        split = pd.read_csv(SPL / f"split_seed{seed}.csv", dtype={"image_id": str})
        for cohort in ("primary", "secondary"):
            q, g = E.cohort_sets(split, cohort)
            for rep in E.REPS:
                orders, d2, lat = E.retrieve(feats, rep, q, g)
                allq.append(E.per_query_metrics(orders, q, g, rep=rep, cohort=cohort, seed=seed))
                lat_rows.append(dict(rep=rep, cohort=cohort, seed=seed, n_queries=len(q), gallery=len(g), dim=feats[E.REPS[rep]].shape[1],
                                     p50_ms=1e3 * np.median(lat), p95_ms=1e3 * np.quantile(lat, .95), mean_ms=1e3 * lat.mean()))
                # FAISS IndexFlatL2 cross-check (squared L2; same ordering as L2)
                pos = {i: k for k, i in enumerate(feats["ids"])}
                X = feats[E.REPS[rep]]
                G = np.ascontiguousarray(X[[pos[i] for i in g["image_id"]]], dtype=np.float32)
                Q = np.ascontiguousarray(X[[pos[i] for i in q["image_id"]]], dtype=np.float32)
                ix = faiss.IndexFlatL2(G.shape[1]); ix.add(G)
                fd, fi = ix.search(Q, 100)
                same = int((fi == orders[:, :100]).all(1).sum())
                faiss_rows.append(dict(rep=rep, cohort=cohort, seed=seed, n_queries=len(q), top100_identical_queries=same,
                                       max_rel_dist_diff=float(np.max(np.abs(fd - d2[:, :100]) / np.maximum(d2[:, :100], 1e-9)))))
    pq = pd.concat(allq, ignore_index=True)
    pq["query_id"] = pq["query_id"].astype(str)
    pq.to_csv(MET / "per_query_all.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(MET / "latency.csv", index=False)
    pd.DataFrame(faiss_rows).to_csv(MET / "faiss_crosscheck.csv", index=False)
    E.macro_micro(pq, by=("rep", "cohort", "seed", "K")).to_csv(MET / "summary_macro_micro.csv", index=False)
    ci, pd_rows = [], []
    for (cohort, seed), sub in pq.groupby(["cohort", "seed"]):
        for rep, rs in sub.groupby("rep"):
            for K, ks in rs.groupby("K"):
                for m in ["P", "Recall", "Hit", "AP_K", "full_AP"]:
                    mean, lo, hi = E.class_bootstrap_ci(ks, m)
                    ci.append(dict(cohort=cohort, seed=seed, rep=rep, K=K, metric=m, macro_mean=mean, ci_lo=lo, ci_hi=hi))
        A, B = sub[sub.rep == "A_logits1000"], sub[sub.rep == "B_pooled2048"]
        for m in ["P", "Recall", "Hit", "AP_K"]:
            d = E.paired_diff_ci(A, B, m)
            d.insert(0, "seed", seed); d.insert(0, "cohort", cohort)
            pd_rows.append(d)
    pd.DataFrame(ci).to_csv(MET / "macro_ci.csv", index=False)
    pd.concat(pd_rows).to_csv(MET / "paired_A_minus_B.csv", index=False)
    storage = {r: dict(dim=feats[c].shape[1], bytes_per_vector=feats[c].shape[1] * 4,
                       total_MB_2263=feats[c].nbytes / 1e6) for r, c in E.REPS.items()}
    (MET / "descriptor_storage.json").write_text(dumps(storage))
    print("evaluate done")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("audit", cmd_audit), ("split", cmd_split), ("extract", cmd_extract), ("evaluate", cmd_evaluate), ("roi", cmd_roi), ("extblocks", cmd_extblocks), ("handimg", cmd_handimg)]:
        p = sp.add_parser(name)
        p.add_argument("--force", action="store_true")
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--threads", type=int, default=4)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
