"""Paper-matched benchmark vs ChatLeafDisease on PlantSeg (protocol: artifacts/chatleaf/benchmark_registered.json). Fixed models; nothing is tuned.
  embed : embed the in-scope PlantSeg images with the 3 saved D5ck checkpoints
  run   : Track S (source-only) and Track T (k-shot) evaluation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .chatleaf_audit import CATS, OUT
from .index import rank

CACHE = Path("data/external/plantseg_256")
M = Path("artifacts/metrics/chatleaf_matched"); M.mkdir(parents=True, exist_ok=True)
CROP_CLASSES = {"tomato": [91, 92, 93, 94, 95], "grape": [41, 42, 43, 44], "strawberry": [79, 80, 81], "pepper": [70, 71, 72, 73]}
HEALTHY = {"tomato": 91, "grape": 41, "strawberry": 79, "pepper": 70}
L2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
TAG = "D5ck_dinoB_p4_supcon_lr3e-5_s{}"


def embed():
    import torch
    from PIL import Image

    from .ext_features import to_tensor
    from .train_embed import DinoEmb, embed_tensors
    g = pd.read_csv(OUT / "audit_inscope_groups.csv")
    X = torch.stack([to_tensor(Image.open(CACHE / (Path(n).stem + ".png")).convert("RGB"), 224) for n in g.Name]).half()
    for s in range(3):
        ck = torch.load(f"artifacts/checkpoints/{TAG.format(s)}.pt", map_location="cpu"); a = ck["args"]
        net = DinoEmb(ck["n_cls"], a["d_emb"], a["mode"], blocks=a["blocks"], variant=a["variant"]).to("cuda"); net.load_state_dict(ck["state"]); net.eval()
        np.savez(f"artifacts/features/chatleaf_inscope_{TAG.format(s)}.npz", name=g.Name.to_numpy(dtype=str), emb=embed_tensors(net, X)); del net; torch.cuda.empty_cache()
    print("embedded", len(g))


class Feats:
    """Query/gallery features for one representation."""
    def __init__(self, kind):
        self.kind = kind
        g = pd.read_csv(OUT / "audit_inscope_groups.csv"); self.q_names = g.Name.to_numpy(dtype=str)
        seg = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
        self.seg = seg; self.seg_ids = seg.image_id.to_numpy(dtype=str)
        if kind == "frozen":
            fx = np.load("artifacts/features/plantseg_dinov2b_frozen_s224.npz"); fs = np.load("artifacts/features/dinov2b_frozen_s224.npz")
            pos = {n: i for i, n in enumerate(fx["name"])}; self.Q = fx["cls"][[pos[n] for n in self.q_names]]
            assert (fs["image_id"] == self.seg_ids).all(); self.G = fs["cls"]
        else:
            s = int(kind[-1]); z = np.load(f"artifacts/features/chatleaf_inscope_{TAG.format(s)}.npz"); assert (z["name"] == self.q_names).all()
            self.Q = z["emb"]; e = np.load(f"artifacts/extensions/B/{TAG.format(s)}_embeddings.npz"); assert (e["image_id"] == self.seg_ids).all(); self.G = e["emb"]
        self.qpos = {n: i for i, n in enumerate(self.q_names)}


def run_methods(Qf, Gf, gl, ql_universe, kind_raw):
    """Qf, Gf: feature matrices; gl: gallery labels. Returns dict method -> (pred, order_or_None)."""
    out = {}
    gids = np.arange(len(Gf)).astype(str)
    Qn, Gn = (Qf, Gf) if kind_raw else (L2(Qf), L2(Gf))
    order, _ = rank(Qn, Gn, gids, k=len(Gf)); out["1nn"] = (gl[order[:, 0]], order)
    sc = StandardScaler().fit(Gf); labs = np.unique(gl)
    if len(labs) > 1:
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Gf), gl); out["lr"] = (lr.predict(sc.transform(Qf)), None)
    else:
        out["lr"] = (np.full(len(Qf), labs[0]), None)
    return out


def metrics(pred, truth, order, gl):
    acc = float((pred == truth).mean()); classes = np.unique(truth)
    f1 = float(f1_score(truth, pred, labels=classes, average="macro", zero_division=0))
    rec = {int(c): float((pred[truth == c] == c).mean()) for c in classes}
    d = dict(acc=acc, macro_f1=f1, per_class_recall=rec)
    if order is not None:
        rel = (gl[order] == truth[:, None]); R = rel.sum(1)
        d["p5"] = float(rel[:, :5].mean()); d["p10"] = float(rel[:, :10].mean())
        cs = np.cumsum(rel, 1); ap = ((cs / np.arange(1, rel.shape[1] + 1)) * rel).sum(1) / np.maximum(R, 1); d["mAP"] = float(ap.mean())
    return d


def boot_ci(correct, n=2000, seed=42):
    rng = np.random.default_rng(seed); b = correct[rng.integers(0, len(correct), (n, len(correct)))].mean(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))


def mcnemar(a_correct, b_correct):
    b = int(((a_correct == 1) & (b_correct == 0)).sum()); c = int(((a_correct == 0) & (b_correct == 1)).sum())
    return dict(a_only=b, b_only=c, p=float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)


def trackS():
    g = pd.read_csv(OUT / "audit_inscope_groups.csv"); cat = pd.DataFrame(CATS, columns=["crop", "paper_cat", "n_paper", "name", "tier", "seg_cls"])
    g = g.merge(cat, left_on="Disease", right_on="name"); conflict = g.groupby("group").Disease.transform("nunique") > 1
    keep = (~g.leak_flag) & (~conflict) & g.seg_cls.notna()
    q = g[keep].sort_values("Name").drop_duplicates("group", keep="first").reset_index(drop=True)
    print("Track S queries:", len(q), q.groupby("name").size().to_dict())
    rows, per_query = [], []
    for kind in ["frozen", "s0", "s1", "s2"]:
        F = Feats(kind)
        for variant in ("crop", "agnostic"):
            truth, pred1, predl, ords, gls = [], [], [], [], []
            for crop, sub in q.groupby("crop"):
                gal = F.seg[F.seg.dup_status != "excluded"]
                gal = gal[gal.class_id.isin(CROP_CLASSES[crop])] if variant == "crop" else gal
                gi = gal.index.to_numpy(); gl = gal.class_id.to_numpy()
                qi = np.array([F.qpos[n] for n in sub.Name]); res = run_methods(F.Q[qi], F.G[gi], gl, None, kind_raw=(kind == "frozen"))
                t = sub.seg_cls.to_numpy().astype(int)
                for meth, (p, o) in res.items():
                    mt = metrics(p, t, o, gl)
                    extra = {}
                    if o is not None:
                        rel = (gl[o] == t[:, None]); cs = np.cumsum(rel, 1)
                        extra = dict(p5=rel[:, :5].mean(1), p10=rel[:, :10].mean(1), ap=((cs / np.arange(1, rel.shape[1] + 1)) * rel).sum(1) / np.maximum(rel.sum(1), 1))
                    per_query.append(pd.DataFrame(dict(feature=kind, variant=variant, method=meth, name=sub.Name.to_numpy(), category=sub.name.to_numpy(), truth=t, pred=p.astype(int), correct=(p == t).astype(int), **extra)))
                    rows.append(dict(feature=kind, variant=variant, method=meth, crop=crop, n=len(sub), gallery=len(gi), n_gallery_classes=len(np.unique(gl)), **{k: v for k, v in mt.items() if k != "per_class_recall"}))
    pq = pd.concat(per_query, ignore_index=True); pq.to_csv(M / "trackS_per_query.csv", index=False)
    pd.DataFrame(rows).to_csv(M / "trackS_by_crop.csv", index=False)
    return q, pq


def summarize_S(pq):
    out = []
    for (feat, var, meth), s in pq.groupby(["feature", "variant", "method"]):
        lo, hi = boot_ci(s.correct.to_numpy()); pc = s.groupby("category").correct.mean()
        out.append(dict(feature=feat, variant=var, method=meth, n=len(s), acc=s.correct.mean(), ci_lo=lo, ci_hi=hi, macro_class_recall=pc.mean(), **{f"recall[{k}]": v for k, v in pc.items()}))
    t = pd.DataFrame(out); t.to_csv(M / "trackS_summary.csv", index=False)
    # paired McNemar (identical queries): CBIR vs LR on the same features, and SupCon vs frozen 1-NN
    pr = []
    key = lambda feat, var, meth: pq[(pq.feature == feat) & (pq.variant == var) & (pq.method == meth)].set_index("name").correct
    for var in ("crop", "agnostic"):
        for feat in ("frozen", "s0", "s1", "s2"):
            a, b = key(feat, var, "1nn"), key(feat, var, "lr"); pr.append(dict(variant=var, comparison=f"1nn vs lr ({feat})", **mcnemar(a.to_numpy(), b.loc[a.index].to_numpy())))
        for s in ("s0", "s1", "s2"):
            a, b = key(s, var, "1nn"), key("frozen", var, "1nn"); pr.append(dict(variant=var, comparison=f"supcon {s} 1nn vs frozen 1nn", **mcnemar(a.to_numpy(), b.loc[a.index].to_numpy())))
    pd.DataFrame(pr).to_csv(M / "trackS_paired.csv", index=False)
    return t, pd.DataFrame(pr)


def trackT():
    g = pd.read_csv(OUT / "audit_inscope_groups.csv"); cat = pd.DataFrame(CATS, columns=["crop", "paper_cat", "n_paper", "name", "tier", "seg_cls"])
    g = g.merge(cat, left_on="Disease", right_on="name"); conflict = g.groupby("group").Disease.transform("nunique") > 1
    pool = g[(~g.leak_flag) & (~conflict)].sort_values("Name").reset_index(drop=True)
    seg_lab = {f"{r.crop}|{r.paper_cat}": 1000 + i for i, r in enumerate(cat.itertuples())}  # synthetic ids for PlantSeg categories (distinct from SegPPD ids)
    pool["lab"] = [seg_lab[f"{c}|{p}"] for c, p in zip(pool.crop, pool.paper_cat)]
    rows = []
    for kind in ["frozen", "s0", "s1", "s2"]:
        F = Feats(kind); seg = F.seg[F.seg.dup_status != "excluded"]
        for k in (1, 5, 10):
            for seed in range(20):
                rng = np.random.default_rng(seed); ref_rows, q_rows = [], []
                for (crop, pc), sub in pool.groupby(["crop", "paper_cat"]):
                    sub = sub.sample(frac=1, random_state=int(rng.integers(1 << 30)))
                    refs = sub.drop_duplicates("group").head(k); rg = set(refs.group)
                    rest = sub[~sub.group.isin(rg)].drop_duplicates("group")
                    n_q = int(cat[(cat.crop == crop) & (cat.paper_cat == pc)].n_paper.iloc[0]); ref_rows.append(refs); q_rows.append(rest.head(n_q))
                refs, qs = pd.concat(ref_rows), pd.concat(q_rows)
                for crop in cat.crop.unique():
                    r, qq = refs[refs.crop == crop], qs[qs.crop == crop]
                    h = seg[seg.class_id == HEALTHY[crop]]
                    Gf = np.vstack([F.Q[[F.qpos[n] for n in r.Name]], F.G[h.index.to_numpy()]]); gl = np.concatenate([r.lab.to_numpy(), np.full(len(h), HEALTHY[crop])])
                    Qf = F.Q[[F.qpos[n] for n in qq.Name]]; res = run_methods(Qf, Gf, gl, None, kind_raw=(kind == "frozen")); t = qq.lab.to_numpy()
                    for meth, (p, o) in res.items(): rows.append(dict(feature=kind, k=k, seed=seed, crop=crop, method=meth, n=len(qq), correct=int((p == t).sum())))
    t = pd.DataFrame(rows); t.to_csv(M / "trackT_raw.csv", index=False)
    s = t.groupby(["feature", "k", "seed", "method"]).agg(n=("n", "sum"), correct=("correct", "sum")).reset_index(); s["acc"] = s.correct / s.n
    a = s.groupby(["feature", "k", "method"]).acc.agg(["mean", "std"]).reset_index(); a.to_csv(M / "trackT_summary.csv", index=False)
    return a


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "embed": embed()
    else:
        q, pq = trackS(); t, pr = summarize_S(pq)
        pd.set_option("display.width", 250); print(t[["feature", "variant", "method", "n", "acc", "ci_lo", "ci_hi", "macro_class_recall"]].round(3).to_string(index=False)); print(pr.round(4).to_string(index=False))
        a = trackT(); print(a.round(3).to_string(index=False))
