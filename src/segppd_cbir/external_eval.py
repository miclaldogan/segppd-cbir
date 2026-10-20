"""ONE-SHOT external evaluation per artifacts/external/external_protocol_registered.json (read that file first; nothing here may be tuned after the run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import feature_extractor as fe
from .ext_features import to_tensor
from .index import rank
from .train_embed import DinoEmb, embed_tensors

OUT = Path("artifacts/metrics/external_plantseg")
CACHE = Path("data/external/plantseg_256")
L2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)


def queries():
    m = pd.read_csv("artifacts/external/plantseg_leak_screen.csv")
    return m[(m.tier != "unmapped") & ~m.leak_flag].reset_index(drop=True)


def gallery():
    b = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str})
    return b[b.role_B == "gallery_B"].sort_values("image_id")


def per_query(order, q, ga):
    gl = ga.class_id.to_numpy()[order[:, :10]]
    hit = (gl == q.class_id.to_numpy()[:, None]).astype(float)
    return pd.DataFrame(dict(name=q.Name, class_id=q.class_id, tier=q.tier, hit1=hit[:, 0], p10=hit.mean(1), hit10=hit.max(1)))


def boot(pq, metric, n=2000, seed=42):
    """Two CIs for the macro mean over classes: (a) resample classes; (b) resample images within each class."""
    rng = np.random.default_rng(seed)
    groups = [g[metric].to_numpy() for _, g in pq.groupby("class_id")]
    cm = np.array([g.mean() for g in groups])
    a = np.array([cm[rng.integers(0, len(cm), len(cm))].mean() for _ in range(n)])
    b = np.array([np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups]) for _ in range(n)])
    q = lambda v: (float(np.quantile(v, .025)), float(np.quantile(v, .975)))
    return float(cm.mean()), q(a), q(b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q, ga = queries(), gallery()
    gid, gl = ga.image_id.to_numpy(dtype=str), ga.class_id.to_numpy()
    imgs = [Image.open(CACHE / (Path(n).stem + ".png")).convert("RGB") for n in q.Name]
    X = torch.stack([to_tensor(im, 224) for im in imgs]).half()
    res, pqs = {}, []
    # trained DINOv2-B + SupCon (checkpoints saved by --save_ckpt)
    for s in range(3):
        tag = f"D5ck_dinoB_p4_supcon_lr3e-5_s{s}"
        ck = torch.load(f"artifacts/checkpoints/{tag}.pt", map_location="cpu")
        a = ck["args"]
        net = DinoEmb(ck["n_cls"], a["d_emb"], a["mode"], blocks=a["blocks"], variant=a["variant"]).to("cuda")
        net.load_state_dict(ck["state"]); net.eval()
        Zq = embed_tensors(net, X)
        z = np.load(f"artifacts/extensions/B/{tag}_embeddings.npz")
        pos = {i: k for k, i in enumerate(z["image_id"])}
        Zg = z["emb"][[pos[i] for i in gid]]
        order, _ = rank(L2(Zq), L2(Zg), gid, k=10)
        res[tag] = (order, Zq)
        del net; torch.cuda.empty_cache()
    # frozen baselines for context
    fs = np.load("artifacts/features/dinov2b_frozen_s224.npz"); fx = np.load("artifacts/features/plantseg_dinov2b_frozen_s224.npz")
    pos = {i: k for k, i in enumerate(fs["image_id"])}
    full = pd.read_csv("artifacts/external/plantseg_leak_screen.csv"); idx = {n: k for k, n in enumerate(fx["name"])}
    res["F_dinov2b_frozen_cls"] = (rank(fx["cls"][[idx[n] for n in q.Name]], fs["cls"][[pos[i] for i in gid]], gid, k=10)[0], None)
    model, tf = fe.load_model("cuda")
    _, Pq = fe.extract(model, tf, imgs, device="cuda", batch_size=32)
    rf = np.load("artifacts/features/resnet50_v2_full.npz", allow_pickle=True)
    rp = {i: k for k, i in enumerate(rf["image_id"].astype(str))}
    res["BASE_resnet50_frozen_pooled2048_raw"] = (rank(Pq, rf["pooled"][[rp[i] for i in gid]], gid, k=10)[0], None)
    rows = []
    for name, (order, _) in res.items():
        pq = per_query(order, q, ga); pq.insert(0, "rep", name); pqs.append(pq)
    pq = pd.concat(pqs, ignore_index=True); pq.to_csv(OUT / "per_query.csv", index=False)
    head = pq[pq.rep.str.startswith("D5ck")].groupby(["name", "class_id", "tier"], as_index=False)[["hit1", "p10", "hit10"]].mean()
    head.insert(0, "rep", "HEADLINE_D5ck_mean3seeds"); pq2 = pd.concat([pq, head], ignore_index=True)
    for (rep, tier), s in pq2.groupby(["rep", "tier"]):
        for met in ("hit1", "p10", "hit10"):
            mean, ca, cb = boot(s, met)
            rows.append(dict(rep=rep, tier=tier, metric=met, n_images=len(s) // 1, n_classes=s.class_id.nunique(), macro_mean=mean,
                             ci_class_lo=ca[0], ci_class_hi=ca[1], ci_image_lo=cb[0], ci_image_hi=cb[1]))
    pd.DataFrame(rows).to_csv(OUT / "summary_ci.csv", index=False)
    pq2.groupby(["rep", "class_id"]).hit1.mean().unstack(0).to_csv(OUT / "per_class_hit1.csv")
    print(pd.DataFrame(rows).query("metric=='hit1'")[["rep", "tier", "n_images", "n_classes", "macro_mean", "ci_class_lo", "ci_class_hi", "ci_image_lo", "ci_image_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
