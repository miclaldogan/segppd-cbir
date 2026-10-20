"""External validation set: PlantSeg v2 (Zenodo 13958858, CC BY-NC-ND 4.0, research evaluation only, never redistributed).

Stage 1 `prepare`: class mapping to SegPPD-101 (fixed before any model sees the images), 256x256 local cache, pHash/dHash, frozen DINOv2-B CLS.
Stage 2 `screen`: near-duplicate screen of EVERY PlantSeg image against ALL 2,263 SegPPD images (hash rule + calibrated embedding rule).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path("data/external/plantseg/plantsegv2")
CACHE = Path("data/external/plantseg_256")
OUT = Path("artifacts/external")

# SegPPD-101 class_id -> PlantSeg disease name. tier "strict": same crop + same condition (accepted synonyms only);
# tier "loose": crop or condition naming differs (reported separately, never in the headline).
MAPPING = {
    4: ("apple rust", "strict"), 5: ("apple scab", "strict"), 7: ("banana cordana leaf spot", "strict"),
    11: ("bean rust", "strict"), 22: ("coffee leaf rust", "strict"), 31: ("cucumber powdery mildew", "strict"),
    42: ("grape black rot", "strict"), 81: ("strawberry leaf scorch", "strict"), 92: ("tomato bacterial leaf spot", "strict"),
    93: ("tomato leaf mold", "strict"), 94: ("tomato late blight", "strict"),
    71: ("bell pepper bacterial spot", "loose"), 60: ("citrus canker", "loose"), 61: ("citrus greening disease", "loose"),
    21: ("coffee brown eye spot", "loose"),
}


EMB_THR = 0.85


def open256(p):
    with Image.open(p) as im:
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
    return im.resize((256, 256), Image.BICUBIC)  # squash, same geometry as the 256x256 SegPPD files


def prepare():
    m = pd.read_csv(ROOT / "Metadatav2.csv", encoding="utf-8-sig")
    inv = {v[0]: (k, v[1]) for k, v in MAPPING.items()}
    m["class_id"] = m.Disease.map(lambda d: inv[d][0] if d in inv else -1)
    m["tier"] = m.Disease.map(lambda d: inv[d][1] if d in inv else "unmapped")
    m["path"] = [str(ROOT / "images" / {"Training": "train", "Validation": "val", "Test": "test"}[s] / n) for n, s in zip(m.Name, m.Split)]
    CACHE.mkdir(parents=True, exist_ok=True)
    ph, dh, ims = [], [], []
    for n, p in zip(m.Name, m.path):
        im = open256(p)
        im.save(CACHE / (Path(n).stem + ".png"))
        ph.append(str(imagehash.phash(im))); dh.append(str(imagehash.dhash(im)))
    m["phash"], m["dhash"] = ph, dh
    m.to_csv(OUT / "plantseg_index.csv", index=False)
    import torch

    from . import dino
    model = dino.load("cuda", name="vit_base_patch14_dinov2.lvd142m")
    feats = dino.frozen_blocks(model, (Image.open(CACHE / (Path(n).stem + ".png")).convert("RGB") for n in m.Name), 224, bs=32)
    np.savez("artifacts/features/plantseg_dinov2b_frozen_s224.npz", name=m.Name.to_numpy(dtype=str), cls=feats["cls"])
    del torch
    print(len(m), "images prepared;", m.tier.value_counts().to_dict())


def _bits(hexes):
    return np.array([imagehash.hex_to_hash(h).hash.flatten() for h in hexes])


def screen():
    """Flag every PlantSeg image that may be a copy of ANY SegPPD-101 image (all 2,263, incl. excluded ones)."""
    m = pd.read_csv(OUT / "plantseg_index.csv")
    inv = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    fe = np.load("artifacts/features/plantseg_dinov2b_frozen_s224.npz")
    fs = np.load("artifacts/features/dinov2b_frozen_s224.npz")
    assert (fs["image_id"] == inv.image_id.to_numpy(dtype=str)).all() and (fe["name"] == m.Name.to_numpy(dtype=str)).all()
    l2n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    E_, S_ = l2n(fe["cls"]), l2n(fs["cls"])
    cos = E_ @ S_.T
    # calibration on known same-photo pairs inside SegPPD (duplicate groups from the audit)
    ss = S_ @ S_.T
    pairs = [(i, j) for _, g in inv.groupby("group_id") if len(g) > 1 for i in g.index for j in g.index if i < j]
    dup_cos = np.array([ss[i, j] for i, j in pairs])
    nondup = ss.copy(); np.fill_diagonal(nondup, -1)
    for i, j in pairs: nondup[i, j] = nondup[j, i] = -1
    thr_calib = float(np.quantile(dup_cos, 0.05))
    thr = EMB_THR  # chosen by visual review of borderline bands: 0.85-0.92 still contains cropped copies, 0.80-0.85 is mostly different photos
    nn = cos.argmax(1); nn_cos = cos.max(1)
    P = (_bits(m.phash)[:, None, :] != _bits(inv.phash)[None, :, :]).sum(2)
    D = (_bits(m.dhash)[:, None, :] != _bits(inv.dhash)[None, :, :]).sum(2)
    hash_hit = ((P <= 12) & (D <= 18)).any(1)
    flag = hash_hit | (nn_cos >= thr)
    m["nn_segppd_id"] = inv.image_id.to_numpy()[nn]; m["nn_segppd_class"] = inv.class_id.to_numpy()[nn]
    m["nn_cos"] = nn_cos; m["hash_dup"] = hash_hit; m["emb_dup"] = nn_cos >= thr; m["leak_flag"] = flag
    m.drop(columns=["URL"]).to_csv(OUT / "plantseg_leak_screen.csv", index=False)
    rep = dict(n_pairs_calibration=len(pairs), calib_dup_cos_min=float(dup_cos.min()), calib_dup_cos_p05=thr_calib, emb_threshold_used=thr,
               calib_nondup_cos_max=float(nondup.max()), calib_nondup_cos_p99=float(np.quantile(nondup.max(1), .99)),
               n_plantseg=len(m), n_flagged=int(flag.sum()), n_hash=int(hash_hit.sum()), n_emb=int((nn_cos >= thr).sum()),
               flagged_by_tier=m[flag].tier.value_counts().to_dict(), nn_cos_quantiles={q: float(np.quantile(nn_cos, q)) for q in (.5, .9, .99, .999)},
               mapped_images=int((m.tier != "unmapped").sum()), mapped_flagged=int((flag & (m.tier != "unmapped")).sum()))
    (OUT / "plantseg_leak_screen.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "screen": screen}[sys.argv[1]]()
