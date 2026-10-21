"""Track B1: audit of the PlantSeg v2 split for a classification-vs-retrieval study. Metadata/features only; no model is trained or evaluated.
Outputs artifacts/research_v2/trackB/audit_*.csv|json and plantseg_groups.csv (visual near-duplicate groups over ALL 11,458 images)."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import imagehash
import numpy as np
import pandas as pd

OUT = Path("artifacts/research_v2/trackB"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    m = pd.read_csv("artifacts/external/plantseg_leak_screen.csv")  # all 11,458 images: hashes + SegPPD leak flags (hash rule or DINOv2-B cos>=0.85)
    meta = pd.read_csv("data/external/plantseg/plantsegv2/Metadatav2.csv", encoding="utf-8-sig")[["Name", "Resolution", "URL", "Mask ratio"]]
    m = m.merge(meta[["Name", "URL"]], on="Name", how="left") if "URL" not in m.columns else m
    fx = np.load("artifacts/features/plantseg_dinov2b_frozen_s224.npz"); assert (fx["name"] == m.Name.to_numpy(dtype=str)).all()
    E = fx["cls"].astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    hb = lambda col: np.array([imagehash.hex_to_hash(h).hash.flatten() for h in m[col]], dtype=np.uint8)
    PH, DH = hb("phash"), hb("dhash"); n = len(m); pairs = []
    for i in range(0, n, 1500):
        cos = E[i:i + 1500] @ E.T
        ph = (PH[i:i + 1500, None, :] != PH[None, :, :]).sum(2); dh = (DH[i:i + 1500, None, :] != DH[None, :, :]).sum(2)
        dup = ((ph <= 12) & (dh <= 18)) | (cos >= 0.95)  # within-PlantSeg rule: 0.85 chains unrelated look-alike photos (visual review of random pairs); 0.95 + hash is conservative but not chaining
        ii, jj = np.nonzero(dup); sel = (ii + i) < jj; pairs += list(zip((ii[sel] + i).tolist(), jj[sel].tolist()))
    p = list(range(n))
    def f(x):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
    for a, b in pairs: p[f(a)] = f(b)
    m["group"] = [f(i) for i in range(n)]; m["group_size"] = m.groupby("group").Name.transform("size")
    m["group_n_split"] = m.groupby("group").Split.transform("nunique"); m["group_n_class"] = m.groupby("group").Disease.transform("nunique")
    m["url_host"] = m.URL.map(lambda u: urlparse(str(u)).netloc)
    m["w"] = m.Resolution.str.split("x").str[0].astype(int); m["h"] = m.Resolution.str.split("x").str[1].astype(int)
    m.drop(columns=["phash", "dhash"]).to_csv(OUT / "plantseg_groups.csv", index=False)
    s = dict(n_images=n, official_split=m.Split.value_counts().to_dict(), n_classes=int(m.Disease.nunique()), n_crops=int(m.Plant.nunique()), dup_pairs=len(pairs),
             images_in_multi_groups=int((m.group_size > 1).sum()), groups_spanning_splits=int(m[m.group_size > 1].groupby("group").Split.nunique().gt(1).sum()),
             images_in_cross_split_groups=int((m.group_n_split > 1).sum()), groups_spanning_classes=int(m[m.group_size > 1].groupby("group").Disease.nunique().gt(1).sum()),
             images_in_cross_class_groups=int((m.group_n_class > 1).sum()), segppd_copies=int(m.leak_flag.sum()), url_unique=int(m.URL.nunique()),
             test_images_sharing_url_with_train=int(m[m.Split == "Test"].URL.isin(set(m[m.Split == "Training"].URL)).sum()), test_images=int((m.Split == "Test").sum()),
             resolution_median_w=int(m.w.median()), resolution_min_w=int(m.w.min()), resolution_max_w=int(m.w.max()))
    json.dump(s, open(OUT / "audit_summary.json", "w"), indent=1); print(json.dumps(s, indent=1))
    # eligibility counts per class under the registered rule (computed WITHOUT any model)
    ok = m[~m.leak_flag & (m.group_n_class == 1)].copy()                       # drop SegPPD copies and label-conflict groups
    tr_groups = set(ok[ok.Split == "Training"].group)
    ok = ok[~((ok.Split != "Training") & ok.group.isin(tr_groups))]
    va_groups = set(ok[ok.Split == "Validation"].group); ok = ok[~((ok.Split == "Test") & ok.group.isin(va_groups))]            # dev/test images whose duplicate group also has a training image are dropped (train keeps the group)
    ok = pd.concat([ok[ok.Split == "Training"], ok[ok.Split != "Training"].drop_duplicates(["group", "Split"])])  # one image per duplicate group inside dev/test
    cnt = ok.groupby(["Disease", "Split"]).size().unstack(fill_value=0)[["Training", "Validation", "Test"]]
    cnt.to_csv(OUT / "audit_class_split_counts_after_filters.csv")
    print(cnt.describe().round(1)); print("classes meeting train>=10, val>=2, test>=5:", int(((cnt.Training >= 10) & (cnt.Validation >= 2) & (cnt.Test >= 5)).sum()), "of", len(cnt))
    print(cnt[~((cnt.Training >= 10) & (cnt.Validation >= 2) & (cnt.Test >= 5))])
    ok.drop(columns=[]).to_csv(OUT / "audit_filtered_images.csv", index=False)


if __name__ == "__main__":
    main()
