"""Dataset audit: decode every image/mask, hash, collect mask semantics evidence."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from .labels import parse_filename, parse_fullname


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def audit(root: Path, out_dir: Path) -> pd.DataFrame:
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = parse_fullname(root / "fullname.txt")
    names = sorted(p.name for p in (root / "image").iterdir())
    mnames = sorted(p.name for p in (root / "mask").iterdir())
    rows, mask_vals = [], collections.Counter()
    for n in tqdm(names, desc="audit"):
        cid, idx = parse_filename(n)
        ip, mp = root / "image" / n, root / "mask" / n
        r = {"image_id": n[:-4], "image_path": f"image/{n}", "mask_path": f"mask/{n}" if n in mnames else "",
             "class_id": cid, "within_class_index": idx, "flags": []}
        c = classes.get(cid)
        if c is None:
            r["flags"].append("class_id_not_in_fullname")
        else:
            r.update(raw_class_label=c["class_name"], crop=c["crop"], condition=c["condition"],
                     healthy_by_name=c["is_healthy_name"])
        r["image_sha256"] = sha256_file(ip)
        r["image_bytes"] = ip.stat().st_size
        try:
            with Image.open(ip) as im:
                r["img_format"], r["img_mode"] = im.format, im.mode
                r["exif_orientation"] = int(im.getexif().get(274, 1))
                im.load()
                r["img_w"], r["img_h"] = im.size
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                r["phash"] = str(imagehash.phash(rgb))
                r["dhash"] = str(imagehash.dhash(rgb))
        except Exception as e:  # noqa: BLE001
            r["flags"].append(f"image_decode_error:{type(e).__name__}")
        if r["mask_path"]:
            try:
                with Image.open(mp) as m:
                    r["mask_mode"] = m.mode
                    m.load()
                    r["mask_w"], r["mask_h"] = m.size
                    a = np.array(m)
                u, cnt = np.unique(a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1, 1),
                                   axis=0, return_counts=True)
                r["mask_n_unique"] = len(u)
                r["mask_values"] = json.dumps({",".join(map(str, v)): int(c) for v, c in zip(u, cnt)}) if len(u) <= 16 else "many"
                for v, c in zip(u, cnt):
                    mask_vals[",".join(map(str, v))] += 1
                r["mask_total_px"] = int(a.shape[0] * a.shape[1])
                nz = [int(v[0]) for v in u if v[0] != 0] if a.ndim == 2 else []
                r["mask_fg_value"] = nz[0] if len(nz) == 1 else -1
                r["mask_fg_px"] = int((a != 0).sum())
                r["occupancy"] = r["mask_fg_px"] / r["mask_total_px"]
                if r["mask_mode"] == "P":
                    r["flags"].append("mask_palette_mode")
                if r["mask_fg_value"] != cid:
                    r["flags"].append("mask_fg_value_ne_class_id")
                r["mask_sha256"] = sha256_file(mp)
            except Exception as e:  # noqa: BLE001
                r["flags"].append(f"mask_decode_error:{type(e).__name__}")
            if (r.get("img_w"), r.get("img_h")) != (r.get("mask_w"), r.get("mask_h")):
                r["flags"].append("image_mask_size_mismatch")
        else:
            r["flags"].append("mask_missing")
        r["flags"] = ";".join(r["flags"])
        rows.append(r)
    extra = sorted(set(mnames) - set(names))
    df = pd.DataFrame(rows)
    df["image_id"] = df["image_id"].astype(str)
    df.to_csv(out_dir / "file_inventory.csv", index=False)
    json.dump({"mask_value_to_n_masks_containing": mask_vals, "masks_without_image": extra},
              open(out_dir / "mask_value_census.json", "w"), indent=1)
    return df
