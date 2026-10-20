import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from segppd_cbir.duplicate_groups import build_groups


def _h(seed, noise=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    im = Image.fromarray(a)
    return str(imagehash.phash(im)), str(imagehash.dhash(im))


def _df(rows):
    return pd.DataFrame([dict(image_id=i, class_id=c, image_sha256=s, phash=p, dhash=d) for i, c, s, p, d in rows])


def test_exact_dup_same_class_keeps_one_and_conflict_excluded():
    p1, d1 = _h(1)
    p2, d2 = _h(2)
    p3, d3 = _h(3)
    df = _df([("00001", 1, "A", p1, d1), ("00002", 1, "A", p1, d1),      # same class exact dup
              ("00003", 2, "B", p2, d2), ("00004", 3, "B", p2, d2),      # label conflict
              ("00005", 4, "C", p3, d3)])
    g = build_groups(df).set_index("image_id")
    assert g.loc["00001", "dup_status"] == "kept_representative" and g.loc["00002", "dup_status"] == "excluded"
    assert g.loc["00003", "dup_reason"] == g.loc["00004", "dup_reason"] == "label_conflict_duplicate_group"
    assert g.loc["00005", "dup_status"] == "unique"
    assert g.loc["00001", "group_id"] == g.loc["00002", "group_id"] != g.loc["00005", "group_id"]
