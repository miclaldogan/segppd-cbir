import numpy as np
import pandas as pd
import pytest
from segppd_cbir import protocols as P
from segppd_cbir.split import make_split


def _inv(n_classes=4, n=16):
    rows = []
    for c in range(1, n_classes + 1):
        for k in range(n):
            iid = f"{c:03d}{k:02d}"
            rows.append(dict(image_id=iid, image_path=f"image/{iid}.png", image_sha256=iid, group_id=f"g{iid}", class_id=c,
                             dup_status="unique", raw_class_label=f"c{c}"))
    return pd.DataFrame(rows)


def test_protocol_b_disjoint_and_counts():
    sp = make_split(_inv(), 42)
    b = P.build_protocol_b(sp)
    roles = b.role_B
    assert set(roles.unique()) >= {"train_B", "gallery_B", "dev_query_B", "test_query_B"}
    sets = {r: set(b[b.role_B == r].image_id) for r in ["train_B", "gallery_B", "dev_query_B", "test_query_B"]}
    names = list(sets)
    for i in range(4):
        for j in range(i + 1, 4):
            assert sets[names[i]].isdisjoint(sets[names[j]])
    tq = b[b.role_B == "test_query_B"]
    assert tq.groupby("class_id").size().eq(3).all()
    # fresh test queries: never an A-test or A-dev query
    assert set(tq.image_id).isdisjoint(set(sp[sp.role == "query_primary"].image_id))
    from segppd_cbir import ext_dev as D
    assert set(tq.image_id).isdisjoint(set(D.make_dev(sp)[0].image_id))


def test_protocol_b_deterministic():
    sp = make_split(_inv(), 42)
    assert P.build_protocol_b(sp).role_B.tolist() == P.build_protocol_b(sp).role_B.tolist()


def test_saved_protocol_b_has_no_overlap():
    import os
    if not os.path.exists("artifacts/splits/protocolB_seed4242.csv"):
        pytest.skip("split not generated")
    b = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str})
    roles = ["train_B", "gallery_B", "dev_query_B", "test_query_B"]
    for i in range(4):
        for j in range(i + 1, 4):
            assert set(b[b.role_B == roles[i]].image_id).isdisjoint(set(b[b.role_B == roles[j]].image_id))
    assert set(b[b.role_B == "test_query_B"].group_id).isdisjoint(set(b[b.role_B.isin(["train_B", "gallery_B"])].group_id))


def test_protocol_c_is_disjoint_from_training_and_gallery():
    import pandas as pd
    from pathlib import Path
    p = Path("artifacts/splits/protocolC_seed777.csv")
    if not p.exists():
        return
    c = pd.read_csv(p, dtype={"image_id": str})
    q, g, tr = (set(c[c.role_C == r].image_id) for r in ("test_query_C", "gallery_C", "train_B"))
    assert not (q & g) and not (q & tr)
    assert set(c[c.role_C == "test_query_C"].group_id).isdisjoint(c[c.role_C.isin(["gallery_C", "train_B"])].group_id)
    assert set(c[c.role_C == "test_query_C"].class_id) <= set(c[c.role_C == "gallery_C"].class_id)
