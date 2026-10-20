import pandas as pd
import pytest
from segppd_cbir.split import check_split, make_split


def _inv():
    rows = []
    for c in range(1, 6):
        n = {1: 20, 2: 14, 3: 12, 4: 3, 5: 1}[c]  # class3 -> secondary (12<13), class4 secondary(3), class5 unscorable
        for k in range(n):
            iid = f"{c:03d}{k:02d}"
            rows.append(dict(image_id=iid, image_path=f"image/{iid}.png", image_sha256=iid, group_id=f"g{iid}",
                             class_id=c, dup_status="unique"))
    rows.append(dict(image_id="00099", image_path="image/00099.png", image_sha256="x", group_id="g00099", class_id=1,
                     dup_status="excluded"))
    return pd.DataFrame(rows)


def test_eligibility_and_roles():
    m = make_split(_inv(), 42)
    cc = m.groupby("class_id").cohort_class.first().to_dict()
    assert cc == {1: "primary", 2: "primary", 3: "secondary", 4: "secondary", 5: "unscorable"}
    q = m[m.role == "query_primary"]
    assert q.groupby("class_id").size().to_dict() == {1: 3, 2: 3}
    assert (m[m.role == "gallery_primary"].groupby("class_id").size() >= 10).all()
    assert m[m.image_id == "00099"].role.iloc[0] == "excluded_duplicate_or_conflict"
    assert m[m.class_id == 5].role.iloc[0] == "unscorable_class"
    assert (m[m.cohort_class == "secondary"].groupby("class_id").role.apply(lambda s: (s == "query_secondary").sum()) == 1).all()


def test_no_leakage_and_determinism():
    inv = _inv()
    a, b = make_split(inv, 42), make_split(inv, 42)
    assert a["role"].tolist() == b["role"].tolist()
    assert a["role"].tolist() != make_split(inv, 7)["role"].tolist()
    check_split(a)


def test_group_straddle_detected():
    m = make_split(_inv(), 42)
    q = m[m.role == "query_primary"].index[0]
    g = m[m.role == "gallery_primary"].index[0]
    m.loc[g, "group_id"] = m.loc[q, "group_id"]
    with pytest.raises(AssertionError):
        check_split(m)
