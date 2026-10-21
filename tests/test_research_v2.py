from pathlib import Path

import pandas as pd


def test_plantseg_protocol_is_group_and_image_disjoint():
    p = Path("artifacts/research_v2/trackB/protocol_P.csv")
    if not p.exists():
        return
    P = pd.read_csv(p); g = {r: set(P[P.role == r].group) for r in ("train", "dev", "test")}; n = {r: set(P[P.role == r].Name) for r in g}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not (g[a] & g[b]) and not (n[a] & n[b])
    cnt = P.groupby(["Disease", "role"]).size().unstack(fill_value=0)
    assert (cnt.train >= 10).all() and (cnt.dev >= 2).all() and (cnt.test >= 5).all()          # registered eligibility
    assert len(cnt) == 98


def test_tie_band_is_one_dev_query():
    from segppd_cbir.rv2 import common as C
    assert abs(C.TIE_EPS - 1 / 150) < 1e-12


def test_segppd_protocol_sets_disjoint():
    from segppd_cbir.rv2 import common as C
    from segppd_cbir.rv2.train import leakage_checks
    assert leakage_checks(C.segppd_sets())
