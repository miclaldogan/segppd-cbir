import numpy as np
import pandas as pd
import pytest
from segppd_cbir import evaluate as E


def _split():
    q = pd.DataFrame({"image_id": ["q1", "q2"], "class_id": [1, 2]})
    g = pd.DataFrame({"image_id": [f"g{i}" for i in range(6)], "class_id": [1, 1, 2, 2, 2, 3]})
    return q, g


def test_per_query_metrics_and_ceiling_and_k_support():
    q, g = _split()
    orders = np.array([[0, 2, 1, 3, 4, 5], [2, 0, 3, 4, 1, 5]])  # q1 rel: 1,0,1,0,0,0 ; q2 rel: 1,0,1,1,0,0
    pq = E.per_query_metrics(orders, q, g, ks=(1, 3, 10), rep="x", cohort="t", n_perm=5)
    assert set(pq.K) == {1, 3}  # K=10 > gallery size(6): unsupported, not invented
    r = pq[(pq.query_id == "q1") & (pq.K == 3)].iloc[0]
    assert r.R == 2 and r.P == pytest.approx(2 / 3) and r.Recall == 1.0 and r.Hit == 1.0 and r.ceiling == pytest.approx(2 / 3)
    assert r.AP_K == pytest.approx((1 + 2 / 3) / 2) and r.full_AP == pytest.approx((1 + 2 / 3) / 2)
    assert r.rand_P == pytest.approx(2 / 6) and r.rand_Recall == pytest.approx(3 / 6)


def test_random_hit_expectation_exact():
    q, g = _split()
    pq = E.per_query_metrics(np.array([[0, 1, 2, 3, 4, 5]] * 2), q, g, ks=(3,), n_perm=3)
    r = pq[pq.query_id == "q1"].iloc[0]  # R=2,N=6,K=3 -> 1 - C(4,3)/C(6,3) = 1-4/20
    assert r.rand_Hit == pytest.approx(0.8)


def test_macro_vs_micro_differ_with_unbalanced_classes():
    pq = pd.DataFrame({"rep": "a", "cohort": "c", "K": 1, "query_id": ["a", "b", "c", "d"], "class_id": [1, 1, 1, 2],
                       **{m: [1.0, 1.0, 1.0, 0.0] for m in E.METRICS}})
    s = E.macro_micro(pq).iloc[0]
    assert s.micro_P == pytest.approx(0.75) and s.macro_P == pytest.approx(0.5)


def test_paired_diff_zero_for_identical():
    pq = pd.DataFrame({"query_id": list("abcd"), "class_id": [1, 1, 2, 2], "K": 1, "P": [1., 0., 1., 0.]})
    d = E.paired_diff_ci(pq, pq, "P", n_boot=50).iloc[0]
    assert d.mean_diff_A_minus_B == 0 and d.ci_lo == 0 and d.ci_hi == 0
