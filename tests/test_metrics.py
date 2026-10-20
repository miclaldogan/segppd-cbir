import math
import numpy as np
import pytest
from segppd_cbir import metrics as m


def test_precision_recall_hit():
    rel = [1, 0, 1, 0, 0]
    assert m.precision_at_k(rel, 1) == 1.0
    assert m.precision_at_k(rel, 5) == 0.4
    assert m.recall_at_k(rel, 3, R=4) == 0.5
    assert m.hit_at_k([0, 0, 1], 2) == 0.0 and m.hit_at_k([0, 0, 1], 3) == 1.0


def test_recall_undefined_when_no_positives():
    assert math.isnan(m.recall_at_k([0, 0], 2, R=0))
    assert math.isnan(m.ap_at_k([0, 0], 2, R=0))
    assert math.isnan(m.full_ap([0, 0], R=0))


def test_ap_hand_computed():
    rel = [1, 0, 1, 0]          # P@1=1, P@3=2/3
    assert m.ap_at_k(rel, 4, R=2) == pytest.approx((1 + 2 / 3) / 2)
    assert m.ap_at_k(rel, 2, R=2) == pytest.approx(1 / 2)       # denom min(R,K)=2
    assert m.ap_at_k(rel, 2, R=5) == pytest.approx(1 / 2)       # sparse: min(5,2)
    assert m.full_ap(rel, R=4) == pytest.approx((1 + 2 / 3) / 4)  # missing positives penalised


def test_ceiling_and_k_guard():
    assert m.precision_ceiling(3, 10) == 0.3
    assert m.precision_ceiling(30, 10) == 1.0
    with pytest.raises(ValueError):
        m.precision_at_k([1, 0], 3)
