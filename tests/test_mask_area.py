import numpy as np
from segppd_cbir.masks import bbox, occupancy, occupancy_bin


def test_occupancy_synthetic():
    m = np.zeros((10, 10), np.uint8)
    m[:2, :5] = 7  # class-id valued foreground
    assert occupancy(m) == 0.10


def test_bin_boundaries():
    assert occupancy_bin(0.20) == "small" and occupancy_bin(0.2000001) == "medium"
    assert occupancy_bin(0.40) == "medium" and occupancy_bin(0.4000001) == "large"
    assert occupancy_bin(1.0) == "large"


def test_empty_or_missing_never_small():
    for v in (0.0, None, float("nan"), -0.1, 1.5):
        assert occupancy_bin(v) == "unknown_invalid"


def test_bbox():
    m = np.zeros((10, 10), np.uint8)
    m[2:4, 3:6] = 1
    assert bbox(m) == (3, 2, 6, 4)
    assert bbox(np.zeros((4, 4))) is None
