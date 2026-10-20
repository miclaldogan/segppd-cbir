"""Mask semantics: foreground = any non-zero pixel (dataset stores class-id at foreground, 0 background)."""
from __future__ import annotations

import numpy as np

BINS = [("small", 0.0, 0.20), ("medium", 0.20, 0.40), ("large", 0.40, 1.0)]


def occupancy(mask: np.ndarray) -> float:
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    return float((mask != 0).sum() / mask.size)


def occupancy_bin(occ: float | None) -> str:
    """small: 0<occ<=0.20, medium: 0.20<occ<=0.40, large: occ>0.40; zero/missing/NaN -> unknown_invalid."""
    if occ is None or not np.isfinite(occ) or occ <= 0 or occ > 1:
        return "unknown_invalid"
    if occ <= 0.20:
        return "small"
    if occ <= 0.40:
        return "medium"
    return "large"


def bbox(mask: np.ndarray, margin_frac: float = 0.0):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    my, mx = int(round(margin_frac * h)), int(round(margin_frac * w))
    return (max(xs.min() - mx, 0), max(ys.min() - my, 0), min(xs.max() + 1 + mx, w), min(ys.max() + 1 + my, h))
