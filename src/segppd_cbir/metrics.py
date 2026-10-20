"""Retrieval metrics implemented from scratch (see docs/RESEARCH_PROTOCOL.md).

rel: 1-D 0/1 array of relevance flags for the ranked list; R: relevant images in the gallery.
"""
from __future__ import annotations

import numpy as np


def _top(rel, k):
    rel = np.asarray(rel, dtype=float)
    if k > len(rel):
        raise ValueError(f"K={k} exceeds ranked list length {len(rel)}")
    return rel[:k]


def precision_at_k(rel, k):
    return float(_top(rel, k).sum() / k)


def recall_at_k(rel, k, R):
    """Undefined (NaN) when R == 0."""
    if R <= 0:
        return float("nan")
    return float(_top(rel, k).sum() / R)


def hit_at_k(rel, k):
    return float(_top(rel, k).sum() > 0)


def precision_ceiling(R, k):
    """Largest attainable P@K given R relevant gallery items: min(R, K) / K."""
    return min(R, k) / k


def ap_at_k(rel, k, R):
    """Truncated AP@K: sum_{i<=K} P@i * rel_i / min(R, K). NaN when R == 0."""
    if R <= 0:
        return float("nan")
    r = _top(rel, k)
    cum = np.cumsum(r)
    prec = cum / np.arange(1, k + 1)
    return float((prec * r).sum() / min(R, k))


def full_ap(rel, R):
    """Full-list AP with denominator R (rel must cover the whole gallery ranking)."""
    if R <= 0:
        return float("nan")
    r = np.asarray(rel, dtype=float)
    prec = np.cumsum(r) / np.arange(1, len(r) + 1)
    return float((prec * r).sum() / R)
