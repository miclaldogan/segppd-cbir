"""Exact Euclidean ranking. Ties broken by (distance, gallery image_id)."""
from __future__ import annotations

import numpy as np


def squared_l2(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Exact squared L2 between rows of q (n,d) and g (m,d), float64 accumulation."""
    q = q.astype(np.float64)
    g = g.astype(np.float64)
    d = (q ** 2).sum(1)[:, None] + (g ** 2).sum(1)[None, :] - 2.0 * q @ g.T
    return np.maximum(d, 0.0)


def rank(q: np.ndarray, g: np.ndarray, gallery_ids, k: int | None = None):
    """Return (order, sq_dist) per query; order indexes the gallery, ties by image_id."""
    d2 = squared_l2(q, g)
    ids = np.asarray(gallery_ids)
    id_rank = np.argsort(np.argsort(ids, kind="stable"), kind="stable")
    orders = np.empty(d2.shape, dtype=np.int64)
    for i in range(d2.shape[0]):
        orders[i] = np.lexsort((id_rank, d2[i]))
    if k is not None:
        orders = orders[:, :k]
    return orders, np.take_along_axis(d2, orders, axis=1)
