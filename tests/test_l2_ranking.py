import numpy as np
import faiss
from segppd_cbir.index import rank, squared_l2


def test_toy_points():
    g = np.array([[0, 0], [3, 4], [1, 0]], dtype=np.float32)
    q = np.array([[0, 0]], dtype=np.float32)
    o, d = rank(q, g, ["a", "b", "c"])
    assert o[0].tolist() == [0, 2, 1]
    assert np.allclose(np.sqrt(d[0]), [0, 1, 5])


def test_ties_broken_by_image_id():
    g = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    o, _ = rank(np.zeros((1, 2), np.float32), g, ["z", "b", "m"])
    assert o[0].tolist() == [1, 2, 0]


def test_matches_faiss_ordering():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(200, 32)).astype(np.float32)
    q = rng.normal(size=(10, 32)).astype(np.float32)
    ix = faiss.IndexFlatL2(32)
    ix.add(g)
    _, fi = ix.search(q, 20)
    o, _ = rank(q, g, [f"{i:04d}" for i in range(200)], k=20)
    assert (fi == o).all()
