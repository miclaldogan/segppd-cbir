import numpy as np
import pytest
import torch
from PIL import Image
from segppd_cbir import feature_extractor as fe


@pytest.fixture(scope="module")
def mt():
    return fe.load_model("cpu")


def _imgs(n=3):
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(n)]


def test_frozen_and_eval(mt):
    model, _ = mt
    fe.assert_frozen(model)
    assert not any(p.requires_grad for p in model.parameters())


def test_dimensions_and_single_pass_consistency(mt):
    model, tf = mt
    lg, pl = fe.extract(model, tf, _imgs(), "cpu", 2)
    assert lg.shape == (3, 1000) and pl.shape == (3, 2048) and lg.dtype == np.float32
    with torch.inference_mode():  # logits from the hooked pass == the stock forward pass
        ref = model(torch.stack([tf(i) for i in _imgs()])).numpy()
    np.testing.assert_allclose(lg, ref, rtol=1e-4, atol=1e-4)
    # fc applied to pooled reproduces logits => A is a deterministic linear map of B (documented, not a new feature)
    with torch.inference_mode():
        np.testing.assert_allclose(model.fc(torch.from_numpy(pl)).numpy(), lg, rtol=1e-4, atol=1e-4)
    assert np.isfinite(lg).all() and np.isfinite(pl).all()


def test_deterministic_and_batch_invariant(mt):
    model, tf = mt
    a = fe.extract(model, tf, _imgs(), "cpu", 1)
    b = fe.extract(model, tf, _imgs(), "cpu", 3)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-4, atol=1e-4)


def test_weights_unchanged_by_inference(mt):
    model, tf = mt
    before = fe.checkpoint_fingerprint(model)
    fe.extract(model, tf, _imgs(), "cpu", 2)
    assert fe.checkpoint_fingerprint(model) == before


def test_preprocessing_is_weight_transform(mt):
    _, tf = mt
    d = fe.transform_description(tf)
    assert list(d["resize_size"]) == [232] and list(d["crop_size"]) == [224]
