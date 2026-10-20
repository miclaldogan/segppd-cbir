"""Frozen torchvision ResNet50 -> two descriptors from ONE forward pass.

A: 1000-D raw fc logits (pre-softmax, ImageNet class-evidence scores)   [primary]
B: 2048-D global-average-pooled features (input to fc)                  [comparison]
No training, no normalisation, no softmax, no projection.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision.models import ResNet50_Weights, resnet50

WEIGHTS_ENUM = ResNet50_Weights.IMAGENET1K_V2


def load_model(device: str = "cpu"):
    model = resnet50(weights=WEIGHTS_ENUM)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), WEIGHTS_ENUM.transforms()


def checkpoint_fingerprint(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def weights_file_sha256() -> str | None:
    p = Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS_ENUM.url).name
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def transform_description(tf) -> dict:
    return {"resize_size": tf.resize_size, "crop_size": tf.crop_size, "interpolation": str(tf.interpolation),
            "mean": list(tf.mean), "std": list(tf.std), "antialias": tf.antialias}


def open_rgb(path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


@torch.inference_mode()
def forward_both(model, x: torch.Tensor):
    """Return (logits[N,1000], pooled[N,2048]) from a single pass; fc is applied to the pooled vector."""
    m = model
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    pooled = torch.flatten(m.avgpool(x), 1)
    logits = m.fc(pooled)
    return logits, pooled


def extract(model, tf, images, device="cpu", batch_size=16):
    """images: iterable of PIL RGB images. Returns float32 arrays (logits, pooled)."""
    L, P = [], []
    batch = []

    def flush():
        x = torch.stack([tf(i) for i in batch]).to(device)
        lg, pl = forward_both(model, x)
        L.append(lg.float().cpu().numpy())
        P.append(pl.float().cpu().numpy())
        batch.clear()

    for im in images:
        batch.append(im)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    return np.concatenate(L), np.concatenate(P)


def assert_frozen(model):
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
