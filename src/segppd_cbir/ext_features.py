"""EXTENSION track (training-free): richer descriptors from the same frozen ResNet50 (no fitting of any kind).
Not the mandated baseline; reported separately. Blocks are per-image vectors, combined later on DEV-selected recipes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import feature_extractor as fe

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_tensor(im: Image.Image, size: int) -> torch.Tensor:
    im = im.resize((size, size), Image.BICUBIC)  # full image, no crop
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - MEAN) / STD


def gem(x, p=3.0):
    return x.clamp(min=1e-6).pow(p).mean((2, 3)).pow(1.0 / p)


@torch.inference_mode()
def blocks(model, x):
    m = model
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    l3 = m.layer3(m.layer2(m.layer1(x)))
    l4 = m.layer4(l3)
    return {"l3_gap": l3.mean((2, 3)), "l3_gem": gem(l3), "l4_gap": l4.mean((2, 3)), "l4_gem": gem(l4),
            "l4_max": l4.amax((2, 3))}


def extract_blocks(model, images, size, batch=16, flip=True):
    out: dict[str, list] = {}
    buf = []

    def flush():
        x = torch.stack([to_tensor(i, size) for i in buf])
        b = blocks(model, x)
        if flip:
            b2 = blocks(model, torch.flip(x, dims=[3]))
            b = {k: (b[k] + b2[k]) / 2 for k in b}
        for k, v in b.items():
            out.setdefault(k, []).append(v.numpy())
        buf.clear()

    for im in images:
        buf.append(im)
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
