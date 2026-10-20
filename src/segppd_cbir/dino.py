"""DINOv2-S (timm vit_small_patch14_dinov2.lvd142m) helpers. Pretrained weights are the official LVD-142M release via the HF hub."""
from __future__ import annotations

import numpy as np
import timm
import torch

NAME = "vit_small_patch14_dinov2.lvd142m"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(device="cuda", img_size=224, name=NAME):
    m = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
    return m.eval().to(device)


def prep(pil_list, size, device):
    x = torch.stack([torch.from_numpy(np.asarray(im.resize((size, size), 3), dtype=np.float32) / 255.0).permute(2, 0, 1) for im in pil_list])
    return ((x - MEAN) / STD).to(device)


def pool(tokens):
    """tokens [B, 1+N, D] -> dict of vectors."""
    cls, p = tokens[:, 0], tokens[:, 1:]
    gem = p.clamp(min=1e-6).pow(3).mean(1).pow(1 / 3) if bool((p > 0).all()) else p.abs().pow(3).mean(1).pow(1 / 3)
    return {"cls": cls, "mean": p.mean(1), "gem": gem}


@torch.inference_mode()
def frozen_blocks(model, pil_iter, size, device="cuda", bs=32, flip=True):
    out: dict[str, list] = {}
    buf = []

    def flush():
        x = prep(buf, size, device)
        t = model.forward_features(x)
        b = pool(t)
        if flip:
            b2 = pool(model.forward_features(torch.flip(x, dims=[3])))
            b = {k: (b[k] + b2[k]) / 2 for k in b}
        for k, v in b.items():
            out.setdefault(k, []).append(v.float().cpu().numpy())
        buf.clear()

    for im in pil_iter:
        buf.append(im)
        if len(buf) == bs:
            flush()
    if buf:
        flush()
    return {k: np.concatenate(v) for k, v in out.items()}
