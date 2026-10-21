"""Training-only augmentation recipes. 'base' = the recipe used in all earlier DINOv2 runs (kept bit-identical)."""
from __future__ import annotations

import io
import random

import torchvision.transforms as T
from PIL import Image, ImageFilter

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class JPEG:
    """Random JPEG re-compression (quality in [q_lo, q_hi]) with probability p."""
    def __init__(self, p=0.5, q_lo=40, q_hi=95):
        self.p, self.q_lo, self.q_hi = p, q_lo, q_hi

    def __call__(self, im):
        if random.random() > self.p:
            return im
        b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(self.q_lo, self.q_hi)); b.seek(0); return Image.open(b).convert("RGB")


class DownUp:
    """Random resolution loss: downscale to a random size then back (simulates low-resolution web photos)."""
    def __init__(self, p=0.5, lo=96, hi=224):
        self.p, self.lo, self.hi = p, lo, hi

    def __call__(self, im):
        if random.random() > self.p:
            return im
        s = random.randint(self.lo, self.hi); return im.resize((s, s), Image.BILINEAR).resize(im.size, Image.BICUBIC)


class Blur:
    def __init__(self, p=0.3, r=(0.3, 1.2)):
        self.p, self.r = p, r

    def __call__(self, im):
        return im.filter(ImageFilter.GaussianBlur(random.uniform(*self.r))) if random.random() < self.p else im


def make_train_aug(kind, size):
    base_tail = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if kind == "base":
        return T.Compose([T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2), T.ColorJitter(0.2, 0.2, 0.2, 0.02)] + base_tail)
    if kind == "robust":  # registered Track-C recipe: moderate colour/illumination, resolution, JPEG, blur, small geometry; no aggressive crops/colours
        return T.Compose([T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2), T.RandomRotation(15),
                          T.ColorJitter(0.3, 0.3, 0.25, 0.03), DownUp(0.5), Blur(0.3), JPEG(0.5)] + base_tail)
    # ---- ablation recipes (each adds ONE family of perturbations to 'base'); all training-only
    base = [T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2)]
    if kind == "color":      # stronger but moderate photometric / illumination variation
        return T.Compose(base + [T.ColorJitter(0.35, 0.35, 0.3, 0.04)] + base_tail)
    if kind == "res":        # resolution loss, blur, JPEG compression
        return T.Compose(base + [T.ColorJitter(0.2, 0.2, 0.2, 0.02), DownUp(0.5), Blur(0.3), JPEG(0.5)] + base_tail)
    if kind == "geo":        # scale/rotation/crop variation
        return T.Compose([T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2), T.RandomRotation(20), T.ColorJitter(0.2, 0.2, 0.2, 0.02)] + base_tail)
    raise ValueError(kind)
