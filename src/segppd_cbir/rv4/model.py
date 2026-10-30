"""Generic metric-embedding model over timm backbones (DINOv2 / DINOv3 ViTs, DINOv3 ConvNeXt) with optional depth truncation and
multi-view (TTA) embedding. Same head as research_v2 (Dropout 0.2 + Linear -> L2-normalise)."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ext_trained import Head

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONES = {  # key -> (timm name, kind, approx params M)
    "dinov2_base": ("vit_base_patch14_dinov2.lvd142m", "vit"), "dinov2_small": ("vit_small_patch14_dinov2.lvd142m", "vit"), "dinov2_small_reg4": ("vit_small_patch14_reg4_dinov2.lvd142m", "vit"),
    "dinov3_vits16": ("vit_small_patch16_dinov3.lvd1689m", "vit"), "dinov3_vits16plus": ("vit_small_plus_patch16_dinov3.lvd1689m", "vit"), "dinov3_vitb16": ("vit_base_patch16_dinov3.lvd1689m", "vit"),
    "dinov3_convnext_tiny": ("convnext_tiny.dinov3_lvd1689m", "convnext"), "dinov3_convnext_small": ("convnext_small.dinov3_lvd1689m", "convnext")}


class Emb(nn.Module):
    def __init__(self, n_cls, d_emb, backbone, blocks=4, truncate=None, img_size=224, full=False, pretrained=True):
        super().__init__()
        name, kind = BACKBONES[backbone]; self.kind, self.size, self.eval_size = kind, img_size or 224, img_size or 224
        kw = ({"dynamic_img_size": True, **({"img_size": img_size} if img_size else {})}) if kind == "vit" else {}  # img_size None = keep the pretrained grid (research_v2 DinoEmb checkpoints) and resample dynamically
        self.m = timm.create_model(name, pretrained=pretrained, num_classes=0, **kw)
        self.patch = self.m.patch_embed.patch_size[0] if kind == "vit" else 32
        if truncate:
            assert kind == "vit"; self.m.blocks = self.m.blocks[:truncate]
        for p in self.m.parameters(): p.requires_grad_(False)
        self.full, self.blocks = full, blocks
        for mod in self.trainable_units(): 
            for p in mod.parameters(): p.requires_grad_(True)
        self.head = Head(self.m.num_features, d_emb, n_cls, drop=0.2)

    def trainable_units(self):
        """Ordered deepest-first list of modules that are trained (layer-wise lr decay follows this order)."""
        if self.kind == "vit":
            bl = list(self.m.blocks); units = list(reversed(bl if self.full else bl[-self.blocks:])); return [self.m.norm] + units
        st = self.m.stages; units = [st[-1]] if not self.full else list(reversed(list(st)))
        return [self.m.head.norm] + units

    def groups(self, lr_bb, lr_head, decay=0.8, proxy_lr_mult=1.0):
        g = [{"params": list(self.head.net.parameters()), "lr": lr_head}, {"params": [self.head.W], "lr": lr_head * proxy_lr_mult}]
        for i, u in enumerate(self.trainable_units()):
            lr = lr_bb if i == 0 else lr_bb * decay ** (i - 1); g.append({"params": list(u.parameters()), "lr": lr})
        return g

    def feat(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f = self.m.forward_features(x); f = f[:, 0] if self.kind == "vit" else self.m.forward_head(f, pre_logits=True)
        return f.float()

    def n_params(self):
        return sum(p.numel() for p in self.m.parameters()) + sum(p.numel() for p in self.head.net.parameters())


def views_of(x, views, size, patch=16):
    """x: [B,3,H,W] at the base size (full image, no crop). Returns list of tensors (each its own view) at `size` or larger."""
    out = [x]
    if "ms" in views: s2 = int(round(size * 1.25 / patch)) * patch; out.append(F.interpolate(x, size=(s2, s2), mode="bicubic", align_corners=False))
    if "5crop" in views:
        big = F.interpolate(x, scale_factor=1.15, mode="bicubic", align_corners=False); H = big.shape[-1]; s = size; c = (H - s) // 2
        out += [big[..., :s, :s], big[..., :s, -s:], big[..., -s:, :s], big[..., -s:, -s:], big[..., c:c + s, c:c + s]]
    return out


@torch.inference_mode()
def embed_tensors(net, X, bs=64, views=()):
    """Flip-averaged unit embedding (research_v2 default) plus optional extra views; every view is flip-averaged too."""
    net.eval(); out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(DEV, non_blocking=True).float(); e = 0
        for v in views_of(x, views, net.eval_size, net.patch):
            e = e + F.normalize(net.head.embed(net.feat(v)) + net.head.embed(net.feat(torch.flip(v, dims=[3]))), dim=1)
        out.append(F.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy()
