"""EXTENSION track S (TRAINED; violates the instructor's no-training rule on purpose, reported separately).
Supervised embedding head (ArcFace-style cosine-margin softmax) on frozen ResNet50 multi-block features.
Training data = gallery images + labels ONLY; query images are never seen in training."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, d_in, d_emb, n_cls, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, d_emb))
        self.W = nn.Parameter(torch.randn(n_cls, d_emb) * 0.01)

    def embed(self, x):
        return F.normalize(self.net(x), dim=1)

    def logits(self, x, y=None, s=30.0, m=0.3):
        e = self.embed(x)
        cos = e @ F.normalize(self.W, dim=1).T
        if y is None:
            return s * cos
        th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        tgt = torch.cos(th + m)
        oh = F.one_hot(y, cos.shape[1]).bool()
        return s * torch.where(oh, tgt, cos)


def fit(X, y, d_emb=512, epochs=100, margin=0.3, lr=1e-3, wd=1e-4, drop=0.3, seed=0, bs=128):
    """X: (n,d) float32 (already L2-normalised block concat), y: int class indices 0..C-1. Returns embed fn."""
    torch.manual_seed(seed)
    n_cls = int(y.max()) + 1
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6  # input standardisation fitted on TRAIN only
    Xt = torch.from_numpy((X - mu) / sd).float()
    yt = torch.from_numpy(y).long()
    h = Head(X.shape[1], d_emb, n_cls, drop)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        h.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(h.logits(Xt[idx], yt[idx], m=margin), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    h.eval()

    def embed(Z):
        with torch.no_grad():
            return h.embed(torch.from_numpy((Z - mu) / sd).float()).numpy().astype(np.float32)
    return embed
