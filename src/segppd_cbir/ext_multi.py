"""EXTENSION (TRAINED): multi-branch embedding head on FROZEN features.
fusion: 'concat' | 'attn' (softmax gate over branch projections, conditioned on all branches)
loss:   'arcface' | 'supcon' | 'proxyanchor'. Train labels = gallery only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHead(nn.Module):
    def __init__(self, dims, d_emb, n_cls, fusion="concat", drop=0.3):
        super().__init__()
        self.fusion = fusion
        self.proj = nn.ModuleList([nn.Sequential(nn.Dropout(drop), nn.Linear(d, d_emb)) for d in dims])
        if fusion == "attn":
            self.gate = nn.Linear(d_emb * len(dims), len(dims))
            self.out = nn.Linear(d_emb, d_emb)
        else:
            self.out = nn.Linear(d_emb * len(dims), d_emb)
        self.W = nn.Parameter(torch.randn(n_cls, d_emb) * 0.01)

    def embed(self, xs):
        z = [p(x) for p, x in zip(self.proj, xs)]
        if self.fusion == "attn":
            a = torch.softmax(self.gate(torch.cat(z, 1)), 1)
            h = sum(a[:, i:i + 1] * z[i] for i in range(len(z)))
        else:
            h = torch.cat(z, 1)
        return F.normalize(self.out(F.relu(h) if self.fusion == "attn" else h), dim=1)


def _arcface(h, e, y, m=0.3, s=30.0):
    cos = e @ F.normalize(h.W, dim=1).T
    th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    oh = F.one_hot(y, cos.shape[1]).bool()
    return F.cross_entropy(s * torch.where(oh, torch.cos(th + m), cos), y)


def _supcon(e, y, t=0.1):
    sim = e @ e.T / t
    n = len(y)
    eye = torch.eye(n, dtype=torch.bool, device=e.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    cnt = pos.sum(1)
    ok = cnt > 0
    return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()


def _proxyanchor(h, e, y, alpha=32.0, delta=0.1):
    P = F.normalize(h.W, dim=1)
    cos = e @ P.T
    oh = F.one_hot(y, P.shape[0]).float()
    pos_exp = torch.exp(-alpha * (cos - delta)) * oh
    neg_exp = torch.exp(alpha * (cos + delta)) * (1 - oh)
    has_pos = oh.sum(0) > 0
    pos_term = torch.log(1 + pos_exp.sum(0))[has_pos].sum() / has_pos.sum()
    neg_term = torch.log(1 + neg_exp.sum(0)).sum() / P.shape[0]
    return pos_term + neg_term


def fit(Xs, y, d_emb=256, epochs=100, loss="arcface", fusion="concat", lr=1e-3, wd=1e-4, drop=0.3, seed=0, bs=128):
    torch.manual_seed(seed)
    stats = [(X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6) for X in Xs]
    Xt = [torch.from_numpy((X - m) / s).float() for X, (m, s) in zip(Xs, stats)]
    yt = torch.from_numpy(y).long()
    h = MultiHead([X.shape[1] for X in Xs], d_emb, int(y.max()) + 1, fusion, drop)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        h.train()
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            e = h.embed([x[idx] for x in Xt])
            l = _arcface(h, e, yt[idx]) if loss == "arcface" else _supcon(e, yt[idx]) if loss == "supcon" else _proxyanchor(h, e, yt[idx])
            opt.zero_grad(); l.backward(); opt.step()
        sched.step()
    h.eval()

    def embed(Zs):
        with torch.no_grad():
            return h.embed([torch.from_numpy((Z - m) / s).float() for Z, (m, s) in zip(Zs, stats)]).numpy().astype(np.float32)
    return embed
