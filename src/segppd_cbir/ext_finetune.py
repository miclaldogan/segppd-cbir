"""EXTENSION track S2 (TRAINED): fine-tune ResNet50 layer3+layer4 with an ArcFace embedding head, then Euclidean retrieval
on the L2-normalised embedding. Training labels = gallery only; BN statistics frozen (eval mode)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from . import ext_dev as D
from . import feature_extractor as fe
from .ext_features import MEAN, STD, to_tensor
from .ext_trained import Head

ROOT = Path("data/extracted/SegPPD-101")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, n_cls, d_emb=256):
        super().__init__()
        self.m, _ = fe.load_model("cpu")
        for p in self.m.parameters():
            p.requires_grad_(False)
        for n in ("layer3", "layer4"):
            for p in getattr(self.m, n).parameters():
                p.requires_grad_(True)
        self.head = Head(2048, d_emb, n_cls, drop=0.2)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.m.modules():  # keep BN statistics fixed always
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        return self

    def feat(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return x.mean((2, 3))


AUG = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(),
                 T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), T.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist())])


@torch.inference_mode()
def embed_all(net, imgs, size=256, bs=32):
    net.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([to_tensor(im, size) for im in imgs[i:i + bs]]).to(DEV)
        e = net.head.embed(net.feat(x)) + net.head.embed(net.feat(torch.flip(x, dims=[3])))
        out.append(F.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def run(mode: str, epochs: int, out_prefix: str, snap=(5, 10, 15, 20, 25, 30), seed=0, threads=8):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    sp = pd.read_csv("artifacts/splits/split_seed42.csv", dtype={"image_id": str})
    dq, dg = D.make_dev(sp)
    train = dg if mode == "dev" else sp[sp["role"] == "gallery_primary"]
    train = train.sort_values("image_id")
    cls = sorted(sp["class_id"].unique())
    cm = {c: i for i, c in enumerate(cls)}
    y = torch.tensor([cm[c] for c in train["class_id"]]).to(DEV)
    imgs = [fe.open_rgb(ROOT / p) for p in train["image_path"]]
    net = Net(len(cls)).to(DEV)
    print("device", DEV, flush=True)
    bb = [p for p in net.m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": 1e-4}, {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    log = []
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(imgs), generator=g).tolist()
        t0, tot = time.time(), 0.0
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            x = torch.stack([AUG(imgs[j]) for j in idx]).to(DEV)
            loss = F.cross_entropy(net.head.logits(net.feat(x), y[idx], m=0.3), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        rec = dict(epoch=ep, loss=tot / len(imgs), sec=round(time.time() - t0))
        if mode == "dev" and ep in snap:
            qi = [fe.open_rgb(ROOT / p) for p in dq["image_path"]]
            gi = [fe.open_rgb(ROOT / p) for p in dg["image_path"]]
            s = D.quick_scores(embed_all(net, qi), embed_all(net, gi), dq["class_id"].to_numpy(), dg["class_id"].to_numpy(),
                               dg["image_id"].to_numpy(dtype=str))
            rec.update({k: float(v) for k, v in s.items()})
        log.append(rec)
        print(rec, flush=True)
        pd.DataFrame(log).to_csv(f"{out_prefix}_log.csv", index=False)
    if mode == "final":
        inv = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
        allimgs = [fe.open_rgb(ROOT / p) for p in inv["image_path"]]
        E_ = embed_all(net, allimgs)
        np.savez(f"{out_prefix}_embeddings.npz", image_id=inv["image_id"].to_numpy(dtype=str), emb=E_)
        print("saved embeddings", E_.shape)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3], seed=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
