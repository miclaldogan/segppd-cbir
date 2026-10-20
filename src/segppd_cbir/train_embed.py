"""Protocol-B embedding training (TRAINED extension). Training images = train_B only. Early stopping on dev (dev_query_B vs gallery_B) only.
Never reads test_query_B labels or scores; it only embeds them after training so that the locked evaluation can run later."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from . import dino
from . import ext_dev as D
from . import feature_extractor as fe
from .ext_features import to_tensor
from .ext_multi import _arcface, _proxyanchor, _supcon
from .ext_trained import Head

ROOT = Path("data/extracted/SegPPD-101")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_aug(size):
    return T.Compose([T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(),
                      T.RandomVerticalFlip(0.2), T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, imgs, y, aug):
        self.imgs, self.y, self.aug = imgs, y, aug

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.aug(self.imgs[i]), self.y[i]


class PKSampler:
    """Class-balanced batches: P classes x K images (sampling with replacement inside small classes)."""
    def __init__(self, labels, P=16, K=4, steps=20, seed=0):
        self.by = {c: np.where(labels == c)[0] for c in np.unique(labels)}
        self.P, self.K, self.steps = P, K, steps
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.steps

    def __iter__(self):
        cls = list(self.by)
        for _ in range(self.steps):
            batch = []
            for c in self.rng.choice(cls, self.P, replace=False):
                ix = self.by[c]
                batch += list(self.rng.choice(ix, self.K, replace=len(ix) < self.K))
            yield batch


class ResNetEmb(nn.Module):
    size = 224
    eval_size = 256

    def __init__(self, n_cls, d_emb, mode, **_):
        super().__init__()
        self.m, _ = fe.load_model("cpu")
        for p in self.m.parameters():
            p.requires_grad_(False)
        if mode != "head":
            for n in ("layer3", "layer4"):
                for p in getattr(self.m, n).parameters():
                    p.requires_grad_(True)
        self.head = Head(2048, d_emb, n_cls, drop=0.2)

    def groups(self, lr_bb, lr_head):
        return [{"params": [p for p in self.m.parameters() if p.requires_grad], "lr": lr_bb}, {"params": self.head.parameters(), "lr": lr_head}]

    def train(self, mode=True):
        super().train(mode)
        for mod in self.m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        return self

    def feat(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        return m.layer4(m.layer3(m.layer2(m.layer1(x)))).mean((2, 3))


class DinoEmb(nn.Module):
    size = 224
    eval_size = 224

    def __init__(self, n_cls, d_emb, mode, blocks=4, variant="small", **_):
        super().__init__()
        self.m = dino.load("cpu", name="vit_base_patch14_dinov2.lvd142m" if variant == "base" else dino.NAME)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.blocks, self.mode = blocks, mode
        if mode != "head":
            for b in self.m.blocks[-blocks:]:
                for p in b.parameters():
                    p.requires_grad_(True)
            for p in self.m.norm.parameters():
                p.requires_grad_(True)
        self.head = Head(self.m.num_features, d_emb, n_cls, drop=0.2)

    def groups(self, lr_bb, lr_head, decay=0.8):
        g = []
        if self.mode != "head":
            for i, b in enumerate(reversed(list(self.m.blocks[-self.blocks:]))):
                g.append({"params": list(b.parameters()), "lr": lr_bb * decay ** i})
            g.append({"params": list(self.m.norm.parameters()), "lr": lr_bb})
        g.append({"params": self.head.parameters(), "lr": lr_head})
        return g

    def feat(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            f = self.m.forward_features(x)[:, 0]
        return f.float()


@torch.inference_mode()
def embed_tensors(net, X, bs=64):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(DEV, non_blocking=True).float()
        e = net.head.embed(net.feat(x)) + net.head.embed(net.feat(torch.flip(x, dims=[3])))
        out.append(F.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy()


def load_tensors(paths, size):
    from PIL import Image  # noqa: F401
    return torch.stack([_norm(to_tensor(fe.open_rgb(ROOT / p), size)) for p in paths]).half()


def _norm(x):  # to_tensor already applies ImageNet mean/std
    return x


def run(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    b = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str})
    tr = b[b.role_B == "train_B"].sort_values("image_id")
    ga = b[b.role_B == "gallery_B"].sort_values("image_id")
    dq = b[b.role_B == "dev_query_B"].sort_values("image_id")
    assert set(tr.image_id).isdisjoint(set(ga.image_id) | set(dq.image_id)), "train overlaps gallery/dev"
    classes = sorted(b[b.role_B.isin(["train_B"])].class_id.unique())
    cm = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cm[c] for c in tr.class_id])
    Net = ResNetEmb if a.backbone == "resnet50" else DinoEmb
    net = Net(len(classes), a.d_emb, a.mode, blocks=a.blocks, variant=a.variant).to(DEV)
    imgs = [fe.open_rgb(ROOT / p) for p in tr.image_path]
    loader = DataLoader(DS(imgs, ytr, make_aug(net.size)), batch_sampler=PKSampler(ytr, 16, 4, a.steps, a.seed), num_workers=6,
                        persistent_workers=True)
    gl = ga.class_id.to_numpy()
    ql = dq.class_id.to_numpy()
    gid = ga.image_id.to_numpy(dtype=str)
    Xg = load_tensors(ga.image_path, net.eval_size)
    Xq = load_tensors(dq.image_path, net.eval_size)
    opt = torch.optim.AdamW(net.groups(a.lr_bb, a.lr_head), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    n_train = sum(p.numel() for g in opt.param_groups for p in g["params"])
    log, best, best_state, bad = [], -1, None, 0
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            e = net.head.embed(net.feat(x))
            loss = _arcface(net.head, e, y) if a.loss == "arcface" else _proxyanchor(net.head, e, y) if a.loss == "proxyanchor" else _supcon(e, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        s = D.quick_scores(embed_tensors(net, Xq), embed_tensors(net, Xg), ql, gl, gid)
        s = {k: float(v) for k, v in s.items()}
        log.append(dict(epoch=ep, loss=tot / len(loader), **s))
        if s["mAP"] > best + 1e-4:
            best, bad, best_state = s["mAP"], 0, copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else:
            bad += 1
        if bad >= a.patience:
            break
    train_sec = time.time() - t_train
    net.load_state_dict(best_state)
    if a.save_ckpt:  # for external-set inference; weights stay local (gitignored)
        Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=best_state, args=vars(a), n_cls=len(classes)), f"artifacts/checkpoints/{a.tag}.pt")
    best_ep = int(np.argmax([r["mAP"] for r in log])) + 1
    inv = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    t0 = time.time()
    Z = embed_tensors(net, load_tensors(inv.image_path, net.eval_size))
    embed_sec = time.time() - t0
    out = Path("artifacts/extensions/B")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.tag}_embeddings.npz", image_id=inv.image_id.to_numpy(dtype=str), emb=Z)
    (out / f"{a.tag}_log.json").write_text(json.dumps(dict(args=vars(a), best_epoch=best_ep, best_dev=log[best_ep - 1], epochs_run=len(log),
                                                           train_seconds=train_sec, embed_all_seconds=embed_sec, n_trainable_params=n_train,
                                                           peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9, history=log), indent=1))
    print(a.tag, "best_ep", best_ep, {k: round(v, 3) for k, v in log[best_ep - 1].items() if k != "epoch"}, f"{train_sec:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["resnet50", "dinov2s"], required=True)
    p.add_argument("--mode", choices=["head", "partial"], default="partial")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--variant", choices=["small", "base"], default="small")
    p.add_argument("--loss", choices=["arcface", "proxyanchor", "supcon"], default="arcface")
    p.add_argument("--d_emb", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr_bb", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--save_ckpt", action="store_true")
    run(p.parse_args())
