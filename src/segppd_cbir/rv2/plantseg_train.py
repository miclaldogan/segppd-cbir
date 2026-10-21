"""Track B training on PlantSeg protocol P (target-supervised). method 'metric' = metric-learned embedding (loss from Track A), method 'ce' = supervised
fine-tuning with a linear head + cross-entropy. Same backbone blocks, optimiser, sampler, augmentation, and selection rule (dev accuracy/Hit@1, tie = one dev image).
python -m segppd_cbir.rv2.plantseg_train <config.json>"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ..ext_features import to_tensor
from ..train_embed import DS, DinoEmb, PKSampler, embed_tensors
from . import common as C
from .augment import make_train_aug
from .plantseg_eval import load_protocol, retrieval
from .train import LOSSES, make_groups

DEV = "cuda"; CACHE = Path("data/external/plantseg_256")


def load_imgs(df):
    return [Image.open(CACHE / (Path(n).stem + ".png")).convert("RGB") for n in df.Name]


def tens(df):
    return torch.stack([to_tensor(Image.open(CACHE / (Path(n).stem + ".png")).convert("RGB"), 224) for n in df.Name]).half()


class CEHead(nn.Module):
    def __init__(self, d, n):
        super().__init__(); self.drop = nn.Dropout(0.2); self.fc = nn.Linear(d, n)

    def forward(self, f):
        return self.fc(self.drop(f))


@torch.inference_mode()
def logits_of(net, ce, X, bs=64):
    net.eval(); ce.eval(); out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(DEV).float(); out.append(((ce(net.feat(x)) + ce(net.feat(torch.flip(x, dims=[3])))) / 2).float().cpu())
    return torch.cat(out).numpy()


def run(cfg_path):
    cfg = json.load(open(cfg_path)); out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True); name = cfg["name"]
    import time as _t
    for _ in range(60):
        gs = C.gpu_status()
        if gs["free_mb"] > 4300: break
        _t.sleep(5)
    assert gs["free_mb"] > 4300, gs
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); P = load_protocol(); tr, dv, te = P["train"], P["dev"], P["test"]
    ytr = tr.label.to_numpy(); n_cls = int(ytr.max()) + 1
    assert not (set(tr.Name) & set(dv.Name)) and not (set(tr.Name) & set(te.Name)) and not (set(dv.Name) & set(te.Name))
    assert not (set(tr.group) & set(dv.group)) and not (set(tr.group) & set(te.group)) and not (set(dv.group) & set(te.group)), "duplicate-group leakage"
    net = DinoEmb(n_cls, cfg.get("d_emb", 256), "partial", blocks=cfg["blocks"], variant="base").to(DEV)
    ce = CEHead(768, n_cls).to(DEV) if cfg["method"] == "ce" else None
    imgs = load_imgs(tr); loader = DataLoader(DS(imgs, ytr, make_train_aug(cfg.get("aug", "base"), 224)), batch_sampler=PKSampler(ytr, cfg["P"], cfg["K"], cfg["steps"], cfg["seed"]), num_workers=6, persistent_workers=True)
    # dev gallery for epoch selection: <=10 train images/class (fixed sample); full train gallery is used for the final evaluation
    rng = np.random.default_rng(0); gsub = pd.concat([d.sample(min(len(d), 10), random_state=int(rng.integers(1 << 30))) for _, d in tr.groupby("label")])
    Xg, Xq = tens(gsub), tens(dv); yg, yq = gsub.label.to_numpy(), dv.label.to_numpy(); gid = gsub.Name.to_numpy(dtype=str)
    groups = make_groups(net, dict(cfg, proxy_lr_mult=cfg.get("proxy_lr_mult", 1.0)))
    if ce is not None: groups.append({"params": list(ce.parameters()), "lr": cfg["lr_head"]})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    lp = {k: v for k, v in cfg.get("loss", {}).items() if k != "type"}; lossf = LOSSES[cfg["loss"]["type"]] if cfg["method"] == "metric" else None
    eps = 1 / len(dv); log, best, max_acc, best_state, since = [], None, -1.0, None, 0; torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        net.train(); tot = 0.0
        if ce is not None: ce.train()
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV); f = net.feat(x)
            loss = F.cross_entropy(ce(f), y) if ce is not None else lossf(net.head, net.head.embed(f), y, **lp)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if ce is not None:
            lg = logits_of(net, ce, Xq); acc = float((lg.argmax(1) == yq).mean()); sec_ = -float(F.cross_entropy(torch.from_numpy(lg), torch.from_numpy(yq)))
            s = dict(acc=acc, secondary=sec_)
        else:
            Eq, Eg = embed_tensors(net, Xq), embed_tensors(net, Xg); ex = retrieval(Eq, Eg, yq, yg, gid); s = dict(acc=float((ex["pred"] == yq).mean()), secondary=float(np.mean(ex["ap"])))
        log.append(dict(epoch=ep, loss=tot / len(loader), **s))
        improved = best is None or s["acc"] > best["acc"] + eps or (s["acc"] >= max_acc - eps and s["secondary"] > best["secondary"] + 1e-5); max_acc = max(max_acc, s["acc"])
        if improved:
            best, since = dict(epoch=ep, **s), 0; best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()}); best_ce = copy.deepcopy(ce.state_dict()) if ce is not None else None
        else:
            since += 1
        print(f"[{name}] ep{ep} loss {log[-1]['loss']:.3f} dev acc {s['acc']:.3f} sec {s['secondary']:.3f} | best ep{best['epoch']} {best['acc']:.3f}", flush=True)
        if since >= cfg["patience"]: break
    train_sec = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 1e9; net.load_state_dict(best_state)
    if ce is not None: ce.load_state_dict(best_ce)
    torch.save(dict(state=best_state, ce=best_ce, cfg=cfg, best=best), out / "best.pt")
    del loader, imgs; import gc; gc.collect()
    embed_sec, n_img = finalize(net, ce, P, out); 
    json.dump(dict(name=name, cfg=cfg, best=best, epochs_run=len(log), train_seconds=train_sec, embed_ms_per_image=1e3 * embed_sec / n_img, peak_vram_gb=peak, ckpt_sha256=hashlib.sha256(open(out / "best.pt", "rb").read()).hexdigest(), gpu_at_start=gs, history=log), open(out / "result.json", "w"), indent=1, default=float)
    print(f"[{name}] DONE best ep{best['epoch']} dev acc {best['acc']:.3f} {train_sec:.0f}s peak {peak:.2f} GB", flush=True)


def finalize(net, ce, P, out, chunk=512):
    """Chunked embedding of every protocol image (memory-light). metric: L2-normalised flip-averaged embeddings; ce: logits and pooled features."""
    t1 = time.time(); n = 0; embs, lgs, fts = {}, {}, {}
    for r, df in P.items():
        E, L, Fe = [], [], []
        for i in range(0, len(df), chunk):
            X = tens(df.iloc[i:i + chunk]); n += len(X)
            if ce is not None: L.append(logits_of(net, ce, X)); Fe.append(embed_tensors_feat(net, X))
            else: E.append(embed_tensors(net, X))
        if ce is not None: lgs[f"logits_{r}"] = np.concatenate(L); fts[r] = np.concatenate(Fe)
        else: embs[r] = np.concatenate(E)
    if ce is not None: np.savez(out / "outputs.npz", **lgs); np.savez(out / "features.npz", **fts)
    else: np.savez(out / "embeddings.npz", **embs)
    return time.time() - t1, n


@torch.inference_mode()
def embed_tensors_feat(net, X, bs=64):
    net.eval(); out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(DEV).float(); out.append(((net.feat(x) + net.feat(torch.flip(x, dims=[3]))) / 2).float().cpu())
    return torch.cat(out).numpy()


def finalize_only(cfg_path):
    """Resume after a crash: reload best.pt and write embeddings/result without retraining."""
    cfg = json.load(open(cfg_path)); out = Path(cfg["out_dir"]); P = load_protocol(); ck = torch.load(out / "best.pt", map_location="cpu"); n_cls = int(P["train"].label.max()) + 1
    net = DinoEmb(n_cls, cfg.get("d_emb", 256), "partial", blocks=cfg["blocks"], variant="base").to(DEV); net.load_state_dict(ck["state"]); net.eval()
    ce = None
    if cfg["method"] == "ce": ce = CEHead(768, n_cls).to(DEV); ce.load_state_dict(ck["ce"])
    sec, n = finalize(net, ce, P, out)
    json.dump(dict(name=cfg["name"], cfg=cfg, best=ck["best"], note="finalised from best.pt after the process died during the embedding stage; training log is in queue.log (epochs 1-30)", embed_ms_per_image=1e3 * sec / n,
                   ckpt_sha256=hashlib.sha256(open(out / "best.pt", "rb").read()).hexdigest()), open(out / "result.json", "w"), indent=1, default=float); print("finalised", cfg["name"], n, "images")


if __name__ == "__main__":
    finalize_only(sys.argv[2]) if sys.argv[1] == "--finalize" else run(sys.argv[1])
