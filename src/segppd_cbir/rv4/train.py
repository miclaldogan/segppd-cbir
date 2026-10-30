"""research_v4 trainer: research_v2 recipe (Proxy-Anchor, dev-Hit@1 selection) over any rv4.model backbone, with optional relational
knowledge distillation from a teacher similarity structure and optional TTA views at embedding time.  python -m segppd_cbir.rv4.train <config.json>
Teacher: npz with image_id + emb (unit vectors); ONLY the train_B rows are read. B/C tests are evaluated after training, exploratory."""
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
from torch.utils.data import DataLoader

from .. import feature_extractor as fe
from ..train_embed import ROOT, PKSampler, load_tensors
from ..rv2 import common as C
from ..rv2.augment import make_train_aug
from ..rv2.train import LOSSES, dev_scores, leakage_checks
from .model import DEV, Emb, embed_tensors


class DSI(torch.utils.data.Dataset):
    def __init__(self, imgs, y, aug): self.imgs, self.y, self.aug = imgs, y, aug
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.aug(self.imgs[i]), self.y[i], i


def kd_loss(e, T, idx):
    """Relational KD: match the student's batch cosine-similarity matrix to the teacher's (off-diagonal pairs)."""
    St = T[idx] @ T[idx].T; Ss = e @ e.T; m = ~torch.eye(len(idx), dtype=torch.bool, device=e.device)
    return ((Ss - St) ** 2)[m].mean()


def run(cfg_path):
    cfg = json.load(open(cfg_path)); out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True); name = cfg["name"]
    if (out / "result.json").exists(): print(f"[{name}] exists, skip"); return
    if (out / "best.pt").exists(): return finalize(cfg, out, name)  # training finished earlier but the process died during embedding: finish from the checkpoint
    for _ in range(90):
        gs = C.gpu_status()
        if gs["free_mb"] > 4300: break
        time.sleep(5)
    assert gs["free_mb"] > 4300, f"not enough free GPU memory: {gs}"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); S = C.segppd_sets(); leakage_checks(S)
    tr, dq, ga = S["train"], S["dev_q"], S["gallery_B"]
    classes = sorted(tr.class_id.unique()); cm = {c: i for i, c in enumerate(classes)}; ytr = np.array([cm[c] for c in tr.class_id])
    net = Emb(len(classes), cfg["d_emb"], cfg["backbone"], blocks=cfg.get("blocks", 4), truncate=cfg.get("truncate"), img_size=cfg.get("img_size", 224), full=cfg.get("full", False)).to(DEV)
    imgs = [fe.open_rgb(ROOT / p) for p in tr.image_path]
    loader = DataLoader(DSI(imgs, ytr, make_train_aug(cfg.get("aug", "base"), net.size)), batch_sampler=PKSampler(ytr, cfg["P"], cfg["K"], cfg["steps"], cfg["seed"]), num_workers=6, persistent_workers=True)
    Xg, Xq = load_tensors(ga.image_path, net.eval_size), load_tensors(dq.image_path, net.eval_size)
    T = None
    if cfg.get("kd"):
        z = np.load(cfg["kd"]["teacher"]); pos = {i: k for k, i in enumerate(z["image_id"])}; T = torch.from_numpy(z["emb"][[pos[i] for i in tr.image_id]]).float().to(DEV)
        T = torch.nn.functional.normalize(T, dim=1); assert T.shape[0] == len(tr)
    opt = torch.optim.AdamW(net.groups(cfg["lr_bb"], cfg["lr_head"], cfg["layer_decay"], cfg.get("proxy_lr_mult", 1.0)), weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    lp = {k: v for k, v in cfg["loss"].items() if k != "type"}; lossf = LOSSES[cfg["loss"]["type"]]; views = tuple(cfg.get("views", ()))
    n_train = sum(p.numel() for g in opt.param_groups for p in g["params"] if p.requires_grad)
    log, best, max_hit, best_state, since = [], None, -1.0, None, 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        net.train(); tot = totk = 0.0
        for x, y, idx in loader:
            x, y, idx = x.to(DEV), y.to(DEV), idx.to(DEV); e = net.head.embed(net.feat(x)); loss = lossf(net.head, e, y, **lp)
            if T is not None: k = kd_loss(e, T, idx); loss = loss + cfg["kd"]["lam"] * k; totk += k.item()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step(); s = dev_scores(embed_tensors(net, Xq, views=views), embed_tensors(net, Xg, views=views), dq, ga); log.append(dict(epoch=ep, loss=tot / len(loader), kd=totk / len(loader), **s))
        improved = best is None or s["hit1"] > best["hit1"] + C.TIE_EPS or (s["hit1"] >= max_hit - C.TIE_EPS and s["mAP"] > best["mAP"] + 1e-4)
        max_hit = max(max_hit, s["hit1"])
        if improved: best, since = dict(epoch=ep, **s), 0; best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else: since += 1
        print(f"[{name}] ep{ep} loss {log[-1]['loss']:.3f} kd {log[-1]['kd']:.4f} dev hit1 {s['hit1']:.3f} mAP {s['mAP']:.3f} | best ep{best['epoch']} {best['hit1']:.3f}/{best['mAP']:.3f}", flush=True)
        if since >= cfg["patience"]: break
    train_sec = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 1e9
    net.load_state_dict(best_state); torch.save(dict(state=best_state, cfg=cfg, n_cls=len(classes), best=best, history=log, train_seconds=train_sec, peak_vram_gb=peak, n_trainable_params=n_train, gpu_at_start=gs), out / "best.pt")
    write_outputs(net, cfg, out, name, best, log, train_sec, peak, n_train, gs, S, views)


def embed_all(net, inv, views, chunk=256):
    """Embed the manifest in chunks (the one-shot load of 2,263 tensors at 336 px exhausted host RAM once)."""
    return np.concatenate([embed_tensors(net, load_tensors(inv.image_path.iloc[i:i + chunk], net.eval_size), views=views) for i in range(0, len(inv), chunk)])


def finalize(cfg, out, name):
    ck = torch.load(out / "best.pt", map_location="cpu"); S = C.segppd_sets(); classes = sorted(S["train"].class_id.unique())
    net = Emb(len(classes), cfg["d_emb"], cfg["backbone"], blocks=cfg.get("blocks", 4), truncate=cfg.get("truncate"), img_size=cfg.get("img_size", 224), full=cfg.get("full", False), pretrained=False).to(DEV)
    net.load_state_dict(ck["state"]); print(f"[{name}] finalizing from best.pt (best ep{ck['best']['epoch']})", flush=True)
    write_outputs(net, cfg, out, name, ck["best"], ck.get("history", []), ck.get("train_seconds", float("nan")), ck.get("peak_vram_gb", float("nan")), ck.get("n_trainable_params", -1), ck.get("gpu_at_start", {}), S, tuple(cfg.get("views", ())), finalized=True)


def write_outputs(net, cfg, out, name, best, log, train_sec, peak, n_train, gs, S, views, finalized=False):
    n_all = net.n_params()
    inv = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    t1 = time.time(); Z = embed_all(net, inv, views); embed_sec = time.time() - t1
    np.savez(out / "embeddings.npz", image_id=inv.image_id.to_numpy(dtype=str), emb=Z)
    res, pqs = C.evaluate_npz(out / "embeddings.npz", S)
    for k, v in pqs.items(): v.to_csv(out / f"per_query_{k}.csv", index=False)
    sha = hashlib.sha256(open(out / "best.pt", "rb").read()).hexdigest()
    json.dump(dict(name=name, cfg=cfg, best_epoch=best["epoch"], best_dev=best, epochs_run=len(log), train_seconds=train_sec, embed_all_seconds=embed_sec, embed_ms_per_image=1e3 * embed_sec / len(inv),
                   peak_vram_gb=peak, n_trainable_params=n_train, n_inference_params=n_all, ckpt_sha256=sha, gpu_at_start=gs, torch=torch.__version__, finalized_from_checkpoint=finalized, history=log, results=res), open(out / "result.json", "w"), indent=1, default=float)
    print(f"[{name}] DONE best ep{best['epoch']} dev hit1 {best['hit1']:.3f} mAP {best['mAP']:.3f} | {train_sec:.0f}s peak {peak:.2f} GB params {n_all/1e6:.1f}M{' (finalized)' if finalized else ''}", flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]: run(p)
