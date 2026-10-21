"""Experiment B mixed-source trainer (X1/X2/X3). Unified labels from cross_protocol (label_alignment.csv). Same backbone, loss, optimiser, budget as Track B M3.
python -m segppd_cbir.rv2.cross_train <variant X1|X2|X3> <seed>"""
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
from PIL import Image
from torch.utils.data import DataLoader

from .. import feature_extractor as fe
from ..train_embed import DS, ROOT, DinoEmb, PKSampler, embed_tensors, load_tensors
from . import common as C
from .augment import make_train_aug
from .plantseg_eval import retrieval
from .plantseg_train import CACHE, tens
from .train import LOSSES, make_groups

DEV = "cuda"; V3 = Path("artifacts/research_v3/cross")


class DomainPK:
    """Domain-balanced class batches: P/2 classes from the SegPPD pool + P/2 from the PlantSeg pool; images drawn from that source only."""
    def __init__(self, y, dom, P, K, steps, seed):
        self.by = {d: {c: np.where((y == c) & (dom == d))[0] for c in np.unique(y[dom == d])} for d in (0, 1)}; self.P, self.K, self.steps, self.rng = P, K, steps, np.random.default_rng(seed)

    def __len__(self): return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            batch, used = [], set()
            for d in (0, 1):
                cl = [c for c in self.by[d] if c not in used]
                for c in self.rng.choice(cl, self.P // 2, replace=False):
                    used.add(c); ix = self.by[d][c]; batch += list(self.rng.choice(ix, self.K, replace=len(ix) < self.K))
            yield batch


def run(variant, seed):
    name = f"cross_{variant}_s{seed}"; out = V3 / "runs" / name; out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open("artifacts/research_v2/trackB/configs/B_M3_metric_proxyanchor_s0.json")); cfg.update(seed=seed, name=name, out_dir=str(out), variant=variant, aug=("color" if variant == "X3" else "base"), sampler=("domain" if variant == "X2" else "class")); json.dump(cfg, open(out / "config.json", "w"), indent=1)
    for _ in range(60):
        gs = C.gpu_status()
        if gs["free_mb"] > 4300: break
        time.sleep(5)
    assert gs["free_mb"] > 4300, gs
    torch.manual_seed(seed); np.random.seed(seed); S = C.segppd_sets(); P = pd.read_csv("artifacts/research_v2/trackB/protocol_P.csv"); M = pd.read_csv(V3 / "label_alignment.csv")
    uni = {(r.source, r.source_label): r.unified for r in M.itertuples()}; tr_s = S["train"]; tr_p = P[P.role == "train"].reset_index(drop=True); dv_p = P[P.role == "dev"].reset_index(drop=True)
    assert not (set(tr_s.image_id) & (set(S["dev_q"].image_id) | set(S["gallery_B"].image_id) | set(S["testB_q"].image_id))), "SegPPD leakage"
    assert not (set(tr_p.Name) & (set(dv_p.Name) | set(P[P.role == "test"].Name))) and not (set(tr_p.group) & (set(dv_p.group) | set(P[P.role == "test"].group))), "PlantSeg leakage"
    y_s = np.array([uni[("seg", c)] for c in tr_s.class_id]); y_p = np.array([uni[("ps", l)] for l in tr_p.label]); y = np.concatenate([y_s, y_p]); dom = np.concatenate([np.zeros(len(y_s), int), np.ones(len(y_p), int)])
    classes = sorted(set(y)); cm = {c: i for i, c in enumerate(classes)}; yt = np.array([cm[c] for c in y]); n_cls = len(classes)
    net = DinoEmb(n_cls, cfg["d_emb"], "partial", blocks=cfg["blocks"], variant="base").to(DEV)
    imgs = [fe.open_rgb(ROOT / p) for p in tr_s.image_path] + [Image.open(CACHE / (Path(n).stem + ".png")).convert("RGB") for n in tr_p.Name]
    sampler = DomainPK(yt, dom, cfg["P"], cfg["K"], cfg["steps"], seed) if variant == "X2" else PKSampler(yt, cfg["P"], cfg["K"], cfg["steps"], seed)
    loader = DataLoader(DS(imgs, yt, make_train_aug(cfg["aug"], 224)), batch_sampler=sampler, num_workers=6, persistent_workers=True)
    # dev: SegPPD dev_query_B vs gallery_B (class ids = unified), PlantSeg dev vs <=10 train images/class
    ga, dq = S["gallery_B"], S["dev_q"]; Xg, Xq = load_tensors(ga.image_path, net.eval_size), load_tensors(dq.image_path, net.eval_size)
    gl_s = np.array([uni[("seg", c)] for c in ga.class_id]); ql_s = np.array([uni[("seg", c)] for c in dq.class_id])
    rng = np.random.default_rng(0); gsub = pd.concat([d.sample(min(len(d), 10), random_state=int(rng.integers(1 << 30))) for _, d in tr_p.groupby("label")]); Xpg, Xpq = tens(gsub), tens(dv_p)
    gl_p, ql_p = np.array([uni[("ps", l)] for l in gsub.label]), np.array([uni[("ps", l)] for l in dv_p.label]); gid_p = gsub.Name.to_numpy(dtype=str); gid_s = ga.image_id.to_numpy(dtype=str)
    opt = torch.optim.AdamW(make_groups(net, dict(cfg, proxy_lr_mult=1.0)), weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    lf = LOSSES["proxyanchor"]; lp = {k: v for k, v in cfg["loss"].items() if k != "type"}; eps = 1 / 300; log, best, mx, bs_, since = [], None, -1.0, None, 0; torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        net.train(); tot = 0.0
        for x, yb in loader:
            x, yb = x.to(DEV), yb.to(DEV); loss = lf(net.head, net.head.embed(net.feat(x)), yb, **lp); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step(); pqs, _ = C.per_query(embed_tensors(net, Xq), embed_tensors(net, Xg), ql_s, gl_s, gid_s, dq.image_id.to_numpy(dtype=str)); ss = C.summarize(pqs)
        ex = retrieval(embed_tensors(net, Xpq), embed_tensors(net, Xpg), ql_p, gl_p, gid_p); h_s, h_p = ss["macro_hit1"], float((ex["pred"] == ql_p).mean())
        s = dict(hit1=(h_s + h_p) / 2, mAP=(ss["macro_mAP"] + float(np.mean(ex["ap"]))) / 2, seg_dev_hit1=h_s, ps_dev_acc=h_p); log.append(dict(epoch=ep, loss=tot / len(loader), **s))
        if best is None or s["hit1"] > best["hit1"] + eps or (s["hit1"] >= mx - eps and s["mAP"] > best["mAP"] + 1e-4):
            best, since = dict(epoch=ep, **s), 0; bs_ = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else: since += 1
        mx = max(mx, s["hit1"]); print(f"[{name}] ep{ep} loss {log[-1]['loss']:.3f} dev seg {h_s:.3f} ps {h_p:.3f} | best ep{best['epoch']} {best['hit1']:.3f}", flush=True)
        if since >= cfg["patience"]: break
    train_sec = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 1e9; net.load_state_dict(bs_); torch.save(dict(state=bs_, cfg=cfg, n_cls=n_cls, best=best), out / "best.pt")
    del loader, imgs; import gc; gc.collect()
    from .cross_embed import embed_all
    embed_all(net, out / "embeddings.npz")
    json.dump(dict(name=name, cfg=cfg, best=best, epochs_run=len(log), train_seconds=train_sec, peak_vram_gb=peak, n_classes=n_cls, n_train=len(y), ckpt_sha256=hashlib.sha256(open(out / "best.pt", "rb").read()).hexdigest(), history=log), open(out / "result.json", "w"), indent=1, default=float)
    print(f"[{name}] DONE best ep{best['epoch']} dev {best['hit1']:.3f} {train_sec:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
