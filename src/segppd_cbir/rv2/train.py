"""Config-driven metric-learning runner (research_v2). One JSON config -> one run directory with config, curves, best checkpoint, embeddings of all
images, evaluation summary and per-query CSVs.   python -m segppd_cbir.rv2.train <config.json>

Selection rule (registered): model selection and early stopping use dev class-macro Hit@1; improvement needs > TIE_EPS; inside the tie band the higher dev mAP wins.
No test set is read for any decision; test sets are only evaluated AFTER the run, labelled exploratory when already consumed."""
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import ext_dev as D
from .. import feature_extractor as fe
from ..train_embed import DS, ROOT, DinoEmb, PKSampler, embed_tensors, load_tensors, make_aug
from . import common as C

DEV = "cuda"


# ------------------------------------------------------------------ losses (own hyper-parameters per loss)
def arcface(head, e, y, s=30.0, m=0.3):
    cos = e @ F.normalize(head.W, dim=1).T
    th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)); oh = F.one_hot(y, cos.shape[1]).bool()
    return F.cross_entropy(s * torch.where(oh, torch.cos(th + m), cos), y)


def proxy_anchor(head, e, y, alpha=32.0, delta=0.1):
    P = F.normalize(head.W, dim=1); cos = e @ P.T; oh = F.one_hot(y, P.shape[0]).float()
    pos = torch.exp(-alpha * (cos - delta)) * oh; neg = torch.exp(alpha * (cos + delta)) * (1 - oh); has = oh.sum(0) > 0
    return torch.log(1 + pos.sum(0))[has].sum() / has.sum() + torch.log(1 + neg.sum(0)).sum() / P.shape[0]


def supcon(head, e, y, tau=0.1):
    sim = e @ e.T / tau; eye = torch.eye(len(y), dtype=torch.bool, device=e.device); sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye; logp = sim - torch.logsumexp(sim, 1, keepdim=True); cnt = pos.sum(1); ok = cnt > 0
    return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()


LOSSES = {"arcface": arcface, "proxyanchor": proxy_anchor, "supcon": supcon}


def make_groups(net, cfg):
    g = []
    blocks = list(net.m.blocks[-cfg["blocks"]:])
    for i, b in enumerate(reversed(blocks)):
        g.append({"params": list(b.parameters()), "lr": cfg["lr_bb"] * cfg["layer_decay"] ** i})
    g.append({"params": list(net.m.norm.parameters()), "lr": cfg["lr_bb"]})
    g.append({"params": list(net.head.net.parameters()), "lr": cfg["lr_head"]})
    g.append({"params": [net.head.W], "lr": cfg["lr_head"] * cfg.get("proxy_lr_mult", 1.0)})
    return g


def dev_scores(Eq, Eg, dq, ga):
    gid = ga.image_id.to_numpy(dtype=str)
    pq, _ = C.per_query(Eq, Eg, dq.class_id.to_numpy(), ga.class_id.to_numpy(), gid, dq.image_id.to_numpy(dtype=str)); s = C.summarize(pq)
    return dict(hit1=s["macro_hit1"], mAP=s["macro_mAP"], P5=s["macro_p5"], P10=s["macro_p10"], micro_hit1=s["micro_hit1"])


def leakage_checks(S):
    ids = {k: set(v.image_id) for k, v in S.items()}; grp = {k: set(v.group_id) for k, v in S.items()}
    for a, b in (("train", "dev_q"), ("train", "gallery_B"), ("train", "testB_q"), ("train", "testC_q"), ("dev_q", "gallery_B"), ("testB_q", "gallery_B"), ("dev_q", "testB_q")):
        assert not (ids[a] & ids[b]), f"image overlap {a}/{b}"
        assert not (grp[a] & grp[b]), f"duplicate-group overlap {a}/{b}"
    assert not (ids["testC_q"] & ids["gallery_C"])
    return True


def run(cfg_path):
    cfg = json.load(open(cfg_path)); out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True); name = cfg["name"]
    for _ in range(90):  # wait up to 7.5 min for the GPU to be free (another short job may be running)
        gs = C.gpu_status()
        if gs["free_mb"] > 4300: break
        time.sleep(5)
    assert gs["free_mb"] > 4300, f"not enough free GPU memory: {gs}"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); S = C.segppd_sets(); leakage_checks(S)
    tr, dq, ga = S["train"], S["dev_q"], S["gallery_B"]
    classes = sorted(tr.class_id.unique()); cm = {c: i for i, c in enumerate(classes)}; ytr = np.array([cm[c] for c in tr.class_id])
    net = DinoEmb(len(classes), cfg["d_emb"], "partial", blocks=cfg["blocks"], variant="base").to(DEV)
    imgs = [fe.open_rgb(ROOT / p) for p in tr.image_path]
    from .augment import make_train_aug
    loader = DataLoader(DS(imgs, ytr, make_train_aug(cfg.get("aug", "base"), net.size)), batch_sampler=PKSampler(ytr, cfg["P"], cfg["K"], cfg["steps"], cfg["seed"]), num_workers=6, persistent_workers=True)
    Xg, Xq = load_tensors(ga.image_path, net.eval_size), load_tensors(dq.image_path, net.eval_size)
    opt = torch.optim.AdamW(make_groups(net, cfg), weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    lp = {k: v for k, v in cfg["loss"].items() if k != "type"}; lossf = LOSSES[cfg["loss"]["type"]]
    n_train = sum(p.numel() for g in opt.param_groups for p in g["params"] if p.requires_grad)
    log, best, max_hit, best_state, since = [], None, -1.0, None, 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        net.train(); tot = 0.0
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV); e = net.head.embed(net.feat(x)); loss = lossf(net.head, e, y, **lp)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step(); s = dev_scores(embed_tensors(net, Xq), embed_tensors(net, Xg), dq, ga); log.append(dict(epoch=ep, loss=tot / len(loader), **s))
        improved = best is None or s["hit1"] > best["hit1"] + C.TIE_EPS or (s["hit1"] >= max_hit - C.TIE_EPS and s["mAP"] > best["mAP"] + 1e-4)
        max_hit = max(max_hit, s["hit1"])
        if improved:
            best, since = dict(epoch=ep, **s), 0; best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else:
            since += 1
        print(f"[{name}] ep{ep} loss {log[-1]['loss']:.3f} dev hit1 {s['hit1']:.3f} mAP {s['mAP']:.3f} | best ep{best['epoch']} {best['hit1']:.3f}/{best['mAP']:.3f}", flush=True)
        if since >= cfg["patience"]:
            break
    train_sec = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 1e9
    net.load_state_dict(best_state); torch.save(dict(state=best_state, cfg=cfg, n_cls=len(classes), best=best), out / "best.pt")
    inv = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    t1 = time.time(); Z = embed_tensors(net, load_tensors(inv.image_path, net.eval_size)); embed_sec = time.time() - t1
    np.savez(out / "embeddings.npz", image_id=inv.image_id.to_numpy(dtype=str), emb=Z)
    res, pqs = C.evaluate_npz(out / "embeddings.npz", S)
    for k, v in pqs.items(): v.to_csv(out / f"per_query_{k}.csv", index=False)
    sha = hashlib.sha256(open(out / "best.pt", "rb").read()).hexdigest()
    json.dump(dict(name=name, cfg=cfg, best_epoch=best["epoch"], best_dev=best, epochs_run=len(log), train_seconds=train_sec, embed_all_seconds=embed_sec, embed_ms_per_image=1e3 * embed_sec / len(inv),
                   peak_vram_gb=peak, n_trainable_params=n_train, ckpt_sha256=sha, gpu_at_start=gs, torch=torch.__version__, history=log, results=res), open(out / "result.json", "w"), indent=1, default=float)
    print(f"[{name}] DONE best ep{best['epoch']} dev hit1 {best['hit1']:.3f} mAP {best['mAP']:.3f} | {train_sec:.0f}s peak {peak:.2f} GB", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
