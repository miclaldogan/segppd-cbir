"""Track C2: global retrieval -> local patch-feature reranking. Deterministic, label-free and mask-free at inference.
Global stage: exact squared-L2 on the 256-D embedding (top-M). Local stage: symmetric Chamfer similarity between L2-normalised last-layer DINOv2 patch tokens
(16x16 = 256 tokens, no flip, tokens of the fine-tuned backbone):  S_loc(q,c) = 0.5*( mean_i max_j <q_i,c_j> + mean_j max_i <q_i,c_j> ).
Final score for the top-M candidates: alpha*cos_global + (1-alpha)*S_loc; ranks below M keep the global order.
Selection of (M, alpha) on the SegPPD dev set only. python -m segppd_cbir.rv2.local_rerank"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..ext_features import to_tensor
from ..index import rank
from ..train_embed import DinoEmb
from . import common as C

OUT = Path("artifacts/research_v2/trackC"); CACHE = Path("data/external/plantseg_256"); ROOT = Path("data/extracted/SegPPD-101"); DEV = "cuda"
MS, ALPHAS = (20, 50), (0.0, 0.25, 0.5, 0.75)


def load_net(path):
    ck = torch.load(path, map_location="cpu"); cfg = ck.get("cfg") or ck["args"]; a = ck.get("args", {})
    net = DinoEmb(ck["n_cls"], 256, "partial", blocks=4, variant="base").to(DEV); net.load_state_dict(ck["state"]); net.eval(); return net


@torch.inference_mode()
def tokens(net, imgs, bs=32):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack(imgs[i:i + bs]).to(DEV).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = net.m.forward_features(x)[:, 1:]
        out.append(torch.nn.functional.normalize(t.float(), dim=-1).half().cpu())
    return torch.cat(out)


def chamfer(q, cands):
    """q [P,D], cands [M,P,D] (fp16 on GPU) -> [M]"""
    s = torch.einsum("pd,mqd->mpq", q, cands).float()
    return 0.5 * (s.max(2).values.mean(1) + s.max(1).values.mean(1))


def local_scores(order_g, qtok, gtok, Mmax):
    """Chamfer similarity of every query to its top-Mmax global candidates (computed once; smaller M reuse the prefix). Returns ([nq,Mmax], median seconds/query)."""
    out = np.zeros((len(order_g), Mmax), np.float32); lat = []
    for i in range(len(order_g)):
        t0 = time.perf_counter(); out[i] = chamfer(qtok[i].to(DEV), gtok[torch.as_tensor(order_g[i, :Mmax])].to(DEV)).cpu().numpy(); torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
    return out, float(np.median(lat))


def apply_rerank(order_g, cos_g, loc, M, alpha):
    new = order_g.copy()
    for i in range(len(order_g)):
        cand = order_g[i, :M]; sc = alpha * cos_g[i, :M] + (1 - alpha) * loc[i, :M]; new[i, :M] = cand[np.argsort(-sc, kind="stable")]
    return new


def rerank_order(order_g, cos_g, qtok, gtok, M, alpha):
    """order_g [nq,Nall] global order; cos_g [nq,Nall] cosine of the same candidates in that order. Returns new order."""
    new = order_g.copy(); lat = []
    for i in range(len(order_g)):
        t0 = time.perf_counter(); cand = order_g[i, :M]; loc = chamfer(qtok[i].to(DEV), gtok[torch.as_tensor(cand)].to(DEV)).cpu().numpy(); torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
        sc = alpha * cos_g[i, :M] + (1 - alpha) * loc; new[i, :M] = cand[np.argsort(-sc, kind="stable")]
    return new, float(np.median(lat))


def metrics_from_order(o, ql, gl):
    rel = (gl[o] == ql[:, None]).astype(float); R = (gl[None, :] == ql[:, None]).sum(1); cs = np.cumsum(rel, 1)
    d = dict(ap=(cs / np.arange(1, rel.shape[1] + 1) * rel).sum(1) / np.maximum(R, 1), pred=gl[o[:, 0]])
    for k in (1, 5, 10): d[f"hit{k}"] = rel[:, :k].max(1); d[f"p{k}"] = rel[:, :k].mean(1)
    return pd.DataFrame(d)


def macro(pq, ql):
    g = pq.assign(c=ql).groupby("c").mean(numeric_only=True).mean(); return dict(hit1=float(g.hit1), p10=float(g.p10), mAP=float(g.ap))


def run(ckpt_fmt="artifacts/checkpoints/D5ck_dinoB_p4_supcon_lr3e-5_s{}.pt", emb_fmt="artifacts/extensions/B/D5ck_dinoB_p4_supcon_lr3e-5_s{}_embeddings.npz", tag="D5ck", seeds=(0, 1, 2)):
    S = C.segppd_sets(); ga = S["gallery_B"]; gl = ga.class_id.to_numpy(); gid = ga.image_id.to_numpy(dtype=str)
    q_ext = pd.read_csv("artifacts/research_v2/trackC/external_per_query_preds.csv") if Path("artifacts/research_v2/trackC/external_per_query_preds.csv").exists() else None
    ext = pd.read_csv("artifacts/research_v2/trackC/organ/sample_ids.csv") if False else None
    import segppd_cbir.rv2.trackC_diagnose as DG
    qe = DG.queries(); sets = {"dev": (S["dev_q"], "seg"), "testB_exploratory": (S["testB_q"], "seg"), "testC_exploratory": (S["testC_q"], "seg"), "external_plantseg_exploratory": (qe, "ps")}
    results = []; sel_rows = []; perq = []
    for s in seeds:
        net = load_net(ckpt_fmt.format(s)); z = np.load(emb_fmt.format(s)); pos = {i: k for k, i in enumerate(z["image_id"])}; G = z["emb"][[pos[i] for i in gid]]
        gtok = tokens(net, [to_tensor(__import__("PIL.Image", fromlist=["Image"]).open(ROOT / p).convert("RGB"), 224) for p in ga.image_path])  # [809,256,768] fp16 (CPU)
        for nm, (qdf, kind) in sets.items():
            if kind == "seg":
                ql = qdf.class_id.to_numpy(); Eq = z["emb"][[pos[i] for i in qdf.image_id]]
                imgs = [to_tensor(__import__("PIL.Image", fromlist=["Image"]).open(ROOT / p).convert("RGB"), 224) for p in qdf.image_path]
            else:
                ql = qdf.class_id.to_numpy()
                imgs = [to_tensor(__import__("PIL.Image", fromlist=["Image"]).open(CACHE / (Path(n).stem + ".png")).convert("RGB"), 224) for n in qdf.Name]
                from ..train_embed import embed_tensors
                Eq = embed_tensors(net, torch.stack(imgs).half())
            qtok = tokens(net, imgs)
            if nm == "testC_exploratory":  # Protocol C queries were carved out of gallery_B: use gallery_C (no self-match)
                posB = {i: k for k, i in enumerate(gid)}; gi = np.array([posB[i] for i in S["gallery_C"].image_id]); Gs, gts, gls, gids = G[gi], gtok[torch.as_tensor(gi)], gl[gi], gid[gi]
            else:
                Gs, gts, gls, gids = G, gtok, gl, gid
            o, d2 = rank(Eq, Gs, gids, k=len(Gs)); cos = 1 - d2 / 2  # rank() already returns the distances in ranked order
            base = metrics_from_order(o, ql, gls); perq.append(base.assign(seed=s, set=nm, M=0, alpha=1.0, class_id=ql, qi=np.arange(len(ql)))); results.append(dict(seed=s, set=nm, M=0, alpha=1.0, **macro(base, ql), latency_ms=0.0))
            loc, lat = local_scores(o, qtok, gts, max(MS))
            for M in MS:
                for a in ALPHAS:
                    new = apply_rerank(o, cos, loc, M, a); pq = metrics_from_order(new, ql, gls); perq.append(pq.assign(seed=s, set=nm, M=M, alpha=a, class_id=ql, qi=np.arange(len(ql)))) if (M, a) in ((20, 0.5), (20, 0.75), (50, 0.5)) else None; results.append(dict(seed=s, set=nm, M=M, alpha=a, **macro(pq, ql), latency_ms=1e3 * lat * (M / max(MS))))
            print(f"seed {s} {nm} done", flush=True); del qtok
        del net, gtok; torch.cuda.empty_cache()
    R = pd.DataFrame(results); R.to_csv(OUT / f"local_rerank_{tag}_all.csv", index=False); pd.concat(perq).to_csv(OUT / f"local_rerank_{tag}_perquery.csv", index=False)
    dev = R[R.set == "dev"].groupby(["M", "alpha"])[["hit1", "mAP", "p10"]].mean().reset_index()
    top = dev.hit1.max(); tied = dev[dev.hit1 >= top - C.TIE_EPS]; best = tied.sort_values("mAP", ascending=False).iloc[0]; sel = dict(M=int(best.M), alpha=float(best.alpha), dev_hit1=float(best.hit1), dev_mAP=float(best.mAP))
    json.dump(sel, open(OUT / f"local_rerank_{tag}_selected.json", "w"), indent=1); print("selected on dev:", sel)
    summ = R.groupby(["set", "M", "alpha"])[["hit1", "p10", "mAP", "latency_ms"]].mean().reset_index(); summ.to_csv(OUT / f"local_rerank_{tag}_summary.csv", index=False)
    print(summ[(summ.M == 0) | ((summ.M == sel["M"]) & (summ.alpha == sel["alpha"]))].round(4).to_string(index=False))


if __name__ == "__main__":
    run(ckpt_fmt="artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s{}/best.pt", emb_fmt="artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s{}/embeddings.npz", tag="PA")
