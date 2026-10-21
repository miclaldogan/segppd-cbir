"""Track C2 evaluation of the augmentation variants: SegPPD dev (selection) + consumed tests (exploratory) from result.json, plus the 1,485 strict external PlantSeg
queries (EXPLORATORY, never used for selection). Caches ext_emb.npz per run directory.  python -m segppd_cbir.rv2.trackC_eval"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..index import rank
from . import common as C
from .trackC_diagnose import CACHE, queries

OUT = Path("artifacts/research_v2/trackC"); FIG = Path("artifacts/research_v2/figures")
RUNS = {"base": [f"artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s{s}" for s in range(3)], "color": [f"{OUT}/runs/C_pa_augcolor_s{s}" for s in range(3)],
        "robust": [f"{OUT}/runs/C_pa_augrobust_s{s}" for s in range(3)], "res": [f"{OUT}/runs/C_pa_augres_s0"], "geo": [f"{OUT}/runs/C_pa_auggeo_s0"]}


def ext_embed(run):
    f = Path(run) / "ext_emb.npz"
    if f.exists(): return np.load(f)["emb"]
    import torch
    from PIL import Image

    from ..ext_features import to_tensor
    from ..train_embed import embed_tensors
    from .local_rerank import load_net
    q = queries(); X = torch.stack([to_tensor(Image.open(CACHE / (Path(n).stem + ".png")).convert("RGB"), 224) for n in q.Name]).half()
    net = load_net(Path(run) / "best.pt"); E = embed_tensors(net, X); np.savez(f, emb=E); del net; torch.cuda.empty_cache(); return E


def ext_metrics(E, q, S):
    ga = S["gallery_B"]; gid = ga.image_id.to_numpy(dtype=str); gl = ga.class_id.to_numpy()
    z = np.load(Path(RUNS_ANY) / "embeddings.npz"); pos = {i: k for k, i in enumerate(z["image_id"])}; G = z["emb"][[pos[i] for i in gid]]
    pq, _ = C.per_query(E, G, q.class_id.to_numpy(), gl, gid, q.Name.to_numpy(dtype=str)); return pq


def main():
    S = C.segppd_sets(); q = queries(); rows, pqs = [], {}
    global RUNS_ANY
    for var, dirs in RUNS.items():
        for d in dirs:
            if not (Path(d) / "best.pt").exists() or not (Path(d) / "result.json").exists(): continue
            RUNS_ANY = d; E = ext_embed(d); pq = ext_metrics(E, q, S); pqs[(var, d)] = pq; r = json.load(open(Path(d) / "result.json"))["results"]; sm = C.summarize(pq)
            rows.append(dict(variant=var, run=Path(d).name, seed=int(Path(d).name.split("_s")[-1]), dev_hit1=r["dev"]["macro_hit1"], dev_mAP=r["dev"]["macro_mAP"], testB_hit1=r["testB_exploratory"]["macro_hit1"], testB_mAP=r["testB_exploratory"]["macro_mAP"],
                             testC_hit1=r["testC_exploratory"]["macro_hit1"], testC_mAP=r["testC_exploratory"]["macro_mAP"], ext_macro_hit1=sm["macro_hit1"], ext_micro_hit1=sm["micro_hit1"], ext_macro_p10=sm["macro_p10"], ext_macro_mAP=sm["macro_mAP"]))
    T = pd.DataFrame(rows); T.to_csv(OUT / "aug_ablation_by_run.csv", index=False)
    A = T.groupby("variant").agg(n_seeds=("seed", "size"), **{f"{c}_mean": (c, "mean") for c in T.columns if c.endswith(("hit1", "mAP", "p10")) and c != "variant"}, **{f"{c}_sd": (c, "std") for c in ("dev_hit1", "testB_hit1", "testC_hit1", "ext_macro_hit1")}).reset_index(); A.to_csv(OUT / "aug_ablation_summary.csv", index=False)
    pd.set_option("display.width", 260); print(A.round(4).to_string(index=False))
    # paired external comparison base vs color/robust (seed-averaged per-query, class bootstrap)
    base = [v for (k, d), v in pqs.items() if k == "base"]
    out = []
    for var in ("color", "robust"):
        vv = [v for (k, d), v in pqs.items() if k == var]
        if not vv or not base: continue
        n = min(len(vv), len(base)); a = np.mean([v.hit1.to_numpy() for v in vv[:n]], 0); b = np.mean([v.hit1.to_numpy() for v in base[:n]], 0)
        d = base[0][["class_id"]].assign(hit1=a - b); m, lo, hi = C.class_boot(d, "hit1")
        out.append(dict(variant=var, vs="base", n_seeds=n, ext_hit1_diff=m, lo=lo, hi=hi, queries_better=int((a > b).sum()), queries_worse=int((a < b).sum())))
    pd.DataFrame(out).to_csv(OUT / "aug_paired_external.csv", index=False); print(pd.DataFrame(out).round(4).to_string(index=False))
    figure(T)


def figure(T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    order = ["base", "color", "res", "geo", "robust"]; sets = [("dev_hit1", "SegPPD dev (selection)"), ("testB_hit1", "Protocol B test (expl.)"), ("testC_hit1", "Protocol C test (expl.)"), ("ext_macro_hit1", "External PlantSeg (expl.)")]
    fig, axs = plt.subplots(1, 4, figsize=(10.2, 3.1), sharey=False); cols = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    for ax, (c, ttl) in zip(axs, sets):
        for i, v in enumerate(order):
            x = T[T.variant == v][c].to_numpy()
            if len(x): ax.bar(i, x.mean(), color=cols[i], yerr=(x.std(ddof=1) if len(x) > 1 else 0), capsize=3); ax.scatter([i] * len(x), x, color="k", s=7, zorder=3)
        ax.set_xticks(range(5), order, fontsize=7.5); ax.set_title(ttl, fontsize=8.5); lo = T[c].min(); ax.set_ylim(max(0, lo - 0.08), T[c].max() + 0.04)
    axs[0].set_ylabel("Hit@1 (class-macro; dots = seeds)"); fig.tight_layout(); fig.savefig(FIG / "C2_augmentation_ablation.png", dpi=180); fig.savefig(FIG / "C2_augmentation_ablation.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
