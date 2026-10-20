"""Figures + cost table for the Protocol B locked evaluation (reads only saved artifacts)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import ext_dev as D
from .hybrid import get, load_protocol_b
from .index import rank

L = Path("artifacts/metrics/protocolB_locked")
F = Path("artifacts/figures")
ROOT = Path("data/extracted/SegPPD-101")
HEAD = [f"D5_dinoB_p4_supcon_s{i}" for i in range(3)]
SHOW = {"BASE_resnet50_frozen_pooled2048_raw": "ResNet50 frozen 2048-D (mandated baseline)",
        "R1_resnet50_ft_arcface_existing_system": "ResNet50 fine-tuned + ArcFace",
        "F_dinov2b_frozen_cls": "DINOv2-B frozen CLS",
        "D4_dinoS_p4_supcon_s1": "DINOv2-S last-4 + SupCon (seed 1)",
        "D5_dinoB_p4_supcon_s0": "DINOv2-B last-4 + SupCon (seed 0)",
        "E_dinoB_supcon_3seed_ensemble": "DINOv2-B + SupCon, 3-seed ensemble"}


def figures():
    F.mkdir(parents=True, exist_ok=True)
    ci = pd.read_csv(L / "macro_ci.csv")
    # 1) Hit@1 with class-bootstrap CI
    fig, ax = plt.subplots(figsize=(8, 3.8))
    reps = list(SHOW)
    for i, r in enumerate(reps):
        c = ci[(ci.rep == r) & (ci.K == 1) & (ci.metric == "P")].iloc[0]
        ax.barh(i, c.macro_mean, xerr=[[c.macro_mean - c.ci_lo], [c.ci_hi - c.macro_mean]], color="#4C78A8", capsize=3)
        ax.text(c.ci_hi + 0.01, i, f"{c.macro_mean:.3f}", va="center", fontsize=8)
    ax.axvline(0.8, color="#E45756", ls="--", lw=1)
    ax.set_yticks(range(len(reps)), [SHOW[r] for r in reps], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Hit@1 (macro, 75 classes, 225 locked queries, 95% CI)", fontsize=8); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(F / "protocolB_hit1_locked.png", dpi=140); plt.close(fig)
    # 2) P@K and Hit@K curves
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    ks = [1, 3, 5, 10, 20, 50, 100]
    for r in SHOW:
        for ax, m in zip(axs, ("P", "Hit")):
            v = [ci[(ci.rep == r) & (ci.K == k) & (ci.metric == m)].macro_mean.iloc[0] for k in ks]
            ax.plot(ks, v, marker="o", ms=3, label=SHOW[r])
    for ax, t in zip(axs, ("Precision@K", "Hit@K")):
        ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_title(t); ax.grid(alpha=.3)
    axs[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(F / "protocolB_curves_locked.png", dpi=140); plt.close(fig)
    # 3) per-class Hit@1
    pc = pd.read_csv(L / "per_class_hit1.csv", index_col=0)
    a, b = pc[HEAD].mean(axis=1), pc["R1_resnet50_ft_arcface_existing_system"]
    order = a.sort_values().index
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(range(len(order)), a[order], color="#4C78A8", label="DINOv2-B + SupCon (3-seed mean)")
    ax.plot(range(len(order)), b[order], "o", ms=3, color="#E45756", label="ResNet50 ft + ArcFace")
    ax.set_xlabel("75 classes sorted by DINOv2-B Hit@1"); ax.set_ylabel("per-class Hit@1 (3 queries/class)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(F / "protocolB_per_class_locked.png", dpi=140); plt.close(fig)


def cost_table():
    rows = []
    for t in ["R1_resnet_arcface_s0", "D2_dino_p4_arcface_lr3e-5_s0", "D3_dino_p4_proxyanchor_lr3e-5_s0", "D4_dino_p4_supcon_lr3e-5_s0",
              "D5_dinoB_p4_supcon_lr3e-5_s0"]:
        d = json.load(open(f"artifacts/extensions/B/{t}_log.json"))
        rows.append(dict(run=t, trainable_params_M=d["n_trainable_params"] / 1e6, epochs_run=d["epochs_run"], train_seconds=d["train_seconds"],
                         peak_vram_gb=d["peak_vram_gb"], embed_2263_images_seconds=d["embed_all_seconds"],
                         embed_ms_per_image=1e3 * d["embed_all_seconds"] / 2263))
    c = pd.DataFrame(rows).merge(pd.DataFrame(), how="left", left_index=True, right_index=True)
    c.to_csv(L / "cost_table.csv", index=False)
    return c


def retrieval_grid(model="D5_dinoB_p4_supcon_lr3e-5_s0", n_ok=3, n_bad=3, k=5):
    dq, ga, tq, b = load_protocol_b()
    z = np.load(f"artifacts/extensions/B/{model}_embeddings.npz")
    Q, G = get(z["image_id"], D.l2n(z["emb"]), tq), get(z["image_id"], D.l2n(z["emb"]), ga)
    o, d2 = rank(Q, G, ga.image_id.to_numpy(dtype=str), k=k)
    ql, gl = tq.class_id.to_numpy(), ga.class_id.to_numpy()
    ok = (gl[o[:, 0]] == ql)
    # transparent selection: first n queries (by image_id) that are correct at rank 1, and first n that are wrong
    sel = list(np.where(ok)[0][:n_ok]) + list(np.where(~ok)[0][:n_bad])
    lab = dict(zip(b.class_id, b.raw_class_label))
    fig, axs = plt.subplots(len(sel), k + 1, figsize=(2 * (k + 1), 2.1 * len(sel)))
    for r, qi in enumerate(sel):
        axs[r, 0].imshow(Image.open(ROOT / tq.image_path.iloc[qi])); axs[r, 0].set_title(f"QUERY {tq.image_id.iloc[qi]}\n{lab[ql[qi]]}", fontsize=6)
        for j in range(k):
            gi = o[qi, j]; good = gl[gi] == ql[qi]
            axs[r, j + 1].imshow(Image.open(ROOT / ga.image_path.iloc[gi]))
            axs[r, j + 1].set_title(f"{'OK' if good else 'WRONG'} d={np.sqrt(d2[qi, j]):.2f}\n{lab[gl[gi]]}", fontsize=6, color="green" if good else "red")
        for a in axs[r]:
            a.axis("off")
    out = Path("artifacts/retrieval_examples"); out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out / "protocolB_D5_s0_examples_LOCAL_ONLY.png", dpi=90); plt.close(fig)
    return int(ok.sum()), len(ok)


if __name__ == "__main__":
    figures()
    print(cost_table().round(2).to_string(index=False))
    print("rank-1 correct (D5 s0):", retrieval_grid())
