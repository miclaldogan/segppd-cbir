"""Precision-Recall curves (full ranking, 11-point interpolated, macro over classes) for the locked Protocol B and C runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import ext_dev as D
from .final_locked import build_matrix
from .hybrid import get, load_protocol_b
from .index import rank

F = Path("artifacts/figures")
LEVELS = np.linspace(0, 1, 11)
SHOW = {"BASE_resnet50_frozen_pooled2048_raw": ("ResNet50 frozen 2048-D", "#9D9D9D"),
        "R1_resnet50_ft_arcface_existing_system": ("ResNet50 fine-tuned + ArcFace", "#F58518"),
        "F_dinov2b_frozen_cls": ("DINOv2-B frozen", "#72B7B2"),
        "D5_dinoB_p4_supcon_s0": ("DINOv2-B last-4 + SupCon (seed 0)", "#4C78A8"),
        "D5_dinoB_p4_supcon_s1": ("DINOv2-B last-4 + SupCon (seed 1)", "#4C78A8"),
        "D5_dinoB_p4_supcon_s2": ("DINOv2-B last-4 + SupCon (seed 2)", "#4C78A8")}


def pr_curves(order, ql, gl):
    """Per query: interpolated precision at 11 recall levels, plus the raw PR points (rank-wise)."""
    out = np.zeros((len(ql), len(LEVELS)))
    for i in range(len(ql)):
        rel = (gl[order[i]] == ql[i]).astype(float)
        R = rel.sum()
        tp = np.cumsum(rel)
        prec, rec = tp / np.arange(1, len(rel) + 1), tp / R
        for j, lv in enumerate(LEVELS):
            m = rec >= lv - 1e-12
            out[i, j] = prec[m].max() if m.any() else 0.0
    return out


def macro(curves, ql):
    df = pd.DataFrame(curves); df["c"] = ql
    return df.groupby("c").mean().mean(axis=0).to_numpy()


def run(cfg_path, tag):
    cfg = json.load(open(cfg_path))
    if cfg.get("protocol") == "C":
        c = pd.read_csv("artifacts/splits/protocolC_seed777.csv", dtype={"image_id": str})
        ga, tq = c[c.role_C == "gallery_C"].sort_values("image_id"), c[c.role_C == "test_query_C"].sort_values("image_id")
    else:
        _, ga, tq, _ = load_protocol_b()
    gid, ql, gl = ga.image_id.to_numpy(dtype=str), tq.class_id.to_numpy(), ga.class_id.to_numpy()
    cache, rows = {}, []
    for name, spec in cfg["candidates"].items():
        cache[name] = build_matrix(spec, cache)
        ids, Z = cache[name]
        order, _ = rank(get(ids, Z, tq), get(ids, Z, ga), gid)
        mp = macro(pr_curves(order, ql, gl), ql)
        rows += [dict(rep=name, recall=float(lv), interp_precision=float(p)) for lv, p in zip(LEVELS, mp)]
    df = pd.DataFrame(rows)
    out = Path(f"artifacts/metrics/protocol{tag}_locked/pr_curves_11pt.csv"); df.to_csv(out, index=False)
    # AP-like summary: mean of the 11 interpolated precisions
    df.groupby("rep").interp_precision.mean().rename("mean_11pt_interp_precision").to_csv(out.with_name("pr_11pt_summary.csv"))
    F.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, (lab, col) in SHOW.items():
        d = df[df.rep == name]
        seed_run = name.startswith("D5_") and not name.endswith("_s0")
        ax.plot(d.recall, d.interp_precision, color=col, lw=1.6 if not seed_run else 1.0, alpha=1 if not seed_run else .6,
                label=None if seed_run else lab, marker="o" if not seed_run else None, ms=3)
    ax.set_xlabel("Recall (macro over classes)"); ax.set_ylabel("Interpolated precision")
    n_q = len(tq)
    ax.set_title(f"Precision-Recall, Protocol {tag} locked ({n_q} queries, gallery {len(ga)})", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.25); ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(F / f"protocol{tag}_PR_curve.png", dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
