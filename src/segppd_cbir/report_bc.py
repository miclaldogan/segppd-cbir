"""Protocol B vs C vs pooled (B u C) Hit@1 for the headline system and baselines + figure. Reads saved per-query files only."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import evaluate as E

SYS = {"BASE_resnet50_frozen_pooled2048_raw": "ResNet50 frozen 2048-D",
       "R1_resnet50_ft_arcface_existing_system": "ResNet50 fine-tuned + ArcFace",
       "F_dinov2b_frozen_cls": "DINOv2-B frozen",
       "HEADLINE": "DINOv2-B last-4 + SupCon\n(mean of 3 seeds, declared headline)"}
SEEDS = [f"D5_dinoB_p4_supcon_s{i}" for i in range(3)]


def load(tag):
    q = pd.read_csv(f"artifacts/metrics/protocol{tag}_locked/per_query.csv", dtype={"query_id": str})
    return q[q.K == 1]


def headline(q):
    h = q[q.rep.isin(SEEDS)].groupby(["query_id", "class_id"], as_index=False).P.mean()  # seed-averaged P@1 per query
    h["rep"] = "HEADLINE"; h["K"] = 1
    return pd.concat([q[q.rep.isin(SYS)], h], ignore_index=True)


def main():
    B, C = headline(load("B")), headline(load("C"))
    P = pd.concat([B, C], ignore_index=True)
    rows = []
    for nm, d in (("B (225 q)", B), ("C (142 q)", C), ("B+C pooled (367 q)", P)):
        for r in SYS:
            m, lo, hi = E.class_bootstrap_ci(d[d.rep == r], "P")
            rows.append(dict(set=nm, rep=r, hit1=m, ci_lo=lo, ci_hi=hi))
    df = pd.DataFrame(rows); df.to_csv("artifacts/metrics/protocolBC_hit1_summary.csv", index=False)
    print(df.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    cols = {"B (225 q)": "#4C78A8", "C (142 q)": "#F58518", "B+C pooled (367 q)": "#54A24B"}
    w = 0.26
    for k, (nm, c) in enumerate(cols.items()):
        d = df[df.set == nm].set_index("rep").loc[list(SYS)]
        x = [i + (k - 1) * w for i in range(len(SYS))]
        ax.bar(x, d.hit1, w, yerr=[d.hit1 - d.ci_lo, d.ci_hi - d.hit1], capsize=2, color=c, label=nm)
        for xi, v in zip(x, d.hit1): ax.text(xi, 0.02, f"{v:.2f}", ha="center", fontsize=7, color="white")
    ax.axhline(0.8, color="#E45756", ls="--", lw=1)
    ax.set_xticks(range(len(SYS)), list(SYS.values()), fontsize=7.5); ax.set_ylim(0, 1)
    ax.set_ylabel("Hit@1 (macro, 95% class-bootstrap CI)"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Two independent locked query sets", fontsize=10)
    fig.tight_layout(); fig.savefig(Path("artifacts/figures/protocolBC_hit1.png"), dpi=150)


if __name__ == "__main__":
    main()
