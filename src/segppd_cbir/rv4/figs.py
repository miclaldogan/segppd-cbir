"""research_v4 figure P1: parameters vs dev Hit@1 (and pooled B+C, exploratory) for every trained candidate. python -m segppd_cbir.rv4.figs"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
FIG = Path("artifacts/research_v4/figures"); FIG.mkdir(exist_ok=True)
COL = {"DINOv2 ViT": "#1f6fb5", "DINOv3 ViT": "#c96a00", "DINOv3 ConvNeXt": "#2e8b57", "ensemble / füzyon (önceki)": "#8e44ad"}


def family(bb):
    return "DINOv3 ConvNeXt" if "convnext" in bb else "DINOv3 ViT" if "dinov3" in bb else "DINOv2 ViT"


def main():
    R = pd.read_csv("artifacts/research_v4/results_all.csv"); R = R[R.dev_hit1 > 0.3]  # drop the diverged full-ft run
    R["fam"] = R.backbone.map(family); R["pool"] = (225 * R.B_hit1 + 142 * R.C_hit1) / 367
    prev = pd.DataFrame([dict(run="PA tek model (3 seed ort.)", params_M=85.9, dev_hit1=0.820, pool=(225 * .825 + 142 * .782) / 367, fam="DINOv2 ViT"),
                         dict(run="PA tek model + ResNet50", params_M=109.9, dev_hit1=0.827, pool=(225 * .839 + 142 * .800) / 367, fam="ensemble / füzyon (önceki)"),
                         dict(run="PA ensemble (3x)", params_M=257.8, dev_hit1=0.813, pool=(225 * .836 + 142 * .813) / 367, fam="ensemble / füzyon (önceki)"),
                         dict(run="SupCon ensemble + ResNet50", params_M=281.8, dev_hit1=0.820, pool=(225 * .840 + 142 * .773) / 367, fam="ensemble / füzyon (önceki)")])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True)
    for ax, col, title in ((axes[0], "dev_hit1", "Geliştirme (seçim) kümesi"), (axes[1], "pool", "B+C birleşik (exploratory)")):
        for fam, c in COL.items():
            d = R[R.fam == fam]; p = prev[prev.fam == fam]
            if len(d): ax.scatter(d.params_M, d[col], s=22, color=c, label=fam, zorder=3, edgecolor="white", linewidth=0.6)
            if len(p): ax.scatter(p.params_M, p[col], s=34, color=c, marker="D", zorder=4, edgecolor="white", linewidth=0.6, label=fam if not len(d) else None)
        ax.axhline(0.85, color="#bbbbbb", lw=0.8, ls="--", zorder=1); ax.text(300, 0.852, "hedef 0,85", fontsize=7, color="#666666", ha="right")
        ax.set_xscale("log"); ax.set_xticks([20, 30, 50, 100, 200, 300]); ax.set_xticklabels(["20", "30", "50", "100", "200", "300"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Çıkarım parametresi (M, log)"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.25, lw=0.5); ax.set_ylim(0.6, 0.87)
        OFF = {"PA tek model (3 seed ort.)": (5, -11), "PA tek model + ResNet50": (-30, 7), "PA ensemble (3x)": (-68, -11), "SupCon ensemble + ResNet50": (-100, 7)}
        for _, r in prev.iterrows(): ax.annotate(r.run, (r.params_M, r[col]), fontsize=6.3, xytext=OFF[r.run], textcoords="offset points", color="#444444")
        fin = R[R.run.str.contains("convnext_tiny_336") & ~R.run.str.startswith("D_")]; ax.scatter([28.0], [fin[col].mean()], s=40, marker="D", color=COL["DINOv3 ConvNeXt"], edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate("ConvNeXt-T 336 px (3 seed ort.)", (28.0, fin[col].mean()), fontsize=6.3, xytext=(6, -3), textcoords="offset points", color="#444444")
    axes[0].set_ylabel("Hit@1 (sınıf-makro)"); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIG / "P1_params_vs_hit1.png", dpi=180); fig.savefig(FIG / "P1_params_vs_hit1.pdf"); print("saved")


if __name__ == "__main__":
    main()
