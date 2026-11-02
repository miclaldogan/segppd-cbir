"""Figures for the written report (Turkish labels). Run: uv run python report/make_figures.py"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
O = Path("report/figs"); O.mkdir(parents=True, exist_ok=True)
RASTER_ONLY = {"dataset_samples", "retrieval_examples"}  # contain dataset photographs


def save(fig, name):
    fig.savefig(O / f"{name}.png", dpi=200)
    if name not in RASTER_ONLY:
        fig.savefig(O / f"{name}.pdf")
    if name.startswith("arch_"):
        fig.savefig(O / f"{name}.svg")
    plt.close(fig)

ROOT = Path("data/extracted/SegPPD-101")
BLUE, ORANGE, GREEN, GREY, RED, TEAL, PURPLE = "#4C78A8", "#F58518", "#54A24B", "#9D9D9D", "#E45756", "#72B7B2", "#B279A2"


# ---------------------------------------------------------------- dataset figures
def dataset_samples():
    m = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str})
    m = m[m.dup_status != "excluded"]
    pick = [(5, "Elma - uyuz"), (94, "Domates - geç yanıklık"), (42, "Üzüm - kara çürüklük"), (24, "Mısır - yanıklık"),
            (16, "Manyok - mozaik"), (22, "Kahve - pas"), (31, "Salatalık - külleme"), (1, "Elma - sağlıklı")]
    fig, ax = plt.subplots(2, len(pick), figsize=(10.2, 3.5))
    for k, (cid, name) in enumerate(pick):
        r = m[m.class_id == cid].sort_values("image_id").iloc[2]
        im = np.asarray(Image.open(ROOT / r.image_path).convert("RGB"))
        mk = np.asarray(Image.open(ROOT / r.mask_path)) > 0
        ax[0, k].imshow(im); ov = im.copy().astype(float)
        ov[mk] = 0.55 * ov[mk] + 0.45 * np.array([228, 87, 86]); ax[1, k].imshow(ov.astype(np.uint8))
        ax[0, k].set_title(name, fontsize=7.5)
        for a in ax[:, k]: a.axis("off")
    ax[0, 0].text(-0.06, 0.5, "Görüntü", rotation=90, transform=ax[0, 0].transAxes, ha="right", va="center", fontsize=8)
    ax[1, 0].text(-0.06, 0.5, "Maske", rotation=90, transform=ax[1, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.tight_layout(w_pad=0.3, h_pad=0.4); save(fig, "dataset_samples")


def class_hist():
    m = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str})
    c = m[m.dup_status != "excluded"].groupby("class_id").size().sort_values(ascending=False).to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.bar(range(len(c)), c, color=[BLUE if v >= 13 else GREY for v in c], width=0.85)
    ax.axhline(13, color=RED, ls="--", lw=1)
    ax.text(len(c) - 1, 14.5, "eşik: 13 görüntü", ha="right", color=RED, fontsize=8)
    ax.bar([], [], color=BLUE, label=f"birincil küme ({int((c >= 13).sum())} sınıf)"); ax.bar([], [], color=GREY, label=f"kapsam dışı ({int((c < 13).sum())} sınıf)"); ax.legend(fontsize=7.5, frameon=False)
    ax.set_xlabel("Sınıflar (görüntü sayısına göre sıralı)"); ax.set_ylabel("Görüntü sayısı"); ax.set_xlim(-1, len(c))
    fig.tight_layout(); save(fig, "class_hist")


# ---------------------------------------------------------------- architecture helpers
def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#444", fs=8, bold=False, lw=1.1, tc="#222", r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.002,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


def arrow(ax, p, q, color="#444", lw=1.3, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); return fig, ax


def arch_baseline():
    fig, ax = canvas(10.2, 3.1)
    box(ax, 0.01, 0.36, 0.12, 0.30, "Giriş\ngörüntüsü\n256×256 RGB", "#F2F2F2")
    box(ax, 0.16, 0.36, 0.13, 0.30, "Preprocessing\nresize 232\ncenter crop 224\nImageNet norm.", "#F2F2F2", fs=7.5)
    # ResNet block
    ax.add_patch(FancyBboxPatch((0.32, 0.12), 0.40, 0.76, boxstyle="round,pad=0.002,rounding_size=0.012", fc="#EAF1F8", ec=BLUE, lw=1.5))
    ax.text(0.52, 0.845, "ResNet50 (ImageNet-1K V2)  -  donmuş, eğitimsiz", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    xs = [0.335, 0.415, 0.495, 0.575]
    for x, t in zip(xs, ["conv1\n+ maxpool", "layer1\nlayer2", "layer3\nlayer4", "global\navg-pool"]):
        box(ax, x, 0.50, 0.07, 0.20, t, "#FFFFFF", BLUE, fs=7)
    for a, b in zip(xs[:-1], xs[1:]): arrow(ax, (a + 0.07, 0.60), (b, 0.60), BLUE)
    box(ax, 0.575, 0.18, 0.07, 0.17, "fc\n2048→1000", "#FFFFFF", BLUE, fs=7)
    arrow(ax, (0.61, 0.50), (0.61, 0.35), BLUE)
    box(ax, 0.655, 0.54, 0.055, 0.12, "2048-B", "#D9E8F5", BLUE, fs=7, bold=True)
    box(ax, 0.655, 0.20, 0.055, 0.12, "1000-B", "#D9E8F5", BLUE, fs=7, bold=True)
    arrow(ax, (0.645, 0.60), (0.655, 0.60), BLUE); arrow(ax, (0.645, 0.265), (0.655, 0.265), BLUE)
    box(ax, 0.755, 0.36, 0.11, 0.30, "Tam Öklid (L2)\nsıralama\n(tie-break:\nmesafe, id)", "#FDF1E3", ORANGE, fs=7.5)
    box(ax, 0.89, 0.36, 0.10, 0.30, "Top-K\nsonuçlar\nP@K, Hit@K,\nmAP", "#EAF4E8", GREEN, fs=7.5)
    arrow(ax, (0.13, 0.51), (0.16, 0.51)); arrow(ax, (0.29, 0.51), (0.335, 0.60)); arrow(ax, (0.71, 0.60), (0.755, 0.52)); arrow(ax, (0.71, 0.265), (0.755, 0.50))
    arrow(ax, (0.865, 0.51), (0.89, 0.51))
    ax.text(0.76, 0.12, "birincil: 1000-B logit  |  karşılaştırma: 2048-B", fontsize=7.2, color="#555", style="italic")
    ax.text(0.765, 0.78, "galeri vektörleri önceden\nhesaplanıp saklanır", fontsize=7.2, color="#555", style="italic")
    save(fig, "arch_baseline")


def arch_final():
    fig, ax = canvas(10.2, 6.3)
    # ---- panel A: training
    ax.add_patch(FancyBboxPatch((0.005, 0.50), 0.99, 0.49, boxstyle="round,pad=0.002,rounding_size=0.012", fc="#FBFBFB", ec="#BBB", lw=1))
    ax.text(0.015, 0.955, "(a) Eğitim  -  yalnızca train_B (772 görüntü), 75 sınıf", fontsize=9, fontweight="bold")
    box(ax, 0.02, 0.70, 0.10, 0.17, "Eğitim\ngörüntüsü", "#F2F2F2")
    box(ax, 0.14, 0.64, 0.14, 0.29, "Sınıf dengeli batch\n16 sınıf × 4 görüntü\n\nAugmentation: kırpma (0,4-1),\nyatay/dikey çevirme,\nrenk oynatma", "#F2F2F2", fs=7.3)
    # DINO block
    ax.add_patch(FancyBboxPatch((0.31, 0.565), 0.45, 0.36, boxstyle="round,pad=0.002,rounding_size=0.012", fc="#EAF1F8", ec=BLUE, lw=1.5))
    ax.text(0.535, 0.895, "DINOv2-B (ViT-B/14, LVD-142M pretraining)", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    box(ax, 0.325, 0.62, 0.085, 0.20, "Patch\nembedding\n+ position", "#FFFFFF", BLUE, fs=7)
    box(ax, 0.425, 0.62, 0.135, 0.20, "Blok 1-8\nDONMUŞ", "#E6E6E6", GREY, fs=7.8, bold=True)
    box(ax, 0.575, 0.62, 0.135, 0.20, "Blok 9-12 + norm\nEĞİTİLİR\nlr 3e-5, katman\nbaşına ×0,8", "#FDE9D5", ORANGE, fs=7, bold=True)
    for a, b in [(0.41, 0.425), (0.56, 0.575)]: arrow(ax, (a, 0.72), (b, 0.72), BLUE)
    ax.text(0.535, 0.585, "CLS token → 768-B", ha="center", fontsize=7.3, color=BLUE, style="italic")
    box(ax, 0.785, 0.70, 0.095, 0.17, "Dropout 0,2\nLinear\n768→256\nL2 norm.", "#FDE9D5", ORANGE, fs=7)
    box(ax, 0.90, 0.70, 0.085, 0.17, "256-B\nembedding\nz", "#D9E8F5", BLUE, fs=7.5, bold=True)
    box(ax, 0.785, 0.53, 0.20, 0.12, "SupCon kaybı (τ = 0,1)\naynı sınıfı yaklaştır, diğerini uzaklaştır", "#F9E1E1", RED, fs=7.2)
    arrow(ax, (0.12, 0.785), (0.14, 0.785)); arrow(ax, (0.28, 0.785), (0.325, 0.74)); arrow(ax, (0.71, 0.72), (0.785, 0.785))
    arrow(ax, (0.88, 0.785), (0.90, 0.785)); arrow(ax, (0.9425, 0.70), (0.9425, 0.65), RED)
    ax.text(0.02, 0.585, "AdamW, cosine LR schedule; early stopping: dev mAP, patience 10", fontsize=7.2, color="#555", style="italic")
    ax.text(0.99, 0.955, "YALNIZCA EĞİTİM", ha="right", fontsize=8, fontweight="bold", color=RED)
    # ---- panel B: retrieval
    ax.add_patch(FancyBboxPatch((0.005, 0.01), 0.99, 0.47, boxstyle="round,pad=0.002,rounding_size=0.012", fc="#FBFBFB", ec="#BBB", lw=1))
    ax.text(0.015, 0.445, "(b) Erişim (CBIR)  -  eğitimsiz, tam Öklid arama", fontsize=9, fontweight="bold")
    box(ax, 0.02, 0.20, 0.10, 0.16, "Sorgu\ngörüntüsü", "#F2F2F2")
    box(ax, 0.02, 0.04, 0.10, 0.12, "Yatay çevrilmiş\nsorgu", "#F2F2F2", fs=7.3)
    box(ax, 0.17, 0.08, 0.22, 0.27, "Aynı ağ: (a)'da eğitilen\nağırlıklar kullanılır\n(DINOv2-B + 256-B projection head)\n224×224, görüntü başına\n2 forward pass", "#EAF1F8", BLUE, fs=7.3)
    box(ax, 0.43, 0.20, 0.11, 0.16, "iki embedding'in\nortalaması\n+ L2 norm.", "#F2F2F2", fs=7.3)
    box(ax, 0.58, 0.20, 0.10, 0.16, "256-B\nsorgu\nvektörü", "#D9E8F5", BLUE, fs=7.5, bold=True)
    box(ax, 0.58, 0.04, 0.10, 0.12, "Galeri\n(809 görüntü\nçevrimdışı)", "#EDEDED", GREY, fs=7)
    box(ax, 0.72, 0.10, 0.12, 0.25, "Tam Öklid (L2)\nsıralama\n(brute-force,\nyaklaşık arama yok)", "#FDF1E3", ORANGE, fs=7.2)
    box(ax, 0.875, 0.10, 0.11, 0.25, "Top-K\nsonuçlar\n(Protokol B:\nHit@1 0,807)", "#EAF4E8", GREEN, fs=7.5, bold=True)
    ax.text(0.99, 0.445, "ÇIKARIM (eğitim yok)", ha="right", fontsize=8, fontweight="bold", color=GREEN)
    ax.text(0.585, 0.395, "isteğe bağlı (ikincil sonuç): 3 seed'in embedding'leri √(1/3) ile ağırlıklanıp birleştirilir (768-B);\nkare mesafe üç modelin kare mesafelerinin ortalaması olur", fontsize=6.8, color="#666", style="italic", ha="center")
    arrow(ax, (0.12, 0.28), (0.17, 0.26)); arrow(ax, (0.12, 0.10), (0.17, 0.17)); arrow(ax, (0.39, 0.225), (0.43, 0.27)); arrow(ax, (0.54, 0.28), (0.58, 0.28))
    arrow(ax, (0.68, 0.28), (0.72, 0.26)); arrow(ax, (0.68, 0.10), (0.72, 0.16)); arrow(ax, (0.84, 0.225), (0.875, 0.225))
    save(fig, "arch_final")


# ---------------------------------------------------------------- result figures
def ablation_bar():
    ci = pd.read_csv("artifacts/metrics/protocolB_locked/macro_ci.csv")
    g = lambda r, m="P": ci[(ci.rep == r) & (ci.K == 1) & (ci.metric == m)].iloc[0]
    seeds = lambda pre: [g(f"{pre}_s{i}") for i in range(3)]
    steps = [("ResNet50 donmuş, 1000-B logit", g("BASE_resnet50_frozen_logits1000_raw"), GREY),
             ("ResNet50 donmuş, 2048-B", g("BASE_resnet50_frozen_pooled2048_raw"), GREY),
             ("ResNet50 fine-tuned + ArcFace", g("R1_resnet50_ft_arcface_existing_system"), ORANGE),
             ("DINOv2-S donmuş", g("F_dinov2s_frozen_cls"), TEAL),
             ("DINOv2-B donmuş", g("F_dinov2b_frozen_cls"), TEAL),
             ("DINOv2-S son 4 blok + ArcFace", g("D2_dinoS_p4_arcface_s0"), BLUE),
             ("DINOv2-S son 4 blok + Proxy-Anchor*", seeds("D3_dinoS_p4_proxyanchor"), BLUE),
             ("DINOv2-S son 4 blok + SupCon*", seeds("D4_dinoS_p4_supcon"), BLUE),
             ("DINOv2-B son 4 blok + SupCon* (önerilen)", seeds("D5_dinoB_p4_supcon"), GREEN)]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for i, (n, c, col) in enumerate(steps):
        if isinstance(c, list):
            mean = np.mean([x.macro_mean for x in c]); lo = np.mean([x.ci_lo for x in c]); hi = np.mean([x.ci_hi for x in c])
        else:
            mean, lo, hi = c.macro_mean, c.ci_lo, c.ci_hi
        ax.barh(i, mean, xerr=[[mean - lo], [hi - mean]], color=col, capsize=2.5, height=0.65)
        ax.text(0.01, i, f"{mean:.3f}".replace(".", ","), va="center", ha="left", fontsize=8, color="white", fontweight="bold")
    ax.axvline(0.8, color=RED, ls="--", lw=1); ax.text(0.805, -0.75, "0,80", color=RED, fontsize=7.5)
    ax.set_yticks(range(len(steps)), [s[0] for s in steps], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Hit@1 (sınıf ortalaması; %95 sınıf-bootstrap GA; * = 3 seed ortalaması)", fontsize=8)
    fig.tight_layout(); fig.savefig(O / "ablation_hit1.png", dpi=200); plt.close(fig)


def roc_fig():
    cv = pd.read_pickle("artifacts/metrics/protocolB_locked/roc_curves.pkl")
    show = [("BASE_resnet50_frozen_pooled2048_raw", "ResNet50 donmuş 2048-B", GREY), ("R1_resnet50_ft_arcface_existing_system", "ResNet50 fine-tuned + ArcFace", ORANGE),
            ("F_dinov2b_frozen_cls", "DINOv2-B donmuş", TEAL), ("D2_dinoS_p4_arcface_s0", "DINOv2-S + ArcFace", PURPLE),
            ("D5_dinoB_p4_supcon_s0", "DINOv2-B + SupCon (önerilen)", GREEN)]
    auc = pd.read_csv("artifacts/metrics/protocolB_locked/classification_roc_metrics.csv").set_index("rep").roc_auc_pairwise
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for a, zoom in zip(axs, (False, True)):
        for k, lab, col in show:
            f, t = cv[k]; a.plot(f, t, color=col, lw=1.6, label=f"{lab} (AUC {str(round(auc[k], 3)).replace(".", ",")})" if not zoom else None)
        a.plot([0, 1], [0, 1], color="#BBB", ls=":", lw=1); a.set_xlabel("Yanlış pozitif oranı"); a.grid(alpha=.25)
        if zoom: a.set_xlim(0, 0.1); a.set_ylim(0, 1.0); a.set_title("Yakınlaştırma (YPO ≤ 0,1)", fontsize=8.5)
        else: a.set_xlim(0, 1); a.set_ylim(0, 1.0); a.set_ylabel("Doğru pozitif oranı"); a.set_title("Tüm aralık", fontsize=8.5)
    axs[0].legend(fontsize=6.6, loc="lower right"); fig.tight_layout(); save(fig, "roc")


def pr_fig():
    d = pd.read_csv("artifacts/metrics/protocolB_locked/pr_curves_11pt.csv")
    show = [("BASE_resnet50_frozen_pooled2048_raw", "ResNet50 donmuş 2048-B", GREY), ("R1_resnet50_ft_arcface_existing_system", "ResNet50 fine-tuned + ArcFace", ORANGE),
            ("F_dinov2b_frozen_cls", "DINOv2-B donmuş", TEAL), ("D2_dinoS_p4_arcface_s0", "DINOv2-S + ArcFace", PURPLE),
            ("D5_dinoB_p4_supcon_s0", "DINOv2-B + SupCon (önerilen)", GREEN)]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for k, lab, col in show:
        s = d[d.rep == k]; ax.plot(s.recall, s.interp_precision, color=col, lw=1.7, marker="o", ms=2.8, label=lab)
    ax.set_xlabel("Recall"); ax.set_ylabel("Interpolated precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=.25); ax.legend(fontsize=6.8, loc="upper right"); fig.tight_layout(); save(fig, "pr_curve")


def pk_fig():
    ci = pd.read_csv("artifacts/metrics/protocolB_locked/macro_ci.csv")
    ks = [1, 3, 5, 10, 20, 50, 100]
    show = [("BASE_resnet50_frozen_pooled2048_raw", "ResNet50 donmuş 2048-B", GREY), ("R1_resnet50_ft_arcface_existing_system", "ResNet50 fine-tuned + ArcFace", ORANGE),
            ("F_dinov2b_frozen_cls", "DINOv2-B donmuş", TEAL), ("D5_dinoB_p4_supcon_s0", "DINOv2-B + SupCon (önerilen)", GREEN)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for a, (m, yl) in zip(axs, (("P", "P@K"), ("Hit", "Hit@K"))):
        for k, lab, col in show:
            a.plot(ks, [ci[(ci.rep == k) & (ci.K == x) & (ci.metric == m)].macro_mean.iloc[0] for x in ks], color=col, marker="o", ms=3, lw=1.6, label=lab)
        a.set_xscale("log"); a.set_xlabel("K"); a.set_ylabel(yl); a.set_ylim(0, 1.02); a.grid(alpha=.25)
    axs[0].legend(fontsize=6.6, loc="upper right"); fig.tight_layout(); save(fig, "pk_hitk")


def generalisation():
    s = pd.read_csv("artifacts/metrics/protocolBC_hit1_summary.csv")
    ex = pd.read_csv("artifacts/metrics/external_plantseg/summary_ci.csv")
    ex = ex[(ex.metric == "hit1") & (ex.tier == "strict")].set_index("rep")
    names = {"BASE_resnet50_frozen_pooled2048_raw": "ResNet50 donmuş\n2048-B", "F_dinov2b_frozen_cls": "DINOv2-B\ndonmuş", "HEADLINE": "DINOv2-B + SupCon\n(önerilen)"}
    ext = {"BASE_resnet50_frozen_pooled2048_raw": "BASE_resnet50_frozen_pooled2048_raw", "F_dinov2b_frozen_cls": "F_dinov2b_frozen_cls", "HEADLINE": "HEADLINE_D5ck_mean3seeds"}
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); w = 0.26
    for j, (lab, col) in enumerate([("SegPPD-101 test B (225 sorgu)", BLUE), ("SegPPD-101 test C (142 sorgu)", TEAL), ("Harici PlantSeg (1485 sorgu, 11 sınıf)", ORANGE)]):
        v, lo, hi = [], [], []
        for k in names:
            if j < 2:
                r = s[(s.rep == k) & (s.set.str.startswith("B (" if j == 0 else "C ("))].iloc[0]; v.append(r.hit1); lo.append(r.ci_lo); hi.append(r.ci_hi)
            else:
                r = ex.loc[ext[k]]; v.append(r.macro_mean); lo.append(r.ci_class_lo); hi.append(r.ci_class_hi)
        x = np.arange(len(names)) + (j - 1) * w
        ax.bar(x, v, w, yerr=[np.array(v) - lo, np.array(hi) - v], capsize=2, color=col, label=lab)
        for xi, vi in zip(x, v): ax.text(xi, 0.02, f"{vi:.2f}".replace(".", ","), ha="center", fontsize=7, color="white", fontweight="bold")
    ax.axhline(0.8, color=RED, ls="--", lw=1); ax.set_xticks(range(len(names)), list(names.values()), fontsize=8); ax.set_ylim(0, 1)
    ax.set_ylabel("Hit@1"); ax.legend(fontsize=7, loc="upper left"); fig.tight_layout(); save(fig, "generalisation")


# ------------------------------------------------------------------ additional figures (v2 report)
from segppd_cbir.index import rank  # noqa: E402

SYS5 = [("BASE_resnet50_frozen_pooled2048_raw", "ResNet50 donmuş 2048-B", GREY), ("R1_resnet50_ft_arcface_existing_system", "ResNet50 fine-tuned + ArcFace", ORANGE),
        ("F_dinov2b_frozen_cls", "DINOv2-B donmuş", TEAL), ("D5_dinoB_p4_supcon_s0", "DINOv2-B + SupCon (seed 0)", GREEN)]


def prk_rk():
    L = "artifacts/metrics/protocolB_locked/"
    ci = pd.read_csv(L + "macro_ci.csv"); sm = pd.read_csv(L + "summary_macro_micro.csv")
    ks = [1, 3, 5, 10, 20, 50, 100]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for a, (m, yl) in zip(axs, (("P", "Precision@K"), ("Recall", "Recall@K"))):
        for k, lab, col in SYS5:
            a.plot(ks, [ci[(ci.rep == k) & (ci.K == x) & (ci.metric == m)].macro_mean.iloc[0] for x in ks], color=col, marker="o", ms=3, lw=1.6, label=lab)
        if m == "P":
            ce = sm[(sm.rep == "D5_dinoB_p4_supcon_s0")].set_index("K").macro_ceiling
            a.plot(ks, [ce[x] for x in ks], color="#999", ls="--", lw=1, label="tavan: min(R,K)/K")
        a.set_xscale("log"); a.set_xlabel("K"); a.set_ylabel(yl); a.set_ylim(0, 1.02); a.grid(alpha=.25)
    axs[0].legend(fontsize=6.3, loc="upper right"); fig.tight_layout(); save(fig, "prk_rk"); 


def hit_map():
    L = "artifacts/metrics/protocolB_locked/"
    ci = pd.read_csv(L + "macro_ci.csv")
    g = lambda r, m, K=1: ci[(ci.rep == r) & (ci.K == K) & (ci.metric == m)].iloc[0]
    sd = lambda p: [f"{p}_s{i}" for i in range(3)]
    steps = [("ResNet50 donmuş, 1000-B logit", ["BASE_resnet50_frozen_logits1000_raw"], GREY), ("ResNet50 donmuş, 2048-B", ["BASE_resnet50_frozen_pooled2048_raw"], GREY),
             ("ResNet50 fine-tuned + ArcFace", ["R1_resnet50_ft_arcface_existing_system"], ORANGE), ("DINOv2-S donmuş", ["F_dinov2s_frozen_cls"], TEAL),
             ("DINOv2-B donmuş", ["F_dinov2b_frozen_cls"], TEAL), ("DINOv2-S son 4 blok + ArcFace", ["D2_dinoS_p4_arcface_s0"], BLUE),
             ("DINOv2-S son 4 blok + Proxy-Anchor*", sd("D3_dinoS_p4_proxyanchor"), BLUE), ("DINOv2-S son 4 blok + SupCon*", sd("D4_dinoS_p4_supcon"), BLUE),
             ("DINOv2-B son 4 blok + SupCon* (önerilen)", sd("D5_dinoB_p4_supcon"), GREEN)]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for a, m, xl in ((axs[0], "P", "Hit@1 (= P@1)"), (axs[1], "full_AP", "mAP")):
        for i, (n, reps, col) in enumerate(steps):
            mean = np.mean([g(r, m).macro_mean for r in reps])
            if m == "P":
                lo, hi = np.mean([g(r, m).ci_lo for r in reps]), np.mean([g(r, m).ci_hi for r in reps]); a.barh(i, mean, xerr=[[mean - lo], [hi - mean]], color=col, capsize=2, height=.65)
            else:
                a.barh(i, mean, color=col, height=.65)
            a.text(0.01, i, f"{mean:.3f}".replace(".", ","), va="center", fontsize=7.5, color="white", fontweight="bold")
        a.set_xlim(0, 1); a.set_xlabel(xl, fontsize=8)
    axs[0].axvline(0.8, color=RED, ls="--", lw=1)
    axs[0].set_yticks(range(len(steps)), [s[0] for s in steps], fontsize=7.5); axs[0].invert_yaxis()
    fig.text(0.5, 0.005, "Protokol B kilitli test (225 sorgu, galeri 809); * = 3 seed ortalaması; çubuk uçları %95 sınıf-bootstrap GA (yalnızca Hit@1)", ha="center", fontsize=6.8, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, "hit_map")


def per_class():
    pc = pd.read_csv("artifacts/metrics/protocolB_locked/per_class_hit1.csv", index_col=0)
    m = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).drop_duplicates("class_id").set_index("class_id").raw_class_label
    sd = [f"D5_dinoB_p4_supcon_s{i}" for i in range(3)]
    fin = pc[sd].mean(axis=1)
    sysd = [("BASE_resnet50_frozen_pooled2048_raw", "ResNet50 donmuş 2048-B", GREY), ("R1_resnet50_ft_arcface_existing_system", "ResNet50 fine-tuned", ORANGE),
            ("F_dinov2b_frozen_cls", "DINOv2-B donmuş", TEAL), (None, "DINOv2-B + SupCon (önerilen)", GREEN)]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    ax = axs[0]; w = 0.2
    for j, (k, lab, col) in enumerate(sysd):
        v = (fin if k is None else pc[k]).round(2); cnt = [int(((v * 3).round() == n).sum()) for n in range(4)]
        ax.bar(np.arange(4) + (j - 1.5) * w, cnt, w, color=col, label=lab)
    ax.set_xticks(range(4), ["0/3", "1/3", "2/3", "3/3"]); ax.set_xlabel("Sınıfın 3 sorgusundan doğru bulunan (Hit@1)"); ax.set_ylabel("Sınıf sayısı (toplam 75)")
    ax.legend(fontsize=6.3, loc="upper center", frameon=False)
    ax = axs[1]; low = fin.sort_values().head(12)
    ax.barh(range(len(low)), low.values, color=GREEN, height=.65)
    ax.set_yticks(range(len(low)), [m[c].replace("_", " ") for c in low.index], fontsize=6.8); ax.invert_yaxis(); ax.set_xlim(0, 1)
    for i, v in enumerate(low.values): ax.text(v + 0.01, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=6.8)
    ax.set_xlabel("Hit@1 (3 seed ort.) - en zayıf 12 sınıf")
    fig.tight_layout(); save(fig, "per_class")


def retrieval_examples():
    b = pd.read_csv("artifacts/splits/protocolB_seed4242.csv", dtype={"image_id": str})
    ga, tq = b[b.role_B == "gallery_B"].sort_values("image_id"), b[b.role_B == "test_query_B"].sort_values("image_id")
    gid = ga.image_id.to_numpy(dtype=str); gl = ga.class_id.to_numpy(); ql = tq.class_id.to_numpy()
    z = np.load("artifacts/extensions/B/D5_dinoB_p4_supcon_lr3e-5_s0_embeddings.npz"); pos = {i: k for k, i in enumerate(z["image_id"])}
    f = lambda df: z["emb"][[pos[i] for i in df.image_id]]
    of, _ = rank(f(tq), f(ga), gid, k=5)  # top-5 kept for ranking; figure shows top-4
    rf = np.load("artifacts/features/resnet50_v2_full.npz", allow_pickle=True); rp = {i: k for k, i in enumerate(rf["image_id"].astype(str))}
    g2 = lambda df: rf["pooled"][[rp[i] for i in df.image_id]]
    ob, _ = rank(g2(tq), g2(ga), gid, k=5)
    ok = gl[of[:, 0]] == ql
    rng = np.random.default_rng(11)  # fixed seed: random, not hand-picked
    def pick(pool, n):  # random, but distinct classes so the figure is not 4 copies of one class
        out, used = [], set()
        for q in rng.permutation(pool):
            if ql[q] not in used: out.append(int(q)); used.add(ql[q])
            if len(out) == n: break
        return out
    sel = pick(np.where(ok)[0], 2) + pick(np.where(~ok)[0], 2)
    names = b.drop_duplicates("class_id").set_index("class_id").raw_class_label
    fig, axs = plt.subplots(4, 11, figsize=(7.6, 6.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 0.12, 1, 1, 1, 1, 1]))  # 2x2 blocks of queries, each block = query + top-4 for 2 methods
    for a in axs[:, 5]: a.axis("off")
    for qi, q in enumerate(sel):
        c0 = 0 if qi % 2 == 0 else 6
        for r, (order, tag) in enumerate(((ob, "ResNet50 (temel)"), (of, "DINOv2-B+SupCon"))):
            row = (qi // 2) * 2 + r
            a = axs[row, c0]; a.imshow(Image.open(ROOT / tq.iloc[q].image_path).convert("RGB")); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(False)
            a.set_title(("SORGU:\n" + textwrap.fill(names[ql[q]].replace("_", " "), 14)) if r == 0 else "", fontsize=6.2, loc="left")
            a.set_xlabel(tag, fontsize=6.0, labelpad=2)
            for j in range(4):
                a = axs[row, c0 + j + 1]; gi = order[q, j]; hit = gl[gi] == ql[q]
                a.imshow(Image.open(ROOT / ga.iloc[gi].image_path).convert("RGB")); a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values(): sp.set_visible(True); sp.set_edgecolor(GREEN if hit else RED); sp.set_linewidth(2.4)
                a.set_title(textwrap.fill(names[gl[gi]].replace("_", " "), 14), fontsize=5.6, color=GREEN if hit else RED)
    fig.tight_layout(h_pad=0.4, w_pad=0.25); save(fig, "retrieval_examples")


def pr_roc():
    d = pd.read_csv("artifacts/metrics/protocolB_locked/pr_curves_11pt.csv")
    cv = pd.read_pickle("artifacts/metrics/protocolB_locked/roc_curves.pkl")
    auc = pd.read_csv("artifacts/metrics/protocolB_locked/classification_roc_metrics.csv").set_index("rep").roc_auc_pairwise
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9))
    for k, lab, col in SYS5:
        s_ = d[d.rep == k]; axs[0].plot(s_.recall, s_.interp_precision, color=col, lw=1.6, marker="o", ms=2.5, label=lab)
        f, t = cv[k]; axs[1].plot(f, t, color=col, lw=1.5, label=f"AUC {str(round(auc[k], 3)).replace('.', ',')}"); axs[2].plot(f, t, color=col, lw=1.5)
    axs[0].set_xlabel("Recall"); axs[0].set_ylabel("Interpolated precision"); axs[0].set_title("(a) Precision-recall", fontsize=8.5); axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1.02)
    axs[1].set_xlabel("Yanlış pozitif oranı"); axs[1].set_ylabel("Doğru pozitif oranı"); axs[1].set_title("(b) ROC", fontsize=8.5); axs[1].plot([0, 1], [0, 1], color="#BBB", ls=":", lw=1)
    axs[2].set_xlabel("Yanlış pozitif oranı"); axs[2].set_title("(c) ROC, YPO ≤ 0,1", fontsize=8.5); axs[2].set_xlim(0, 0.1)
    for a in axs: a.grid(alpha=.25); a.set_ylim(0, 1.02)
    axs[0].legend(fontsize=5.8, loc="upper right"); axs[1].legend(fontsize=6, loc="lower right")
    fig.tight_layout(); save(fig, "pr_roc")


if __name__ == "__main__":
    for f in (pr_roc, dataset_samples, class_hist, arch_baseline, arch_final, roc_fig, pr_fig, generalisation, prk_rk, hit_map, per_class, retrieval_examples):
        f(); print("ok", f.__name__)
