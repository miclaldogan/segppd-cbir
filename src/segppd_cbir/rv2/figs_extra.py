"""Extra research_v2 figures: A3 PR curves, C1 error factors, C3 internal-vs-external, C4 global-vs-local reranking, C5 examples (LOCAL ONLY: dataset photographs)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ..index import rank
from . import common as C
from .trackC_diagnose import CACHE, queries

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
FIG = Path("artifacts/research_v2/figures"); TC = Path("artifacts/research_v2/trackC"); TA = Path("artifacts/research_v2/trackA")
COL = {"arcface": "#9467bd", "proxyanchor": "#ff7f0e", "supcon": "#2ca02c"}


def save(fig, name, vector=True):
    fig.savefig(FIG / f"{name}.png", dpi=180)
    if vector: fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def pr_curves(order_rel, R):
    """11-point interpolated precision vs recall for one query set from the relevance matrix (rows = queries)."""
    lv = np.linspace(0, 1, 11); out = np.zeros((len(order_rel), 11))
    for i, rel in enumerate(order_rel):
        tp = np.cumsum(rel); prec, rec = tp / np.arange(1, len(rel) + 1), tp / max(R[i], 1)
        for j, l in enumerate(lv): m = rec >= l - 1e-12; out[i, j] = prec[m].max() if m.any() else 0
    return lv, out


def A3():
    S = C.segppd_sets(); sel = json.load(open(TA / "selected_configs.json")); fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, (nm, qk, gk) in zip(axs, (("Protocol B test (exploratory)", "testB_q", "gallery_B"), ("SegPPD dev", "dev_q", "gallery_B"))):
        for loss, n0 in sel.items():
            curves = []
            for s in range(3):
                d = TA / "runs" / n0.replace("_s0", f"_s{s}"); z = np.load(d / "embeddings.npz"); pos = {i: k for k, i in enumerate(z["image_id"])}; f = lambda df: z["emb"][[pos[i] for i in df.image_id]]
                q, g = S[qk], S[gk]; gid = g.image_id.to_numpy(dtype=str); o, _ = rank(f(q), f(g), gid); gl = g.class_id.to_numpy(); ql = q.class_id.to_numpy()
                rel = (gl[o] == ql[:, None]).astype(float); R = (gl[None, :] == ql[:, None]).sum(1); lv, c = pr_curves(rel, R); curves.append(pd.DataFrame(c).assign(c=ql).groupby("c").mean().mean().to_numpy())
            ax.plot(lv, np.mean(curves, 0), marker="o", ms=3, color=COL[loss], label=loss)
        ax.set_xlabel("Recall"); ax.set_ylabel("Interpolated precision"); ax.set_title(nm, fontsize=8.5); ax.grid(alpha=.25); ax.set_ylim(0, 1.02)
    axs[0].legend(fontsize=7, frameon=False); fig.tight_layout(); save(fig, "A3_pr_curves")


def C1():
    f = lambda n: pd.read_csv(TC / f"diag_by_{n}.csv")
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 3.2))
    t = f("Plant").sort_values("hit1"); axs[0].barh(t.Plant, t.hit1, color="#4C78A8"); axs[0].set_title("by crop (n = %d queries)" % t.n.sum(), fontsize=8.5)
    for i, (n, h) in enumerate(zip(t.n, t.hit1)): axs[0].text(h + .01, i, f"n={n}", va="center", fontsize=6.5)
    t = f("mask_bin"); axs[1].bar(t.mask_bin, t.hit1, color="#72B7B2"); axs[1].set_title("by PlantSeg mask-area ratio", fontsize=8.5); [axs[1].text(i, h + .01, f"n={n}", ha="center", fontsize=6.5) for i, (n, h) in enumerate(zip(t.n, t.hit1))]
    o = pd.read_csv(TC / "diag_by_organ_provisional.csv"); o = o[o.organ.isin(["L", "F", "M", "W"])]; axs[2].bar(o.organ.map({"L": "leaf", "F": "fruit", "M": "mixed", "W": "whole plant"}), o.hit1, color="#F58518")
    [axs[2].text(i, h + .01, f"n={n}", ha="center", fontsize=6.5) for i, (n, h) in enumerate(zip(o.n, o.hit1))]; axs[2].set_title("by organ (PROVISIONAL, 1 annotator, 121 imgs)", fontsize=7.5)
    cf = pd.read_csv(TC / "diag_top_confusions.csv").head(8); axs[3].barh([f"{a} -> {b}".replace("_", " ")[:38] for a, b in zip(cf.t, cf.p)][::-1], cf.n_mean_per_seed[::-1], color="#d62728"); axs[3].set_title("top confusions (queries/seed)", fontsize=8.5); axs[3].tick_params(axis="y", labelsize=6)
    for a in axs[:3]: a.set_ylim(0, 1) if a is not axs[0] else a.set_xlim(0, 1.15)
    axs[1].set_ylabel("Hit@1 (seed mean)")
    for a in axs[1:3]: a.tick_params(axis="x", labelsize=6.5, rotation=25)
    axs[3].set_title("top confusions (queries / seed)", fontsize=8); fig.tight_layout(w_pad=1.2); save(fig, "C1_error_factors")


def C3():
    S = pd.read_csv(TA / "final_summary.csv"); ev = pd.read_csv(TC / "aug_ablation_by_run.csv"); ind = pd.read_csv(TC / "diag_indomain_same_classes.csv")
    pa = S[S.model.str.startswith("proxyanchor")].set_index("set"); sc = S[S.model.str.contains("HISTORICAL")].set_index("set"); base = ev[ev.variant == "base"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); labels = ["SegPPD dev", "Prot. B test", "Prot. C test", "same 11 classes\n(B test)", "External PlantSeg\n(11 classes)"]
    cls11 = set(queries().class_id); pa11 = np.mean([pd.read_csv(TA / "runs" / f"A_pro_alpha32_delta0.3_s{s}" / "per_query_testB_exploratory.csv", dtype={"query_id": str}).query("class_id in @cls11").groupby("class_id").hit1.mean().mean() for s in range(3)])
    ex = pd.read_csv("artifacts/metrics/external_plantseg/summary_ci.csv"); sc_ext = float(ex[(ex.rep == "HEADLINE_D5ck_mean3seeds") & (ex.tier == "strict") & (ex.metric == "hit1")].macro_mean.iloc[0])
    pav = [pa.loc["dev", "macro_hit1"], pa.loc["testB_exploratory", "macro_hit1"], pa.loc["testC_exploratory", "macro_hit1"], pa11, base.ext_macro_hit1.mean()]
    scv = [sc.loc["dev", "macro_hit1"], sc.loc["testB_exploratory", "macro_hit1"], sc.loc["testC_exploratory", "macro_hit1"], ind[ind.set == "testB_exploratory"].hit1_11classes.mean(), sc_ext]
    x = np.arange(5); ax.bar(x - .2, pav, .4, color="#ff7f0e", label="Proxy-Anchor (3 seeds)"); ax.bar(x + .2, scv, .4, color="#2ca02c", label="SupCon tau 0.1, historical D5ck (3 seeds)")
    for xi, v in list(zip(x - .2, pav)) + list(zip(x + .2, scv)):
        if not np.isnan(v): ax.text(xi, v + .01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, fontsize=7); ax.set_ylim(0.4, 0.9); ax.set_ylabel("Hit@1 (class-macro)"); ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Internal vs external Hit@1 (all exploratory except dev); \"same 11 classes\" = in-domain, external classes only", fontsize=7.5); fig.tight_layout(); save(fig, "C3_internal_vs_external")


def C4():
    a = pd.read_csv(TC / "local_rerank_PA_perquery.csv"); fig, axs = plt.subplots(1, 4, figsize=(10.2, 3.0)); rows = []
    for ax, st in zip(axs, ("dev", "testB_exploratory", "testC_exploratory", "external_plantseg_exploratory")):
        g = a[a.set == st]; vals = {}
        for (M, al), d in g.groupby(["M", "alpha"]):
            per = d.groupby(["seed", "class_id"]).hit1.mean().groupby("class_id").mean(); vals[(M, al)] = (per.mean(), d.groupby("seed").hit1.mean().std(ddof=1))
        lab = {(0, 1.0): "global only", (20, 0.5): "rerank M=20\nalpha=.5", (20, 0.75): "rerank M=20\nalpha=.75", (50, 0.5): "rerank M=50\nalpha=.5"}
        for i, k in enumerate(lab): ax.bar(i, vals[k][0], yerr=vals[k][1], capsize=3, color=["#7f7f7f", "#1f77b4", "#17becf", "#bcbd22"][i])
        ax.set_xticks(range(4), list(lab.values()), fontsize=6.3); ax.set_ylim(min(v[0] for v in vals.values()) - .05, max(v[0] for v in vals.values()) + .03); ax.set_title(st.replace("_exploratory", " (expl.)"), fontsize=8)
        base = g[(g.M == 0)].groupby("qi").hit1.mean(); 
        for (M, al) in ((20, 0.5), (20, 0.75), (50, 0.5)):
            d = g[(g.M == M) & (g.alpha == al)]; diff = (d.groupby(["class_id", "qi"]).hit1.mean() - g[g.M == 0].groupby(["class_id", "qi"]).hit1.mean()).rename("hit1").reset_index()
            m, lo, hi = C.class_boot(diff[["class_id", "hit1"]], "hit1"); rows.append(dict(set=st, M=M, alpha=al, hit1_diff=m, lo=lo, hi=hi))
    axs[0].set_ylabel("Hit@1 (3-seed mean +/- SD)"); fig.tight_layout(); save(fig, "C4_global_vs_local_rerank"); pd.DataFrame(rows).to_csv(TC / "local_rerank_paired.csv", index=False); print(pd.DataFrame(rows).round(4).to_string(index=False))


def C5():
    q = queries(); S = C.segppd_sets(); ga = S["gallery_B"]; gid = ga.image_id.to_numpy(dtype=str); gl = ga.class_id.to_numpy(); d = TA / "runs" / "A_pro_alpha32_delta0.3_s0"
    z = np.load(d / "embeddings.npz"); pos = {i: k for k, i in enumerate(z["image_id"])}; G = z["emb"][[pos[i] for i in gid]]; E = np.load(TC / "runs" / "C_pa_augcolor_s0" / "ext_emb.npz")["emb"] if False else np.load(d / "ext_emb.npz")["emb"]
    o, _ = rank(E, G, gid, k=4); names = dict(zip(S["train"].class_id, S["train"].raw_class_label)); ok = gl[o[:, 0]] == q.class_id.to_numpy(); rng = np.random.default_rng(5)
    org = pd.read_csv(TC / "organ" / "organ_labels.csv").set_index("Name").organ; pick = list(rng.choice(np.where(ok)[0], 2, replace=False)) + list(rng.choice(np.where(~ok)[0], 2, replace=False))
    gf = [i for i in np.where((q.Disease == "grape black rot") & ~ok)[0] if org.get(q.Name[i]) == "F"][:1]; tm = [i for i in np.where((q.Disease == "tomato leaf mold") & ~ok)[0]][:1]; pick += gf + tm
    fig, axs = plt.subplots(len(pick), 5, figsize=(6.4, 1.35 * len(pick)))
    for r, qi in enumerate(pick):
        axs[r, 0].imshow(Image.open(CACHE / (Path(q.Name[qi]).stem + ".png")).convert("RGB")); axs[r, 0].axis("off")
        axs[r, 0].set_title("PlantSeg query:\n" + textwrap.fill(names[q.class_id[qi]].replace("_", " "), 20), fontsize=5.3, loc="left")
        for j in range(4):
            gi = o[qi, j]; hit = gl[gi] == q.class_id[qi]; a = axs[r, j + 1]; a.imshow(Image.open("data/extracted/SegPPD-101/" + ga.iloc[gi].image_path).convert("RGB")); a.set_xticks([]); a.set_yticks([])
            [sp.set_edgecolor("#2ca02c" if hit else "#d62728") or sp.set_linewidth(2) for sp in a.spines.values()]; a.set_title(names[gl[gi]].replace("_", " ")[:26], fontsize=5, color="#2ca02c" if hit else "#d62728")
    fig.tight_layout(h_pad=0.4, w_pad=0.2); save(fig, "C5_external_examples_LOCAL_ONLY", vector=False)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]: globals()[f]()
