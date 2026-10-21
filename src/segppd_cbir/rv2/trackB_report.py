"""Track B aggregation over seeds (M3/M4/M5 repeated for every available seed; M1/M2 are deterministic) + figures. python -m segppd_cbir.rv2.trackB_report"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .plantseg_eval import OUT, fit_lr, img_boot, load_protocol, mcnemar, retrieval, summarize

FIG = Path("artifacts/research_v2/figures"); FIG.mkdir(parents=True, exist_ok=True)


def seeds_available():
    return [s for s in range(3) if (OUT / "runs" / f"B_M3_metric_proxyanchor_s{s}" / "embeddings.npz").exists() and (OUT / "runs" / f"B_M5_ce_finetune_s{s}" / "outputs.npz").exists()]


def main():
    P = load_protocol(); ytr, ydv, yte = (P[r].label.to_numpy() for r in ("train", "dev", "test")); gid = P["train"].Name.to_numpy(dtype=str)
    rows, corr, preds = [], {}, {}
    fz = pd.read_csv(OUT / "results_frozen.csv")
    for k, nm in (("M1", "M1 frozen + linear"), ("M2", "M2 frozen + CBIR")):
        c = pd.read_csv(OUT / f"pred_{k}.csv"); corr[(k, 0)] = c.correct.to_numpy(); preds[(k, 0)] = c.pred.to_numpy()
        r = fz[fz.method.str.startswith(k)].iloc[0].to_dict(); r.update(model=k, seed=0); rows.append(r)
    for s in seeds_available():
        Z = np.load(OUT / "runs" / f"B_M3_metric_proxyanchor_s{s}" / "embeddings.npz"); t0 = __import__("time").time(); ex = retrieval(Z["test"], Z["train"], yte, ytr, gid); lat = (__import__("time").time() - t0) / len(yte)
        r, pc, c = summarize("M3", yte, ex["pred"], ex); r.update(model="M3", seed=s, search_ms_per_query=1e3 * lat); rows.append(r); corr[("M3", s)] = c; preds[("M3", s)] = ex["pred"]
        fn, C, sec, dacc = fit_lr(Z["train"], ytr, Z["dev"], ydv); pr = fn(Z["test"]); r, pc4, c = summarize("M4", yte, pr); r.update(model="M4", seed=s, best_C=C, fit_seconds=sec); rows.append(r); corr[("M4", s)] = c; preds[("M4", s)] = pr
        o = np.load(OUT / "runs" / f"B_M5_ce_finetune_s{s}" / "outputs.npz"); pr = o["logits_test"].argmax(1); r, pc5, c = summarize("M5", yte, pr); r.update(model="M5", seed=s); rows.append(r); corr[("M5", s)] = c; preds[("M5", s)] = pr
        if s == 0: pc.to_csv(OUT / "per_class_M3_s0.csv", index=False); pc5.to_csv(OUT / "per_class_M5_s0.csv", index=False); pc4.to_csv(OUT / "per_class_M4_s0.csv", index=False)
    R = pd.DataFrame(rows); R.to_csv(OUT / "results_by_seed.csv", index=False)
    agg = R.groupby("model")[["accuracy", "macro_f1", "macro_recall", "macro_precision", "mAP", "p5", "p10", "hit10"]].agg(["mean", "std"]); agg.to_csv(OUT / "results_seed_mean_sd.csv")
    pairs = []
    for a, b in (("M3", "M4"), ("M3", "M5"), ("M4", "M5"), ("M3", "M1"), ("M4", "M1"), ("M5", "M1"), ("M3", "M2")):
        for s in seeds_available():
            ca = corr[(a, s if a not in ("M1", "M2") else 0)]; cb = corr[(b, s if b not in ("M1", "M2") else 0)]; pairs.append(dict(a=a, b=b, seed=s, acc_a=float(ca.mean()), acc_b=float(cb.mean()), **mcnemar(ca, cb)))
    pp = pd.DataFrame(pairs); pp.to_csv(OUT / "paired_by_seed.csv", index=False)
    pd.set_option("display.width", 250); print(agg.round(4).to_string()); print(pp.round(4).to_string(index=False))
    # confusion pairs (seed 0 of M3 and M5) over the 98 classes
    names = P["train"].drop_duplicates("label").set_index("label").Disease; cp = {}
    for k in ("M3", "M4", "M5"):
        d = pd.DataFrame(dict(t=yte, p=preds[(k, 0)])); e = d[d.t != d.p].groupby(["t", "p"]).size().sort_values(ascending=False).head(12)
        cp[k] = [dict(true=names[a], pred=names[b], n=int(n)) for (a, b), n in e.items()]
    json.dump(cp, open(OUT / "top_confusions_seed0.json", "w"), indent=1)
    figures(R, corr, preds, P, names)


def figures(R, corr, preds, P, names):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    lab = {"M1": "M1 frozen +\nlinear", "M2": "M2 frozen +\nCBIR 1-NN", "M3": "M3 metric-learned\n+ CBIR 1-NN", "M4": "M4 linear on\nM3 embedding", "M5": "M5 CE\nfine-tuning"}
    col = {"M1": "#72B7B2", "M2": "#9D9D9D", "M3": "#2ca02c", "M4": "#4C78A8", "M5": "#F58518"}
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, met, yl in ((axs[0], "accuracy", "Top-1 accuracy (= CBIR Hit@1 for M2/M3)"), (axs[1], "macro_f1", "Macro-F1")):
        for i, m in enumerate(lab):
            v = R[R.model == m][met].to_numpy(); ax.bar(i, v.mean(), color=col[m], yerr=(v.std(ddof=1) if len(v) > 1 else 0), capsize=3); ax.scatter([i] * len(v), v, color="k", s=8, zorder=3)
            ax.text(i, v.mean() + 0.012, f"{v.mean():.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(5), list(lab.values()), fontsize=7); ax.set_ylim(0.5, 0.95); ax.set_ylabel(yl, fontsize=8)
    fig.suptitle("PlantSeg protocol P: 98 classes, 1,646 identical test images (dots = seeds; M1/M2 deterministic)", fontsize=8.5); fig.tight_layout(); fig.savefig(FIG / "B1_classification_vs_cbir.png", dpi=180); fig.savefig(FIG / "B1_classification_vs_cbir.pdf"); plt.close(fig)
    # per-class recall: CBIR vs classifier (seed 0)
    pc3, pc5 = pd.read_csv(OUT / "per_class_M3_s0.csv"), pd.read_csv(OUT / "per_class_M5_s0.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8)); ax = axs[0]; ax.scatter(pc5.recall, pc3.recall, s=10, color="#2ca02c"); ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("per-class recall, M5 (CE fine-tuning)"); ax.set_ylabel("per-class recall, M3 (CBIR)"); ax.set_title("98 classes, seed 0", fontsize=8.5)
    d = (pc3.recall - pc5.recall); idx = np.r_[d.sort_values().index[:6], d.sort_values().index[-6:]]; ax = axs[1]
    ax.barh(range(12), d[idx].to_numpy(), color=["#d62728" if x < 0 else "#2ca02c" for x in d[idx]]); ax.set_yticks(range(12), [names[int(pc3.label[i])][:34] for i in idx], fontsize=6.5); ax.set_xlabel("recall(M3 CBIR) - recall(M5 classifier)")
    fig.tight_layout(); fig.savefig(FIG / "B2_per_class_cbir_vs_classifier.png", dpi=180); fig.savefig(FIG / "B2_per_class_cbir_vs_classifier.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
