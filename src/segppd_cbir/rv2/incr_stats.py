"""Experiment A statistics + figures from per_query_predictions.csv.gz. Class-macro Hit@1 over novel classes (each class novel exactly once), averaged over 5 support draws;
class-level bootstrap (2000) CIs; paired differences on identical classes/queries/supports."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("artifacts/research_v3/incr"); FIG = Path("artifacts/research_v3/figures"); FIG.mkdir(exist_ok=True)
pq = pd.read_csv(R / "per_query_predictions.csv.gz"); cf = pd.read_csv(R / "class_folds.csv").set_index("label")
SH = (1, 3, 5, 10); rng = np.random.default_rng(42)
# per (enc, scen, method, k, class): mean over draws and queries of correct / hit5 / ap...
g = pq.groupby(["enc", "scen", "method", "k", "truth"]).agg(hit1=("correct", "mean"), hit5=("hit5", "mean"), ap=("ap", "mean"), p5=("p5", "mean"), r10=("r10", "mean")).reset_index()
classes = np.sort(g.truth.unique()); assert len(classes) == 98, len(classes)
B = rng.integers(0, len(classes), (2000, len(classes)))


def ci(v):  # v indexed by `classes`
    b = v[B].mean(1); return float(v.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def vec(enc, scen, method, k, col="hit1"):
    d = g[(g.enc == enc) & (g.scen == scen) & (g.method == method) & (g.k == k)].set_index("truth")[col]; return d.reindex(classes).to_numpy()


rows = []
for enc in ("base-trained", "frozen-DINOv2-B"):
    for scen in ("S1", "S2"):
        for m in ("CBIR", "NCP", "LR"):
            for k in SH:
                for col in (("hit1", "hit5", "ap", "p5", "r10") if m == "CBIR" else ("hit1", "hit5")):
                    v = vec(enc, scen, m, k, col); mu, lo, hi = ci(v); rows.append(dict(enc=enc, scen=scen, method=m, k=k, metric=col, mean=mu, lo=lo, hi=hi, n_classes=len(v)))
T = pd.DataFrame(rows); T.to_csv(R / "summary_by_shot.csv", index=False)
# chance levels
nov_n = cf.groupby("fold").size().mean(); print("chance S1 ~", 1 / nov_n, " S2 ~", 1 / 98)
# paired differences
pr = []
for enc in ("base-trained", "frozen-DINOv2-B"):
    for scen in ("S1", "S2"):
        for k in SH:
            for a, b in (("CBIR", "NCP"), ("CBIR", "LR"), ("NCP", "LR")):
                d = vec(enc, scen, a, k) - vec(enc, scen, b, k); mu, lo, hi = ci(d); pr.append(dict(enc=enc, scen=scen, k=k, a=a, b=b, diff=mu, lo=lo, hi=hi))
    for scen in ("S1", "S2"):
        for k in SH:
            for m in ("CBIR", "NCP", "LR"):
                d = vec("base-trained", scen, m, k) - vec("frozen-DINOv2-B", scen, m, k); mu, lo, hi = ci(d); pr.append(dict(enc="base-trained minus frozen", scen=scen, k=k, a=m, b=m + "(frozen)", diff=mu, lo=lo, hi=hi))
PR = pd.DataFrame(pr).drop_duplicates(); PR.to_csv(R / "paired_differences.csv", index=False)
# k trend (CBIR S2 base-trained): 1->10 gain
# base check
cb = pd.read_csv(R / "cost_and_base_check.csv"); bc = cb[cb.scen == "BASE_CHECK"].groupby("enc")[["base_hit1_before", "base_hit1_after"]].mean(); bc.to_csv(R / "base_interference.csv")
cost = cb[cb.scen.isin(["S1", "S2"])].groupby(["enc", "scen", "k"])[["n_gallery", "gallery_bytes_f32", "ncp_bytes_f32", "lr_bytes_f32", "concat_seconds", "ncp_seconds", "lr_fit_seconds"]].mean().reset_index(); cost.to_csv(R / "cost_summary.csv", index=False)
# oracle: Track B M3 (labels of these classes seen in training) on the same novel classes, seed-0 predictions
o = pd.read_csv("artifacts/research_v2/trackB/pred_M3.csv"); oc = o.groupby("truth").correct.mean().reindex(classes).to_numpy(); orc = ci(oc)
# per-fold consistency
fold_of = pq.drop_duplicates("truth").set_index("truth").fold; pf = []
for f in range(5):
    cl = classes[[fold_of[c] == f for c in classes]]
    for k in SH:
        v = g[(g.enc == "base-trained") & (g.scen == "S2") & (g.method == "CBIR") & (g.k == k) & g.truth.isin(cl)].hit1.mean(); pf.append(dict(fold=f, k=k, cbir_S2_hit1=v))
pd.DataFrame(pf).to_csv(R / "per_fold_cbir_S2.csv", index=False)
# per-class k=5 table
pc = pd.DataFrame(dict(label=classes, fold=[fold_of[c] for c in classes]))
for k in (1, 5, 10): pc[f"CBIR_S2_k{k}"] = vec("base-trained", "S2", "CBIR", k)
pc["oracle_M3"] = oc; names = pd.read_csv("artifacts/research_v2/trackB/protocol_P.csv").drop_duplicates("label").set_index("label").Disease; pc["Disease"] = pc.label.map(names); pc.to_csv(R / "per_class_novel.csv", index=False)
json.dump(dict(oracle_M3_on_same_classes=orc, chance_S1=1 / nov_n, chance_S2=1 / 98), open(R / "reference_levels.json", "w"))
# figures
C = {"CBIR": "#4C78A8", "NCP": "#F58518", "LR": "#54A24B"}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for ax, scen, ttl in zip(axs, ("S2", "S1"), ("Açık galeri (98 sınıf; temel + yeni)", "Yalnızca yeni sınıflar (~20 sınıf)")):
    for m in ("CBIR", "NCP", "LR"):
        d = T[(T.enc == "base-trained") & (T.scen == scen) & (T.method == m) & (T.metric == "hit1")].sort_values("k"); ax.errorbar(d.k, d["mean"], yerr=[d["mean"] - d.lo, d.hi - d["mean"]], marker="o", ms=4, lw=1.4, capsize=2, color=C[m], label=m)
    d = T[(T.enc == "frozen-DINOv2-B") & (T.scen == scen) & (T.method == "CBIR") & (T.metric == "hit1")].sort_values("k"); ax.plot(d.k, d["mean"], "--", color="#999", label="CBIR, donmuş DINOv2-B")
    ax.axhline(orc[0], color="#E45756", ls=":", lw=1, label="üst sınır (sınıflar etiketli eğitildi)") if scen == "S2" else None
    ax.set_xscale("log"); ax.set_xticks(SH, [str(k) for k in SH]); ax.set_xlabel("Yeni sınıf başına destek görüntüsü (k)"); ax.set_title(ttl, fontsize=8.5); ax.grid(alpha=.25); ax.set_ylim(0, 1)
axs[0].set_ylabel("Hit@1 (sınıf-makro, yeni sınıflar)"); axs[1].legend(*axs[0].get_legend_handles_labels(), fontsize=6, loc="lower right"); fig.tight_layout(); fig.savefig(FIG / "A1_support_size.png", dpi=200); fig.savefig(FIG / "A1_support_size.pdf"); plt.close(fig)
# cost figure: insertion time
fig, ax = plt.subplots(figsize=(4.6, 2.9)); cs = cost[(cost.enc == "base-trained") & (cost.scen == "S2")]
ax.plot(cs.k, cs.lr_fit_seconds, "o-", color=C["LR"], label="LR: sınıflandırıcıyı yeniden eğit"); ax.plot(cs.k, cs.ncp_seconds + 1e-6, "o-", color=C["NCP"], label="NCP: prototipleri yeniden hesapla"); ax.plot(cs.k, cs.concat_seconds + 1e-6, "o-", color=C["CBIR"], label="CBIR: galeriye ekle (concat)")
ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xticks(SH, [str(k) for k in SH]); ax.set_xlabel("k"); ax.set_ylabel("saniye (kodlayıcı dışı)"); ax.grid(alpha=.25); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "A2_update_cost.png", dpi=200); fig.savefig(FIG / "A2_update_cost.pdf"); plt.close(fig)
pd.set_option("display.width", 250)
print(T[(T.metric == "hit1")].assign(v=lambda d: d["mean"].round(3).astype(str) + " [" + d.lo.round(3).astype(str) + "," + d.hi.round(3).astype(str) + "]").pivot_table(index=["enc", "scen", "method"], columns="k", values="v", aggfunc="first").to_string())
print(T[(T.method == "CBIR") & (T.enc == "base-trained") & (T.scen == "S2") & (T.k.isin([1, 10]))][["k", "metric", "mean"]].round(3).to_string())
print("oracle", orc); print(bc.round(4)); print(cost[cost.enc == "base-trained"].round(4).to_string()); print(PR[(PR.enc == "base-trained") & (PR.scen == "S2")].round(3).to_string()); print(pd.DataFrame(pf).pivot(index="fold", columns="k", values="cbir_S2_hit1").round(3))
