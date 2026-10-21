"""Track A report: aggregates the final-stage runs (3 seeds x 3 losses) + the historical mAP-selected SupCon baseline; paired comparisons; figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C

ROOT = Path("artifacts/research_v2/trackA"); SETS = ("dev", "testB_exploratory", "testC_exploratory")
FIG = Path("artifacts/research_v2/figures"); FIG.mkdir(parents=True, exist_ok=True)


def runs():
    sel = json.load(open(ROOT / "selected_configs.json")); out = {}
    for loss, nm in sel.items():
        out[loss] = [ROOT / "runs" / nm.replace("_s0", f"_s{s}") for s in range(3)]
    return out


def hist_baseline():
    """Historical mAP-selected SupCon (tau 0.1): original D5 runs (checkpoints not kept; embeddings kept) evaluated with the same code."""
    S = C.segppd_sets(); res = []
    for s in range(3):
        o, pq = C.evaluate_npz(f"artifacts/extensions/B/D5_dinoB_p4_supcon_lr3e-5_s{s}_embeddings.npz", S); res.append((o, pq))
    return res


def per_query_mean(dirs, nm):
    fr = [pd.read_csv(d / f"per_query_{nm}.csv", dtype={"query_id": str}) for d in dirs]; base = fr[0][["query_id", "class_id"]].copy()
    for c in [c for c in fr[0].columns if c == "ap" or (c.startswith("hit") and c[3:].isdigit()) or (c[0] in "pr" and c[1:].isdigit())]:
        base[c] = np.mean([f[c].to_numpy() for f in fr], axis=0)
    return base, fr


def table():
    R = runs(); rows, pairs = [], []
    hist = hist_baseline(); data = {}
    for loss, dirs in R.items():
        res = [json.load(open(d / "result.json")) for d in dirs]
        for nm in SETS:
            pm, fr = per_query_mean(dirs, nm); data[(loss, nm)] = (pm, fr)
            per_seed = [r["results"][nm] for r in res]
            row = dict(model=f"{loss} (Hit@1-selected)", set=nm, n=len(pm))
            for k in ("macro_hit1", "macro_p5", "macro_p10", "macro_hit5", "macro_hit10", "macro_r5", "macro_r10", "macro_mAP", "micro_hit1", "micro_mAP"):
                v = [p[k] for p in per_seed]; row[k] = float(np.mean(v)); row[k + "_sd"] = float(np.std(v, ddof=1))
            row["hit1_seeds"] = [round(p["macro_hit1"], 4) for p in per_seed]; row["hit1_ci_lo"], row["hit1_ci_hi"] = C.class_boot(pm, "hit1")[1:]
            row["intra_over_inter"] = float(np.mean([p["diag"]["intra_over_inter"] for p in per_seed])); row["nn_intra_closer_frac"] = float(np.mean([p["diag"]["nn_intra_closer_frac"] for p in per_seed]))
            rows.append(row)
    for nm in SETS:
        v = [h[0][nm] for h in hist]; row = dict(model="supcon tau0.1 (HISTORICAL, mAP-selected)", set=nm, n=v[0]["n_queries"])
        for k in ("macro_hit1", "macro_p5", "macro_p10", "macro_hit5", "macro_hit10", "macro_r5", "macro_r10", "macro_mAP", "micro_hit1", "micro_mAP"):
            x = [p[k] for p in v]; row[k] = float(np.mean(x)); row[k + "_sd"] = float(np.std(x, ddof=1))
        row["hit1_seeds"] = [round(p["macro_hit1"], 4) for p in v]; row["intra_over_inter"] = float(np.mean([p["diag"]["intra_over_inter"] for p in v])); rows.append(row)
        pm = pd.concat([h[1][nm] for h in hist]).groupby("query_id").mean(numeric_only=True).reset_index(); data[("hist", nm)] = (pm, [h[1][nm] for h in hist])
    t = pd.DataFrame(rows); t.to_csv(ROOT / "final_summary.csv", index=False)
    # paired comparisons on identical queries (seed-averaged per-query difference, class bootstrap) + per-seed McNemar
    names = list(R) + ["hist"]
    for a, b in (("arcface", "supcon"), ("proxyanchor", "supcon"), ("arcface", "proxyanchor"), ("arcface", "hist"), ("proxyanchor", "hist"), ("supcon", "hist")):
        for nm in SETS:
            pa, fa = data[(a, nm)]; pb, fb = data[(b, nm)]; m = pa.merge(pb, on="query_id", suffixes=("_a", "_b")); d = m.assign(diff=m.hit1_a - m.hit1_b, class_id=m.class_id_a)
            mean, lo, hi = C.class_boot(d.rename(columns={"diff": "hit1"})[["class_id", "hit1"]], "hit1")
            mapd = m.assign(d=m.ap_a - m.ap_b, class_id=m.class_id_a).rename(columns={"d": "hit1"})[["class_id", "hit1"]]; mm, ml, mh = C.class_boot(mapd, "hit1")
            mc = [C.mcnemar_exact(fa[s].hit1.to_numpy(), fb[s].hit1.to_numpy()) for s in range(3)]
            pairs.append(dict(a=a, b=b, set=nm, hit1_diff=mean, hit1_lo=lo, hit1_hi=hi, mAP_diff=mm, mAP_lo=ml, mAP_hi=mh, mcnemar_p_per_seed=[round(x["p"], 4) for x in mc]))
    pd.DataFrame(pairs).to_csv(ROOT / "paired_comparisons.csv", index=False)
    # cost
    cost = []
    for loss, dirs in R.items():
        res = [json.load(open(d / "result.json")) for d in dirs]
        cost.append(dict(loss=loss, train_s_mean=np.mean([r["train_seconds"] for r in res]), epochs_mean=np.mean([r["epochs_run"] for r in res]), peak_vram_gb=max(r["peak_vram_gb"] for r in res), embed_ms_per_image=np.mean([r["embed_ms_per_image"] for r in res]),
                         best_epochs=[r["best_epoch"] for r in res], ckpt_sha256_12=[r["ckpt_sha256"][:12] for r in res]))
    pd.DataFrame(cost).to_csv(ROOT / "cost_table.csv", index=False)
    conf = {loss: [json.load(open(d / "result.json"))["results"]["dev"]["top_confusions"][:5] for d in dirs][0] for loss, dirs in R.items()}
    json.dump(conf, open(ROOT / "dev_top_confusions_seed0.json", "w"), indent=1)
    return t, pd.DataFrame(pairs), data


def figures(t, data):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"arcface": "#9467bd", "proxyanchor": "#ff7f0e", "supcon": "#2ca02c", "hist": "#7f7f7f"}
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), sharex=False)
    for ax, nm, ttl in zip(axs, SETS, ("dev (selection)", "Protocol B test (exploratory)", "Protocol C test (exploratory)")):
        s = t[t.set == nm]
        for _, r in s.iterrows():
            k = "hist" if "HISTORICAL" in r.model else r.model.split()[0]
            ax.errorbar(r.macro_mAP, r.macro_hit1, xerr=r.macro_mAP_sd, yerr=r.macro_hit1_sd, fmt="o", color=col[k], capsize=3, label=r.model.replace(" (Hit@1-selected)", ""))
        ax.set_xlabel("mAP"); ax.set_title(ttl, fontsize=8.5); ax.grid(alpha=.25)
    axs[0].set_ylabel("Hit@1 (class-macro; mean +/- SD of 3 seeds)"); axs[0].legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "A1_hit1_vs_map.png", dpi=180); fig.savefig(FIG / "A1_hit1_vs_map.pdf"); plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.0))
    for j, nm in enumerate(SETS):
        for i, (m, yl) in enumerate((("p", "Precision@K"), ("r", "Recall@K"))):
            for k in ("arcface", "proxyanchor", "supcon", "hist"):
                pm, _ = data[(k, nm)]; ks = (1, 3, 5, 10, 20, 50)
                axs[i, j].plot(ks, [pm[f"{m}{x}"].mean() for x in ks], marker="o", ms=3, color=col[k], label=k if k != "hist" else "supcon (historical)")
            axs[i, j].set_xscale("log"); axs[i, j].set_xlabel("K"); axs[i, j].set_ylabel(yl if j == 0 else ""); axs[i, j].grid(alpha=.25)
        axs[0, j].set_title(nm, fontsize=8.5)
    axs[0, 0].legend(fontsize=6.5, frameon=False); fig.tight_layout(); fig.savefig(FIG / "A2_pk_rk.png", dpi=180); fig.savefig(FIG / "A2_pk_rk.pdf"); plt.close(fig)


if __name__ == "__main__":
    t, p, d = table(); figures(t, d); pd.set_option("display.width", 250)
    print(t[["model", "set", "macro_hit1", "macro_hit1_sd", "macro_mAP", "macro_p10", "hit1_ci_lo", "hit1_ci_hi", "hit1_seeds"]].round(4).to_string(index=False)); print(p.round(4).to_string(index=False))
