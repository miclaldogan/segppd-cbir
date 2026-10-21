"""Experiment B evaluation (CPU): E1 PlantSeg target, E2 SegPPD target, E3 shared-class PlantSeg queries into the SegPPD gallery, for S0/T0/X1/X2/X3 x seeds.
Per-query files, class-bootstrap CIs, paired differences on identical queries. python -m segppd_cbir.rv2.cross_eval"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common as C
from .plantseg_eval import retrieval

V = Path("artifacts/research_v3/cross"); FIG = Path("artifacts/research_v3/figures"); FIG.mkdir(exist_ok=True)
M = pd.read_csv(V / "label_alignment.csv"); uni = {(r.source, r.source_label): r.unified for r in M.itertuples()}; SH = set(M[M.shared].unified)
S = C.segppd_sets(); P = pd.read_csv("artifacts/research_v2/trackB/protocol_P.csv"); Ptr, Pte = P[P.role == "train"], P[P.role == "test"]
CONDS = ["S0", "T0", "X1", "X2", "X3"]; NAMES = {"S0": "S0 SegPPD-only", "T0": "T0 PlantSeg-only", "X1": "X1 karışık", "X2": "X2 karışık+alan dengeli", "X3": "X3 karışık+renk aug."}


def load(cond, s):
    p = V / "emb_single" / f"{cond}_s{s}.npz" if cond in ("S0", "T0") else V / "runs" / f"cross_{cond}_s{s}" / "embeddings.npz"
    return np.load(p) if p.exists() else None


def evaluate():
    seg_pos = None; rows, pqs = [], []
    ql_p = np.array([uni[("ps", l)] for l in Pte.label]); gl_p = np.array([uni[("ps", l)] for l in Ptr.label]); in_sh = np.isin(ql_p, list(SH))
    gl_s = np.array([uni[("seg", c)] for c in S["gallery_B"].class_id]); ql_s = np.array([uni[("seg", c)] for c in S["testB_q"].class_id])
    for cond in CONDS:
        for s in range(3):
            z = load(cond, s)
            if z is None: print("missing", cond, s); continue
            sp = {i: k for k, i in enumerate(z["seg_id"])}; pp = {n: k for k, n in enumerate(z["ps_name"])}
            S_ = lambda df: z["seg"][[sp[i] for i in df.image_id]]; P_ = lambda df: z["ps"][[pp[n] for n in df.Name]]
            # E1
            ex = retrieval(P_(Pte), P_(Ptr), ql_p, gl_p, Ptr.Name.to_numpy(dtype=str))
            pqs.append(pd.DataFrame(dict(eval="E1", cond=cond, seed=s, qid=Pte.Name.to_numpy(), truth=ql_p, pred=ex["pred"], top1=ex["top1_ref"], ap=ex["ap"], p10=ex["p10"], shared=in_sh)))
            # E2
            ga, tq = S["gallery_B"], S["testB_q"]; o, _ = C.per_query(S_(tq), S_(ga), ql_s, gl_s, ga.image_id.to_numpy(dtype=str), tq.image_id.to_numpy(dtype=str))
            pqs.append(pd.DataFrame(dict(eval="E2", cond=cond, seed=s, qid=o.query_id, truth=ql_s, pred=o.pred, top1=o.top1_id, ap=o.ap, p10=o.p10, shared=np.isin(ql_s, list(SH)))))
            # E3
            q3 = Pte[in_sh]; ex3 = retrieval(P_(q3), S_(ga), ql_p[in_sh], gl_s, ga.image_id.to_numpy(dtype=str))
            pqs.append(pd.DataFrame(dict(eval="E3", cond=cond, seed=s, qid=q3.Name.to_numpy(), truth=ql_p[in_sh], pred=ex3["pred"], top1=ex3["top1_ref"], ap=ex3["ap"], p10=ex3["p10"], shared=True)))
    pq = pd.concat(pqs); pq["correct"] = (pq.truth == pq.pred).astype(int); pq.to_csv(V / "per_query_predictions.csv.gz", index=False); return pq


def main():
    pq = evaluate(); rng = np.random.default_rng(42); out, pair = [], []
    views = {"E1 all (98 sınıf)": ("E1", False), "E1 shared-10": ("E1", True), "E2 (75 sınıf)": ("E2", False), "E3 shared-10 -> SegPPD galeri": ("E3", True)}
    for vn, (ev, sh) in views.items():
        d = pq[(pq["eval"] == ev) & (pq.shared if sh else True)]
        for cond in CONDS:
            dd = d[d.cond == cond]
            if dd.empty: continue
            per_seed = dd.groupby(["seed", "truth"])[["correct", "ap", "p10"]].mean().groupby("seed").mean(); cm = dd.groupby("truth")[["correct", "ap", "p10"]].mean()  # seed-averaged per class
            b = cm.correct.to_numpy()[rng.integers(0, len(cm), (2000, len(cm)))].mean(1)
            out.append(dict(view=vn, cond=cond, n_queries=int(dd[dd.seed == dd.seed.min()].shape[0]), n_classes=len(cm), hit1_class_macro=float(cm.correct.mean()), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)), seed_sd=float(per_seed.correct.std(ddof=1)) if len(per_seed) > 1 else np.nan,
                            hit1_query_weighted=float(dd.groupby("seed").correct.mean().mean()), mAP=float(per_seed.ap.mean()), P10=float(per_seed.p10.mean()), n_seeds=dd.seed.nunique()))
        for a, bb in (("X1", "T0"), ("X1", "S0"), ("X2", "X1"), ("X3", "X1"), ("T0", "S0")):
            da, db = d[d.cond == a].groupby("truth").correct.mean(), d[d.cond == bb].groupby("truth").correct.mean()
            if da.empty or db.empty: continue
            diff = (da - db).dropna().to_numpy(); bt = diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(1); pair.append(dict(view=vn, a=a, b=bb, diff=float(diff.mean()), lo=float(np.quantile(bt, .025)), hi=float(np.quantile(bt, .975)), n_classes=len(diff)))
    R = pd.DataFrame(out); R.to_csv(V / "results_summary.csv", index=False); Pp = pd.DataFrame(pair); Pp.to_csv(V / "paired_differences.csv", index=False)
    pd.set_option("display.width", 250); print(R.round(3).to_string()); print(Pp.round(3).to_string())
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.9), sharey=False); col = {"S0": "#9D9D9D", "T0": "#4C78A8", "X1": "#F58518", "X2": "#E45756", "X3": "#54A24B"}
    for ax, vn in zip(axs, views):
        r = R[R.view == vn].set_index("cond").reindex(CONDS).dropna(subset=["hit1_class_macro"])
        for i, (c, row) in enumerate(r.iterrows()): ax.bar(i, row.hit1_class_macro, color=col[c], yerr=[[row.hit1_class_macro - row.lo], [row.hi - row.hit1_class_macro]], capsize=2, error_kw=dict(lw=.8))
        ax.set_xticks(range(len(r)), list(r.index), fontsize=6.5); ax.set_title(vn, fontsize=6.8); ax.grid(axis="y", alpha=.25); ax.set_ylim(0, 1)
    axs[0].set_ylabel("Hit@1 (sınıf-makro, 3 tohum ort.)", fontsize=7); fig.tight_layout(); fig.savefig(FIG / "B1_source_conditions.png", dpi=200); fig.savefig(FIG / "B1_source_conditions.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
