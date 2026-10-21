"""Seed-averaged summaries, count-matched subsampling, sensitivity and the single comparison table for the ChatLeafDisease-matched benchmark."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

M = "artifacts/metrics/chatleaf_matched/"
SUP = ["s0", "s1", "s2"]
pq = pd.read_csv(M + "trackS_per_query.csv")
au = json.load(open("artifacts/chatleaf/audit_summary.json")); ct = pd.read_csv("artifacts/chatleaf/audit_category_table.csv")
rng = np.random.default_rng(0)


def seedmean(df, col="correct"):
    return df[df.feature.isin(SUP)].groupby("name")[col].mean()


def ci(x, n=2000, seed=42):
    r = np.random.default_rng(seed); b = x[r.integers(0, len(x), (n, len(x)))].mean(1); return float(np.quantile(b, .025)), float(np.quantile(b, .975))


out = {}
for var in ("crop", "agnostic"):
    for meth in ("1nn", "lr"):
        s = pq[(pq.variant == var) & (pq.method == meth)]
        x = seedmean(s); cats = s.drop_duplicates("name").set_index("name").category.loc[x.index]
        lo, hi = ci(x.to_numpy()); pc = x.groupby(cats).mean()
        out[f"S|{var}|{meth}|supcon_mean3"] = dict(n=len(x), acc=float(x.mean()), lo=lo, hi=hi, macro_class_recall=float(pc.mean()), per_class={k: float(v) for k, v in pc.items()},
                                                 per_seed=[float(s[s.feature == f].correct.mean()) for f in SUP])
        if meth == "1nn":
            for col in ("p5", "p10", "ap"): out[f"S|{var}|1nn|supcon_mean3"][col] = float(seedmean(s, col).mean())
        f = pq[(pq.variant == var) & (pq.method == meth) & (pq.feature == "frozen")]; out[f"S|{var}|{meth}|frozen"] = dict(n=len(f), acc=float(f.correct.mean()), ci=ci(f.correct.to_numpy()))
# paired test on seed-averaged correctness is not binary; use the per-seed McNemar from trackS_paired.csv; additionally pool the 3 seeds' discordant counts (conservative: seeds are dependent, so reported only descriptively)
pr = pd.read_csv(M + "trackS_paired.csv"); out["paired_S"] = pr.to_dict("records")
# count-matched subsample (Table S2 counts, capped by availability), 500 draws
cat_n = dict(zip(ct.plantseg_name, ct.paper_count_S2)); s = pq[(pq.variant == "crop") & (pq.method == "1nn") & pq.feature.isin(SUP)]
x = seedmean(s).rename("c").reset_index().merge(s.drop_duplicates("name")[["name", "category"]], on="name")
res = []
for d in range(500):
    pick = pd.concat([g.sample(min(len(g), cat_n[c]), random_state=int(rng.integers(1 << 30))) for c, g in x.groupby("category")]); res.append((pick.c.mean(), pick.groupby("category").c.mean().mean()))
res = np.array(res); out["S_countmatched_supcon_1nn"] = dict(n_per_draw=int(sum(min(len(g), cat_n[c]) for c, g in x.groupby("category"))), micro_mean=float(res[:, 0].mean()), micro_sd=float(res[:, 0].std()), macro_mean=float(res[:, 1].mean()))
# sensitivity (post hoc, NOT part of the registered headline): drop grape black rot (PlantSeg images are mostly berries, SegPPD references are leaves)
y = x[x.category != "grape black rot"]; out["S_excl_grape_supcon_1nn_post_hoc"] = dict(n=len(y), acc=float(y.c.mean()), ci=ci(y.c.to_numpy()))
# ChatLD context on the same 5 categories (different images!)
cov = ct[ct.segppd_class.notna()]
out["chatld_context"] = dict(reported_overall=0.773, S5_implied_micro=au["implied_micro_accuracy"], S5_macro_recall=au["macro_recall"], on_covered_micro=au["chatld_micro_on_covered"], on_covered_macro_recall=au["chatld_macro_recall_on_covered"],
                             per_class_recall={r.plantseg_name: r.chatld_recall_S5 for r in cov.itertuples()})
t = pd.read_csv(M + "trackT_summary.csv"); out["trackT"] = t.to_dict("records")
json.dump(out, open(M + "summary_all.json", "w"), indent=1, default=float)
g = lambda k: out[k]
print("S crop 1nn supcon:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g("S|crop|1nn|supcon_mean3").items() if k != "per_class"})
print("S crop lr supcon:", round(g("S|crop|lr|supcon_mean3")["acc"], 3), "frozen 1nn", round(g("S|crop|1nn|frozen")["acc"], 3), "frozen lr", round(g("S|crop|lr|frozen")["acc"], 3))
print("S agnostic 1nn supcon:", round(g("S|agnostic|1nn|supcon_mean3")["acc"], 3), "lr", round(g("S|agnostic|lr|supcon_mean3")["acc"], 3))
print("count-matched:", g("S_countmatched_supcon_1nn")); print("excl grape:", g("S_excl_grape_supcon_1nn_post_hoc")); print("per-class 1nn crop:", {k: round(v, 3) for k, v in g("S|crop|1nn|supcon_mean3")["per_class"].items()})
