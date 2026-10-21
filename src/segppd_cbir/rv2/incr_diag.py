"""Experiment A post-hoc DIAGNOSTICS (not registered methods, not used for any selection): (1) where do S2 novel-query errors go (base vs other novel class), (2) does gallery imbalance explain the CBIR gap:
CBIR over a base gallery thinned to k images per base class (same k as the novel supports)."""
import numpy as np
import pandas as pd

from .plantseg_eval import retrieval

R = "artifacts/research_v3/incr/"; pq = pd.read_csv(R + "per_query_predictions.csv.gz"); cf = pd.read_csv(R + "class_folds.csv").set_index("label"); S = pd.read_csv(R + "support_draws.csv")
x = pq[(pq.enc == "base-trained") & (pq.scen == "S2") & (pq.method == "CBIR")].copy(); x["nov_fold"] = x.fold
x["pred_is_novel"] = [cf.fold.get(p, -1) == f for p, f in zip(x.pred, x.fold)]; x["pred_is_base"] = ~x.pred_is_novel
err = x[x.correct == 0].groupby("k").pred_is_base.mean(); print("share of S2 CBIR errors that land on a BASE class:\n", err.round(3))
rows = []
for f in range(5):
    run = f"{R}runs/fold{f}_s0/"; man = pd.read_csv(run + "manifest.csv"); emb = np.load(run + "embeddings.npz"); meta = {p: g.reset_index(drop=True) for p, g in man.groupby("part")}
    bt = meta["train"]; Gb, gb, idb = emb["train"], bt.label_orig.to_numpy(), bt.Name.to_numpy(dtype=str); pos = {n: i for i, n in enumerate(meta["nov_train"].Name)}; ql = meta["nov_test"].label_orig.to_numpy()
    for k in (1, 3, 5, 10):
        for d in range(5):
            r = np.random.default_rng([7, f, k, d]); keep = np.concatenate([r.permutation(np.where(gb == c)[0])[:k] for c in np.unique(gb)])
            sd = S[(S.fold == f) & (S.draw == d) & (S.rank_in_draw <= k)]; G = np.concatenate([Gb[keep], emb["nov_train"][[pos[n] for n in sd.Name]]]); gl = np.concatenate([gb[keep], sd.label.to_numpy()]); gid = np.concatenate([idb[keep], sd.Name.to_numpy(dtype=str)])
            ex = retrieval(emb["nov_test"], G, ql, gl, gid); rows.append(dict(fold=f, k=k, draw=d, hit1=float((ex["pred"] == ql).mean())))
D = pd.DataFrame(rows); D.to_csv(R + "diag_balanced_gallery.csv", index=False)
full = pd.read_csv(R + "summary_by_shot.csv"); ff = full[(full.enc == "base-trained") & (full.scen == "S2") & (full.method == "CBIR") & (full.metric == "hit1")].set_index("k")["mean"]
print(pd.DataFrame(dict(balanced_gallery_query_mean=D.groupby("k").hit1.mean().round(3), full_gallery_class_macro=ff.round(3))))
pd.DataFrame(dict(error_share_on_base=err)).to_csv(R + "diag_error_destination.csv")
