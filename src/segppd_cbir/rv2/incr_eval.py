"""Experiment A evaluation (CPU only). For each finished fold: frozen embeddings -> scenarios S1/S2 x shots x support draws x methods (CBIR, NCP, LR) x encoders (base-trained, raw frozen DINOv2-B).
Writes per-query predictions, summary tables and bootstrap CIs. python -m segppd_cbir.rv2.incr_eval"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..index import rank, squared_l2
from .plantseg_eval import retrieval

R = Path("artifacts/research_v3/incr"); SHOTS = (1, 3, 5, 10); L2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
P = pd.read_csv("artifacts/research_v2/trackB/protocol_P.csv"); S = pd.read_csv(R / "support_draws.csv"); CF = pd.read_csv(R / "class_folds.csv")
FX = np.load("artifacts/features/plantseg_dinov2b_frozen_s224.npz"); FPOS = {n: i for i, n in enumerate(FX["name"])}


def frozen(names): return L2(FX["cls"][[FPOS[n] for n in names]].astype(np.float32))


def ncp(Q, G, gl, ql_cand=None, k=5):
    labs = np.unique(gl); Pr = np.stack([G[gl == c].mean(0) for c in labs]); d = squared_l2(Q, Pr); o = np.argsort(d, 1, kind="stable"); return labs, Pr, labs[o[:, 0]], labs[o[:, :k]]


def lr_fit(G, gl):
    t0 = time.time(); m = LogisticRegression(C=10.0, max_iter=200, class_weight="balanced").fit(G, gl); return m, time.time() - t0


def one(fold, enc, E, meta, rows, perq, topk):
    """E: dict part -> embeddings aligned with meta[part] (DataFrame with Name,label_orig)."""
    bt, nt, bte, nte = meta["train"], meta["nov_train"], meta["test"], meta["nov_test"]
    nov_labels = sorted(nt.label_orig.unique()); Gb = E["train"]; gb = bt.label_orig.to_numpy(); idb = bt.Name.to_numpy(dtype=str); pos = {n: i for i, n in enumerate(nt.Name)}
    base_only_hit = float((retrieval(E["test"], Gb, bte.label_orig.to_numpy(), gb, idb)["pred"] == bte.label_orig.to_numpy()).mean())
    for d in sorted(S[S.fold == fold].draw.unique()):
        for k in SHOTS:
            sd = S[(S.fold == fold) & (S.draw == d) & (S.rank_in_draw <= k)]; sn = sd.Name.to_numpy(dtype=str); ix = [pos[n] for n in sn]; Gs = E["nov_train"][ix]; gs = sd.label.to_numpy(); q, ql, qn = E["nov_test"], nte.label_orig.to_numpy(), nte.Name.to_numpy(dtype=str)
            for scen in ("S1", "S2"):
                if scen == "S1": G, gl, gid = Gs, gs, sn
                else:
                    t0 = time.time(); G = np.concatenate([Gb, Gs]); gl = np.concatenate([gb, gs]); gid = np.concatenate([idb, sn]); t_ins = time.time() - t0
                ex = retrieval(q, G, ql, gl, gid)
                for i in range(len(q)):
                    perq.append((fold, enc, scen, d, k, "CBIR", qn[i], ql[i], ex["pred"][i], ex["top1_ref"][i], ex["ap"][i], ex["hit5"][i], ex["p5"][i], ex["r10"][i]))
                if scen == "S2" and d == 0:
                    o, _ = rank(q, G, gid, k=10)
                    for i in range(len(q)): topk.append((fold, enc, k, qn[i], ql[i], ";".join(gid[o[i]]), ";".join(map(str, gl[o[i]]))))
                t0 = time.time(); labs, Pr, pn, p5 = ncp(q, G, gl); t_ncp = time.time() - t0
                for i in range(len(q)): perq.append((fold, enc, scen, d, k, "NCP", qn[i], ql[i], pn[i], "", np.nan, float((p5[i] == ql[i]).any()), np.nan, np.nan))
                m, t_fit = lr_fit(G, gl); pl = m.predict(q); top5 = m.classes_[np.argsort(-m.decision_function(q), 1)[:, :5]]
                for i in range(len(q)): perq.append((fold, enc, scen, d, k, "LR", qn[i], ql[i], pl[i], "", np.nan, float((top5[i] == ql[i]).any()), np.nan, np.nan))
                rows.append(dict(fold=fold, enc=enc, scen=scen, draw=d, k=k, n_gallery=len(G), n_classes=len(np.unique(gl)), gallery_bytes_f32=G.size * 4, ncp_bytes_f32=Pr.size * 4, lr_bytes_f32=int(m.coef_.size * 4),
                                 concat_seconds=(t_ins if scen == "S2" else 0.0), ncp_seconds=t_ncp, lr_fit_seconds=t_fit, n_support=len(gs)))
        # base interference: base queries against base+supports (k=10 draw, the largest insertion)
        G = np.concatenate([Gb, E["nov_train"][[pos[n] for n in S[(S.fold == fold) & (S.draw == d) & (S.rank_in_draw <= 10)].Name]]]); gl = np.concatenate([gb, S[(S.fold == fold) & (S.draw == d) & (S.rank_in_draw <= 10)].label.to_numpy()])
        gid = np.concatenate([idb, S[(S.fold == fold) & (S.draw == d) & (S.rank_in_draw <= 10)].Name.to_numpy(dtype=str)])
        h = float((retrieval(E["test"], G, bte.label_orig.to_numpy(), gl, gid)["pred"] == bte.label_orig.to_numpy()).mean())
        rows.append(dict(fold=fold, enc=enc, scen="BASE_CHECK", draw=d, k=10, base_hit1_before=base_only_hit, base_hit1_after=h))


def main():
    rows, perq, topk = [], [], []
    for fold in range(5):
        run = R / "runs" / f"fold{fold}_s0"
        if not (run / "result.json").exists(): print("fold", fold, "not finished, skipped"); continue
        man = pd.read_csv(run / "manifest.csv"); meta = {p: g.reset_index(drop=True) for p, g in man.groupby("part")}; emb = np.load(run / "embeddings.npz")
        for p in meta: assert len(meta[p]) == len(emb[p]), p
        E_t = {p: emb[p] for p in meta}; E_f = {p: frozen(meta[p].Name) for p in meta}
        for enc, E in (("base-trained", E_t), ("frozen-DINOv2-B", E_f)):
            print("fold", fold, enc, flush=True); one(fold, enc, E, meta, rows, perq, topk)
    cols = ["fold", "enc", "scen", "draw", "k", "method", "Name", "truth", "pred", "top1_ref", "ap", "hit5", "p5", "r10"]
    pq = pd.DataFrame(perq, columns=cols); pq["correct"] = (pq.truth == pq.pred).astype(int); pq.to_csv(R / "per_query_predictions.csv.gz", index=False)
    pd.DataFrame(topk, columns=["fold", "enc", "k", "query", "truth", "top10_ids", "top10_labels"]).to_csv(R / "ranking_top10_draw0.csv.gz", index=False)
    pd.DataFrame(rows).to_csv(R / "cost_and_base_check.csv", index=False); print("saved", len(pq), "per-query rows")


if __name__ == "__main__":
    main()
