"""Single DINOv2-B model (+ ResNet50-ft ArcFace) fusion, exact L2 on sqrt-weighted concat of unit vectors. Weight w on ResNet50 in {0, .05, .1, .2, .3}; selected on dev only
(Hit@1, tie band 1/150, then mAP). B/C tests consumed -> exploratory. Both single models (Proxy-Anchor, SupCon), seeds averaged. python -m segppd_cbir.rv2.single_fusion"""
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from . import common as C
from .metric_study import OUT, RUN, S, SUP, pq_from_D

L2n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True); WS = (0.0, 0.05, 0.1, 0.2, 0.3); rows, pqs = [], []
rn = np.load("artifacts/extensions/B/R1_resnet_arcface_s0_embeddings.npz"); rnid = rn["image_id"]
for model, tpl in (("PA_single", RUN), ("SUPCON_single", SUP)):
    for seed in range(3):
        z = np.load(tpl.format(seed)); assert (z["image_id"] == rnid).all(); pos = {i: k for k, i in enumerate(rnid)}
        for w in WS:
            X = np.concatenate([np.sqrt(1 - w) * L2n(z["emb"].astype(np.float64)), np.sqrt(w) * L2n(rn["emb"].astype(np.float64))], axis=1); E = lambda df: X[[pos[i] for i in df.image_id]]
            for name, qs, gs in (("dev", "dev_q", "gallery_B"), ("testB_exploratory", "testB_q", "gallery_B"), ("testC_exploratory", "testC_q", "gallery_C")):
                q, g = S[qs], S[gs]; pq = pq_from_D(cdist(E(q), E(g), "sqeuclidean"), q.class_id.to_numpy(), g.class_id.to_numpy(), g.image_id.to_numpy(dtype=str), q.image_id.to_numpy(dtype=str)); s = C.summarize(pq)
                rows.append(dict(model=model, w_resnet=w, seed=seed, set=name, hit1=s["macro_hit1"], mAP=s["macro_mAP"], P10=s["macro_p10"])); pqs.append(pq.assign(model=model, w=w, seed=seed, set=name))
R = pd.DataFrame(rows); R.to_csv(OUT / "single_fusion_by_seed.csv", index=False); pd.concat(pqs).to_csv(OUT / "per_query_single_fusion.csv.gz", index=False)
M = R.groupby(["model", "w_resnet", "set"])[["hit1", "mAP", "P10"]].mean().reset_index(); M.to_csv(OUT / "single_fusion_mean.csv", index=False); sel = {}
for m in M.model.unique():
    d = M[(M.model == m) & (M.set == "dev")].set_index("w_resnet"); band = d[d.hit1 >= d.hit1.max() - C.TIE_EPS + 1e-12]; sel[m] = float(band.mAP.idxmax())
    print(m); print(M[M.model == m].pivot(index="w_resnet", columns="set", values="hit1").round(3).to_string()); print("dev-selected w =", sel[m], "| band", list(band.index))
json.dump(dict(rule="dev Hit@1 (mean 3 seeds), tie band 1/150, then dev mAP", selected_w=sel), open(OUT / "single_fusion_selection_dev_only.json", "w"), indent=1)
rng = np.random.default_rng(42); P = pd.concat(pqs)
for m in sel:
    for w in sorted({sel[m], 0.1}):
        for s in ("dev", "testB_exploratory", "testC_exploratory"):
            f = lambda ww: P[(P.model == m) & (P.w == ww) & (P.set == s)].groupby("class_id").hit1.mean(); d = (f(w) - f(0.0)).to_numpy(); b = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
            print(m, "w", w, s, "diff vs no-ResNet", round(float(d.mean()), 3), np.round(np.quantile(b, [.025, .975]), 3))
