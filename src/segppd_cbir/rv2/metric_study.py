"""Distance-metric comparison for the reference model (DINOv2-B + Proxy-Anchor a32 d0.3, 3 seeds, existing embeddings; no training).
Variants: L2 (registered metric), cosine and inner product (identical ranking on L2-normalised embeddings: verified, not assumed), L1, PCA-whitening (fit on train_B embeddings only) + L2 with power alpha and dims.
Selection among variants: dev class-macro Hit@1 (mean over seeds; tie band 1/150, inside it higher dev mAP). B/C tests are consumed -> EXPLORATORY.  python -m segppd_cbir.rv2.metric_study"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from . import common as C

OUT = Path("artifacts/research_v3/metrics"); OUT.mkdir(exist_ok=True); S = C.segppd_sets(); RUN = "artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s{}/embeddings.npz"; SUP = "artifacts/extensions/B/D5_dinoB_p4_supcon_lr3e-5_s{}_embeddings.npz"
L2n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)


def pq_from_D(D, ql, gl, gid, qid):
    idr = np.argsort(np.argsort(gid, kind="stable"), kind="stable"); o = np.stack([np.lexsort((idr, D[i])) for i in range(len(D))]); rel = (gl[o] == ql[:, None]).astype(float); R = (gl[None, :] == ql[:, None]).sum(1)
    cs = np.cumsum(rel, 1); prec = cs / np.arange(1, rel.shape[1] + 1); d = dict(query_id=qid, class_id=ql, ap=(prec * rel).sum(1) / np.maximum(R, 1), pred=gl[o[:, 0]])
    for k in C.KS: d[f"hit{k}"] = rel[:, :k].max(1); d[f"p{k}"] = rel[:, :k].mean(1); d[f"r{k}"] = rel[:, :k].sum(1) / np.maximum(R, 1)
    return pd.DataFrame(d)


def transform(variant, Ztr):
    kind = variant[0]
    if kind != "whiten": return lambda x: x
    _, alpha, dims = variant; mu = Ztr.mean(0); U, s, Vt = np.linalg.svd(Ztr - mu, full_matrices=False); W = Vt[:dims].T / (s[:dims] / np.sqrt(len(Ztr) - 1)) ** alpha
    return lambda x: L2n((x - mu) @ W)


def distance(variant, Q, G):
    k = variant[0]
    if k in ("l2", "whiten"): return cdist(Q, G, "sqeuclidean")
    if k == "cosine": return 1 - L2n(Q) @ L2n(G).T
    if k == "dot": return -(Q @ G.T)
    if k == "l1": return cdist(Q, G, "cityblock")


VARS = [("l2",), ("cosine",), ("dot",), ("l1",)] + [("whiten", a, d) for a, d in itertools.product((0.25, 0.5, 1.0), (256, 128, 64))]
vname = lambda v: v[0] if v[0] != "whiten" else f"whiten(alpha={v[1]},dims={v[2]})+L2"
def load_model(model, seed):
    """-> (image_id array, embeddings). Ensembles = concat of sqrt(1/3)*each unit-norm 256-d seed embedding (squared L2 = mean of the three squared L2)."""
    if model == "PA_single": z = np.load(RUN.format(seed)); return z["image_id"], z["emb"]
    tpl = RUN if model == "PA_ensemble" else SUP; zs = [np.load(tpl.format(s)) for s in range(3)]; assert all((zs[0]["image_id"] == z["image_id"]).all() for z in zs)
    return zs[0]["image_id"], np.concatenate([np.sqrt(1 / 3) * z["emb"] for z in zs], axis=1)


def main(model):
    rows = []; pqf = OUT / f"per_query_{model}.csv"; pqf.unlink(missing_ok=True)
    for seed in range(3 if model == "PA_single" else 1):
        ids, emb = load_model(model, seed); pos = {i: k for k, i in enumerate(ids)}; E = lambda df: emb[[pos[i] for i in df.image_id]].astype(np.float64); Ztr = E(S["train"])
        for v in VARS:
            if v[0] == "whiten" and v[2] > Ztr.shape[1]: continue
            T = transform(v, Ztr)
            for name, qs, gs in (("dev", "dev_q", "gallery_B"), ("testB_exploratory", "testB_q", "gallery_B"), ("testC_exploratory", "testC_q", "gallery_C")):
                q, g = S[qs], S[gs]; D = distance(v, T(E(q)), T(E(g))); pq = pq_from_D(D, q.class_id.to_numpy(), g.class_id.to_numpy(), g.image_id.to_numpy(dtype=str), q.image_id.to_numpy(dtype=str)); s = C.summarize(pq)
                rows.append(dict(variant=vname(v), seed=seed, set=name, hit1=s["macro_hit1"], mAP=s["macro_mAP"], P10=s["macro_p10"], hit10=s["macro_hit10"])); pq.assign(variant=vname(v), seed=seed, set=name).to_csv(pqf, mode="a", header=not pqf.exists(), index=False)
    R = pd.DataFrame(rows); R.to_csv(OUT / f"metric_study_by_seed_{model}.csv", index=False); M = R.groupby(["variant", "set"])[["hit1", "mAP", "P10", "hit10"]].mean().reset_index(); M.to_csv(OUT / f"metric_study_mean_{model}.csv", index=False)
    dev = M[M.set == "dev"].set_index("variant"); mx = dev.hit1.max(); band = dev[dev.hit1 >= mx - C.TIE_EPS + 1e-12]; sel = band.mAP.idxmax()
    json.dump(dict(model=model, rule="dev Hit@1 (mean over seeds); tie band 1/150; higher dev mAP inside the band", selected=sel, dev_best_hit1=float(mx), band=list(band.index)), open(OUT / f"metric_selection_dev_only_{model}.json", "w"), indent=1)
    pd.set_option("display.width", 220); print(model); print(M.pivot(index="variant", columns="set", values="hit1").round(3).to_string()); print("selected on dev:", sel)
    l2 = R[R.variant == "l2"].hit1.to_numpy(); print("cosine/dot identical to l2:", [bool(np.allclose(l2, R[R.variant == v].hit1.to_numpy())) for v in ("cosine", "dot")])


if __name__ == "__main__":
    import sys
    for m in sys.argv[1:]: main(m)
