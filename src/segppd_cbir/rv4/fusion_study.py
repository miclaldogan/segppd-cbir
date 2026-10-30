"""Zero-training fusion of existing embeddings (sqrt-weighted concat of unit vectors, exact L2). Pairs of small models and small + ResNet50-ft, to see whether
two light models replace one DINOv2-B. Weight w on the second member in {0.1..0.5}; selection on dev only (Hit@1, tie band 1/150, then mAP). B/C exploratory.
python -m segppd_cbir.rv4.fusion_study"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ..rv2 import common as C
from ..rv2.metric_study import pq_from_D

OUT = Path("artifacts/research_v4/fusion"); OUT.mkdir(parents=True, exist_ok=True); S = C.segppd_sets(); L2n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
R4 = "artifacts/research_v4/{}/runs/{}/embeddings.npz"
EMB = {"convnext336": R4.format("followup", "F_dinov3_convnext_tiny_336"), "convnext224": R4.format("sweep", "S_dinov3_convnext_tiny"), "reg4_336": R4.format("followup", "F_dinov2_small_reg4_336"),
       "reg4_224": R4.format("sweep", "S_dinov2_small_reg4"), "vits16plus": R4.format("sweep", "S_dinov3_vits16plus"), "dinov3_vitb": R4.format("sweep", "S_dinov3_vitb16"),
       "resnet50ft": "artifacts/extensions/B/R1_resnet_arcface_s0_embeddings.npz", "dinov2_base_PA_s0": "artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s0/embeddings.npz"}
PARAMS = {"convnext336": 28.0, "convnext224": 28.0, "reg4_336": 21.9, "reg4_224": 21.7, "vits16plus": 28.8, "dinov3_vitb": 85.8, "resnet50ft": 24.0, "dinov2_base_PA_s0": 85.9}
PAIRS = [("convnext336", "reg4_336"), ("convnext336", "resnet50ft"), ("reg4_336", "resnet50ft"), ("convnext336", "vits16plus"), ("convnext224", "reg4_224"), ("convnext336", "reg4_336", "resnet50ft"),
         ("dinov3_vitb", "resnet50ft"), ("dinov2_base_PA_s0", "resnet50ft")]
WS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def load(k):
    z = np.load(EMB[k]); return z["image_id"], L2n(z["emb"].astype(np.float64))


rows = []
for pair in PAIRS:
    ids0, E0 = load(pair[0]); mem = [E0]
    for k in pair[1:]:
        ids, E = load(k); assert (ids == ids0).all(); mem.append(E)
    pos = {i: j for j, i in enumerate(ids0)}
    for w in WS:
        if len(pair) == 2: X = np.concatenate([np.sqrt(1 - w) * mem[0], np.sqrt(w) * mem[1]], 1)
        else: X = np.concatenate([np.sqrt(1 - w) * mem[0], np.sqrt(w / 2) * mem[1], np.sqrt(w / 2) * mem[2]], 1)  # 3-way: w split equally over the two extras
        G = lambda df: X[[pos[i] for i in df.image_id]]
        for name, qs, gs in (("dev", "dev_q", "gallery_B"), ("testB_exploratory", "testB_q", "gallery_B"), ("testC_exploratory", "testC_q", "gallery_C")):
            q, g = S[qs], S[gs]; pq = pq_from_D(cdist(G(q), G(g), "sqeuclidean"), q.class_id.to_numpy(), g.class_id.to_numpy(), g.image_id.to_numpy(dtype=str), q.image_id.to_numpy(dtype=str)); s = C.summarize(pq)
            rows.append(dict(pair="+".join(pair), params_M=sum(PARAMS[k] for k in pair), w=w, set=name, hit1=s["macro_hit1"], mAP=s["macro_mAP"], P10=s["macro_p10"]))
R = pd.DataFrame(rows); R.to_csv(OUT / "fusion_pairs.csv", index=False); sel = {}
for p in R.pair.unique():
    d = R[(R.pair == p) & (R.set == "dev")].set_index("w"); band = d[d.hit1 >= d.hit1.max() - C.TIE_EPS + 1e-12]; sel[p] = float(band.mAP.idxmax())
json.dump(dict(rule="dev Hit@1, tie band 1/150, then dev mAP", selected_w=sel), open(OUT / "selection_dev_only.json", "w"), indent=1)
pd.set_option("display.width", 250); T = R[R.apply(lambda r: r.w in (0.0, sel[r.pair]), axis=1)].pivot_table(index=["pair", "params_M", "w"], columns="set", values=["hit1", "mAP"]).round(3)
print(T.to_string()); print("dev-selected w:", sel)
