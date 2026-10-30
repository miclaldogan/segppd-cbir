"""Table-4-format metrics (Hit@1, P@10, Hit@10, mAP, macro-F1 of the 1-NN class, pairwise ROC-AUC with score = -distance over all query-gallery pairs) on the Protocol B test, for the
distance variants of metric_study. EXPLORATORY (B test consumed). Seeds averaged for the single model. python -m segppd_cbir.rv2.metric_table_b"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from . import common as C
from .metric_study import NSEEDS, OUT, S, VARS, distance, load_model, pq_from_D, transform, vname

SEL = {"PA_single": ("whiten", 0.25, 128), "PA_ensemble": ("whiten", 0.25, 128), "SUPCON_ensemble": ("whiten", 0.5, 64), "SUPCON_ens_RESNET": None, "PA_single_RESNET": None, "SUPCON_single_RESNET": None}
SHOW = [("l2",), ("l1"), ]
rows = []
for model, sel in SEL.items():
    if sel is None: sel = tuple(json.load(open(OUT / f"metric_selection_dev_only_{model}.json"))["selected_tuple"])
    for v in (("l2",), ("l1",), sel):
        vals = []
        for seed in range(NSEEDS(model)):
            ids, emb = load_model(model, seed); pos = {i: k for k, i in enumerate(ids)}; E = lambda df: emb[[pos[i] for i in df.image_id]].astype(np.float64); T = transform(v, E(S["train"]))
            q, g = S["testB_q"], S["gallery_B"]; D = distance(v, T(E(q)), T(E(g))); ql, gl = q.class_id.to_numpy(), g.class_id.to_numpy()
            pq = pq_from_D(D, ql, gl, g.image_id.to_numpy(dtype=str), q.image_id.to_numpy(dtype=str)); s = C.summarize(pq); pred = pq.pred.to_numpy()
            vals.append(dict(hit1=s["macro_hit1"], p10=s["macro_p10"], hit10=s["macro_hit10"], mAP=s["macro_mAP"], f1=float(f1_score(ql, pred, labels=np.unique(gl), average="macro", zero_division=0)),
                             auc=float(roc_auc_score((ql[:, None] == gl[None, :]).ravel(), -D.ravel()))))
        rows.append(dict(model=model, metric=vname(v), **pd.DataFrame(vals).mean().to_dict()))
R = pd.DataFrame(rows); R.to_csv(OUT / "metric_study_tableB_format.csv", index=False); print(R.round(3).to_string(index=False))
