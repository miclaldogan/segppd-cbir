"""Track C1: error diagnosis of the source-only external PlantSeg evaluation (EXPLORATORY: these 1,485 images were inspected before).
Re-embeds the 1,485 strict external queries with the three D5ck checkpoints (GPU), retrieves from gallery_B, joins PlantSeg metadata and the small
provisional organ annotation, and writes factor tables. python -m segppd_cbir.rv2.trackC_diagnose"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("artifacts/research_v2/trackC"); CACHE = Path("data/external/plantseg_256"); TAG = "D5ck_dinoB_p4_supcon_lr3e-5_s{}"


def queries():
    g = pd.read_csv("artifacts/research_v2/trackB/plantseg_groups.csv"); s = pd.read_csv("artifacts/external/plantseg_leak_screen.csv")
    q = s[(s.tier == "strict") & ~s.leak_flag][["Name", "Disease", "Plant", "class_id"]].merge(g[["Name", "w", "h", "url_host", "Mask ratio", "URL"]], on="Name"); return q.reset_index(drop=True)


def embed(q):
    import torch
    from PIL import Image

    from ..ext_features import to_tensor
    from ..train_embed import DinoEmb, embed_tensors
    X = torch.stack([to_tensor(Image.open(CACHE / (Path(n).stem + ".png")).convert("RGB"), 224) for n in q.Name]).half()
    for s in range(3):
        f = OUT / f"ext_emb_s{s}.npz"
        if f.exists(): continue
        ck = torch.load(f"artifacts/checkpoints/{TAG.format(s)}.pt", map_location="cpu"); a = ck["args"]
        net = DinoEmb(ck["n_cls"], a["d_emb"], a["mode"], blocks=a["blocks"], variant=a["variant"]).to("cuda"); net.load_state_dict(ck["state"]); net.eval()
        np.savez(f, name=q.Name.to_numpy(dtype=str), emb=embed_tensors(net, X)); del net; torch.cuda.empty_cache()


def analyse():
    from ..index import rank
    from .common import segppd_sets
    q = queries(); S = segppd_sets(); ga = S["gallery_B"]; gid = ga.image_id.to_numpy(dtype=str); gl = ga.class_id.to_numpy()
    names = dict(zip(S["train"].class_id, S["train"].raw_class_label)); rows = []
    for s in range(3):
        z = np.load(OUT / f"ext_emb_s{s}.npz"); e = np.load(f"artifacts/extensions/B/{TAG.format(s)}_embeddings.npz"); pos = {i: k for k, i in enumerate(e["image_id"])}
        G = e["emb"][[pos[i] for i in gid]]; o, _ = rank(z["emb"], G, gid, k=len(G)); pred = gl[o[:, 0]]; rel = (gl[o] == q.class_id.to_numpy()[:, None])
        rows.append(pd.DataFrame(dict(seed=s, Name=q.Name, truth=q.class_id, pred=pred, hit1=(pred == q.class_id).astype(int), p10=rel[:, :10].mean(1), top1_id=gid[o[:, 0]])))
    pq = pd.concat(rows); pq.to_csv(OUT / "external_per_query_preds.csv", index=False)
    m = pq.groupby("Name").agg(hit1=("hit1", "mean"), p10=("p10", "mean")).reset_index().merge(q, on="Name")
    m["res_bin"] = pd.cut(np.minimum(m.w, m.h), [0, 300, 500, 800, 1e9], labels=["<300", "300-500", "500-800", ">800"]); m["mask_bin"] = pd.cut(m["Mask ratio"], [-1, .05, .15, .30, 1.1], labels=["<=0.05", "0.05-0.15", "0.15-0.30", ">0.30"])
    top_hosts = m.url_host.value_counts().head(6).index; m["host"] = np.where(m.url_host.isin(top_hosts), m.url_host, "other")
    tabs = {}
    for fct in ("Plant", "Disease", "res_bin", "mask_bin", "host"):
        t = m.groupby(fct, observed=True).agg(n=("hit1", "size"), hit1=("hit1", "mean"), p10=("p10", "mean")).reset_index().sort_values("n", ascending=False); t.to_csv(OUT / f"diag_by_{fct}.csv", index=False); tabs[fct] = t
    org = pd.read_csv(OUT / "organ" / "organ_labels.csv").merge(m[["Name", "hit1"]], on="Name"); to = org.groupby("organ").agg(n=("hit1", "size"), hit1=("hit1", "mean")).reset_index(); to.to_csv(OUT / "diag_by_organ_provisional.csv", index=False)
    o2 = org[org.organ.isin(["L", "F"])]; to2 = o2.groupby(["Disease", "organ"]).agg(n=("hit1", "size"), hit1=("hit1", "mean")).reset_index(); to2.to_csv(OUT / "diag_by_disease_organ_provisional.csv", index=False)
    conf = pq[pq.pred != pq.truth].assign(t=lambda d: d.truth.map(names), p=lambda d: d.pred.map(names)).groupby(["t", "p"]).size().sort_values(ascending=False).head(12) / 3
    conf.rename("n_mean_per_seed").reset_index().to_csv(OUT / "diag_top_confusions.csv", index=False)
    # in-domain comparison on the same 11 classes (B and C test queries, seed-mean)
    from . import common as C
    cls11 = sorted(q.class_id.unique()); ind = []
    for s in range(3):
        o, pqs = C.evaluate_npz(f"artifacts/extensions/B/{TAG.format(s)}_embeddings.npz", S)
        for nm in ("testB_exploratory", "testC_exploratory"): d = pqs[nm][pqs[nm].class_id.isin(cls11)]; ind.append(dict(seed=s, set=nm, hit1_11classes=float(d.groupby("class_id").hit1.mean().mean()), hit1_all75=float(pqs[nm].groupby("class_id").hit1.mean().mean())))
    pd.DataFrame(ind).to_csv(OUT / "diag_indomain_same_classes.csv", index=False)
    summ = dict(n_queries=len(q), external_hit1_seed_mean=float(m.hit1.mean()), external_hit1_class_macro=float(m.groupby("class_id").hit1.mean().mean()), indomain=pd.DataFrame(ind).groupby("set")[["hit1_11classes", "hit1_all75"]].mean().to_dict(),
                organ_counts=to.set_index("organ").n.to_dict(), organ_hit1=to.set_index("organ").hit1.round(3).to_dict())
    json.dump(summ, open(OUT / "diag_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
    for k, t in tabs.items(): print(k); print(t.round(3).to_string(index=False))
    print(to.round(3).to_string(index=False)); print(to2.round(3).to_string(index=False)); print(conf.round(1))


if __name__ == "__main__":
    import sys
    q = queries(); print(len(q), "strict external queries")
    if "embed" in sys.argv: embed(q)
    if "analyse" in sys.argv: analyse()
