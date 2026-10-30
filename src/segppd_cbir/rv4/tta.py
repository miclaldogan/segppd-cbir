"""Stage T: test-time multi-view embedding (zero parameters) for saved checkpoints. Views: flip (baseline) | flip+ms | flip+5crop | all.
Loads research_v2 DinoEmb checkpoints (cfg with backbone dinov2_base) or rv4 Emb checkpoints. Selection on dev only; B/C exploratory.
python -m segppd_cbir.rv4.tta <run_dir> [<run_dir> ...]   (one GPU job at a time: run when no training is active)"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..train_embed import load_tensors
from ..rv2 import common as C
from .model import DEV, Emb, embed_tensors

OUT = Path("artifacts/research_v4/tta"); OUT.mkdir(parents=True, exist_ok=True); S = C.segppd_sets()
VIEWS = {"flip": (), "flip+ms": ("ms",), "flip+5crop": ("5crop",), "all": ("ms", "5crop")}
SETS = (("dev", "dev_q", "gallery_B"), ("testB_exploratory", "testB_q", "gallery_B"), ("testC_exploratory", "testC_q", "gallery_C"))


def load_net(run_dir):
    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu"); cfg = ck["cfg"]
    if cfg.get("dataset") == "segppd_B" and "truncate" not in cfg and "img_size" not in cfg:  # research_v2 DinoEmb checkpoint: pretrained 518 grid, dynamic resampling at 224
        net = Emb(ck["n_cls"], cfg["d_emb"], "dinov2_base", blocks=cfg["blocks"], img_size=None, pretrained=False)
    else:
        net = Emb(ck["n_cls"], cfg["d_emb"], cfg["backbone"], blocks=cfg.get("blocks", 4), truncate=cfg.get("truncate"), img_size=cfg.get("img_size", 224), full=cfg.get("full", False), pretrained=False)
    missing, unexpected = net.load_state_dict(ck["state"], strict=False); assert not unexpected and not missing, (missing[:5], unexpected[:5])
    return net.to(DEV).eval()


def main(run_dirs):
    rows = []
    for rd in run_dirs:
        net = load_net(rd); name = Path(rd).name; X = {k: load_tensors(S[k].image_path, net.eval_size) for k in ("dev_q", "gallery_B", "testB_q", "testC_q", "gallery_C")}
        for vname, views in VIEWS.items():
            t0 = time.time(); E = {k: embed_tensors(net, x, views=views) for k, x in X.items()}; ms = 1e3 * (time.time() - t0) / sum(len(x) for x in X.values())
            np.savez(OUT / f"emb_{name}_{vname}.npz", **{k: v for k, v in E.items()}, **{f"id_{k}": S[k].image_id.to_numpy(dtype=str) for k in E})  # protocol-set embeddings for later fusion studies
            for sname, qs, gs in SETS:
                q, g = S[qs], S[gs]; pq, _ = C.per_query(E[qs], E[gs], q.class_id.to_numpy(), g.class_id.to_numpy(), g.image_id.to_numpy(dtype=str), q.image_id.to_numpy(dtype=str)); s = C.summarize(pq)
                rows.append(dict(run=name, views=vname, set=sname, hit1=s["macro_hit1"], mAP=s["macro_mAP"], P10=s["macro_p10"], embed_ms_per_image=ms)); pq.assign(run=name, views=vname, set=sname).to_csv(OUT / f"per_query_{name}_{vname}_{sname}.csv", index=False)
            print(f"[{name}] {vname}: " + " ".join(f"{r['set'][:4]} {r['hit1']:.3f}/{r['mAP']:.3f}" for r in rows[-3:]) + f" ({ms:.1f} ms/img)", flush=True)
        del net; torch.cuda.empty_cache()
    R = pd.DataFrame(rows); f = OUT / "tta_by_run.csv"; R.to_csv(f, mode="a", header=not f.exists(), index=False)
    M = R.groupby(["views", "set"])[["hit1", "mAP"]].mean().reset_index(); print(M.pivot(index="views", columns="set", values="hit1").round(3).to_string())
    dev = M[M.set == "dev"].set_index("views"); mx = dev.hit1.max(); band = dev[dev.hit1 >= mx - C.TIE_EPS + 1e-12]; sel = band.mAP.idxmax()
    json.dump(dict(runs=[Path(r).name for r in run_dirs], rule="dev Hit@1 (mean over runs), tie band 1/150, then dev mAP", selected=sel), open(OUT / f"selection_dev_only_{Path(run_dirs[0]).name.split('_s')[0]}.json", "w"), indent=1); print("selected on dev:", sel)


if __name__ == "__main__":
    main(sys.argv[1:])
