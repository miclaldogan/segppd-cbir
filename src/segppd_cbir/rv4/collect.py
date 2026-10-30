"""Collect rv4 run results into one table. python -m segppd_cbir.rv4.collect [stage dirs...]"""
import json
import sys
from pathlib import Path

import pandas as pd

rows = []
for d in (sys.argv[1:] or ["artifacts/research_v4/sweep", "artifacts/research_v4/kd", "artifacts/research_v4/followup", "artifacts/research_v4/final"]):
    for f in sorted(Path(d, "runs").glob("*/result.json")) if Path(d, "runs").exists() else []:
        r = json.load(open(f)); c = r["cfg"]; res = r["results"]
        rows.append(dict(run=r["name"], backbone=c["backbone"], full=c.get("full", False), trunc=c.get("truncate"), img=c.get("img_size", 224), lr_bb=c["lr_bb"], kd=(c.get("kd") or {}).get("lam"), teacher=Path((c.get("kd") or {}).get("teacher", "")).stem or None,
                         params_M=r["n_inference_params"] / 1e6, best_ep=r["best_epoch"], dev_hit1=r["best_dev"]["hit1"], dev_mAP=r["best_dev"]["mAP"],
                         B_hit1=res["testB_exploratory"]["macro_hit1"], B_mAP=res["testB_exploratory"]["macro_mAP"], C_hit1=res["testC_exploratory"]["macro_hit1"], C_mAP=res["testC_exploratory"]["macro_mAP"],
                         train_s=r["train_seconds"], embed_ms=r["embed_ms_per_image"], vram_GB=r["peak_vram_gb"]))
R = pd.DataFrame(rows); R.to_csv("artifacts/research_v4/results_all.csv", index=False); pd.set_option("display.width", 250)
print(R.drop(columns=["teacher"]).round(3).to_string(index=False))
