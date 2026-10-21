"""Experiment A encoder: DINOv2-B + Proxy-Anchor trained on BASE-class train images of one fold only; epochs selected on BASE-class dev images only.
Reuses the Track B trainer unchanged (patched protocol loader). Novel-class train/test images are embedded only at the end by the frozen encoder.
python -m segppd_cbir.rv2.incr_train <fold> [seed]"""
import json
import sys
from pathlib import Path

import pandas as pd

from . import plantseg_train as T

fold = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
P = pd.read_csv("artifacts/research_v2/trackB/protocol_P.csv"); cf = pd.read_csv("artifacts/research_v3/incr/class_folds.csv")
nov = set(cf[cf.fold == fold].label); base = sorted(set(P.label) - nov); remap = {c: i for i, c in enumerate(base)}
B = P[~P.label.isin(nov)].copy(); B["label_orig"] = B.label; B["label"] = B.label.map(remap); N = P[P.label.isin(nov)].copy(); N["label_orig"] = N.label
proto = {"train": B[B.role == "train"].reset_index(drop=True), "dev": B[B.role == "dev"].reset_index(drop=True), "test": B[B.role == "test"].reset_index(drop=True),
         "nov_train": N[N.role == "train"].reset_index(drop=True), "nov_test": N[N.role == "test"].reset_index(drop=True)}
assert not (set(proto["train"].label_orig) & nov) and not (set(proto["dev"].label_orig) & nov), "novel class leaked into training/selection"
T.load_protocol = lambda: proto
out = Path(f"artifacts/research_v3/incr/runs/fold{fold}_s{seed}"); out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open("artifacts/research_v2/trackB/configs/B_M3_metric_proxyanchor_s0.json")); cfg.update(seed=seed, name=f"incr_fold{fold}_s{seed}", out_dir=str(out), fold=fold)
json.dump(cfg, open(out / "config.json", "w"), indent=1)
pd.concat([v.assign(part=k)[["Name", "label", "label_orig", "role", "part"]] for k, v in proto.items()]).to_csv(out / "manifest.csv", index=False)
T.run(out / "config.json")
