"""Uniform embeddings for Experiment B: every SegPPD image (dataset_manifest order) and every protocol-P image, flip-averaged, from any DinoEmb checkpoint.
python -m segppd_cbir.rv2.cross_embed <best.pt> <out.npz>   (used for the existing single-source encoders S0 and T0)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..train_embed import DinoEmb, embed_tensors, load_tensors
from .plantseg_train import tens

DEV = "cuda"


@torch.inference_mode()
def embed_all(net, out, chunk=512):
    net.eval(); inv = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    P = pd.read_csv("artifacts/research_v2/trackB/protocol_P.csv"); seg = []
    for i in range(0, len(inv), chunk): seg.append(embed_tensors(net, load_tensors(inv.image_path.iloc[i:i + chunk], net.eval_size)))
    ps = [embed_tensors(net, tens(P.iloc[i:i + chunk])) for i in range(0, len(P), chunk)]
    np.savez(out, seg_id=inv.image_id.to_numpy(dtype=str), seg=np.concatenate(seg), ps_name=P.Name.to_numpy(dtype=str), ps=np.concatenate(ps))


if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu"); cfg = ck["cfg"]
    n_cls = ck.get("n_cls") or int(ck["state"]["head.W"].shape[0]); net = DinoEmb(n_cls, cfg.get("d_emb", 256), "partial", blocks=cfg["blocks"], variant="base").to(DEV); net.load_state_dict(ck["state"]); embed_all(net, Path(sys.argv[2])); print("ok", sys.argv[2])
