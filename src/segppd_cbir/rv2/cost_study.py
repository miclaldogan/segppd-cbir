"""Cost of the 'ensemble + ResNet50 fusion' architecture vs a normal single trained model. CPU only (the GPU is busy with training): parameters and forward FLOPs are counted analytically from
the architectures (random init, no download); exact-L2 search latency is timed with numpy on the real gallery size and on a synthetic 100k gallery; GPU embed times and training times are taken from
earlier measurements (cost_table / result.json) and combined by addition, NOT re-measured as a fused pipeline."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision
from torch.utils.flop_counter import FlopCounterMode

OUT = Path("artifacts/research_v3/cost"); torch.set_num_threads(2)
vit = timm.create_model("vit_base_patch14_dinov2", pretrained=False, num_classes=0, img_size=224).eval(); rn = torchvision.models.resnet50(); rn.fc = torch.nn.Identity(); rn.eval()
head_dino = 768 * 256 + 256; head_rn = 2048 * 256 + 256
npar = lambda m: sum(p.numel() for p in m.parameters())


def flops(m, size):
    x = torch.zeros(1, 3, size, size)
    with FlopCounterMode(display=False) as f, torch.inference_mode(): m(x)
    return f.get_total_flops() / 2 / 1e9  # multiply-accumulates in G (1 MAC = 2 FLOP)


gm_dino, gm_rn = flops(vit, 224), flops(rn, 256)
meas = dict(dino_embed_ms=10.6, resnet_embed_ms=10.0, dino_train_s=(600, 770), resnet_train_s=660, search_256_ms=0.26, search_768_ms=0.75)  # earlier measurements (GPU embed incl. flip-averaging)
r = np.random.default_rng(0); res = {}
for n in (809, 100_000):
    G = r.standard_normal((n, 1024)).astype(np.float32); q = r.standard_normal((1, 1024)).astype(np.float32)
    for d in (256, 512, 768, 1024):
        g, qq = np.ascontiguousarray(G[:, :d]), q[:, :d]; ts = []
        for _ in range(200 if n < 1e5 else 20):
            t0 = time.perf_counter(); dist = (g * g).sum(1) - 2 * g @ qq[0]; np.argpartition(dist, 10)[:10]; ts.append(time.perf_counter() - t0)
        res[(n, d)] = float(np.median(ts) * 1e3)
rows = []
def add(name, n_dino, n_rn, note):
    p = n_dino * (npar(vit) + head_dino) + n_rn * (npar(rn) + head_rn); dim = 256 * n_dino + 256 * n_rn; macs = n_dino * gm_dino + n_rn * gm_rn
    ms = n_dino * meas["dino_embed_ms"] + n_rn * meas["resnet_embed_ms"]; lo = n_dino * meas["dino_train_s"][0] + n_rn * meas["resnet_train_s"]; hi = n_dino * meas["dino_train_s"][1] + n_rn * meas["resnet_train_s"]
    rows.append(dict(system=name, n_dino=n_dino, n_resnet=n_rn, params_M=p / 1e6, fp32_weights_MB=p * 4 / 1e6, fwd_GMACs_per_view=macs, embed_ms_per_image_gpu=ms, train_s_low=lo, train_s_high=hi, descriptor_dim=dim,
                     gallery_MB_per_100k=dim * 4 * 1e5 / 1e6, search_ms_809=res[(809, dim)], search_ms_100k=res[(100000, dim)], note=note))
add("Single DINOv2-B + head (normal trained model)", 1, 0, "reference")
add("3-seed ensemble (DINOv2-B x3)", 3, 0, "")
add("Ensemble + ResNet50 fusion (3x DINOv2-B + ResNet50-ft)", 3, 1, "report Tablo 4 last row")
add("Single DINOv2-B + ResNet50 fusion (1x DINOv2-B + ResNet50-ft)", 1, 1, "single_fusion w=0.05; 0.839 Hit@1 on B (expl.)")
add("ResNet50-ft + ArcFace alone", 0, 1, "earlier system; 0.511 Hit@1 on B")
D = pd.DataFrame(rows); D.to_csv(OUT / "cost_comparison.csv", index=False); pd.set_option("display.width", 250); print(D.round(2).T.to_string()); print("dino GMACs", gm_dino, "resnet GMACs", gm_rn, "params dino/resnet", npar(vit) / 1e6, npar(rn) / 1e6)
for p in ("artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s0/best.pt", "artifacts/research_v2/trackB/runs/B_M3_metric_proxyanchor_s0/best.pt"): print(p.split("/")[-2], Path(p).stat().st_size / 1e6, "MB")
