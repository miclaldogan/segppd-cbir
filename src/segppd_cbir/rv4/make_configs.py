"""Stage S configs (registered grid, seed 0). python -m segppd_cbir.rv4.make_configs"""
import json
from pathlib import Path

ROOT = "artifacts/research_v4/sweep"
BASE = dict(dataset="segppd_B", blocks=4, lr_bb=3e-5, lr_head=1e-3, layer_decay=0.8, d_emb=256, epochs=60, patience=10, steps=20, P=16, K=4, aug="base", seed=0, proxy_lr_mult=1.0, img_size=224,
            loss=dict(type="proxyanchor", alpha=32.0, delta=0.3))
SWEEP = {"S_dinov3_vits16plus": dict(backbone="dinov3_vits16plus"), "S_dinov3_vits16": dict(backbone="dinov3_vits16"), "S_dinov2_small": dict(backbone="dinov2_small"),
         "S_dinov2_small_reg4": dict(backbone="dinov2_small_reg4"), "S_dinov3_convnext_tiny": dict(backbone="dinov3_convnext_tiny"),
         "S_dinov2_base_trunc9": dict(backbone="dinov2_base", truncate=9), "S_dinov2_base_trunc8": dict(backbone="dinov2_base", truncate=8), "S_dinov3_vitb16": dict(backbone="dinov3_vitb16")}


def write(configs, root=ROOT, queue="queue.txt"):
    Path(root, "configs").mkdir(parents=True, exist_ok=True); names = []
    for name, p in configs.items():
        c = dict(BASE, **p, name=name, out_dir=f"{root}/runs/{name}"); json.dump(c, open(f"{root}/configs/{name}.json", "w"), indent=1); names.append(name)
    Path(root, queue).write_text("\n".join(names) + "\n"); print("\n".join(names)); return names


if __name__ == "__main__":
    write(SWEEP)
