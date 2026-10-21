"""Writes the registered Track-A configs (search stage, seed 0). Final stage configs are written by select_and_final()."""
import json
from pathlib import Path

BASE = dict(dataset="segppd_B", backbone="dinov2_base", blocks=4, lr_bb=3e-5, lr_head=1e-3, layer_decay=0.8, d_emb=256, epochs=60, patience=10, steps=20, P=16, K=4, aug="base", seed=0, proxy_lr_mult=1.0)
GRID = {"arcface": [dict(s=30.0, m=0.2), dict(s=30.0, m=0.3), dict(s=30.0, m=0.5)],
        "proxyanchor": [dict(alpha=32.0, delta=0.1), dict(alpha=16.0, delta=0.1), dict(alpha=32.0, delta=0.3), dict(alpha=32.0, delta=0.1, _proxy_lr_mult=10.0)],
        "supcon": [dict(tau=0.05), dict(tau=0.1), dict(tau=0.2)]}


def tag(loss, p):
    return loss[:3] + "_" + "_".join(f"{k.strip('_')}{v:g}" for k, v in p.items())


def main(root="artifacts/research_v2/trackA"):
    Path(root, "configs").mkdir(parents=True, exist_ok=True); names = []
    for loss, cfgs in GRID.items():
        for p in cfgs:
            p = dict(p); mult = p.pop("_proxy_lr_mult", 1.0); name = f"A_{tag(loss, dict(p, **({'plr': mult} if mult != 1 else {})))}_s0"
            c = dict(BASE, name=name, loss=dict(type=loss, **p), proxy_lr_mult=mult, out_dir=f"{root}/runs/{name}"); json.dump(c, open(f"{root}/configs/{name}.json", "w"), indent=1); names.append(name)
    Path(root, "search_queue.txt").write_text("\n".join(names) + "\n"); print("\n".join(names))


if __name__ == "__main__":
    main()
