"""Track A: apply the registered selection rule to the seed-0 search runs (dev only) and write the final-stage configs (seeds 1,2) + queue."""
import json
from pathlib import Path

import pandas as pd

from . import common as C

ROOT = Path("artifacts/research_v2/trackA")


def main():
    rows = []
    for rj in sorted((ROOT / "runs").glob("A_*_s0/result.json")):
        r = json.load(open(rj)); b = r["best_dev"]
        rows.append(dict(name=r["name"], loss=r["cfg"]["loss"]["type"], params=json.dumps({k: v for k, v in r["cfg"]["loss"].items() if k != "type"}), proxy_lr_mult=r["cfg"]["proxy_lr_mult"], best_epoch=b["epoch"], dev_hit1=b["hit1"], dev_mAP=b["mAP"],
                         dev_p10=b["P10"], epochs_run=r["epochs_run"], train_s=r["train_seconds"]))
    t = pd.DataFrame(rows).sort_values(["loss", "dev_hit1"], ascending=[True, False]); t.to_csv(ROOT / "search_summary.csv", index=False)
    sel = {}
    for loss, g in t.groupby("loss"):
        top = g.dev_hit1.max(); tied = g[g.dev_hit1 >= top - C.TIE_EPS]; w = tied.sort_values("dev_mAP", ascending=False).iloc[0]; sel[loss] = w.name_ if False else w["name"]
        print(loss, "selected", w["name"], f"dev hit1 {w.dev_hit1:.3f} mAP {w.dev_mAP:.3f}; tied candidates: {list(tied['name'])}")
    json.dump(sel, open(ROOT / "selected_configs.json", "w"), indent=1)
    q = []
    for loss, nm in sel.items():
        c = json.load(open(ROOT / "configs" / f"{nm}.json"))
        for s in (1, 2):
            c2 = dict(c, seed=s, name=nm.replace("_s0", f"_s{s}"), out_dir=str(ROOT / "runs" / nm.replace("_s0", f"_s{s}"))); json.dump(c2, open(ROOT / "configs" / f"{c2['name']}.json", "w"), indent=1); q.append(c2["name"])
    (ROOT / "final_queue.txt").write_text("\n".join(q) + "\n"); print("final-stage queue:", q); print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
