"""Phase 1 audit for the ChatLeafDisease (Pan et al. 2025) paper-matched benchmark. Reads only saved metadata/features; no model is run.
Writes artifacts/chatleaf/audit_*.csv|json and the near-duplicate groups of in-scope PlantSeg images."""
from __future__ import annotations

import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd

OUT = Path("artifacts/chatleaf"); OUT.mkdir(parents=True, exist_ok=True)
# (crop, paper category, paper Table S2 count, PlantSeg v2 disease name, mapping tier, SegPPD class id or None)
CATS = [
    ("tomato", "Bacterial spot", 59, "tomato bacterial leaf spot", "strict", 92),
    ("tomato", "Yellow Leaf Curl Virus", 64, "tomato yellow leaf curl virus", "strict", None),
    ("tomato", "Late Blight", 100, "tomato late blight", "strict", 94),
    ("tomato", "Mosaic Virus", 43, "tomato mosaic virus", "strict", None),
    ("tomato", "Early Blight", 112, "tomato early blight", "strict", None),
    ("grape", "Black rot", 13, "grape black rot", "strict", 42),
    ("grape", "Leaf Blight", 46, "grape leaf spot", "assumed", None),
    ("strawberry", "Leaf Scorch", 26, "strawberry leaf scorch", "strict", 81),
    ("pepper", "Bacterial Spot", 44, "bell pepper bacterial spot", "loose", 71)]
# ChatLD Table S5 (PlantSeg): precision, recall (%)
S5 = {"Bacterial spot": (87.8, 61.0, 72.0), "Early Blight": (80.8, 87.4, 84.0), "Late Blight": (91.5, 84.4, 87.8), "Mosaic Virus": (91.3, 48.8, 63.6),
      "Yellow Leaf Curl Virus": (69.6, 100.0, 82.1), "Black rot": (75.0, 69.2, 72.0), "Leaf Blight": (100.0, 73.8, 84.9), "Leaf Scorch": (100.0, 96.2, 98.0), "Bacterial Spot": (100.0, 100.0, 100.0)}


def uf_groups(n, pairs):
    p = list(range(n))
    def f(x):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
    for a, b in pairs: p[f(a)] = f(b)
    return np.array([f(i) for i in range(n)])


def main():
    ps = pd.read_csv("artifacts/external/plantseg_leak_screen.csv")  # includes phash/dhash + SegPPD leak flags (hash rule or DINOv2-B cos >= 0.85)
    meta = pd.read_csv("data/external/plantseg/plantsegv2/Metadatav2.csv", encoding="utf-8-sig")
    seg = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}); seg = seg[seg.dup_status != "excluded"]
    rows = []
    for crop, cat, n_paper, name, tier, sc in CATS:
        d = ps[ps.Disease == name]; mt = meta[meta.Disease == name]
        s5 = S5[cat]
        rows.append(dict(crop=crop, paper_category=cat, paper_count_S2=n_paper, plantseg_name=name, tier=tier, local_total=len(d), local_test_split=int((mt.Split == "Test").sum()),
                         count_match=bool(len(d) == n_paper), ratio_paper_to_local=round(n_paper / len(d), 3), flagged_vs_segppd=int(d.leak_flag.sum()),
                         segppd_class=sc, segppd_usable_images=int((seg.class_id == sc).sum()) if sc else 0, chatld_recall_S5=s5[1], chatld_precision_S5=s5[0], chatld_f1_S5=s5[2]))
    t = pd.DataFrame(rows); t.to_csv(OUT / "audit_category_table.csv", index=False)
    # what accuracy does S5 imply? (recall x count, the only per-image quantity recoverable)
    corr = (t.paper_count_S2 * t.chatld_recall_S5 / 100)
    imp = dict(total_images_S2=int(t.paper_count_S2.sum()), implied_micro_accuracy=float(corr.sum() / t.paper_count_S2.sum()), macro_recall=float(t.chatld_recall_S5.mean()),
               macro_f1=float(t.chatld_f1_S5.mean()), reported_in_text=0.773)
    covered = t[t.segppd_class.notna()]
    imp.update(covered_categories=int(len(covered)), covered_paper_images=int(covered.paper_count_S2.sum()),
               chatld_micro_on_covered=float((covered.paper_count_S2 * covered.chatld_recall_S5 / 100).sum() / covered.paper_count_S2.sum()), chatld_macro_recall_on_covered=float(covered.chatld_recall_S5.mean()),
               uncovered_paper_images=int(t[t.segppd_class.isna()].paper_count_S2.sum()))
    # within-PlantSeg near-duplicate groups among the in-scope categories (same rule as the SegPPD screen)
    names = set(t.plantseg_name)
    sc = ps[ps.Disease.isin(names)].reset_index(drop=True)
    fx = np.load("artifacts/features/plantseg_dinov2b_frozen_s224.npz"); idx = {n: i for i, n in enumerate(fx["name"])}
    E = fx["cls"][[idx[n] for n in sc.Name]]; E = E / np.linalg.norm(E, axis=1, keepdims=True); cos = E @ E.T
    hb = lambda col: np.array([imagehash.hex_to_hash(h).hash.flatten() for h in sc[col]])
    P = (hb("phash")[:, None, :] != hb("phash")[None, :, :]).sum(2); D = (hb("dhash")[:, None, :] != hb("dhash")[None, :, :]).sum(2)
    dup = ((P <= 12) & (D <= 18)) | (cos >= 0.85); np.fill_diagonal(dup, False)
    ii, jj = np.nonzero(np.triu(dup, 1)); sc["group"] = uf_groups(len(sc), zip(ii, jj))
    sc["group_size"] = sc.groupby("group").Name.transform("size")
    cross = sc.groupby("group").Disease.nunique(); imp.update(inscope_images=len(sc), within_plantseg_dup_pairs=int(len(ii)), images_in_multi_groups=int((sc.group_size > 1).sum()),
                                                            groups_spanning_categories=int((cross > 1).sum()))
    sc[["Name", "Disease", "Split", "leak_flag", "group", "group_size"]].to_csv(OUT / "audit_inscope_groups.csv", index=False)
    json.dump(imp, open(OUT / "audit_summary.json", "w"), indent=1)
    print(t.drop(columns=["chatld_precision_S5"]).to_string(index=False)); print(json.dumps(imp, indent=1))


if __name__ == "__main__":
    main()
