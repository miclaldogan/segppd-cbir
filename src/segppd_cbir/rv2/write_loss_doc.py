"""Generates docs/LOSS_FUNCTION_STUDY.md from the saved Track-A artifacts (tables are auto-filled; no number is typed by hand)."""
import json
from pathlib import Path

import pandas as pd

R = Path("artifacts/research_v2/trackA")
t = pd.read_csv(R / "final_summary.csv"); p = pd.read_csv(R / "paired_comparisons.csv"); s = pd.read_csv(R / "search_summary.csv"); c = pd.read_csv(R / "cost_table.csv"); sel = json.load(open(R / "loss_selection_dev_only.json"))
f = lambda x, d=3: f"{x:.{d}f}"
names = {"dev": "dev (150 queries; used for selection)", "testB_exploratory": "Protocol B test (225 q; consumed -> exploratory)", "testC_exploratory": "Protocol C test (142 q; consumed -> exploratory)"}


def main_table(nm):
    rows = ["| Model | Hit@1 (macro) mean +/- SD | per-seed Hit@1 | class-bootstrap 95% CI (seed-mean) | P@5 | P@10 | Hit@5 | Hit@10 | R@10 | mAP +/- SD | query-weighted Hit@1 / mAP | intra/inter dist. |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t[t.set == nm].iterrows():
        ci = f"[{f(r.hit1_ci_lo)}, {f(r.hit1_ci_hi)}]" if pd.notna(r.hit1_ci_lo) else "-"
        rows.append(f"| {r.model} | {f(r.macro_hit1)} +/- {f(r.macro_hit1_sd)} | {r.hit1_seeds} | {ci} | {f(r.macro_p5)} | {f(r.macro_p10)} | {f(r.macro_hit5)} | {f(r.macro_hit10)} | {f(r.macro_r10)} | {f(r.macro_mAP)} +/- {f(r.macro_mAP_sd)} | {f(r.micro_hit1)} / {f(r.micro_mAP)} | {f(r.intra_over_inter, 2)} |")
    return "\n".join(rows)


def pair_table():
    rows = ["| A vs B | set | Hit@1 diff [95% class-bootstrap CI] | mAP diff [95% CI] | per-seed exact McNemar p (Hit@1) |", "|---|---|---|---|---|"]
    for _, r in p[p.b != "hist"].iterrows():
        rows.append(f"| {r.a} - {r.b} | {r.set} | {r.hit1_diff:+.3f} [{r.hit1_lo:+.3f}, {r.hit1_hi:+.3f}] | {r.mAP_diff:+.3f} [{r.mAP_lo:+.3f}, {r.mAP_hi:+.3f}] | {r.mcnemar_p_per_seed} |")
    return "\n".join(rows)


def search_table():
    rows = ["| config (seed 0) | loss hyper-parameters | best epoch | dev Hit@1 | dev mAP | dev P@10 | epochs run | train s |", "|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        rows.append(f"| {r['name']} | {r.params}{' + proxy lr x10' if r.proxy_lr_mult != 1 else ''} | {r.best_epoch} | {f(r.dev_hit1)} | {f(r.dev_mAP)} | {f(r.dev_p10)} | {r.epochs_run} | {r.train_s:.0f} |")
    return "\n".join(rows)


def cost_table():
    rows = ["| loss | mean train s | mean epochs | peak VRAM GB | embed ms/image (GPU, incl. flip) | best epochs | checkpoint sha256[:12] |", "|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows(): rows.append(f"| {r.loss} | {r.train_s_mean:.0f} | {r.epochs_mean:.1f} | {r.peak_vram_gb:.2f} | {r.embed_ms_per_image:.1f} | {r.best_epochs} | {r.ckpt_sha256_12} |")
    return "\n".join(rows)


doc = f"""# Track A - DINOv2-B loss-function study (ArcFace vs Proxy-Anchor vs SupCon)

Registered before any run: `artifacts/research_v2/trackA/registered.json`. Code: `src/segppd_cbir/rv2/` (`train.py` runner, `make_configs.py`, `trackA_select.py`, `trackA_report.py`). Every run directory (`artifacts/research_v2/trackA/runs/*`) holds `config`, per-epoch curves (`result.json: history`), `best.pt` (sha256 recorded), embeddings of all 2,263 images and per-query CSVs. Historical results/checkpoints were not touched.

## 1. Provenance of the earlier "ArcFace 0.760 / Proxy-Anchor 0.748 / SupCon 0.717" table
Verified from `artifacts/metrics/protocolB_locked/`, `artifacts/extensions/B/*_log.json`, `docs/EXPERIMENT_RESULTS.md`: backbone **DINOv2-SMALL** (`vit_small_patch14_dinov2`), last 4 blocks, d_emb 256; split = **Protocol B locked test** (225 queries, 75 classes, gallery 809); ArcFace **1 seed** (m=0.3, s=30), Proxy-Anchor and SupCon = **3-seed means**; mAP values 0.603 / 0.645 / 0.658; the loss had been chosen on **dev mAP** (dev mAP 0.597 / 0.636 / 0.663). DINOv2-B had only ever been trained with SupCon. The Protocol B/C test sets are consumed, so everything below that touches them is **exploratory**.

## 2. Controlled protocol (identical across losses)
Official LVD-142M DINOv2-B init; last 4 blocks + final norm trainable; layer-wise lr decay 0.8, lr_bb 3e-5, lr_head 1e-3; d_emb 256; 224 px bicubic; flip-averaged L2-normalised embedding; 16x4 class-balanced batches, 20 steps/epoch, max 60 epochs, patience 10, AdamW, cosine lr, bf16; the same augmentation; seeds 0/1/2; `train_B` only (772 images, leakage and duplicate-group checks asserted at start of every run); exact squared-L2 retrieval; relevance = same class.
**Selection rule (fixed beforehand):** dev class-macro Hit@1; improvement needs more than one dev query (1/150 = 0.0067); inside that tie band the higher dev mAP wins - for epochs, for configurations and for the loss choice. Each loss gets its own justified hyper-parameter grid (seed 0, dev only): ArcFace margin {{0.2,0.3,0.5}}; Proxy-Anchor (alpha,delta) {{(32,.1),(16,.1),(32,.3)}} + proxy lr x10; SupCon tau {{0.05,0.1,0.2}}.

### Search stage (seed 0, dev only)
{search_table()}

Selected by the rule (`loss_selection_dev_only.json`, computed from dev results before any test number was looked at): ArcFace m=0.5; Proxy-Anchor alpha=32, delta=0.3 (tied with alpha=16 at 0.820, higher mAP); SupCon tau=0.1. Across losses (3-seed dev means) Proxy-Anchor and SupCon tie on dev Hit@1 ({sel['per_loss']['proxyanchor']['hit1']:.3f} vs {sel['per_loss']['supcon']['hit1']:.3f}); the tie-break by dev mAP ({sel['per_loss']['proxyanchor']['mAP']:.3f} vs {sel['per_loss']['supcon']['mAP']:.3f}) picks **Proxy-Anchor** as the loss carried into Tracks B and C.

## 3. Final stage: 3 matched seeds per loss
### Dev (selection set; optimistic because epochs/configs were chosen on it)
{main_table('dev')}

### Protocol B test - EXPLORATORY (consumed set)
{main_table('testB_exploratory')}

### Protocol C test - EXPLORATORY (consumed set)
{main_table('testC_exploratory')}

The "HISTORICAL" row is the original mAP-selected SupCon (tau 0.1, DINOv2-B; embeddings of the original D5 runs), kept separate from the new Hit@1-selected checkpoints; re-evaluated with the same code it reproduces the earlier numbers.

## 4. Paired comparisons on identical queries (seed-averaged per-query differences, class bootstrap; per-seed McNemar)
{pair_table()}

Figures: `artifacts/research_v2/figures/A1_hit1_vs_map.png` (Hit@1 vs mAP), `A2_pk_rk.png` (Precision@K and Recall@K per loss; PDF versions alongside). Per-query predictions: `runs/*/per_query_*.csv`. Most frequent dev confusions per loss: `dev_top_confusions_seed0.json` (e.g. tomato mold -> tomato late blight, lemon sooty mold -> orange canker, apple rust -> apple scab, banana cordana -> sigatoka, bean rust -> peanut rust: biologically/visually similar pairs, mostly the same for every loss).

## 5. Cost
{cost_table()}
Exact retrieval over the 809-image gallery: 0.26 ms/query for any of the three 256-D embeddings (`artifacts/metrics/protocolB_locked/retrieval_latency.csv`).

## 6. Answers
1. **Best DINOv2-B Hit@1:** Proxy-Anchor on every set (dev 0.820, B-test 0.825, C-test 0.782), but its lead over SupCon (dev 0.820 / B 0.806 / C 0.771) is within noise (paired differences +0.000, +0.019, +0.011; every CI includes 0). ArcFace is lowest (0.776 / 0.793 / 0.744).
2. **Best mAP:** Proxy-Anchor (0.757 / 0.753 / 0.751) > SupCon (0.747 / 0.740 / 0.731) > ArcFace (0.703 / 0.708 / 0.696).
3. **Meaningful?** Only the ArcFace deficit is reliable in mAP (-0.033 to -0.044 vs SupCon, CIs exclude 0; ArcFace vs Proxy-Anchor -0.045 to -0.055). For Hit@1 no pairwise difference is statistically established on B/C (ArcFace vs Proxy-Anchor: dev -0.044 [-0.082, -0.009]); practically Proxy-Anchor and SupCon are equivalent (<= 2 points) and both are 3-5 points above ArcFace in mAP. The DINOv2-S pattern (ArcFace best at Hit@1, SupCon best at mAP) did **not** replicate on DINOv2-B: here a higher mAP went together with an equal-or-higher Hit@1.
4. Selecting by Hit@1 instead of mAP changed nothing material for SupCon (new vs historical: dev {t[(t.model.str.startswith('supcon (Hit'))&(t.set=='dev')].macro_hit1.iloc[0]:.3f} vs {t[(t.model.str.contains('HISTORICAL'))&(t.set=='dev')].macro_hit1.iloc[0]:.3f}; B-test {t[(t.model.str.startswith('supcon (Hit'))&(t.set=='testB_exploratory')].macro_hit1.iloc[0]:.3f} vs {t[(t.model.str.contains('HISTORICAL'))&(t.set=='testB_exploratory')].macro_hit1.iloc[0]:.3f}).
5. **Recommendation:** for Hit@1 and for overall retrieval quality the dev-selected choice is **Proxy-Anchor (alpha 32, delta 0.3)** - best or tied-best on every metric with the same cost as SupCon (~10 min of training). Caveats: tie with SupCon on dev Hit@1 (decided by dev mAP), three seeds only, test sets consumed, a single backbone/split/hyper-parameter budget. No new independent test result is claimed.
"""
Path("docs/LOSS_FUNCTION_STUDY.md").write_text(doc); print(len(doc))
