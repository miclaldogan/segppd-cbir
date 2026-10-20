# Resume notes (written when the session was paused)

Project path: `/home/misa/projects/university/myildrim/segppd-cbir`
Environments: `.venv` (CPU torch, `uv run ...`) and `.venv-gpu` (CUDA 13 torch, `.venv-gpu/bin/python ...`, RTX 4050 6 GB).

## Done
- Dataset downloaded + MD5 verified; audit, duplicate groups, Protocol A split (`artifacts/splits/split_seed*.csv`).
- Protocol A audit passed (`artifacts/audits/protocol_audit.json`); Protocol B built (`artifacts/splits/protocolB_seed4242.csv`):
  train 772 / gallery 809 / dev queries 150 / LOCKED test queries 225 (never evaluated, never used for selection).
- Frozen features: ResNet50, DINOv2-S/B (`artifacts/features/`).
- Protocol B dev runs saved in `artifacts/extensions/B/`: R1 (ResNet50 ft), D1 (DINOv2-S head only), D2 (DINOv2-S last 4 blocks, ArcFace, lr 3e-5).
- `MISLABELED_rep{1,2,3}_*` = three replicates of (DINOv2-S, last 4 blocks, ArcFace, lr 5e-5); a shell bug dropped the intended args. Kept as noise estimate.

## NOT done (next)
1. D3 (Proxy-Anchor) and D4 (SupCon) on DINOv2-S -- `data/processed/run_dino_b.sh` lines for D3/D4 (D1, D2 already done; do not rerun).
2. Pick config on DEV only; 3 seeds for the best; optional DINOv2-B partial fine-tune.
3. Dev-selected fusion (`segppd_cbir/hybrid.py`); ResNet50 added little on dev (oracle union 0.76 vs DINO-ft 0.70).
4. ONE locked evaluation on `test_query_B` for all final candidates (per-query metrics, class-bootstrap CIs, paired diffs), then docs
   (DATASET_AUDIT, RETRIEVAL_PROTOCOL, FEATURE_REPRESENTATIONS, EXPERIMENT_RESULTS, LIMITATIONS, FINAL_HANDOFF), figures, optional Streamlit demo.
5. Optional external validation set (needs web verification of licence/classes + near-duplicate screen against all 2,263 images).

## Resume commands
```
cd /home/misa/projects/university/myildrim/segppd-cbir
P=".venv-gpu/bin/python -m segppd_cbir.train_embed --backbone dinov2s"
$P --tag D3_dino_p4_proxyanchor_lr3e-5_s0 --mode partial --lr_bb 3e-5 --loss proxyanchor --seed 0
$P --tag D4_dino_p4_supcon_lr3e-5_s0      --mode partial --lr_bb 3e-5 --loss supcon      --seed 0
uv run pytest -q          # 31 tests
```


## UPDATE (later session)
All steps above were completed except the optional external validation and demo. D3/D4 (Proxy-Anchor/SupCon, 3 seeds each), D5 (DINOv2-B + SupCon, 3 seeds) trained; locked Protocol B evaluation was run ONCE (`artifacts/metrics/protocolB_locked/`, candidates frozen in `artifacts/metrics/locked_candidates.json`). The locked set is consumed - see docs/FINAL_HANDOFF.md.

## UPDATE (Protocol C + external set)
- Protocol C (142 locked queries carved from gallery_B) built, audited, evaluated ONCE; see EXPERIMENT_RESULTS.md. Pooled B+C Hit@1 0.794 [0.741, 0.844].
- P-R curves: `python -m segppd_cbir.report_pr <candidates.json> B|C`; B-vs-C figure: `python -m segppd_cbir.report_bc`.
- External validation candidate: PlantSeg (Zenodo 13958858, plantsegv2.zip 1.7 GB, md5 16cb69a5d97a61bc553a572f929490bf, CC BY-NC-ND 4.0: research evaluation only, never redistribute). Download: `data/external/dl.sh` (background, resumable with curl -C -; ~55 KB/s). Completion marker `data/external/dl.done`.
  TODO after download: verify md5; map class names to SegPPD-101 classes; screen ALL external images against all 2,263 SegPPD images (SHA, pHash/dHash, plus DINOv2 near-neighbour check) because provenance of SegPPD-101 images is not documented and may overlap PlantSeg; drop overlaps; evaluate frozen models once as a pre-registered extra test.
- External PlantSeg evaluation DONE (one shot). Models retrained with checkpoints: `data/processed/run_D5ck.sh` -> `artifacts/checkpoints/` (gitignored). Results `artifacts/metrics/external_plantseg/`.
