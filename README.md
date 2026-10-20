# SegPPD-101 content-based retrieval (frozen ResNet50 baseline + trained extensions)

Exact-Euclidean CBIR on the official SegPPD-101 archive (Zenodo 10.5281/zenodo.15114160). The instructor-mandated baseline is training-free (frozen ResNet50, 1000-D logits vs 2048-D pooled features); trained DINOv2 extensions are reported separately under a leakage-controlled protocol.
Start with `docs/EXPERIMENT_RESULTS.md`, `docs/RETRIEVAL_PROTOCOL.md`, `docs/DATASET_AUDIT.md`, `docs/LIMITATIONS.md`, `docs/FINAL_HANDOFF.md` (status, commands, Turkish summary). Data and weights are not committed; raw dataset images must not be redistributed (rights of the underlying photos are undocumented).
Environment: `uv sync` (CPU torch). GPU training uses a second environment (`.venv-gpu`, CUDA torch) - see `docs/RESUME.md`. Tests: `uv run pytest -q`.
Figures: `artifacts/figures/` (Hit@1 with CIs, P@K/Hit@K curves, Precision-Recall curves for Protocols B and C, B-vs-C comparison, per-class results). Locked results: `artifacts/metrics/protocolB_locked/`, `artifacts/metrics/protocolC_locked/`.
