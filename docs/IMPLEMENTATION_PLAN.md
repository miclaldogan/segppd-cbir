# Implementation plan

Authoritative spec: `SEGPPD101_CLAUDE_CODE_MASTER_PROMPT.md`. Inference-only: frozen torchvision ResNet50, 2048-D pooled features, exact L2 retrieval. No training.

Project path: `/home/misa/projects/university/myildrim/segppd-cbir` (directory was relocated from `.../university/segppd-cbir` mid-session by something outside this session; contents intact, work continues here).

Environment (inspected 2026-10-19): Arch Linux, system Python 3.14 (project pins 3.12 via uv), RTX 4050 Laptop 6 GB, 15 GB RAM, 61 GB free disk, extractors `7z`/`bsdtar` present. Project lives in the existing dedicated directory (it held only the master prompt).

| # | Phase | Status |
|---|-------|--------|
| 1 | Env + Zenodo metadata | DONE (record read; MD5 `2242cf24...`, 247,220,581 B; license field = MIT) |
| 2 | Download + MD5 verify | IN PROGRESS (server ~50 KB/s, resumable curl) |
| 3 | Safe extract + schema discovery + mask audit | TODO |
| 4 | Duplicate audit + leakage-safe split (freeze before scores) | TODO |
| 5 | Frozen ResNet50 features | TODO |
| 6 | Exact L2 retrieval + metrics + random baseline | TODO |
| 7 | Lesion-occupancy analysis | TODO |
| 8 | Oracle ROI diagnostic | TODO |
| 9 | Tests, reports, optional Streamlit demo, handoff | TODO |

Status values used in `FINAL_HANDOFF.md`: DONE / PARTIAL / BLOCKED, each with evidence.
