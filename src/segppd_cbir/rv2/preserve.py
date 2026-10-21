"""Preservation manifest: sha256 of every Research-V2 checkpoint, registration, split manifest and result table, then marks them read-only (chmod a-w, reversible).
Run before any Research-V3 experiment so later work cannot overwrite historical results. Writes artifacts/research_v3/PRESERVATION_MANIFEST.json."""
import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path("artifacts/research_v2"); pats = ["**/best.pt", "**/registered.json", "**/protocol_P.csv", "**/results*.csv", "**/per_query*.csv", "**/pred_M*.csv", "**/*selection*.json", "**/result.json", "**/audit_summary.json"]
files = sorted({p for pat in pats for p in ROOT.glob(pat) if p.is_file()})
extra = [Path(p) for p in ("docs/RESEARCH_V2_SUMMARY.md", "docs/LOSS_FUNCTION_STUDY.md", "docs/PLANTSEG_CLASSIFICATION_VS_CBIR.md", "docs/DOMAIN_ROBUST_CBIR.md", "docs/CBIR_RESEARCH_REPORT_V2_SECTIONS_TR.md") if Path(p).exists()]
man = {}
for p in files + extra:
    h = hashlib.sha256(); h.update(p.read_bytes()); man[str(p)] = dict(sha256=h.hexdigest(), bytes=p.stat().st_size)
    os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
Path("artifacts/research_v3").mkdir(exist_ok=True); json.dump(man, open("artifacts/research_v3/PRESERVATION_MANIFEST.json", "w"), indent=1)
print(len(man), "files hashed and set read-only;", sum(v["bytes"] for v in man.values()) / 1e9, "GB")
