from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata


def environment_report() -> dict:
    pk = {}
    for p in ["torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn", "pillow", "faiss-cpu", "imagehash", "matplotlib"]:
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pk[p] = None
    cpu = ""
    try:
        cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": cpu,
            "packages": pk}


def dumps(d) -> str:
    return json.dumps(d, indent=2, default=str)
