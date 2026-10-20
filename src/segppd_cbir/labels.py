"""Class-name table from the dataset's own fullname.txt (never inferred from image content)."""
from __future__ import annotations

import re
from pathlib import Path

FILE_RE = re.compile(r"^(\d{3})(\d{2})\.png$")


def parse_fullname(path: Path) -> dict[int, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r"(\d{3})\s+(.+)", line)
        if not m:
            raise ValueError(f"unparseable fullname.txt line: {line!r}")
        cid, name = int(m.group(1)), m.group(2).strip()
        if cid in out:
            raise ValueError(f"duplicate class id {cid}")
        crop, _, cond = name.partition("_")
        out[cid] = {"class_id": cid, "class_name": name, "crop": crop, "condition": cond,
                    "is_healthy_name": "healthy" in cond.lower()}
    return out


def parse_filename(name: str) -> tuple[int, int]:
    m = FILE_RE.match(name)
    if not m:
        raise ValueError(f"filename does not match CCCNN.png: {name}")
    return int(m.group(1)), int(m.group(2))
