from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def overlay(img: np.ndarray, mask: np.ndarray, alpha=0.45) -> np.ndarray:
    out = img.astype(float).copy()
    m = mask != 0
    out[m] = (1 - alpha) * out[m] + alpha * np.array([255, 0, 0])
    return out.astype(np.uint8)


def contact_sheet(df, root: Path, out: Path, ncols=6, title_col="raw_class_label"):
    """df rows -> grid; each cell: image with red mask overlay, title = id + label + occupancy."""
    n = len(df)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.9))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), df.itertuples()):
        im = np.array(Image.open(root / r.image_path).convert("RGB"))
        mk = np.array(Image.open(root / r.mask_path))
        ax.imshow(overlay(im, mk))
        ax.set_title(f"{r.image_id} {getattr(r, title_col)}\nocc={r.occupancy:.2f}", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
