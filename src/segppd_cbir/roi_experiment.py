"""ORACLE diagnostic: crops from ground-truth mask bounding boxes (+margin). Privileged information, not deployable."""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import feature_extractor as fe
from .masks import bbox

MARGIN = 0.10  # fraction of image side added on every side of the union bounding box


def crops(root, inv, margin=MARGIN):
    for r in inv.itertuples():
        im = fe.open_rgb(root / r.image_path)
        m = np.array(Image.open(root / r.mask_path))
        b = bbox(m, margin)
        yield im.crop(b)
