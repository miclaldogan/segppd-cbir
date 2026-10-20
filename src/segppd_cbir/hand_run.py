from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .hand_features import describe

ROOT = Path("data/extracted/SegPPD-101")


def _f(p):
    return describe(ROOT / p)


if __name__ == "__main__":
    inv = pd.read_csv("artifacts/audits/dataset_manifest.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
    with Pool(4) as pool:
        rows = pool.map(_f, list(inv["image_path"]), chunksize=32)
    out = {k: np.stack([r[k] for r in rows]).astype(np.float32) for k in rows[0]}
    np.savez("artifacts/features/hand_blocks.npz", image_id=inv["image_id"].to_numpy(dtype=str), **out)
    print({k: v.shape for k, v in out.items()})
