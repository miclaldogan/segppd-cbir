"""Classical hand-crafted descriptors (EXTENSION track). Every block is a per-image vector; no fitting involved."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

S = 128  # working resolution for speed


def _grid(a, n=2):
    h, w = a.shape[:2]
    return [a[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n] for i in range(n) for j in range(n)]


def _hist(x, bins, rng):
    h, _ = np.histogram(x, bins=bins, range=rng)
    h = h.astype(np.float32)
    return np.sqrt(h / max(h.sum(), 1))  # sqrt of L1-normalised hist => Euclid == Hellinger


def hsv_hist(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
    return np.sqrt(h / h.sum())


def lab_moments(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = []
    for c in _grid(lab, 3):
        out += list(c.reshape(-1, 3).mean(0) / 255) + list(c.reshape(-1, 3).std(0) / 255)
    return np.array(out, np.float32)


def lbp(gray):
    out = []
    for P, R in ((8, 1), (16, 2)):
        l = local_binary_pattern(gray, P, R, "uniform")
        for c in [l] + _grid(l, 2):
            out.append(_hist(c, P + 2, (0, P + 2)))
    return np.concatenate(out)


def hog_feat(gray):
    return hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True).astype(np.float32)


def glcm(gray):
    q = (gray // 8).astype(np.uint8)  # 32 levels
    m = graycomatrix(q, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    return np.concatenate([graycoprops(m, p).flatten() for p in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity")]).astype(np.float32)


_GABOR = [cv2.getGaborKernel((15, 15), s, t, l, 0.5) for s in (2.0, 4.0) for l in (4.0, 8.0) for t in np.arange(0, np.pi, np.pi / 6)]


def gabor(gray):
    g = gray.astype(np.float32) / 255
    out = []
    for k in _GABOR:
        r = cv2.filter2D(g, -1, k)
        out += [r.mean(), r.std()]
    return np.array(out, np.float32)


def edge_hist(gray):
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx)
    return np.concatenate([_hist(ang[c], 12, (-np.pi, np.pi)) * 1.0 for c in [np.ones_like(ang, bool)]] +
                          [np.array([mag.mean() / 255, mag.std() / 255], np.float32)])


BLOCKS = {"hsv": hsv_hist, "labmom": lab_moments, "lbp": lbp, "hog": hog_feat, "glcm": glcm, "gabor": gabor, "edge": edge_hist}


def describe(path_or_rgb):
    rgb = path_or_rgb if isinstance(path_or_rgb, np.ndarray) else cv2.cvtColor(cv2.imread(str(path_or_rgb)), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {"hsv": hsv_hist(rgb), "labmom": lab_moments(rgb), "lbp": lbp(gray), "hog": hog_feat(gray), "glcm": glcm(gray),
            "gabor": gabor(gray), "edge": edge_hist(gray)}
