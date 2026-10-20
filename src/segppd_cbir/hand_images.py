"""Hand-crafted feature MAPS rendered as 3-channel images, to be fed to the frozen ResNet50 (EXTENSION track)."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def _norm(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _u8(ch):
    return (np.stack(ch, -1) * 255).astype(np.uint8)


def v_lbp(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return _u8([local_binary_pattern(g, 8, 1, "uniform") / 9.0, local_binary_pattern(g, 16, 2, "uniform") / 17.0, g / 255.0])


def v_gabor(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    ch = [_norm(np.abs(cv2.filter2D(g, -1, cv2.getGaborKernel((21, 21), 4.0, t, 8.0, 0.5)))) for t in (0, np.pi / 3, 2 * np.pi / 3)]
    return _u8(ch)


def v_lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def v_hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def v_grad(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return _u8([_norm(mag), _norm(np.abs(cv2.Laplacian(g, cv2.CV_32F))), g / 255.0])


def v_veg(rgb):
    f = rgb.astype(np.float32) + 1e-3
    s = f.sum(2)
    r, g, b = f[..., 0] / s, f[..., 1] / s, f[..., 2] / s
    return _u8([_norm(2 * g - r - b), _norm(1.4 * r - g), _norm((g - b) / (g + b + 1e-6))])


def v_clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


VARIANTS = {"lbp": v_lbp, "gabor": v_gabor, "lab": v_lab, "hsv": v_hsv, "grad": v_grad, "veg": v_veg, "clahe": v_clahe}
