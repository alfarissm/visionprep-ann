"""Tahap 1: Color-to-Grayscale Conversion.

gray = 0.299 R + 0.587 G + 0.114 B (luminance ITU-R BT.601).
"""

import numpy as np

W_R, W_G, W_B = 0.299, 0.587, 0.114


def to_grayscale(pic):
    """Konversi citra (row, col, ch) menjadi grayscale 2 dimensi."""
    if pic.ndim == 2:
        return pic.astype(np.float64)
    pic = pic.astype(np.float64)
    return W_R * pic[:, :, 0] + W_G * pic[:, :, 1] + W_B * pic[:, :, 2]
