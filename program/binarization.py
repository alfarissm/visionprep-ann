"""Tahap 3: Binarization.

Citra grayscale diinversi (opsional) lalu di-threshold menjadi 0/255 sehingga
objek menjadi putih pada latar hitam.
"""

import numpy as np


def binarize(gray, threshold, invert):
    """Binarisasi citra grayscale 2D menjadi 0/255."""
    gray = gray.astype(np.float64)
    if invert:
        gray = 255 - gray
    result = np.zeros(shape=gray.shape, dtype=np.uint8)
    result[gray >= threshold] = 255
    return result
