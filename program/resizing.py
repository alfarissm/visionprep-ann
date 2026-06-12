"""Tahap 2: Resizing dengan Average Pooling atau Max Pooling.

Citra dibagi menjadi wilayah-wilayah pooling yang proporsional terhadap rasio
ukuran asal dan ukuran hasil. Setiap piksel hasil diambil dari satu wilayah:
rata-rata (Average) atau nilai maksimum (Max).
"""

import numpy as np

AVERAGE = "average"
MAX = "max"


def resize_pooling(gray, out_rows, out_cols, method):
    """Perkecil citra grayscale menjadi (out_rows, out_cols) dengan pooling."""
    in_rows, in_cols = gray.shape
    if out_rows > in_rows or out_cols > in_cols:
        raise ValueError("Ukuran hasil harus lebih kecil dari ukuran asal.")
    gray = gray.astype(np.float64)
    result = np.zeros(shape=(out_rows, out_cols), dtype=np.float64)
    for i in range(out_rows):
        r0 = (i * in_rows) // out_rows
        r1 = max(((i + 1) * in_rows) // out_rows, r0 + 1)
        for j in range(out_cols):
            c0 = (j * in_cols) // out_cols
            c1 = max(((j + 1) * in_cols) // out_cols, c0 + 1)
            region = gray[r0:r1, c0:c1]
            if method == AVERAGE:
                result[i, j] = np.sum(region) / region.size
            else:
                result[i, j] = np.max(region)
    return result
