"""Tahap 4: Creating Dataset (inputs + labels).

Setiap gambar biner di-flatten menjadi satu baris. Kolom 0 berisi nomor sampel,
kolom berikutnya nilai piksel. Label dibuat one-hot; kelas dideteksi otomatis
dari awalan nama berkas.
"""

import numpy as np

import common


def build_dataset(sample_paths):
    """Bangun (inputs, labels, classes, rows, cols) dari berkas _bin.jpg."""
    classes = sorted(set(common.class_of(p) for p in sample_paths))
    samples = sorted(sample_paths,
                     key=lambda p: (common.class_of(p), common.sample_index(p)))
    first = common.read_channel(common.stem(samples[0]) + common.SUF_BIN + ".jpg")
    rows, cols = first.shape[0], first.shape[1]
    npix = rows * cols
    n = len(samples)
    inputs = np.zeros(shape=(n, npix + 1), dtype=np.uint16)
    labels = np.zeros(shape=(n, len(classes) + 1), dtype=float)
    for k, p in enumerate(samples):
        b = common.read_channel(common.stem(p) + common.SUF_BIN + ".jpg")
        inputs[k, 0] = k
        inputs[k, 1:] = b.reshape(npix)
        labels[k, 0] = k
        labels[k, 1 + classes.index(common.class_of(p))] = 1
    return inputs, labels, classes, rows, cols
