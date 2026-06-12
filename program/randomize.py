"""Tahap 5: Randomize Dataset.

Baris inputs dan labels diacak bersamaan memakai satu deret indeks acak yang
sama, sehingga pasangan gambar-label tetap konsisten.
"""

import numpy as np


def randomize(inputs, labels):
    """Acak baris inputs & labels dengan indeks yang sama. Mengembalikan keduanya."""
    m = inputs.shape[0]
    seeds = list(range(m))
    np.random.shuffle(seeds)
    inputs_r = np.zeros(shape=inputs.shape, dtype=inputs.dtype)
    labels_r = np.zeros(shape=labels.shape, dtype=labels.dtype)
    for i in range(m):
        inputs_r[i, :] = inputs[seeds[i], :]
        labels_r[i, :] = labels[seeds[i], :]
    return inputs_r, labels_r
