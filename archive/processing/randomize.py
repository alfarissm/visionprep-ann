"""Tahap 5: Randomize Dataset.

Baris-baris inputs dan labels diacak bersamaan memakai satu deret indeks
acak yang sama, sehingga pasangan gambar-label tetap konsisten.
Mengikuti pola materi M10-M14 (np.random.shuffle pada deret indeks).
"""

import os

import numpy as np


def find_dataset_files(folder):
    """Cari pasangan inputs_*.npy dan labels_*.npy (bukan random_*)."""
    inputs_name, labels_name = None, None
    for filename in sorted(os.listdir(folder)):
        if filename.startswith("inputs_") and filename.endswith(".npy"):
            inputs_name = filename
        if filename.startswith("labels_") and filename.endswith(".npy"):
            labels_name = filename
    return inputs_name, labels_name


def process_folder(folder, log=print):
    """Acak dataset menjadi random_inputs_*.npy dan random_labels_*.npy."""
    inputs_name, labels_name = find_dataset_files(folder)
    if inputs_name is None or labels_name is None:
        log("Dataset inputs/labels .npy tidak ditemukan. "
            "Jalankan Create Dataset dulu.")
        return None

    inputs = np.load(os.path.join(folder, inputs_name))
    labels = np.load(os.path.join(folder, labels_name))
    m = inputs.shape[0]
    if labels.shape[0] != m:
        log(f"Jumlah baris inputs ({m}) dan labels ({labels.shape[0]}) "
            "tidak sama. Dibatalkan.")
        return None

    # Deret indeks 0..m-1 diacak, lalu dipakai untuk kedua dataset.
    seeds = list(range(m))
    np.random.shuffle(seeds)

    inputs_random = np.zeros(shape=inputs.shape, dtype=inputs.dtype)
    labels_random = np.zeros(shape=labels.shape, dtype=labels.dtype)
    for i in range(m):
        inputs_random[i, :] = inputs[seeds[i], :]
        labels_random[i, :] = labels[seeds[i], :]

    out_inputs = "random_" + inputs_name
    out_labels = "random_" + labels_name
    np.save(os.path.join(folder, out_inputs), inputs_random)
    np.save(os.path.join(folder, out_labels), labels_random)

    log(f"Urutan {m} sampel diacak.")
    log(f"  {inputs_name} -> {out_inputs}")
    log(f"  {labels_name} -> {out_labels}")
    log("Contoh urutan baru (10 pertama): "
        + ", ".join(str(int(x)) for x in inputs_random[:10, 0]))
    return out_inputs, out_labels
