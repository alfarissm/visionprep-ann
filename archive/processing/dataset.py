"""Tahap 4: Creating Dataset (inputs + labels).

Setiap gambar <nama>_bin.jpg menjadi satu baris dataset:
- Kolom 0      : nomor sampel.
- Kolom 1..N   : nilai piksel hasil flatten (row-major), N = row*col.

Label dibuat one-hot per kelas:
- Kolom 0      : nomor sampel.
- Kolom 1..K   : one-hot, K = jumlah kelas.

Kelas dideteksi otomatis dari prefix nama file (sebelum nomor urut),
misal "lingkaran (3)" -> kelas "lingkaran". Mengikuti format dataset
materi M10-M14 (kolom 0 nomor sampel, file .npy).
"""

import os

import numpy as np

from . import imgio


def collect_samples(folder):
    """Kelompokkan file _bin.jpg per kelas, urut kelas lalu nomor sampel."""
    files = imgio.list_jpg(folder, suffix=imgio.SUFFIX_BINARY)
    classes = []
    for filename in files:
        cls = imgio.class_of(imgio.base_name(filename))
        if cls not in classes:
            classes.append(cls)
    classes.sort()
    ordered = []
    for cls in classes:
        for filename in files:
            if imgio.class_of(imgio.base_name(filename)) == cls:
                ordered.append((filename, cls))
    return ordered, classes


def process_folder(folder, log=print):
    """Bangun inputs_<n>_<kolom>.npy, labels_<n>_<K+1>.npy, classes.txt."""
    samples, classes = collect_samples(folder)
    if not samples:
        log("Tidak ada gambar _bin.jpg. Jalankan Binarization dulu.")
        return None
    if len(classes) < 2:
        log(f"Hanya ditemukan 1 kelas ({classes[0]}). "
            "Dataset butuh minimal 2 kelas.")
        return None

    # Ukuran gambar diambil dari sampel pertama; semua harus sama.
    first = imgio.read_image(os.path.join(folder, samples[0][0]))
    rows = first.shape[0]
    cols = first.shape[1]
    n_pixels = rows * cols
    n_samples = len(samples)
    n_classes = len(classes)

    inputs = np.zeros(shape=(n_samples, n_pixels + 1), dtype=np.uint16)
    labels = np.zeros(shape=(n_samples, n_classes + 1), dtype=float)

    for k, (filename, cls) in enumerate(samples):
        pic = imgio.read_image(os.path.join(folder, filename))
        if pic.shape[0] != rows or pic.shape[1] != cols:
            log(f"Ukuran {filename} ({pic.shape[0]}x{pic.shape[1]}) beda "
                f"dari sampel pertama ({rows}x{cols}). Dibatalkan.")
            return None
        gray = pic[:, :, 0] if pic.ndim == 3 else pic  # R=G=B, satu channel
        inputs[k, 0] = k
        inputs[k, 1:] = gray.reshape(n_pixels)
        labels[k, 0] = k
        labels[k, 1 + classes.index(cls)] = 1

    inputs_name = f"inputs_{n_samples}_{n_pixels + 1}.npy"
    labels_name = f"labels_{n_samples}_{n_classes + 1}.npy"
    np.save(os.path.join(folder, inputs_name), inputs)
    np.save(os.path.join(folder, labels_name), labels)

    # Simpan urutan kelas + ukuran gambar untuk tahap ANN.
    with open(os.path.join(folder, "classes.txt"), "w", encoding="utf-8") as f:
        f.write(f"{rows} {cols}\n")
        for cls in classes:
            f.write(cls + "\n")

    log(f"Kelas terdeteksi ({n_classes}): {', '.join(classes)}")
    for cls in classes:
        jumlah = sum(1 for _, c in samples if c == cls)
        log(f"  {cls}: {jumlah} sampel")
    log(f"Dataset inputs : {inputs_name}  (shape {inputs.shape})")
    log(f"Dataset labels : {labels_name}  (shape {labels.shape})")
    log("Disimpan di folder yang sama dengan gambar.")
    return inputs_name, labels_name
