"""Tahap 3: Binarization.

Gambar grayscale diubah menjadi hitam-putih murni (0 atau 255).
Mengikuti materi M10-M14: gambar diinversi dulu (latar menjadi hitam,
objek menjadi putih) lalu di-threshold:
  piksel >= threshold -> 255 (putih / objek)
  piksel <  threshold -> 0   (hitam / latar)
"""

import os

import numpy as np

from . import imgio


def binarize(gray, threshold, invert=True):
    """Binarisasi array grayscale 2 dimensi menjadi 0/255."""
    gray = gray.astype(np.float64)
    if invert:
        gray = 255 - gray
    result = np.zeros(shape=gray.shape, dtype=np.uint8)
    result[gray >= threshold] = 255
    return result


def process_folder(folder, threshold, invert=True, log=print):
    """Binarisasi semua <nama>_rs.jpg di folder menjadi <nama>_bin.jpg."""
    files = imgio.list_jpg(folder, suffix=imgio.SUFFIX_RESIZE)
    if not files:
        log("Tidak ada gambar _rs.jpg. Jalankan Resizing dulu.")
        return 0

    count = 0
    for filename in files:
        pic = imgio.read_image(os.path.join(folder, filename))
        gray = pic[:, :, 0] if pic.ndim == 3 else pic
        binary = binarize(gray, threshold, invert)
        out_name = imgio.base_name(filename) + imgio.SUFFIX_BINARY + ".jpg"
        imgio.save_gray_jpg(os.path.join(folder, out_name), binary)
        count += 1
        log(f"  {filename} -> {out_name}")

    log(f"Selesai. {count} gambar dibinarisasi (threshold={threshold}, "
        f"inversi={'ya' if invert else 'tidak'}).")
    return count
