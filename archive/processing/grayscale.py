"""Tahap 1: Color-to-Grayscale Conversion.

Materi M10-M14 memakai rata-rata sederhana gray = (R + G + B) / 3.
Modifikasi pada project ini: memakai rumus luminance terbobot
gray = 0.299 R + 0.587 G + 0.114 B (standar ITU-R BT.601), yang lebih
sesuai dengan sensitivitas mata manusia sehingga kontras objek lebih baik.
"""

import os

import numpy as np

from . import imgio

# Bobot luminance ITU-R BT.601.
W_R, W_G, W_B = 0.299, 0.587, 0.114


def to_grayscale(pic):
    """Konversi array gambar (row, col, ch) menjadi grayscale 2 dimensi."""
    if pic.ndim == 2:
        return pic.astype(np.float64)
    pic = pic.astype(np.float64)
    gray = W_R * pic[:, :, 0] + W_G * pic[:, :, 1] + W_B * pic[:, :, 2]
    return gray


def process_folder(folder, log=print):
    """Konversi semua .jpg mentah di folder menjadi <nama>_gs.jpg."""
    files = imgio.list_jpg(folder, suffix="")
    if not files:
        log("Tidak ada gambar .jpg mentah di folder ini.")
        return 0

    count = 0
    for filename in files:
        pic = imgio.read_image(os.path.join(folder, filename))
        gray = to_grayscale(pic)
        out_name = imgio.base_name(filename) + imgio.SUFFIX_GRAYSCALE + ".jpg"
        imgio.save_gray_jpg(os.path.join(folder, out_name), gray)
        count += 1
        log(f"  {filename} -> {out_name}")

    log(f"Selesai. {count} gambar dikonversi ke grayscale.")
    return count
