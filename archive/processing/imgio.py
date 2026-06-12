"""Utilitas baca/simpan gambar .jpg dan parsing nama file.

Hanya memakai NumPy dasar dan Matplotlib dasar sesuai ketentuan project.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt

# Suffix penanda hasil tiap tahap. Output tahap X tidak boleh ikut
# terbaca sebagai input mentah pada tahap berikutnya.
SUFFIX_GRAYSCALE = "_gs"
SUFFIX_RESIZE = "_rs"
SUFFIX_BINARY = "_bin"
ALL_SUFFIXES = (SUFFIX_GRAYSCALE, SUFFIX_RESIZE, SUFFIX_BINARY)


def read_image(path):
    """Baca file .jpg menjadi array NumPy uint8."""
    return plt.imread(path)


def save_gray_jpg(path, gray):
    """Simpan array grayscale 2 dimensi sebagai .jpg 3 channel (R=G=B)."""
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    arr = np.zeros(shape=(gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    arr[:, :, 0] = gray
    arr[:, :, 1] = gray
    arr[:, :, 2] = gray
    plt.imsave(path, arr)


def strip_extension(filename):
    return os.path.splitext(filename)[0]


def base_name(filename):
    """Nama file tanpa ekstensi dan tanpa suffix tahap.

    "huruf (3)_gs.jpg" -> "huruf (3)"
    """
    name = strip_extension(filename)
    for suffix in ALL_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def stage_suffix(filename):
    """Suffix tahap dari nama file, atau "" jika gambar mentah."""
    name = strip_extension(filename)
    for suffix in ALL_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""


def class_of(base):
    """Nama kelas dari nama dasar file: prefix sebelum nomor urut.

    "lingkaran (3)" -> "lingkaran",  "Hangul7" -> "Hangul"
    """
    match = re.match(r"^(.*?)[\s_\-]*\(?(\d+)\)?$", base)
    if match and match.group(1):
        return match.group(1).strip()
    return base


def sample_index(base):
    """Nomor urut sampel dari nama dasar file (untuk pengurutan)."""
    match = re.match(r"^.*?[\s_\-]*\(?(\d+)\)?$", base)
    if match:
        return int(match.group(1))
    return 0


def list_jpg(folder, suffix=""):
    """Daftar file .jpg di folder dengan suffix tahap tertentu.

    suffix "" berarti gambar mentah (tanpa suffix tahap apa pun).
    Hasil diurutkan per kelas lalu per nomor sampel.
    """
    files = []
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg")):
            continue
        if stage_suffix(filename) != suffix:
            continue
        files.append(filename)
    files.sort(key=lambda f: (class_of(base_name(f)), sample_index(base_name(f))))
    return files
