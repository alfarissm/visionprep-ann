"""Konstanta dan utilitas bersama (NumPy + Matplotlib).

Berisi penanda akhiran berkas, pembaca/penyimpan citra, serta bantu parsing
nama berkas. Dipakai oleh berkas-berkas fitur dan oleh gui.py.
"""

import numpy as np
import matplotlib.pyplot as plt

# Akhiran nama berkas hasil tiap tahap.
SUF_GS = "_gs"
SUF_RS = "_rs"
SUF_BIN = "_bin"


def stem(path):
    """Path tanpa ekstensi: 'a/b (3).jpg' -> 'a/b (3)'."""
    return path[:path.rfind(".")]


def name_of(path):
    """Nama berkas tanpa folder."""
    i = max(path.rfind("/"), path.rfind("\\"))
    return path[i + 1:]


def name_stem(path):
    """Nama berkas tanpa folder dan tanpa ekstensi."""
    n = name_of(path)
    return n[:n.rfind(".")]


def folder_of(path):
    i = max(path.rfind("/"), path.rfind("\\"))
    return path[:i] if i >= 0 else "."


def class_of(path):
    """Nama kelas dari awalan nama berkas: 'katakana_a (3)' -> 'katakana_a'."""
    nm = name_stem(path)
    return nm.split(" (")[0].strip() if " (" in nm else nm


def sample_index(path):
    """Nomor sampel di dalam kurung: 'katakana_a (3)' -> 3."""
    nm = name_stem(path)
    if "(" in nm and ")" in nm:
        try:
            return int(nm[nm.find("(") + 1:nm.rfind(")")])
        except ValueError:
            return 0
    return 0


def save_gray_jpg(path, gray):
    """Simpan array grayscale 2D sebagai .jpg tiga kanal (R=G=B)."""
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    arr = np.zeros(shape=(gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    arr[:, :, 0] = gray
    arr[:, :, 1] = gray
    arr[:, :, 2] = gray
    plt.imsave(path, arr)


def read_channel(path):
    """Baca .jpg dan ambil satu kanal (citra grayscale/biner: R=G=B)."""
    img = plt.imread(path)
    return img[:, :, 0] if img.ndim == 3 else img
