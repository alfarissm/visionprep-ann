"""Tahap 2: Resizing dengan Average Pooling atau Max Pooling.

Gambar dibagi menjadi wilayah-wilayah pooling. Setiap piksel hasil diambil
dari satu wilayah pada gambar asal:
- Average pooling: nilai rata-rata wilayah.
- Max pooling   : nilai maksimum wilayah.

Batas wilayah dihitung proporsional sehingga seluruh piksel gambar asal
tercakup walaupun ukurannya tidak habis dibagi ukuran hasil.
"""

import os

import numpy as np

from . import imgio

AVERAGE = "average"
MAX = "max"


def resize_pooling(gray, out_rows, out_cols, method):
    """Perkecil array grayscale 2 dimensi menjadi (out_rows, out_cols)."""
    in_rows, in_cols = gray.shape
    if out_rows > in_rows or out_cols > in_cols:
        raise ValueError(
            f"Ukuran hasil ({out_rows}x{out_cols}) harus lebih kecil dari "
            f"ukuran asal ({in_rows}x{in_cols})."
        )

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
            elif method == MAX:
                result[i, j] = np.max(region)
            else:
                raise ValueError(f"Metode pooling tidak dikenal: {method}")

    return result


def process_folder(folder, out_rows, out_cols, method, log=print):
    """Resize semua <nama>_gs.jpg di folder menjadi <nama>_rs.jpg."""
    files = imgio.list_jpg(folder, suffix=imgio.SUFFIX_GRAYSCALE)
    if not files:
        log("Tidak ada gambar _gs.jpg. Jalankan Color-to-Grayscale dulu.")
        return 0

    count = 0
    for filename in files:
        pic = imgio.read_image(os.path.join(folder, filename))
        gray = pic[:, :, 0] if pic.ndim == 3 else pic  # R=G=B, ambil satu channel
        small = resize_pooling(gray, out_rows, out_cols, method)
        out_name = imgio.base_name(filename) + imgio.SUFFIX_RESIZE + ".jpg"
        imgio.save_gray_jpg(os.path.join(folder, out_name), small)
        count += 1
        log(f"  {filename} -> {out_name}")

    log(f"Selesai. {count} gambar di-resize ke {out_rows}x{out_cols} "
        f"({method} pooling).")
    return count
