"""Generator diagram alur kerja (gaya draw.io): kotak siku hitam-putih,
tata letak memutar (serpentine) 2 kolom agar tidak terlalu panjang.

Jalankan:  python generate_diagram.py
Hasil   :  diagram_alur.png
"""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

STEPS = [
    "Pengambilan Kebutuhan Pengguna",
    "Perancangan Program dan GUI",
    "Color-to-Grayscale Conversion",
    "Resizing (Average / Max Pooling)",
    "Binarization",
    "Creating Dataset (+ Label)",
    "Randomize Dataset",
    "Penyiapan Antarmuka (GUI)",
    "Pengujian Kotak Hitam",
    "Validasi Dataset via Pelatihan ANN",
]

COLS = 5
BOX_W = 4.0
BOX_H = 1.25
COL_X = [2.5, 7.2, 11.9, 16.6, 21.3]   # pusat horizontal tiap kolom
ROW_PITCH = 2.4                         # jarak vertikal antar baris
LINE_W = 1.4
FONT = 8.5
WRAP = 15


def cell(i):
    """Posisi (baris, kolom) untuk kotak ke-i, pola memutar."""
    row = i // COLS
    pos = i % COLS
    col = pos if row % 2 == 0 else (COLS - 1 - pos)
    return row, col


def draw():
    n = len(STEPS)
    n_rows = (n + COLS - 1) // COLS
    fig, ax = plt.subplots(figsize=(13.0, 3.2), facecolor="white")
    ax.set_xlim(0, 23.8)
    top = n_rows * ROW_PITCH
    ax.set_ylim(0, top + 0.4)
    ax.axis("off")

    centers = []
    for i in range(n):
        row, col = cell(i)
        cx = COL_X[col]
        cy = top - row * ROW_PITCH - BOX_H / 2
        centers.append((cx, cy))

    # Kotak + teks.
    for text, (cx, cy) in zip(STEPS, centers):
        ax.add_patch(Rectangle((cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
                               facecolor="white", edgecolor="black",
                               linewidth=LINE_W, zorder=2))
        ax.text(cx, cy, textwrap.fill(text, width=WRAP), ha="center",
                va="center", fontsize=FONT, color="black", zorder=3)

    # Panah mengikuti urutan, belok di ujung baris.
    for i in range(n - 1):
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        if abs(y1 - y2) < 1e-6:                      # horizontal
            if x2 > x1:
                start, end = (x1 + BOX_W / 2, y1), (x2 - BOX_W / 2, y2)
            else:
                start, end = (x1 - BOX_W / 2, y1), (x2 + BOX_W / 2, y2)
        else:                                        # vertikal (belok turun)
            start, end = (x1, y1 - BOX_H / 2), (x2, y2 + BOX_H / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=16, color="black",
                                     linewidth=LINE_W, zorder=1))

    plt.tight_layout(pad=0.4)
    plt.savefig("diagram_alur.png", dpi=150, facecolor="white",
                bbox_inches="tight")
    print("Tersimpan: diagram_alur.png")


if __name__ == "__main__":
    draw()
