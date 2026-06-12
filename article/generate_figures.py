"""Generator gambar hasil untuk artikel (selain screenshot & diagram alur).

Menghasilkan:
  ilustrasi_pooling.png   - cara kerja Average vs Max Pooling (contoh angka)
  io_grayscale.png        - input berwarna -> output grayscale
  io_resizing.png         - grayscale -> Average Pooling
  io_binarization.png     - hasil resize -> citra biner
  montase_dataset.png     - contoh sampel biner 3 kelas (A, I, U)
  grafik_akurasi.png      - akurasi pelatihan ANN per epoch

Jalankan:  python generate_figures.py
Sumber gambar: folder dataset (katakana ア イ ウ).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from processing import grayscale, resizing, binarization, imgio, ann

FOLDER = "dataset"
CLASSES = ["katakana_a", "katakana_i", "katakana_u"]
ROWS, COLS = 20, 30
THRESHOLD = 70

# Sampel bersih per kelas untuk ilustrasi before/after.
SAMPLE = {"katakana_a": 3, "katakana_i": 0, "katakana_u": 0}


def load_raw(cls, n):
    return imgio.read_image(f"{FOLDER}/{cls} ({n}).jpg")


def pipeline(pic, method=resizing.AVERAGE):
    g = grayscale.to_grayscale(pic)
    r = resizing.resize_pooling(g, ROWS, COLS, method)
    b = binarization.binarize(r, THRESHOLD, True)
    return g, r, b


# --------------------------------------------------------- ilustrasi pooling
def fig_pooling():
    src = np.array([[10, 20, 30, 40],
                    [12, 18, 33, 38],
                    [50, 60, 70, 80],
                    [52, 58, 72, 78]], dtype=float)
    avg = np.zeros((2, 2))
    mx = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            blk = src[i*2:i*2+2, j*2:j*2+2]
            avg[i, j] = blk.mean()
            mx[i, j] = blk.max()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), facecolor="white")
    for ax, data, title in zip(
            axes, [src, avg, mx],
            ["Input (4x4)", "Average Pooling (2x2)", "Max Pooling (2x2)"]):
        ax.imshow(data, cmap="gray", vmin=0, vmax=80)
        ax.set_title(title, fontsize=11)
        for (i, j), v in np.ndenumerate(data):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="red", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Ilustrasi Resizing 4x4 menjadi 2x2 (wilayah pooling 2x2)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("ilustrasi_pooling.png", dpi=150, facecolor="white")
    plt.close()
    print("Tersimpan: ilustrasi_pooling.png")


# --------------------------------------------------------- before / after
def show(ax, img, title, color=False):
    if color:
        ax.imshow(img.astype(np.uint8))
    else:
        ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def fig_io():
    pic = load_raw("katakana_a", SAMPLE["katakana_a"])
    g, r_avg, b = pipeline(pic, resizing.AVERAGE)

    # Grayscale
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.4), facecolor="white")
    show(ax[0], pic, "Input (berwarna)", color=True)
    show(ax[1], g, "Output (grayscale)")
    plt.tight_layout(); plt.savefig("io_grayscale.png", dpi=150,
                                    facecolor="white"); plt.close()
    print("Tersimpan: io_grayscale.png")

    # Resizing (Average Pooling, metode yang dipakai pada dataset)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.4), facecolor="white")
    show(ax[0], g, "Input (grayscale)")
    show(ax[1], r_avg, f"Output Average Pooling ({ROWS}x{COLS})")
    plt.tight_layout(); plt.savefig("io_resizing.png", dpi=150,
                                    facecolor="white"); plt.close()
    print("Tersimpan: io_resizing.png")

    # Binarization
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.4), facecolor="white")
    show(ax[0], r_avg, "Input (hasil resize)")
    show(ax[1], b, f"Output (biner, threshold {THRESHOLD})")
    plt.tight_layout(); plt.savefig("io_binarization.png", dpi=150,
                                    facecolor="white"); plt.close()
    print("Tersimpan: io_binarization.png")


# --------------------------------------------------------- montase dataset
def fig_montage():
    per = 6
    fig, axes = plt.subplots(len(CLASSES), per, figsize=(9, 4.6),
                             facecolor="white")
    label = {"katakana_a": "A", "katakana_i": "I", "katakana_u": "U"}
    for row, cls in enumerate(CLASSES):
        for col in range(per):
            pic = load_raw(cls, col)
            _, _, b = pipeline(pic, resizing.AVERAGE)
            ax = axes[row, col]
            ax.imshow(b, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"Kelas {label[cls]}", fontsize=11)
    fig.suptitle("Contoh sampel biner dataset (3 kelas katakana)", fontsize=12)
    plt.tight_layout(); plt.savefig("montase_dataset.png", dpi=150,
                                    facecolor="white"); plt.close()
    print("Tersimpan: montase_dataset.png")


# --------------------------------------------------------- grafik akurasi ANN
def fig_accuracy():
    inputs, labels = [], []
    for ci, cls in enumerate(CLASSES):
        for n in range(20):
            pic = load_raw(cls, n)
            _, _, b = pipeline(pic, resizing.AVERAGE)
            inputs.append(b.reshape(-1) / 255.0)
            onehot = np.zeros(len(CLASSES)); onehot[ci] = 1
            labels.append(onehot)
    inputs = np.array(inputs)
    labels = np.array(labels)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(inputs))
    inputs, labels = inputs[idx], labels[idx]

    i_n, h_n, o_n = inputs.shape[1], 20, len(CLASSES)
    w_i_h = rng.uniform(-0.5, 0.5, (h_n, i_n))
    w_h_o = rng.uniform(-0.5, 0.5, (o_n, h_n))
    b_i_h = np.zeros((h_n, 1)); b_h_o = np.zeros((o_n, 1))
    lr, epochs = 0.001, 150
    history = []
    for _ in range(epochs):
        correct = 0
        for inp, lab in zip(inputs, labels):
            inp = inp.reshape(-1, 1); lab = lab.reshape(-1, 1)
            h = ann.sigmoid(b_i_h + w_i_h @ inp)
            o = ann.sigmoid(b_h_o + w_h_o @ h)
            correct += int(np.argmax(o) == np.argmax(lab))
            d_o = o - lab
            w_h_o += -lr * d_o @ h.T; b_h_o += -lr * d_o
            d_h = w_h_o.T @ d_o * (h * (1 - h))
            w_i_h += -lr * d_h @ inp.T; b_i_h += -lr * d_h
        history.append(correct / len(inputs) * 100)

    plt.figure(figsize=(7, 4), facecolor="white")
    plt.plot(range(1, epochs + 1), history, color="black", linewidth=1.6)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Akurasi pelatihan (%)", fontsize=11)
    plt.title(f"Akurasi Pelatihan ANN ({len(inputs)} sampel, "
              f"akhir = {history[-1]:.0f}%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(0, 105)
    plt.tight_layout(); plt.savefig("grafik_akurasi.png", dpi=150,
                                    facecolor="white"); plt.close()
    print(f"Tersimpan: grafik_akurasi.png (akurasi akhir {history[-1]:.0f}%)")


def fig_randomize():
    """Ilustrasi urutan sampel sebelum dan sesudah diacak (warna per kelas)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n, per = 60, 20
    before = np.array([i // per for i in range(n)])
    rng = np.random.default_rng(1)
    after = before.copy()
    rng.shuffle(after)

    colors = ["#4C72B0", "#DD8452", "#55A868"]  # A, I, U
    cmap = ListedColormap(colors)
    labels = ["A", "I", "U"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 2.6), facecolor="white")
    for ax, data, title in zip(
            axes, [before, after],
            ["Sebelum diacak (terurut per kelas)", "Sesudah diacak"]):
        ax.imshow(data.reshape(1, n), cmap=cmap, vmin=0, vmax=2,
                  aspect="auto")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_yticks([])
        ax.set_xticks(range(0, n, 10))
        ax.set_xlabel("nomor baris dataset", fontsize=9)
    handles = [Patch(facecolor=colors[i], label=f"Kelas {labels[i]}")
               for i in range(3)]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=3,
               frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("randomize_order.png", dpi=150, facecolor="white")
    plt.close()
    print("Tersimpan: randomize_order.png")


if __name__ == "__main__":
    fig_pooling()
    fig_io()
    fig_montage()
    fig_accuracy()
    fig_randomize()
