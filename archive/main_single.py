"""Tool Preprocessing Gambar untuk Pembuatan Dataset Model Vision.

UAS INF322 Image Processing. Aplikasi desktop semi-otomatis:
1. Color-to-Grayscale Conversion
2. Resizing (Average Pooling / Max Pooling)
3. Binarization
4. Creating Dataset (inputs + labels)
5. Randomize Dataset
Plus pembuktian: Training & Testing ANN dengan dataset yang dihasilkan.

Mengikuti gaya kode dosen (M10-M14): HANYA memakai NumPy dan Matplotlib,
ditambah Tkinter untuk antarmuka. Seluruh program berada pada satu berkas ini,
tanpa pustaka lain. Penanganan berkas dilakukan melalui dialog Tkinter dan
operasi string biasa.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

# ========================================================================
# BAGIAN 1 — FUNGSI PENGOLAHAN CITRA (NumPy + Matplotlib)
# ========================================================================

AVERAGE = "average"
MAX = "max"
SUF_GS = "_gs"
SUF_RS = "_rs"
SUF_BIN = "_bin"

# Bobot luminance ITU-R BT.601 untuk konversi grayscale.
W_R, W_G, W_B = 0.299, 0.587, 0.114


def stem(path):
    """Path tanpa ekstensi: 'a/b (3).jpg' -> 'a/b (3)'."""
    return path[:path.rfind(".")]


def name_of(path):
    """Nama berkas tanpa folder: 'a/b (3).jpg' -> 'b (3).jpg'."""
    i = max(path.rfind("/"), path.rfind("\\"))
    return path[i + 1:]


def name_stem(path):
    """Nama berkas tanpa folder & tanpa ekstensi."""
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
    """Nomor sampel di dalam tanda kurung: 'katakana_a (3)' -> 3."""
    nm = name_stem(path)
    if "(" in nm and ")" in nm:
        try:
            return int(nm[nm.find("(") + 1:nm.rfind(")")])
        except ValueError:
            return 0
    return 0


def to_grayscale(pic):
    """Konversi citra (row, col, ch) menjadi grayscale dengan luminance."""
    if pic.ndim == 2:
        return pic.astype(np.float64)
    pic = pic.astype(np.float64)
    return W_R * pic[:, :, 0] + W_G * pic[:, :, 1] + W_B * pic[:, :, 2]


def resize_pooling(gray, out_rows, out_cols, method):
    """Perkecil citra grayscale menjadi (out_rows, out_cols) dengan pooling."""
    in_rows, in_cols = gray.shape
    if out_rows > in_rows or out_cols > in_cols:
        raise ValueError("Ukuran hasil harus lebih kecil dari ukuran asal.")
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
            else:
                result[i, j] = np.max(region)
    return result


def binarize(gray, threshold, invert):
    """Ubah citra grayscale menjadi biner 0/255 (inversi lalu threshold)."""
    gray = gray.astype(np.float64)
    if invert:
        gray = 255 - gray
    result = np.zeros(shape=gray.shape, dtype=np.uint8)
    result[gray >= threshold] = 255
    return result


def save_gray_jpg(path, gray):
    """Simpan array grayscale 2D sebagai .jpg tiga kanal (R=G=B)."""
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    arr = np.zeros(shape=(gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    arr[:, :, 0] = gray
    arr[:, :, 1] = gray
    arr[:, :, 2] = gray
    plt.imsave(path, arr)


def read_channel(path):
    """Baca .jpg dan ambil satu kanal (untuk citra grayscale/biner)."""
    img = plt.imread(path)
    return img[:, :, 0] if img.ndim == 3 else img


def safe_read(path):
    try:
        return plt.imread(path)
    except Exception:
        return None


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


# ========================================================================
# BAGIAN 2 — ANTARMUKA (Tkinter)
# ========================================================================

BG = "#0C0F14"
SURFACE = "#141922"
SURFACE2 = "#1B222E"
BORDER = "#2A3340"
TEXT = "#EDF2F8"
MUTED = "#9DABBF"
ACCENT = "#C8FF3C"
ACCENT_HI = "#D6FF6B"
INK = "#0C0F14"
DANGER = "#FF6B6B"

STAGES = [
    ("Color-to-Grayscale", "grayscale"),
    ("Resizing", "resizing"),
    ("Binarization", "binarization"),
    ("Creating Dataset", "dataset"),
    ("Randomize Dataset", "randomize"),
    ("Train ANN", "ann_train"),
    ("Test ANN (insample)", "ann_test"),
    ("Test ANN (custom)", "ann_custom"),
]

IMAGE_STAGE_IO = {
    "grayscale": ("", SUF_GS),
    "resizing": (SUF_GS, SUF_RS),
    "binarization": (SUF_RS, SUF_BIN),
}


def round_points(x1, y1, x2, y2, r):
    return [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]


class Card(tk.Canvas):
    """Panel sudut-bulat; konten di self.body."""

    def __init__(self, master, radius=12, pad=10, fill=SURFACE,
                 border=BORDER, bg=BG, expand=False):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0)
        self.radius, self.pad, self.fill, self.border = radius, pad, fill, border
        self.expand = expand
        self.body = tk.Frame(self, bg=fill)
        self._win = self.create_window(pad, pad, anchor="nw", window=self.body)
        self.bind("<Configure>", self._on_canvas)
        if not expand:
            self.body.bind("<Configure>", self._on_body)

    def _draw(self, w, h):
        self.delete("shape")
        pts = round_points(1, 1, w - 1, h - 1, self.radius)
        self.create_polygon(pts, smooth=True, splinesteps=32, fill=self.fill,
                            outline=self.border, tags="shape")
        self.tag_lower("shape")

    def _on_canvas(self, event):
        self._draw(event.width, event.height)
        self.itemconfigure(self._win, width=event.width - 2 * self.pad)
        if self.expand:
            self.itemconfigure(self._win, height=event.height - 2 * self.pad)

    def _on_body(self, event):
        self.configure(height=event.height + 2 * self.pad)


class RoundedButton(tk.Canvas):
    """Tombol sudut-bulat dengan efek hover."""

    def __init__(self, master, text, command, fill=ACCENT, fg=INK,
                 hover=ACCENT_HI, font=("Bahnschrift", 13), height=40,
                 radius=11, bg=BG):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0,
                         height=height)
        self.command, self.fill, self.hover, self.radius = command, fill, hover, radius
        self._text = self.create_text(0, 0, text=text, fill=fg, font=font)
        self.bind("<Configure>", self._on_configure)
        self.bind("<Button-1>", lambda e: self._fire())
        self.bind("<Enter>", lambda e: self._paint(self.hover))
        self.bind("<Leave>", lambda e: self._paint(self.fill))
        self.configure(cursor="hand2")
        self._enabled = True

    def _on_configure(self, event):
        self.delete("shape")
        pts = round_points(1, 1, event.width - 1, event.height - 1, self.radius)
        self.create_polygon(pts, smooth=True, splinesteps=28,
                            fill=self.fill, outline="", tags="shape")
        self.tag_lower("shape")
        self.coords(self._text, event.width / 2, event.height / 2)

    def _paint(self, color):
        if self._enabled:
            self.itemconfigure("shape", fill=color)

    def _fire(self):
        if self._enabled:
            self.command()

    def set_enabled(self, enabled, text=None):
        self._enabled = enabled
        self.itemconfigure("shape", fill=self.fill if enabled else BORDER)
        if text is not None:
            self.itemconfigure(self._text, text=text)
        self.configure(cursor="hand2" if enabled else "arrow")


class StageButton(tk.Frame):
    """Baris tahap klik-able dengan nomor + judul dan status aktif."""

    def __init__(self, master, index, title, command):
        super().__init__(master, bg=SURFACE, cursor="hand2")
        self.command = command
        self.active = False
        self.bar = tk.Frame(self, bg=SURFACE, width=3)
        self.bar.pack(side="left", fill="y")
        self.inner = tk.Frame(self, bg=SURFACE)
        self.inner.pack(side="left", fill="x", expand=True, padx=(10, 8), pady=6)
        self.num = tk.Label(self.inner, text=f"{index:02d}", bg=SURFACE,
                            fg=MUTED, font=("Cascadia Mono", 10))
        self.num.pack(side="left")
        self.label = tk.Label(self.inner, text=title, bg=SURFACE, fg=TEXT,
                             font=("Segoe UI", 11), anchor="w")
        self.label.pack(side="left", padx=(8, 0))
        for w in (self, self.inner, self.num, self.label, self.bar):
            w.bind("<Button-1>", lambda e: self.command())
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)

    def _enter(self, _e):
        if not self.active:
            self._paint(SURFACE2, TEXT, MUTED)

    def _leave(self, _e):
        if not self.active:
            self._paint(SURFACE, TEXT, MUTED)

    def set_active(self, active):
        self.active = active
        if active:
            self._paint(SURFACE2, ACCENT, ACCENT, bar=ACCENT)
        else:
            self._paint(SURFACE, TEXT, MUTED, bar=SURFACE)

    def _paint(self, bg, label_fg, num_fg, bar=None):
        for w in (self, self.inner, self.num, self.label):
            w.configure(bg=bg)
        self.num.configure(fg=num_fg)
        self.label.configure(fg=label_fg)
        if bar is not None:
            self.bar.configure(bg=bar)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VISIONPREP · Image Preprocessing Tool — INF322")
        self.geometry("1200x800")
        self.minsize(1080, 720)
        self.configure(bg=BG)

        # Status dataset (dipakai antar-tahap, disimpan di memori).
        self.raw_paths = []
        self.classes = []
        self.rows = self.cols = None
        self.inputs_path = self.labels_path = None
        self.rand_inputs_path = self.rand_labels_path = None
        self.weights = None
        self.last_pred = None

        self.stage_key = STAGES[0][1]
        self.stage_buttons = {}

        self.var_folder = tk.StringVar()
        self.var_pooling = tk.StringVar(value=AVERAGE)
        self.var_row = tk.StringVar(value="20")
        self.var_col = tk.StringVar(value="30")
        self.var_threshold = tk.StringVar(value="100")
        self.var_invert = tk.BooleanVar(value=True)
        self.var_hidden = tk.StringVar(value="20")
        self.var_epochs = tk.StringVar(value="100")
        self.var_sample = tk.StringVar(value="0")

        self._build()
        self.select_stage(self.stage_key)

    # --------------------------------------------------------------- layout
    def _build(self):
        self._build_header()
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(8, 12))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_sidebar(body)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_preview(right)
        self._build_params(right)

    def _build_header(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=14, pady=(12, 8))
        left = tk.Frame(header, bg=BG)
        left.pack(side="left")
        row = tk.Frame(left, bg=BG)
        row.pack(anchor="w")
        tk.Label(row, text="VISION", bg=BG, fg=TEXT,
                 font=("Bahnschrift", 22)).pack(side="left")
        tk.Label(row, text="PREP", bg=BG, fg=ACCENT,
                 font=("Bahnschrift", 22)).pack(side="left")
        tk.Label(left, text="PREPROCESSING PIPELINE · DATASET MODEL VISION",
                 bg=BG, fg=MUTED, font=("Cascadia Mono", 7)).pack(anchor="w")
        self.lbl_status = tk.Label(header, text="● IDLE", bg=BG, fg=MUTED,
                                   font=("Cascadia Mono", 9))
        self.lbl_status.pack(side="right", pady=(6, 0))

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=BG, width=300)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)

        self._section_label(side, "Folder (.jpg)")
        folder = Card(side, radius=11, pad=6)
        folder.pack(fill="x", pady=(4, 14))
        tk.Entry(folder.body, textvariable=self.var_folder, bg=SURFACE, fg=TEXT,
                 relief="flat", font=("Segoe UI", 11),
                 insertbackground=ACCENT).pack(side="left", fill="x",
                                               expand=True, padx=(8, 6), pady=5)
        browse = RoundedButton(folder.body, "BROWSE", self.choose_files,
                               fill=SURFACE2, fg=TEXT, hover=BORDER,
                               font=("Cascadia Mono", 10), height=30, radius=8,
                               bg=SURFACE)
        browse.configure(width=84)
        browse.pack(side="right", padx=(0, 3), pady=3)

        self._section_label(side, "TAHAP PROSES")
        stages = Card(side, radius=11, pad=6)
        stages.pack(fill="x", pady=(4, 14))
        for i, (title, key) in enumerate(STAGES, start=1):
            btn = StageButton(stages.body, i, title,
                              command=lambda k=key: self.select_stage(k))
            btn.pack(fill="x")
            self.stage_buttons[key] = btn
            if i < len(STAGES):
                tk.Frame(stages.body, bg=BG, height=1).pack(fill="x")

        self.btn_start = RoundedButton(side, "▶  START", self.start,
                                       font=("Bahnschrift", 15), height=50)
        self.btn_start.pack(fill="x", pady=(2, 12))

        self._build_log(side)

    def _build_preview(self, parent):
        card = Card(parent, radius=12, pad=8, expand=True)
        card.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        bar = tk.Frame(card.body, bg=SURFACE)
        bar.pack(fill="x", padx=4, pady=(2, 4))
        tk.Label(bar, text="PREVIEW", bg=SURFACE, fg=TEXT,
                 font=("Bahnschrift", 15)).pack(side="left")
        self.lbl_preview_stage = tk.Label(bar, text="", bg=SURFACE, fg=ACCENT,
                                          font=("Cascadia Mono", 10))
        self.lbl_preview_stage.pack(side="right")

        self.fig = Figure(figsize=(5.6, 3.8), dpi=100, facecolor=SURFACE)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.03,
                                 wspace=0.06)
        self.ax_before = self.fig.add_subplot(1, 2, 1)
        self.ax_after = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=card.body)
        self.canvas.get_tk_widget().configure(bg=SURFACE, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._render_blank()

    def _build_params(self, parent):
        rowf = tk.Frame(parent, bg=BG)
        rowf.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        rowf.columnconfigure((0, 1, 2), weight=1, uniform="param")

        c1 = Card(rowf, radius=12, pad=10)
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._param_title(c1.body, "RESIZING")
        g1 = tk.Frame(c1.body, bg=SURFACE)
        g1.pack(fill="x")
        self._radio(g1, "Average Pooling", AVERAGE, 0, 0, columnspan=2)
        self._radio(g1, "Max Pooling", MAX, 1, 0, columnspan=2)
        tk.Label(g1, text="Row", bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 11)).grid(row=2, column=0, sticky="w",
                                            pady=(8, 0))
        self._entry(g1, self.var_row, 2, 1)
        tk.Label(g1, text="Col", bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 11)).grid(row=3, column=0, sticky="w",
                                            pady=(2, 0))
        self._entry(g1, self.var_col, 3, 1)

        c2 = Card(rowf, radius=12, pad=10)
        c2.grid(row=0, column=1, sticky="nsew", padx=4)
        self._param_title(c2.body, "BINARIZATION")
        g2 = tk.Frame(c2.body, bg=SURFACE)
        g2.pack(fill="x")
        tk.Label(g2, text="Threshold", bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self._entry(g2, self.var_threshold, 0, 1)
        self._check(g2, "Inversi (latar hitam, objek putih)", 1, 0,
                    columnspan=2)

        c3 = Card(rowf, radius=12, pad=10)
        c3.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        self._param_title(c3.body, "ANN")
        g3 = tk.Frame(c3.body, bg=SURFACE)
        g3.pack(fill="x")
        for i, (lbl, var) in enumerate([("Hidden", self.var_hidden),
                                        ("Epochs", self.var_epochs),
                                        ("Sampel uji", self.var_sample)]):
            tk.Label(g3, text=lbl, bg=SURFACE, fg=TEXT,
                     font=("Segoe UI", 11)).grid(row=i, column=0, sticky="w",
                                                pady=2)
            self._entry(g3, var, i, 1)

    def _build_log(self, parent):
        card = Card(parent, radius=12, pad=8, fill="#0A0D12",
                   border=BORDER, expand=True)
        card.pack(fill="both", expand=True)
        bar = tk.Frame(card.body, bg="#0A0D12")
        bar.pack(fill="x", padx=4, pady=(2, 2))
        tk.Label(bar, text="● CONSOLE", bg="#0A0D12", fg=ACCENT,
                 font=("Cascadia Mono", 10)).pack(side="left")
        self.text_log = tk.Text(card.body, bg="#0A0D12", fg="#D4DEEA",
                                relief="flat", font=("Cascadia Mono", 11),
                                insertbackground=ACCENT, padx=8, pady=6,
                                state="disabled", wrap="word",
                                highlightthickness=0)
        self.text_log.pack(fill="both", expand=True)
        self.text_log.tag_configure("head", foreground=ACCENT)
        self.text_log.tag_configure("err", foreground=DANGER)

    # ----------------------------------------------------------- komponen UI
    def _section_label(self, parent, text):
        tk.Label(parent, text=text, bg=BG, fg=MUTED,
                 font=("Cascadia Mono", 10)).pack(anchor="w")

    def _param_title(self, parent, text):
        tk.Label(parent, text=text, bg=SURFACE, fg=ACCENT,
                 font=("Cascadia Mono", 10)).pack(anchor="w", pady=(0, 8))

    def _entry(self, parent, var, row, column, sticky="w"):
        wrap = Card(parent, radius=7, pad=0, fill=SURFACE2, bg=SURFACE)
        wrap.configure(width=82)
        wrap.grid(row=row, column=column, sticky=sticky, padx=2, pady=2)
        tk.Entry(wrap.body, textvariable=var, bg=SURFACE2, fg=ACCENT, width=5,
                 relief="flat", font=("Cascadia Mono", 13),
                 insertbackground=ACCENT, justify="center").pack(padx=4, pady=4)

    def _radio(self, parent, text, value, row, column, columnspan=1):
        tk.Radiobutton(parent, text=text, value=value, variable=self.var_pooling,
                       bg=SURFACE, fg=TEXT, selectcolor=SURFACE2,
                       activebackground=SURFACE, activeforeground=ACCENT,
                       font=("Segoe UI", 11), anchor="w", highlightthickness=0,
                       bd=0).grid(row=row, column=column, columnspan=columnspan,
                                  sticky="w")

    def _check(self, parent, text, row, column, columnspan=1):
        tk.Checkbutton(parent, text=text, variable=self.var_invert, bg=SURFACE,
                       fg=TEXT, selectcolor=SURFACE2, activebackground=SURFACE,
                       activeforeground=ACCENT, font=("Segoe UI", 11),
                       anchor="w", justify="left", wraplength=150,
                       highlightthickness=0, bd=0).grid(
            row=row, column=column, columnspan=columnspan, sticky="w",
            pady=(6, 0))

    # ----------------------------------------------------------- preview viz
    def _style_axis(self, ax, caption, color):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(SURFACE2)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.set_title(caption, color=color, fontsize=13, fontfamily="monospace",
                     pad=8, loc="left", fontweight="bold")

    def _placeholder(self, ax, caption, text):
        ax.clear()
        self._style_axis(ax, caption, MUTED)
        ax.text(0.5, 0.5, text, color=MUTED, fontsize=12, ha="center",
                va="center", fontfamily="monospace", transform=ax.transAxes)

    def _render_blank(self):
        self._placeholder(self.ax_before, "INPUT", "—")
        self._placeholder(self.ax_after, "OUTPUT", "—")
        self.canvas.draw()

    def _show_array(self, ax, caption, img, color):
        ax.clear()
        self._style_axis(ax, caption, color)
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    def refresh_preview(self):
        stage = self.stage_key
        self.lbl_preview_stage.configure(
            text=dict(STAGES).get(stage, "").upper())

        if stage in IMAGE_STAGE_IO:
            if not self.raw_paths:
                self._render_blank()
                return
            in_suf, out_suf = IMAGE_STAGE_IO[stage]
            p0 = self.raw_paths[0]
            in_path = p0 if in_suf == "" else stem(p0) + in_suf + ".jpg"
            out_path = stem(p0) + out_suf + ".jpg"
            img_in = safe_read(in_path)
            if img_in is not None:
                self._show_array(self.ax_before, "INPUT", img_in, TEXT)
            else:
                self._placeholder(self.ax_before, "INPUT",
                                  "tahap sebelumnya belum dijalankan")
            img_out = safe_read(out_path)
            if img_out is not None:
                self._show_array(self.ax_after, "OUTPUT", img_out, ACCENT)
            else:
                self._placeholder(self.ax_after, "OUTPUT", "klik START")
        elif stage in ("ann_test", "ann_custom") and self.last_pred is not None:
            picture, actual, predicted = self.last_pred
            self.ax_before.clear()
            if actual is None:
                self._style_axis(self.ax_before, "DATA UJI (CUSTOM)", TEXT)
                self.ax_before.imshow(picture, cmap="gray", vmin=0, vmax=255)
                self._placeholder(self.ax_after, "HASIL",
                                  f"ANN : {predicted}\n(gambar luar dataset)")
            else:
                ok = actual == predicted
                self._style_axis(self.ax_before, "DATA UJI", TEXT)
                self.ax_before.imshow(picture, cmap="gray")
                self._placeholder(
                    self.ax_after, "HASIL",
                    f"asli  : {actual}\nANN   : {predicted}\n"
                    f"status: {'BENAR' if ok else 'SALAH'}")
        else:
            note = {
                "dataset": "menghasilkan\ninputs_*.npy + labels_*.npy",
                "randomize": "menghasilkan\nrandom_*.npy",
                "ann_train": "melatih ANN —\nakurasi di console",
                "ann_test": "jalankan Test ANN\nuntuk prediksi",
                "ann_custom": "klik START lalu pilih\ngambar .jpg dari disk",
            }.get(stage, "—")
            self._placeholder(self.ax_before, "INFO", note)
            self._placeholder(self.ax_after, "OUTPUT", "lihat console / folder")
        self.canvas.draw()

    # --------------------------------------------------------------- perilaku
    def select_stage(self, key):
        self.stage_key = key
        for k, btn in self.stage_buttons.items():
            btn.set_active(k == key)
        self.refresh_preview()

    def choose_files(self):
        import os
        folder = filedialog.askdirectory(title="Pilih folder berisi gambar .jpg")
        if not folder:
            return
        paths = [folder + "/" + f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg"))]
        if not paths:
            self.log("Tidak ada gambar .jpg di folder ini.", "err")
            return
        # Abaikan berkas hasil tahap (_gs/_rs/_bin) agar hanya gambar mentah.
        raw = [p for p in paths
               if not name_stem(p).endswith((SUF_GS, SUF_RS, SUF_BIN))]
        if not raw:
            self.log("Tidak ada gambar mentah terpilih.", "err")
            return
        self.raw_paths = sorted(raw, key=lambda p: (class_of(p),
                                                    sample_index(p)))
        self.var_folder.set(f"{folder}  ({len(self.raw_paths)} gambar)")
        # Reset status dataset karena masukan berganti.
        self.inputs_path = self.labels_path = None
        self.rand_inputs_path = self.rand_labels_path = None
        self.weights = None
        self.last_pred = None
        self.refresh_preview()

    def log(self, message, tag=None):
        self.text_log.configure(state="normal")
        self.text_log.insert("end", str(message) + "\n", tag or ())
        self.text_log.see("end")
        self.text_log.configure(state="disabled")
        self.text_log.update_idletasks()

    def set_status(self, text, color=MUTED):
        self.lbl_status.configure(text=text, fg=color)
        self.lbl_status.update_idletasks()

    def read_int(self, var, name, minimum=1):
        try:
            value = int(var.get())
        except ValueError:
            raise ValueError(f"{name} harus berupa angka.")
        if value < minimum:
            raise ValueError(f"{name} minimal {minimum}.")
        return value

    def start(self):
        stage = self.stage_key
        custom_path = None
        if stage == "ann_custom":
            custom_path = filedialog.askopenfilename(
                title="Pilih gambar uji (.jpg) dari luar dataset",
                filetypes=[("JPEG", "*.jpg *.jpeg")])
            if not custom_path:
                return
        elif not self.raw_paths:
            self.log("Pilih gambar dulu (tombol Browse).", "err")
            return

        title = dict((k, t) for t, k in STAGES)[stage]
        self.log("")
        self.log(f"=== {title} ===", "head")
        self.set_status("● RUNNING", ACCENT)
        self.btn_start.set_enabled(False, "…  WORKING")
        self.update_idletasks()
        try:
            self.run_stage(stage, custom_path)
        except ValueError as error:
            self.log(f"Input tidak valid: {error}", "err")
        except Exception as error:
            self.log(f"Terjadi error: {error}", "err")
        finally:
            self.btn_start.set_enabled(True, "▶  START")
            self.set_status("● DONE", ACCENT)
            self.refresh_preview()

    def run_stage(self, stage, custom_path):
        if stage == "grayscale":
            self.do_grayscale()
        elif stage == "resizing":
            rows = self.read_int(self.var_row, "Row")
            cols = self.read_int(self.var_col, "Col")
            self.do_resizing(rows, cols, self.var_pooling.get())
        elif stage == "binarization":
            th = self.read_int(self.var_threshold, "Threshold", 0)
            self.do_binarization(th, self.var_invert.get())
        elif stage == "dataset":
            self.do_dataset()
        elif stage == "randomize":
            self.do_randomize()
        elif stage == "ann_train":
            hidden = self.read_int(self.var_hidden, "Hidden neuron")
            epochs = self.read_int(self.var_epochs, "Epochs")
            self.do_ann_train(hidden, epochs)
        elif stage == "ann_test":
            index = self.read_int(self.var_sample, "Nomor sampel", 0)
            self.do_ann_test(index)
        elif stage == "ann_custom":
            th = self.read_int(self.var_threshold, "Threshold", 0)
            self.do_ann_custom(custom_path, self.var_pooling.get(), th,
                               self.var_invert.get())

    # ----------------------------------------------------- implementasi tahap
    def do_grayscale(self):
        for p in self.raw_paths:
            pic = plt.imread(p)
            save_gray_jpg(stem(p) + SUF_GS + ".jpg", to_grayscale(pic))
            self.log("  " + name_of(p) + " -> " + name_stem(p) + SUF_GS + ".jpg")
        self.log(f"Selesai. {len(self.raw_paths)} gambar dikonversi ke grayscale.")

    def do_resizing(self, rows, cols, method):
        for p in self.raw_paths:
            gray = read_channel(stem(p) + SUF_GS + ".jpg")
            small = resize_pooling(gray, rows, cols, method)
            save_gray_jpg(stem(p) + SUF_RS + ".jpg", small)
            self.log("  " + name_stem(p) + SUF_GS + " -> " + name_stem(p) + SUF_RS)
        self.log(f"Selesai. {len(self.raw_paths)} gambar di-resize ke "
                 f"{rows}x{cols} ({method} pooling).")

    def do_binarization(self, threshold, invert):
        for p in self.raw_paths:
            gray = read_channel(stem(p) + SUF_RS + ".jpg")
            save_gray_jpg(stem(p) + SUF_BIN + ".jpg",
                          binarize(gray, threshold, invert))
            self.log("  " + name_stem(p) + SUF_RS + " -> " + name_stem(p) + SUF_BIN)
        self.log(f"Selesai. {len(self.raw_paths)} gambar dibinarisasi "
                 f"(threshold={threshold}, inversi={'ya' if invert else 'tidak'}).")

    def do_dataset(self):
        classes = sorted(set(class_of(p) for p in self.raw_paths))
        if len(classes) < 2:
            self.log("Dataset butuh minimal 2 kelas (nama berkas berbeda).",
                     "err")
            return
        samples = sorted(self.raw_paths,
                         key=lambda p: (class_of(p), sample_index(p)))
        first = read_channel(stem(samples[0]) + SUF_BIN + ".jpg")
        rows, cols = first.shape[0], first.shape[1]
        npix = rows * cols
        n = len(samples)
        inputs = np.zeros(shape=(n, npix + 1), dtype=np.uint16)
        labels = np.zeros(shape=(n, len(classes) + 1), dtype=float)
        for k, p in enumerate(samples):
            b = read_channel(stem(p) + SUF_BIN + ".jpg")
            inputs[k, 0] = k
            inputs[k, 1:] = b.reshape(npix)
            labels[k, 0] = k
            labels[k, 1 + classes.index(class_of(p))] = 1
        folder = folder_of(samples[0])
        self.inputs_path = f"{folder}/inputs_{n}_{npix + 1}.npy"
        self.labels_path = f"{folder}/labels_{n}_{len(classes) + 1}.npy"
        np.save(self.inputs_path, inputs)
        np.save(self.labels_path, labels)
        self.classes, self.rows, self.cols = classes, rows, cols
        self.log(f"Kelas terdeteksi ({len(classes)}): {', '.join(classes)}")
        for c in classes:
            self.log(f"  {c}: {sum(1 for p in samples if class_of(p) == c)} sampel")
        self.log(f"Dataset inputs : {name_of(self.inputs_path)}  {inputs.shape}")
        self.log(f"Dataset labels : {name_of(self.labels_path)}  {labels.shape}")

    def do_randomize(self):
        if self.inputs_path is None:
            self.log("Jalankan Create Dataset dulu.", "err")
            return
        inputs = np.load(self.inputs_path)
        labels = np.load(self.labels_path)
        m = inputs.shape[0]
        seeds = list(range(m))
        np.random.shuffle(seeds)
        inputs_r = np.zeros(shape=inputs.shape, dtype=inputs.dtype)
        labels_r = np.zeros(shape=labels.shape, dtype=labels.dtype)
        for i in range(m):
            inputs_r[i, :] = inputs[seeds[i], :]
            labels_r[i, :] = labels[seeds[i], :]
        folder = folder_of(self.inputs_path)
        self.rand_inputs_path = f"{folder}/random_{name_of(self.inputs_path)}"
        self.rand_labels_path = f"{folder}/random_{name_of(self.labels_path)}"
        np.save(self.rand_inputs_path, inputs_r)
        np.save(self.rand_labels_path, labels_r)
        self.log(f"Urutan {m} sampel diacak.")
        self.log("Contoh urutan baru (10 pertama): "
                 + ", ".join(str(int(x)) for x in inputs_r[:10, 0]))

    def do_ann_train(self, hidden, epochs, learn_rate=0.001):
        if self.rand_inputs_path is None:
            self.log("Jalankan Randomize Dataset dulu.", "err")
            return
        inputs = np.load(self.rand_inputs_path)[:, 1:].astype(float) / 255.0
        labels = np.load(self.rand_labels_path)[:, 1:]
        m, i_n = inputs.shape
        o_n = labels.shape[1]
        self.log(f"Data latih: {m} sampel, {i_n} input neuron, "
                 f"{hidden} hidden neuron, {o_n} output neuron.")
        w_i_h = np.random.uniform(-0.5, 0.5, (hidden, i_n))
        w_h_o = np.random.uniform(-0.5, 0.5, (o_n, hidden))
        b_i_h = np.zeros((hidden, 1))
        b_h_o = np.zeros((o_n, 1))
        accuracy = 0.0
        for epoch in range(1, epochs + 1):
            correct = 0
            for inp, label in zip(inputs, labels):
                inp = inp.reshape(-1, 1)
                label = label.reshape(-1, 1)
                h = sigmoid(b_i_h + w_i_h @ inp)
                o = sigmoid(b_h_o + w_h_o @ h)
                correct += int(np.argmax(o) == np.argmax(label))
                delta_o = o - label
                w_h_o += -learn_rate * delta_o @ h.T
                b_h_o += -learn_rate * delta_o
                delta_h = w_h_o.T @ delta_o * (h * (1 - h))
                w_i_h += -learn_rate * delta_h @ inp.T
                b_i_h += -learn_rate * delta_h
            accuracy = round(correct / m * 100, 2)
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                self.log(f"  epoch {epoch}: akurasi {accuracy}%")
                self.update_idletasks()
        self.weights = (w_i_h, b_i_h, w_h_o, b_h_o)
        self.log(f"Training selesai. Akurasi akhir: {accuracy}%.")

    def do_ann_test(self, index):
        if self.weights is None:
            self.log("Jalankan Train ANN dulu.", "err")
            return
        inputs = np.load(self.rand_inputs_path)[:, 1:].astype(float)
        labels = np.load(self.rand_labels_path)[:, 1:]
        if not 0 <= index < inputs.shape[0]:
            self.log(f"Nomor sampel harus 0 - {inputs.shape[0] - 1}.", "err")
            return
        w_i_h, b_i_h, w_h_o, b_h_o = self.weights
        inp = (inputs[index] / 255.0).reshape(-1, 1)
        o = sigmoid(b_h_o + w_h_o @ sigmoid(b_i_h + w_i_h @ inp))
        actual = self.classes[int(np.argmax(labels[index]))]
        predicted = self.classes[int(np.argmax(o))]
        self.log(f"Sampel {index}: label asli = {actual}, "
                 f"prediksi ANN = {predicted}.")
        self.last_pred = (inputs[index].reshape(self.rows, self.cols),
                          actual, predicted)

    def do_ann_custom(self, path, method, threshold, invert):
        if self.weights is None or self.rows is None:
            self.log("Jalankan Create Dataset & Train ANN dulu.", "err")
            return
        pic = plt.imread(path)
        gray = to_grayscale(pic)
        small = resize_pooling(gray, self.rows, self.cols, method)
        binary = binarize(small, threshold, invert)
        w_i_h, b_i_h, w_h_o, b_h_o = self.weights
        inp = (binary.reshape(-1, 1).astype(float)) / 255.0
        o = sigmoid(b_h_o + w_h_o @ sigmoid(b_i_h + w_i_h @ inp))
        predicted = self.classes[int(np.argmax(o))]
        self.log(f"Gambar uji '{name_of(path)}' (luar dataset) -> "
                 f"prediksi ANN = {predicted}.")
        self.last_pred = (binary, None, predicted)


if __name__ == "__main__":
    App().mainloop()
