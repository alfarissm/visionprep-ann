"""Tool Preprocessing Gambar untuk Pembuatan Dataset Model Vision.

UAS INF322 Image Processing.
Aplikasi desktop (Tkinter) untuk preprocessing semi-otomatis:
1. Color-to-Grayscale Conversion
2. Resizing (Average Pooling / Max Pooling)
3. Binarization
4. Creating Dataset (inputs + labels)
5. Randomize Dataset
Plus pembuktian: Training & Testing ANN dengan dataset yang dihasilkan.

GUI bertema "computer-vision lab console": latar gelap, satu aksen acid-lime,
kartu sudut-bulat (rounded), layout padat, panel preview INPUT -> OUTPUT.
Hanya memakai NumPy dasar, Matplotlib dasar (di-embed via FigureCanvasTkAgg),
dan Tkinter (bawaan Python).

CATATAN: versi ini modular (memakai folder processing/). Versi setara dalam
satu berkas dan hanya numpy+matplotlib+tkinter ada di main_single.py.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from processing import imgio, grayscale, resizing, binarization
from processing import dataset, randomize, ann

# --------------------------------------------------------------------- tema
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
    "grayscale": ("", imgio.SUFFIX_GRAYSCALE),
    "resizing": (imgio.SUFFIX_GRAYSCALE, imgio.SUFFIX_RESIZE),
    "binarization": (imgio.SUFFIX_RESIZE, imgio.SUFFIX_BINARY),
}


def first_file(folder, suffix):
    files = imgio.list_jpg(folder, suffix)
    return files[0] if files else None


def round_points(x1, y1, x2, y2, r):
    """Titik untuk polygon halus membentuk rounded rectangle."""
    return [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]


class Card(tk.Canvas):
    """Panel sudut-bulat. Konten ditaruh di `self.body`.

    expand=False -> tinggi mengikuti isi (auto). expand=True -> isi mengisi
    seluruh kartu (untuk panel yang di-stretch oleh layout).
    """

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
    """Tombol sudut-bulat dengan teks tengah dan efek hover."""

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

    def set_text(self, text):
        self.itemconfigure(self._text, text=text)


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

        self.log_queue = queue.Queue()
        self.worker = None
        self.stage_key = STAGES[0][1]
        self.stage_buttons = {}
        self.last_pred = None
        self._custom_path = None

        self.var_folder = tk.StringVar()
        self.var_pooling = tk.StringVar(value=resizing.AVERAGE)
        self.var_row = tk.StringVar(value="20")
        self.var_col = tk.StringVar(value="30")
        self.var_threshold = tk.StringVar(value="100")
        self.var_invert = tk.BooleanVar(value=True)
        self.var_hidden = tk.StringVar(value="20")
        self.var_epochs = tk.StringVar(value="100")
        self.var_sample = tk.StringVar(value="0")

        self._build()
        self.select_stage(self.stage_key)
        self.poll_log_queue()

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
        right.rowconfigure(0, weight=1)   # preview penuh
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

        self._section_label(side, "FOLDER GAMBAR (.jpg)")
        folder = Card(side, radius=11, pad=6)
        folder.pack(fill="x", pady=(4, 14))
        tk.Entry(folder.body, textvariable=self.var_folder, bg=SURFACE, fg=TEXT,
                 relief="flat", font=("Segoe UI", 11),
                 insertbackground=ACCENT).pack(side="left", fill="x",
                                               expand=True, padx=(8, 6), pady=5)
        browse = RoundedButton(folder.body, "BROWSE", self.choose_folder,
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
        self._radio(g1, "Average Pooling", resizing.AVERAGE, 0, 0, columnspan=2)
        self._radio(g1, "Max Pooling", resizing.MAX, 1, 0, columnspan=2)
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

    def _show_image(self, ax, caption, path, color):
        ax.clear()
        self._style_axis(ax, caption, color)
        img = plt.imread(path)
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    def refresh_preview(self):
        stage = self.stage_key
        folder = self.var_folder.get().strip()
        self.lbl_preview_stage.configure(
            text=dict(STAGES).get(stage, "").upper())

        if not folder or not os.path.isdir(folder):
            self._render_blank()
            return

        if stage in IMAGE_STAGE_IO:
            in_suf, out_suf = IMAGE_STAGE_IO[stage]
            in_file = first_file(folder, in_suf)
            out_file = first_file(folder, out_suf)
            if in_file:
                self._show_image(self.ax_before, "INPUT",
                                 os.path.join(folder, in_file), TEXT)
            else:
                self._placeholder(self.ax_before, "INPUT",
                                  "tahap sebelumnya belum dijalankan")
            if out_file:
                self._show_image(self.ax_after, "OUTPUT",
                                 os.path.join(folder, out_file), ACCENT)
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

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Pilih folder berisi gambar .jpg")
        if folder:
            self.var_folder.set(folder)
            self.refresh_preview()

    def log(self, message, tag=None):
        self.log_queue.put((str(message), tag))

    def poll_log_queue(self):
        while not self.log_queue.empty():
            message, tag = self.log_queue.get_nowait()
            self.text_log.configure(state="normal")
            self.text_log.insert("end", message + "\n", tag or ())
            self.text_log.see("end")
            self.text_log.configure(state="disabled")
        self.after(100, self.poll_log_queue)

    def set_status(self, text, color=MUTED):
        self.lbl_status.configure(text=text, fg=color)

    def read_int(self, var, name, minimum=1):
        try:
            value = int(var.get())
        except ValueError:
            raise ValueError(f"{name} harus berupa angka.")
        if value < minimum:
            raise ValueError(f"{name} minimal {minimum}.")
        return value

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            self.log("Masih ada proses berjalan, tunggu selesai.", "err")
            return
        folder = self.var_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            self.log("Pilih folder gambar dulu (tombol Browse).", "err")
            return
        if self.stage_key == "ann_custom":
            path = filedialog.askopenfilename(
                title="Pilih gambar uji (.jpg) dari luar dataset",
                filetypes=[("JPEG", "*.jpg *.jpeg")])
            if not path:
                return
            self._custom_path = path
        title = dict((k, t) for t, k in STAGES)[self.stage_key]
        self.log("")
        self.log(f"=== {title} ===", "head")
        self.set_status("● RUNNING", ACCENT)
        self.btn_start.set_enabled(False, "…  WORKING")
        self.worker = threading.Thread(target=self.run_stage,
                                       args=(self.stage_key, folder),
                                       daemon=True)
        self.worker.start()

    def run_stage(self, stage, folder):
        try:
            if stage == "grayscale":
                grayscale.process_folder(folder, log=self.log)
            elif stage == "resizing":
                rows = self.read_int(self.var_row, "Row")
                cols = self.read_int(self.var_col, "Col")
                resizing.process_folder(folder, rows, cols,
                                        self.var_pooling.get(), log=self.log)
            elif stage == "binarization":
                threshold = self.read_int(self.var_threshold, "Threshold", 0)
                binarization.process_folder(folder, threshold,
                                            self.var_invert.get(), log=self.log)
            elif stage == "dataset":
                dataset.process_folder(folder, log=self.log)
            elif stage == "randomize":
                randomize.process_folder(folder, log=self.log)
            elif stage == "ann_train":
                hidden = self.read_int(self.var_hidden, "Hidden neuron")
                epochs = self.read_int(self.var_epochs, "Epochs")
                ann.train(folder, hidden, epochs, log=self.log)
            elif stage == "ann_test":
                index = self.read_int(self.var_sample, "Nomor sampel", 0)
                result = ann.predict(folder, index, log=self.log)
                if result is not None:
                    self.last_pred = result
            elif stage == "ann_custom":
                result = ann.predict_external(
                    folder, self._custom_path, self.var_pooling.get(),
                    self.read_int(self.var_threshold, "Threshold", 0),
                    self.var_invert.get(), log=self.log)
                if result is not None:
                    self.last_pred = result
        except ValueError as error:
            self.log(f"Input tidak valid: {error}", "err")
        except Exception as error:
            self.log(f"Terjadi error: {error}", "err")
        finally:
            self.after(0, self._stage_done)

    def _stage_done(self):
        self.btn_start.set_enabled(True, "▶  START")
        self.set_status("● DONE", ACCENT)
        self.refresh_preview()


if __name__ == "__main__":
    App().mainloop()
