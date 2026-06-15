"""Antarmuka Tkinter untuk tool preprocessing — UAS INF322.

Memanggil fungsi inti pada preprocessing.py. Hanya memakai NumPy, Matplotlib,
dan Tkinter. Jalankan: python gui.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

import preprocessing as pp

# --------------------------------------------------------------------- tema
BG = "#F4F6FB"
SURFACE = "#FFFFFF"
SURFACE2 = "#EEF1F7"
BORDER = "#D9DFEA"
TEXT = "#1A2230"
MUTED = "#6B7686"
ACCENT = "#3B6CF6"
ACCENT_HI = "#1E4FD8"
INK = "#FFFFFF"
DANGER = "#E5484D"
CONSOLE = "#F1F3F9"
CONSOLE_FG = "#2A3344"

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
    "grayscale": ("", pp.SUF_GS),
    "resizing": (pp.SUF_GS, pp.SUF_RS),
    "binarization": (pp.SUF_RS, pp.SUF_BIN),
}


def safe_read(path):
    try:
        return plt.imread(path)
    except Exception:
        return None


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
        self.var_pooling = tk.StringVar(value=pp.AVERAGE)
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
        self._radio(g1, "Average Pooling", pp.AVERAGE, 0, 0, columnspan=2)
        self._radio(g1, "Max Pooling", pp.MAX, 1, 0, columnspan=2)
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
        card = Card(parent, radius=12, pad=8, fill=CONSOLE,
                   border=BORDER, expand=True)
        card.pack(fill="both", expand=True)
        bar = tk.Frame(card.body, bg=CONSOLE)
        bar.pack(fill="x", padx=4, pady=(2, 2))
        tk.Label(bar, text="● CONSOLE", bg=CONSOLE, fg=ACCENT,
                 font=("Cascadia Mono", 10)).pack(side="left")
        self.text_log = tk.Text(card.body, bg=CONSOLE, fg=CONSOLE_FG,
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
            in_path = p0 if in_suf == "" else pp.stem(p0) + in_suf + ".jpg"
            out_path = pp.stem(p0) + out_suf + ".jpg"
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

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Pilih folder berisi gambar .jpg")
        if not folder:
            return
        # Ambil gambar mentah saja (abaikan hasil tahap _gs/_rs/_bin).
        names = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg"))
                 and not pp.name_stem(f).endswith((pp.SUF_GS, pp.SUF_RS,
                                                   pp.SUF_BIN))]
        if not names:
            self.log("Tidak ada gambar .jpg mentah di folder ini.", "err")
            return
        raw = [folder + "/" + f for f in names]
        self.raw_paths = sorted(raw, key=lambda p: (pp.class_of(p),
                                                    pp.sample_index(p)))
        self.var_folder.set(f"{folder}  ({len(self.raw_paths)} gambar)")
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
            self.do_resizing(self.read_int(self.var_row, "Row"),
                             self.read_int(self.var_col, "Col"),
                             self.var_pooling.get())
        elif stage == "binarization":
            self.do_binarization(self.read_int(self.var_threshold, "Threshold", 0),
                                 self.var_invert.get())
        elif stage == "dataset":
            self.do_dataset()
        elif stage == "randomize":
            self.do_randomize()
        elif stage == "ann_train":
            self.do_ann_train(self.read_int(self.var_hidden, "Hidden neuron"),
                              self.read_int(self.var_epochs, "Epochs"))
        elif stage == "ann_test":
            self.do_ann_test(self.read_int(self.var_sample, "Nomor sampel", 0))
        elif stage == "ann_custom":
            self.do_ann_custom(custom_path, self.var_pooling.get(),
                               self.read_int(self.var_threshold, "Threshold", 0),
                               self.var_invert.get())

    # ----------------------------------------------------- implementasi tahap
    def do_grayscale(self):
        for p in self.raw_paths:
            pic = plt.imread(p)
            pp.save_gray_jpg(pp.stem(p) + pp.SUF_GS + ".jpg",
                             pp.to_grayscale(pic))
            self.log("  " + pp.name_of(p) + " -> " + pp.name_stem(p) + pp.SUF_GS)
        self.log(f"Selesai. {len(self.raw_paths)} gambar dikonversi grayscale.")

    def do_resizing(self, rows, cols, method):
        for p in self.raw_paths:
            gray = pp.read_channel(pp.stem(p) + pp.SUF_GS + ".jpg")
            pp.save_gray_jpg(pp.stem(p) + pp.SUF_RS + ".jpg",
                             pp.resize_pooling(gray, rows, cols, method))
            self.log("  " + pp.name_stem(p) + pp.SUF_RS)
        self.log(f"Selesai. {len(self.raw_paths)} gambar di-resize ke "
                 f"{rows}x{cols} ({method} pooling).")

    def do_binarization(self, threshold, invert):
        for p in self.raw_paths:
            gray = pp.read_channel(pp.stem(p) + pp.SUF_RS + ".jpg")
            pp.save_gray_jpg(pp.stem(p) + pp.SUF_BIN + ".jpg",
                             pp.binarize(gray, threshold, invert))
            self.log("  " + pp.name_stem(p) + pp.SUF_BIN)
        self.log(f"Selesai. {len(self.raw_paths)} gambar dibinarisasi "
                 f"(threshold={threshold}).")

    def do_dataset(self):
        classes = sorted(set(pp.class_of(p) for p in self.raw_paths))
        if len(classes) < 2:
            self.log("Dataset butuh minimal 2 kelas.", "err")
            return
        inputs, labels, classes, rows, cols = pp.build_dataset(self.raw_paths)
        folder = pp.folder_of(self.raw_paths[0])
        n = inputs.shape[0]
        self.inputs_path = f"{folder}/inputs_{n}_{inputs.shape[1]}.npy"
        self.labels_path = f"{folder}/labels_{n}_{labels.shape[1]}.npy"
        np.save(self.inputs_path, inputs)
        np.save(self.labels_path, labels)
        self.classes, self.rows, self.cols = classes, rows, cols
        self.log(f"Kelas terdeteksi ({len(classes)}): {', '.join(classes)}")
        for c in classes:
            self.log(f"  {c}: {sum(1 for p in self.raw_paths if pp.class_of(p) == c)} sampel")
        self.log(f"Dataset inputs : {pp.name_of(self.inputs_path)}  {inputs.shape}")
        self.log(f"Dataset labels : {pp.name_of(self.labels_path)}  {labels.shape}")

    def do_randomize(self):
        if self.inputs_path is None:
            self.log("Jalankan Create Dataset dulu.", "err")
            return
        inputs = np.load(self.inputs_path)
        labels = np.load(self.labels_path)
        inputs_r, labels_r = pp.randomize(inputs, labels)
        folder = pp.folder_of(self.inputs_path)
        self.rand_inputs_path = f"{folder}/random_{pp.name_of(self.inputs_path)}"
        self.rand_labels_path = f"{folder}/random_{pp.name_of(self.labels_path)}"
        np.save(self.rand_inputs_path, inputs_r)
        np.save(self.rand_labels_path, labels_r)
        self.log(f"Urutan {inputs.shape[0]} sampel diacak.")
        self.log("Contoh urutan baru (10 pertama): "
                 + ", ".join(str(int(x)) for x in inputs_r[:10, 0]))

    def do_ann_train(self, hidden, epochs):
        if self.rand_inputs_path is None:
            self.log("Jalankan Randomize Dataset dulu.", "err")
            return
        inputs = np.load(self.rand_inputs_path)[:, 1:].astype(float) / 255.0
        labels = np.load(self.rand_labels_path)[:, 1:]
        self.weights = pp.ann_train(inputs, labels, hidden, epochs,
                                    log=self.log)

    def do_ann_test(self, index):
        if self.weights is None:
            self.log("Jalankan Train ANN dulu.", "err")
            return
        inputs = np.load(self.rand_inputs_path)[:, 1:].astype(float)
        labels = np.load(self.rand_labels_path)[:, 1:]
        if not 0 <= index < inputs.shape[0]:
            self.log(f"Nomor sampel harus 0 - {inputs.shape[0] - 1}.", "err")
            return
        o = pp.ann_predict(self.weights, inputs[index] / 255.0)
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
        gray = pp.to_grayscale(plt.imread(path))
        small = pp.resize_pooling(gray, self.rows, self.cols, method)
        binary = pp.binarize(small, threshold, invert)
        o = pp.ann_predict(self.weights, binary.reshape(-1).astype(float) / 255.0)
        predicted = self.classes[int(np.argmax(o))]
        self.log(f"Gambar uji '{pp.name_of(path)}' (luar dataset) -> "
                 f"prediksi ANN = {predicted}.")
        self.last_pred = (binary, None, predicted)


if __name__ == "__main__":
    App().mainloop()
