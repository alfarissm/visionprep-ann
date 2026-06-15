"""Antarmuka Tkinter untuk tool preprocessing — UAS INF322.

Memanggil fungsi inti pada preprocessing.py. Hanya memakai NumPy, Matplotlib,
dan Tkinter. Gaya tampilan klasik Windows (ttk tema 'vista'): tombol timbul,
group box, dan daftar native. Jalankan: python gui.py
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
from tkinter import ttk

import preprocessing as pp

# ----------------------------------------------------------- tema klasik Win7
FACE = "#F0F0F0"      # warna dialog/jendela (Windows 7)
FIELD = "#FFFFFF"     # latar kolom isian / daftar
TEXT = "#1A1A1A"
MUTED = "#5A5A5A"
ACCENT = "#1565C0"    # biru judul (gaya tautan Win7)
BORDER = "#C2C2C2"
SEL_BG = "#CCE4F7"    # sorot pilihan (Win7)
DANGER = "#C42B1C"

F_UI = ("Segoe UI", 9)
F_BOLD = ("Segoe UI", 9, "bold")
F_TITLE = ("Segoe UI", 18, "bold")
F_SUB = ("Segoe UI", 8)
F_MONO = ("Consolas", 9)

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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VisionPrep - Image Preprocessing Tool (INF322)")
        self.geometry("1180x800")
        self.minsize(1060, 720)
        self.configure(bg=FACE)
        self._init_style()

        self.raw_paths = []
        self.classes = []
        self.rows = self.cols = None
        self.inputs_path = self.labels_path = None
        self.rand_inputs_path = self.rand_labels_path = None
        self.weights = None
        self.last_pred = None

        self.stage_key = STAGES[0][1]

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

    # ------------------------------------------------------------------ gaya
    def _init_style(self):
        style = ttk.Style(self)
        for theme in ("vista", "xpnative", "winnative", "clam"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure(".", background=FACE, foreground=TEXT, font=F_UI)
        style.configure("TFrame", background=FACE)
        style.configure("TLabel", background=FACE, foreground=TEXT)
        style.configure("TLabelframe", background=FACE)
        style.configure("TLabelframe.Label", background=FACE, foreground=ACCENT,
                        font=F_BOLD)
        style.configure("TCheckbutton", background=FACE)
        style.configure("TRadiobutton", background=FACE)
        style.configure("Big.TButton", font=("Segoe UI", 12, "bold"),
                        padding=(10, 10))

    # --------------------------------------------------------------- layout
    def _build(self):
        self._build_header()
        body = ttk.Frame(self, padding=(10, 4, 10, 10))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_sidebar(body)
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_preview(right)
        self._build_params(right)

    def _build_header(self):
        header = tk.Frame(self, bg=FACE)
        header.pack(fill="x", padx=14, pady=(12, 2))
        row = tk.Frame(header, bg=FACE)
        row.pack(side="left")
        tk.Label(row, text="Vision", bg=FACE, fg=TEXT,
                 font=F_TITLE).pack(side="left")
        tk.Label(row, text="Prep", bg=FACE, fg=ACCENT,
                 font=F_TITLE).pack(side="left")
        tk.Label(header, text="Preprocessing pipeline · dataset model vision",
                 bg=FACE, fg=MUTED, font=F_SUB).pack(side="left", padx=(10, 0),
                                                     pady=(10, 0))
        self.lbl_status = tk.Label(header, text="●  IDLE", bg=FACE,
                                   fg=MUTED, font=F_BOLD)
        self.lbl_status.pack(side="right", pady=(8, 0))
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=12,
                                                      pady=(6, 0))

    def _build_sidebar(self, parent):
        side = ttk.Frame(parent, width=300)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)

        # ---- folder
        gf = ttk.Labelframe(side, text="Folder gambar (.jpg)", padding=8)
        gf.pack(fill="x", pady=(0, 8))
        ttk.Entry(gf, textvariable=self.var_folder, font=F_UI).pack(
            side="left", fill="x", expand=True, padx=(0, 6), ipady=2)
        ttk.Button(gf, text="Browse…", width=9,
                   command=self.choose_folder).pack(side="right")

        # ---- tahap proses
        gs = ttk.Labelframe(side, text="Tahap proses", padding=6)
        gs.pack(fill="x", pady=(0, 8))
        self.listbox = tk.Listbox(
            gs, height=len(STAGES), font=("Segoe UI", 10), bd=1,
            relief="sunken", bg=FIELD, fg=TEXT, highlightthickness=1,
            highlightbackground=BORDER, selectbackground=SEL_BG,
            selectforeground=TEXT, activestyle="none", exportselection=False)
        for i, (title, _key) in enumerate(STAGES, start=1):
            self.listbox.insert("end", f"  {i:02d}    {title}")
        self.listbox.pack(fill="x")
        self.listbox.bind("<<ListboxSelect>>", self._on_stage_select)

        # ---- tombol start (timbul)
        self.btn_start = ttk.Button(side, text="▶  START",
                                    style="Big.TButton", command=self.start)
        self.btn_start.pack(fill="x", pady=(2, 10))

        # ---- console
        self._build_log(side)

    def _build_preview(self, parent):
        gp = ttk.Labelframe(parent, text="Preview", padding=8)
        gp.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        gp.rowconfigure(1, weight=1)
        gp.columnconfigure(0, weight=1)
        topbar = ttk.Frame(gp)
        topbar.grid(row=0, column=0, sticky="ew")
        self.lbl_preview_stage = tk.Label(topbar, text="", bg=FACE, fg=ACCENT,
                                          font=F_BOLD)
        self.lbl_preview_stage.pack(side="right")

        self.fig = Figure(figsize=(5.6, 3.8), dpi=100, facecolor=FACE)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.03,
                                 wspace=0.06)
        self.ax_before = self.fig.add_subplot(1, 2, 1)
        self.ax_after = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=gp)
        w = self.canvas.get_tk_widget()
        w.configure(bg=FACE, highlightthickness=0)
        w.grid(row=1, column=0, sticky="nsew")
        self._render_blank()

    def _build_params(self, parent):
        rowf = ttk.Frame(parent)
        rowf.grid(row=1, column=0, sticky="ew")
        rowf.columnconfigure((0, 1, 2), weight=1, uniform="param")

        # RESIZING
        c1 = ttk.Labelframe(rowf, text="Resizing", padding=10)
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        ttk.Radiobutton(c1, text="Average Pooling", value=pp.AVERAGE,
                        variable=self.var_pooling).grid(row=0, column=0,
                                                        columnspan=2, sticky="w")
        ttk.Radiobutton(c1, text="Max Pooling", value=pp.MAX,
                        variable=self.var_pooling).grid(row=1, column=0,
                                                        columnspan=2, sticky="w")
        ttk.Label(c1, text="Row").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self._entry(c1, self.var_row, 2, 1)
        ttk.Label(c1, text="Col").grid(row=3, column=0, sticky="w", pady=(2, 0))
        self._entry(c1, self.var_col, 3, 1)

        # BINARIZATION
        c2 = ttk.Labelframe(rowf, text="Binarization", padding=10)
        c2.grid(row=0, column=1, sticky="nsew", padx=4)
        ttk.Label(c2, text="Threshold").grid(row=0, column=0, sticky="w")
        self._entry(c2, self.var_threshold, 0, 1)
        ttk.Checkbutton(c2, text="Inversi (latar hitam,\nobjek putih)",
                        variable=self.var_invert).grid(row=1, column=0,
                                                       columnspan=2, sticky="w",
                                                       pady=(8, 0))

        # ANN
        c3 = ttk.Labelframe(rowf, text="ANN", padding=10)
        c3.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        for i, (lbl, var) in enumerate([("Hidden", self.var_hidden),
                                        ("Epochs", self.var_epochs),
                                        ("Sampel uji", self.var_sample)]):
            ttk.Label(c3, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
            self._entry(c3, var, i, 1)

    def _build_log(self, parent):
        gl = ttk.Labelframe(parent, text="Console", padding=6)
        gl.pack(fill="both", expand=True)
        wrap = tk.Frame(gl, bg=BORDER, bd=0)
        wrap.pack(fill="both", expand=True)
        self.text_log = tk.Text(wrap, bg=FIELD, fg=TEXT, relief="flat",
                                font=F_MONO, padx=8, pady=6, state="disabled",
                                wrap="word", highlightthickness=0, bd=0)
        scroll = ttk.Scrollbar(wrap, orient="vertical",
                               command=self.text_log.yview)
        self.text_log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text_log.pack(side="left", fill="both", expand=True, padx=1, pady=1)
        self.text_log.tag_configure("head", foreground=ACCENT, font=F_BOLD)
        self.text_log.tag_configure("err", foreground=DANGER)

    # ----------------------------------------------------------- komponen UI
    def _entry(self, parent, var, row, column):
        ttk.Entry(parent, textvariable=var, width=7, justify="center",
                  font=F_MONO).grid(row=row, column=column, sticky="w",
                                    padx=4, pady=2, ipady=1)

    # ----------------------------------------------------------- preview viz
    def _style_axis(self, ax, caption, color):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(FIELD)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.set_title(caption, color=color, fontsize=12, fontfamily="sans-serif",
                     pad=8, loc="left", fontweight="bold")

    def _placeholder(self, ax, caption, text):
        ax.clear()
        self._style_axis(ax, caption, MUTED)
        ax.text(0.5, 0.5, text, color=MUTED, fontsize=11, ha="center",
                va="center", fontfamily="sans-serif", transform=ax.transAxes)

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
    def _on_stage_select(self, _event):
        sel = self.listbox.curselection()
        if sel:
            self.select_stage(STAGES[sel[0]][1])

    def select_stage(self, key):
        self.stage_key = key
        idx = [k for _t, k in STAGES].index(key)
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
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
        self.set_status("●  RUNNING", ACCENT)
        self.btn_start.configure(text="…  WORKING", state="disabled")
        self.update_idletasks()
        try:
            self.run_stage(stage, custom_path)
        except ValueError as error:
            self.log(f"Input tidak valid: {error}", "err")
        except Exception as error:
            self.log(f"Terjadi error: {error}", "err")
        finally:
            self.btn_start.configure(text="▶  START", state="normal")
            self.set_status("●  DONE", "#1E7D34")
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
