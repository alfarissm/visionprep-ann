"""Render kode inti tiap fitur menjadi gambar (syntax-highlight) untuk artikel.

Menghasilkan code_*.png — tangkapan layar kode bergaya editor gelap.
Jalankan:  python generate_code_images.py
"""

import inspect

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter

from processing import grayscale, resizing, binarization, dataset, randomize, ann

# (nama file keluaran, objek fungsi yang dirender)
TARGETS = [
    ("code_grayscale.png", grayscale.to_grayscale),
    ("code_resizing.png", resizing.resize_pooling),
    ("code_binarization.png", binarization.binarize),
    ("code_dataset.png", dataset.process_folder),
    ("code_randomize.png", randomize.process_folder),
    ("code_ann.png", ann.train),
]

FONT_CANDIDATES = ["Consolas", "Cascadia Mono", "DejaVu Sans Mono"]


def make_formatter():
    """ImageFormatter dengan font monospace pertama yang tersedia."""
    for font in FONT_CANDIDATES:
        try:
            fmt = ImageFormatter(font_name=font, font_size=24, style="monokai",
                                 line_numbers=True, image_pad=18,
                                 line_number_bg="#2B2B2B",
                                 line_number_fg="#888888")
            # Uji cepat render.
            highlight("x = 1\n", PythonLexer(), fmt)
            return font
        except Exception:
            continue
    return None


def render(path, func, font):
    code = inspect.getsource(func)
    fmt = ImageFormatter(font_name=font, font_size=24, style="monokai",
                         line_numbers=True, image_pad=18,
                         line_number_bg="#2B2B2B", line_number_fg="#888888")
    png = highlight(code, PythonLexer(), fmt)
    with open(path, "wb") as f:
        f.write(png)
    print(f"Tersimpan: {path}")


if __name__ == "__main__":
    font = make_formatter()
    if font is None:
        print("Tidak menemukan font monospace yang cocok.")
    else:
        print(f"Font dipakai: {font}")
        for path, func in TARGETS:
            render(path, func, font)
