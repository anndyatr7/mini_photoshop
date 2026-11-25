import copy
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Warna tema
BG = "#0f1117"
PANEL = "#171a23"
CARD = "#222734"
BORDER = "#2b3140"
TEXT = "#eef0f6"
MUTED = "#8b93a7"
ACCENT = "#7c5cff"
BEFORE = "#ffb454"
AFTER = "#34d399"
FONT = "Segoe UI"

PREVIEW_MAX = 1000   # sisi terpanjang gambar preview (px)
MAX_UNDO = 50
MAX_ZOOM = 16.0


# ----------------------------------------------------------------------
# Fungsi pengolahan citra
# Semua operasi: fungsi(image, scale, **param) -> image baru.
# `scale` = rasio resolusi penuh terhadap preview (1.0 saat preview), dipakai
# agar ukuran kernel/blok proporsional, jadi hasil simpan = hasil preview.
# ----------------------------------------------------------------------
def to_gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def odd(value):
    value = max(1, int(round(value)))
    return value if value % 2 else value + 1


def same_image(a, b):
    a, b = to_bgr(a), to_bgr(b)
    return a.shape == b.shape and np.array_equal(a, b)


def op_brightness(img, s, amount):
    return np.clip(img.astype(np.int16) + int(amount), 0, 255).astype(np.uint8)


def op_contrast(img, s, factor):
    result = (img.astype(np.float32) - 128) * factor + 128
    return np.clip(result, 0, 255).astype(np.uint8)


def op_saturation(img, s, factor):
    hsv = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def op_grayscale(img, s):
    return to_gray(img)


def op_equalization(img, s):
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def op_threshold(img, s, threshold):
    return cv2.threshold(to_gray(img), int(threshold), 255, cv2.THRESH_BINARY)[1]


def op_negative(img, s):
    return cv2.bitwise_not(img)


def op_sharpen(img, s, amount):
    # amount 0 = gambar tidak berubah (jumlah elemen kernel selalu 1)
    kernel = np.array([[0, -amount, 0],
                       [-amount, 1 + 4 * amount, -amount],
                       [0, -amount, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def op_blur(img, s, mode, size):
    if not mode or size <= 1:
        return img
    k = odd(size * s)
    if mode == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    if mode == "averaging":
        return cv2.blur(img, (k, k))
    if mode == "median":
        return cv2.medianBlur(img, k)
    if mode == "bilateral":
        return cv2.bilateralFilter(to_bgr(img), min(k, 15), 75, 75)
    kernel = np.ones((k, k), np.float32) / (k * k)
    return cv2.filter2D(img, -1, kernel)


def op_sketch(img, s, size):
    gray = to_gray(img)
    k = odd(size * s)
    blurred = cv2.GaussianBlur(cv2.bitwise_not(gray), (k, k), 0)
    return cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)


def op_edge(img, s, method, gain):
    gray = to_gray(img)
    if method == "canny":
        k = odd(3 * s)
        return cv2.Canny(cv2.GaussianBlur(gray, (k, k), 0), 100, 200)
    if method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        edge = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5,
                               cv2.convertScaleAbs(gy), 0.5, 0)
    else:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        edge = cv2.filter2D(gray, cv2.CV_16S, kernel).clip(0, 255)
    return np.clip(edge.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def op_sepia(img, s):
    matrix = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]], dtype=np.float32)  # urutan BGR
    return cv2.transform(to_bgr(img), matrix)


def _shift_temperature(img, amount):
    result = to_bgr(img).astype(np.int16)
    result[:, :, 2] += int(amount)   # merah
    result[:, :, 0] -= int(amount)   # biru
    return np.clip(result, 0, 255).astype(np.uint8)


def op_warm(img, s, amount):
    return _shift_temperature(img, amount)


def op_cool(img, s, amount):
    return _shift_temperature(img, -amount)


def op_vignette(img, s, strength):
    height, width = img.shape[:2]
    y, x = np.ogrid[-1:1:height * 1j, -1:1:width * 1j]
    mask = (1 - (strength / 100) * np.clip((x * x + y * y) / 2, 0, 1)).astype(np.float32)
    if img.ndim == 3:
        mask = mask[:, :, None]
    return (img * mask).astype(np.uint8)


def op_cartoon(img, s):
    bgr = to_bgr(img)
    gray = cv2.medianBlur(to_gray(bgr), odd(5 * s))
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, max(3, odd(9 * s)), 9)
    smooth = cv2.bilateralFilter(bgr, min(odd(9 * s), 15), 150, 150)
    return cv2.bitwise_and(smooth, smooth, mask=edges)


def op_emboss(img, s):
    kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float32)
    return cv2.filter2D(to_gray(img), -1, kernel, delta=128)


def op_pixelate(img, s, size):
    height, width = img.shape[:2]
    block = max(1, int(round(size * s)))
    small = cv2.resize(img, (max(1, width // block), max(1, height // block)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def op_mirror(img, s, mode):
    code = {"h": 1, "v": 0, "b": -1}.get(mode)
    return img if code is None else cv2.flip(img, code)


def op_rotate(img, s, angle):
    code = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "180": cv2.ROTATE_180}.get(angle)
    return img if code is None else cv2.rotate(img, code)


def op_crop(img, s, left, top, right, bottom):
    height, width = img.shape[:2]
    x1, y1 = int(width * left / 100), int(height * top / 100)
    x2 = max(x1 + 1, width - int(width * right / 100))
    y2 = max(y1 + 1, height - int(height * bottom / 100))
    return img[y1:y2, x1:x2].copy()


def op_resize(img, s, percent):
    factor = percent / 100
    if abs(factor - 1) < 1e-6:
        return img
    height, width = img.shape[:2]
    size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if factor < 1
                      else cv2.INTER_CUBIC)


# Spesifikasi parameter: slider (kunci, label, min, max, default, langkah)
# atau pilihan (kunci, label, [(teks, nilai)], default). Nilai "" = belum memilih.
def _slider(key, text, low, high, default, step=1):
    return ("slider", key, text, low, high, default, step)


def _choice(key, text, options, default=""):
    return ("choice", key, text, options, default)


FEATURES = {
    "Brightness": (op_brightness, [_slider("amount", "Brightness (-255 s/d 255):", -255, 255, 0)]),
    "Contrast": (op_contrast, [_slider("factor", "Kontras (1 = normal):", 0.2, 3.0, 1.0, 0.1)]),
    "Saturation": (op_saturation, [_slider("factor", "Saturasi (1 = normal):", 0.0, 3.0, 1.0, 0.1)]),
    "Grayscale": (op_grayscale, []),
    "Equalization": (op_equalization, []),
    "Thresholding": (op_threshold, [_slider("threshold", "Nilai threshold (0-255):", 0, 255, 128)]),
    "Negative": (op_negative, []),
    "Sharpening": (op_sharpen, [_slider("amount", "Intensitas sharpening (0-5):", 0.0, 5.0, 1.0, 0.1)]),
    "Blur": (op_blur, [
        _choice("mode", "Jenis blur:", [
            ("Gaussian Blur", "gaussian"), ("Averaging Blur", "averaging"),
            ("Median Blur", "median"), ("Bilateral Blur", "bilateral"),
            ("Standard Blur (2D)", "standard")]),
        _slider("size", "Ukuran kernel:", 1, 31, 5, 2)]),
    "Sketch": (op_sketch, [_slider("size", "Ketebalan garis (blur):", 3, 101, 21, 2)]),
    "Sepia": (op_sepia, []),
    "Warm": (op_warm, [_slider("amount", "Kehangatan:", 0, 100, 30)]),
    "Cool": (op_cool, [_slider("amount", "Kesejukan:", 0, 100, 30)]),
    "Vignette": (op_vignette, [_slider("strength", "Kekuatan vignette:", 0, 100, 60)]),
    "Cartoon": (op_cartoon, []),
    "Emboss": (op_emboss, []),
    "Pixelate": (op_pixelate, [_slider("size", "Ukuran blok piksel:", 2, 50, 10)]),
    "Edge Detection": (op_edge, [
        _choice("method", "Metode:", [("Laplacian", "laplacian"), ("Sobel", "sobel"),
                                      ("Canny", "canny")], "laplacian"),
        _slider("gain", "Penguatan (Laplacian/Sobel):", 0.5, 5.0, 1.0, 0.5)]),
    "Mirroring": (op_mirror, [_choice("mode", "Jenis mirroring:", [
        ("Flip Horizontal", "h"), ("Flip Vertikal", "v"), ("Flip Keduanya", "b")])]),
    "Rotate": (op_rotate, [_choice("angle", "Putar gambar:", [
        ("90° ke kanan", "cw"), ("90° ke kiri", "ccw"), ("180°", "180")])]),
    "Crop": (op_crop, [
        _slider("left", "Potong kiri (%):", 0, 45, 0),
        _slider("top", "Potong atas (%):", 0, 45, 0),
        _slider("right", "Potong kanan (%):", 0, 45, 0),
        _slider("bottom", "Potong bawah (%):", 0, 45, 0)]),
    "Resize": (op_resize, [_slider("percent", "Skala (%):", 10, 200, 100)]),
}

GROUPS = (
    ("Penyesuaian", ("Brightness", "Contrast", "Saturation", "Grayscale",
                     "Equalization", "Thresholding", "Negative", "Sharpening")),
    ("Efek", ("Blur", "Sketch", "Sepia", "Warm", "Cool", "Vignette", "Cartoon",
              "Emboss", "Pixelate", "Edge Detection")),
    ("Transformasi", ("Mirroring", "Rotate", "Crop", "Resize")),
)

VIEW_MODES = (("1 Gambar", "single"), ("Berdampingan", "side"),
              ("Geser", "slide"), ("Peta Beda", "diff"))


def default_params(name):
    params = {}
    for spec in FEATURES[name][1]:
        params[spec[1]] = spec[5] if spec[0] == "slider" else spec[4]
    return params


class MiniPhotoshop:
    """Editor citra sederhana: pipeline langkah non-destruktif, preview cepat,
    Undo/Redo, zoom/pan, dan simpan dalam resolusi penuh."""

    def __init__(self, root):
        self.root = root
        self.root.title("Mini Photoshop - Pengolahan Citra")
        self.root.geometry("1320x820")
        self.root.minsize(1100, 700)
        self.root.configure(bg=BG)

        self.filename = None
        self.full_image = None       # resolusi penuh (dipakai saat menyimpan)
        self.original_image = None   # preview (sisi terpanjang <= PREVIEW_MAX)
        self.current_image = None    # hasil pipeline pada preview
        self.scale = 1.0             # rasio full / preview

        self.steps = []              # [{"name": ..., "params": {...}}]
        self.active = None           # langkah yang sedang diedit (draft)
        self.undo_stack = []
        self.redo_stack = []
        self._cache = None           # (index, gambar setelah steps[:index])
        self._job = None
        self._vars = []
        self.feature_buttons = {}

        self.view_mode = tk.StringVar(value="single")
        self.show_data = tk.BooleanVar(value=True)
        self.zoom = 1.0
        self.center = [0.5, 0.5]     # titik tengah tampilan (pecahan 0..1)
        self._fit_size = (1.0, 1.0)
        self._pan_last = None
        self._peek = False
        self.split = 0.5
        self._split_box = None
        self._photos = {}
        self._pending_data = None

        self._setup_style()
        self._build_menu()
        self._build_layout()
        self._show_empty_state()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TButton", background=CARD, foreground=TEXT, bordercolor=BORDER,
                        lightcolor=CARD, darkcolor=CARD, focuscolor=ACCENT,
                        font=(FONT, 10), padding=(8, 4), relief="flat")
        style.map("TButton", background=[("active", BORDER)])
        for name, color in (("Accent.TButton", ACCENT), ("Active.TButton", ACCENT),
                            ("Danger.TButton", "#a83a48")):
            style.configure(name, background=color, foreground="white",
                            lightcolor=color, darkcolor=color)
            style.map(name, background=[("active", color)])

        style.configure("TRadiobutton", background=PANEL, foreground=TEXT,
                        font=(FONT, 10), indicatorbackground=CARD,
                        indicatorforeground=ACCENT, padding=3)
        style.map("TRadiobutton", background=[("active", PANEL)])
        style.configure("Mode.Toolbutton", background=CARD, foreground=MUTED,
                        font=(FONT, 9), padding=(10, 4), bordercolor=BORDER,
                        lightcolor=CARD, darkcolor=CARD)
        style.map("Mode.Toolbutton",
                  background=[("selected", ACCENT), ("active", BORDER)],
                  foreground=[("selected", "white")])

        style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=TEXT, font=(FONT, 9), rowheight=21, borderwidth=0)
        style.configure("Treeview.Heading", background=CARD, foreground=TEXT,
                        font=(FONT, 9, "bold"), relief="flat")
        style.map("Treeview.Heading", background=[("active", CARD)])

    def _label(self, parent, text="", size=10, color=TEXT, bold=False, bg=PANEL, **kw):
        font = (FONT, size, "bold") if bold else (FONT, size)
        return tk.Label(parent, text=text, font=font, fg=color, bg=bg, **kw)

    def _card(self, parent, bg=PANEL):
        return tk.Frame(parent, bg=bg, highlightthickness=1, highlightbackground=BORDER)

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        def menu(title):
            item = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label=title, menu=item)
            return item

        file_menu = menu("File")
        file_menu.add_command(label="Buka Gambar", accelerator="Ctrl+O", command=self.open_file)
        file_menu.add_command(label="Simpan Hasil", accelerator="Ctrl+S", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Keluar", command=self.root.quit)

        edit_menu = menu("Edit")
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        edit_menu.add_command(label="Reset Gambar", command=self.reset_image)

        view_menu = menu("Tampilan")
        for text, value in VIEW_MODES:
            view_menu.add_radiobutton(label=text, value=value, variable=self.view_mode,
                                      command=self._render_views)
        view_menu.add_separator()
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        view_menu.add_checkbutton(label="Tampilkan Data & Histogram",
                                  variable=self.show_data, command=self._toggle_data)

        for title, names in GROUPS:
            group_menu = menu(title)
            for name in names:
                group_menu.add_command(label=name, command=lambda n=name: self.select_feature(n))

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda event: self.open_file())
        self.root.bind("<Control-s>", lambda event: self.save_file())
        self.root.bind("<Control-z>", lambda event: self.undo())
        self.root.bind("<Control-y>", lambda event: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda event: self.redo())

    def _build_layout(self):
        # Toolbar atas
        toolbar = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self._label(toolbar, "Mini Photoshop", 15, bold=True).pack(side=tk.LEFT)
        self.file_label = self._label(toolbar, "", 9, MUTED)
        self.file_label.pack(side=tk.LEFT, padx=14)
        for text, command, style in (
            ("Reset", self.reset_image, "Danger.TButton"),
            ("Redo", self.redo, "TButton"),
            ("Undo", self.undo, "TButton"),
            ("Simpan Hasil", self.save_file, "TButton"),
            ("Buka Gambar", self.open_file, "Accent.TButton"),
        ):
            ttk.Button(toolbar, text=text, command=command, style=style).pack(
                side=tk.RIGHT, padx=4)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Sidebar: fitur per kategori + daftar langkah
        sidebar = self._card(body)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        for title, names in GROUPS:
            self._label(sidebar, title.upper(), 8, MUTED, True).pack(
                anchor=tk.W, padx=12, pady=(8, 2))
            group = tk.Frame(sidebar, bg=PANEL)
            group.pack(padx=8)
            for index, name in enumerate(names):
                button = ttk.Button(group, text=name, width=13,
                                    command=lambda n=name: self.select_feature(n))
                button.grid(row=index // 2, column=index % 2, padx=2, pady=2, sticky="ew")
                self.feature_buttons[name] = button

        self._label(sidebar, "LANGKAH (klik untuk edit)", 8, MUTED, True).pack(
            anchor=tk.W, padx=12, pady=(10, 2))
        self.step_list = tk.Listbox(
            sidebar, height=6, bg=CARD, fg=TEXT, selectbackground=ACCENT,
            selectforeground="white", highlightthickness=0, bd=0,
            font=(FONT, 9), activestyle="none", exportselection=False)
        self.step_list.pack(fill=tk.X, padx=10)
        self.step_list.bind("<<ListboxSelect>>", self._on_step_select)
        ttk.Button(sidebar, text="Hapus Langkah", command=self.delete_step).pack(
            fill=tk.X, padx=10, pady=(4, 10))

        main = tk.Frame(body, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bagian data (dipasang dulu di bawah supaya tidak terpotong)
        data_frame = tk.Frame(main, bg=BG, height=270)
        self.data_frame = data_frame
        data_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        data_frame.pack_propagate(False)

        table_card = self._card(data_frame)
        table_card.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self._label(table_card, "Data Before / After", 11, bold=True).pack(
            anchor=tk.W, padx=10, pady=(6, 2))
        self.table = ttk.Treeview(table_card, columns=("name", "before", "after", "diff"),
                                  show="headings", height=7, selectmode="none")
        for col, title, width in (("name", "Data", 150), ("before", "Before", 95),
                                  ("after", "After", 95), ("diff", "Selisih", 75)):
            self.table.heading(col, text=title)
            self.table.column(col, width=width, anchor=tk.W if col == "name" else tk.E,
                              stretch=False)
        self.table.pack(padx=10)
        self.summary_label = self._label(table_card, "", 9, MUTED, justify=tk.LEFT,
                                         wraplength=410, anchor=tk.NW)
        self.summary_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        hist_card = self._card(data_frame)
        hist_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(5, 2.6), dpi=90, facecolor=PANEL)
        self.ax_before, self.ax_after = self.figure.subplots(2, 1, sharex=True, sharey=True)
        self.figure.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.17, hspace=0.45)
        self.hist_canvas = FigureCanvasTkAgg(self.figure, master=hist_card)
        self.hist_canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Bagian gambar + panel pengaturan
        self.view_row = tk.Frame(main, bg=BG)
        self.view_row.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        settings = self._card(self.view_row)
        settings.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self._label(settings, "Pengaturan", 11, bold=True).pack(anchor=tk.W, padx=12, pady=(8, 4))
        self.frame_dynamic = tk.Frame(settings, bg=PANEL, width=210)
        self.frame_dynamic.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        self.frame_dynamic.pack_propagate(False)

        view_card = self._card(self.view_row)
        view_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar = tk.Frame(view_card, bg=PANEL)
        bar.pack(fill=tk.X, padx=10, pady=6)
        for text, value in VIEW_MODES:
            ttk.Radiobutton(bar, text=text, value=value, variable=self.view_mode,
                            style="Mode.Toolbutton", command=self._render_views).pack(
                side=tk.LEFT, padx=(0, 4))
        self._label(bar, "Scroll: zoom  |  Klik 2x: reset zoom", 8, MUTED).pack(side=tk.RIGHT)
        self.history_label = self._label(view_card, "", 9, MUTED)
        self.history_label.configure(justify=tk.LEFT, anchor=tk.W, wraplength=560)
        self.history_label.pack(fill=tk.X, padx=10, pady=(0, 6), after=bar)
        self.history_label.bind(
            "<Configure>", lambda e: self.history_label.configure(wraplength=e.width - 8))

        self.canvas = tk.Canvas(view_card, bg=BG, highlightthickness=0, height=200)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda event: self._render_views())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", lambda event: self.reset_zoom())
        self.canvas.bind("<MouseWheel>", self._on_wheel)   # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_wheel)     # Linux
        self.canvas.bind("<Button-5>", self._on_wheel)

        self._show_message("Buka gambar, lalu pilih fitur di sebelah kiri.")

    def _show_empty_state(self):
        self.table.delete(*self.table.get_children())
        self.summary_label.configure(text="Data akan muncul setelah gambar dibuka.")
        self._draw_histograms()
        self._render_views()

    def _clear_dynamic_frame(self):
        for widget in self.frame_dynamic.winfo_children():
            widget.destroy()
        self._vars = []

    def _show_message(self, text):
        self._clear_dynamic_frame()
        self._label(self.frame_dynamic, text, 10, MUTED, justify=tk.LEFT,
                    wraplength=200).pack(anchor=tk.W, pady=4)

    def _require_image(self):
        if self.current_image is None:
            messagebox.showwarning("Belum ada gambar", "Silakan buka gambar terlebih dahulu.")
            return False
        return True

    def _update_buttons(self):
        name = self.active["name"] if self.active else None
        for label, button in self.feature_buttons.items():
            button.configure(style="Active.TButton" if label == name else "TButton")

    # ------------------------------------------------------------------
    # Panel pengaturan (dibuat dari spesifikasi FEATURES)
    # ------------------------------------------------------------------
    def _build_controls(self):
        self._clear_dynamic_frame()
        active = self.active
        if active is None:
            return
        frame = self.frame_dynamic
        title = active["name"] if active["is_new"] else \
            f"{active['name']} (langkah {active['index'] + 1})"
        self._label(frame, title, 10, ACCENT, True).pack(anchor=tk.W)

        buttons = tk.Frame(frame, bg=PANEL)
        buttons.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Terapkan", style="Accent.TButton",
                   command=self.apply_active).pack(side=tk.LEFT, expand=True,
                                                    fill=tk.X, padx=(0, 3))
        ttk.Button(buttons, text="Batal", command=self.cancel_active).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        specs = FEATURES[active["name"]][1]
        if not specs:
            self._label(frame, "Efek ini tidak punya pengaturan.", 9, MUTED,
                        wraplength=200, justify=tk.LEFT).pack(anchor=tk.W, pady=4)
        for spec in specs:
            if spec[0] == "slider":
                self._add_slider(active["params"], spec)
            else:
                self._add_choice(active["params"], spec)

    def _add_slider(self, params, spec):
        _, key, text, low, high, _default, step = spec
        self._label(self.frame_dynamic, text, 9).pack(anchor=tk.W, pady=(4, 0))
        slider = tk.Scale(
            self.frame_dynamic, from_=low, to=high, orient=tk.HORIZONTAL,
            resolution=step, bg=PANEL, fg=TEXT, troughcolor=CARD,
            activebackground=ACCENT, highlightthickness=0, bd=0,
            sliderrelief=tk.FLAT, font=(FONT, 9))
        slider.set(params[key])
        is_int = float(step).is_integer()

        def on_change(value):
            number = float(value)
            params[key] = int(number) if is_int else round(number, 2)
            self._schedule_recompute()

        # command dipasang setelah set() agar nilai awal tidak memicu perhitungan
        slider.configure(command=on_change)
        slider.pack(fill=tk.X)

    def _add_choice(self, params, spec):
        _, key, text, options, _default = spec
        self._label(self.frame_dynamic, text, 9).pack(anchor=tk.W, pady=(4, 0))
        var = tk.StringVar(value=params[key])
        self._vars.append(var)

        def on_change(var=var, key=key):
            params[key] = var.get()
            self._schedule_recompute()

        for label, value in options:
            ttk.Radiobutton(self.frame_dynamic, text=label, variable=var, value=value,
                            command=on_change).pack(anchor=tk.W, padx=8)

    # ------------------------------------------------------------------
    # Pipeline langkah
    # ------------------------------------------------------------------
    def _apply(self, image, name, params, scale=1.0):
        try:
            return FEATURES[name][0](image, scale, **params)
        except (cv2.error, ValueError, TypeError, MemoryError) as error:
            messagebox.showerror("Kesalahan pemrosesan", str(error))
            return image

    def _prefix(self, index):
        """Gambar setelah steps[:index] diterapkan (di-cache)."""
        if self._cache and self._cache[0] == index:
            return self._cache[1]
        image = self.original_image
        for step in self.steps[:index]:
            image = self._apply(image, step["name"], step["params"])
        self._cache = (index, image)
        return image

    def _pipeline_preview(self):
        active = self.active
        if active is None:
            return self._prefix(len(self.steps))
        index = active["index"]
        image = self._apply(self._prefix(index), active["name"], active["params"])
        for step in self.steps[index + 1:]:      # langkah sesudahnya ikut dihitung ulang
            image = self._apply(image, step["name"], step["params"])
        return image

    def _schedule_recompute(self):
        if self._job is not None:
            self.root.after_cancel(self._job)
        self._job = self.root.after(20, self._recompute)

    def _recompute(self):
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None
        if self.original_image is None:
            return
        self.current_image = self._pipeline_preview()
        self._refresh_result()

    def _active_changed(self, active):
        if not active["is_new"]:
            return active["params"] != active["orig"]
        base = self._prefix(active["index"])
        return not same_image(base, self._apply(base, active["name"], active["params"]))

    def _finish_active(self, commit):
        """Tutup langkah aktif. commit=True menyimpannya ke pipeline bila ada perubahan."""
        active = self.active
        if active is None:
            return
        changed = commit and self._active_changed(active)
        self.active = None
        if changed:
            self._push_undo()
            if active["is_new"]:
                self.steps.append({"name": active["name"], "params": dict(active["params"])})
            else:
                self.steps[active["index"]]["params"] = dict(active["params"])
            self._cache = None
        self._refresh_steps_list()

    def _close_active(self, commit, message):
        if self.active is None:
            return
        self._finish_active(commit)
        self._update_buttons()
        self._show_message(message)
        self._recompute()

    def select_feature(self, name):
        if not self._require_image():
            return
        self._finish_active(True)    # pindah fitur = langkah sebelumnya diterapkan
        self.active = {"index": len(self.steps), "name": name,
                       "params": default_params(name), "is_new": True, "orig": None}
        self._update_buttons()
        self._build_controls()
        self._recompute()

    def apply_active(self):
        self._close_active(True, "Langkah diterapkan. Pilih fitur lain untuk lanjut.")

    def cancel_active(self):
        self._close_active(False, "Perubahan dibatalkan.")

    def _on_step_select(self, event=None):
        selection = self.step_list.curselection()
        if not selection:
            return
        index = selection[0]
        self._finish_active(True)
        if index >= len(self.steps):
            return
        step = self.steps[index]
        self.active = {"index": index, "name": step["name"], "params": dict(step["params"]),
                       "is_new": False, "orig": dict(step["params"])}
        self._refresh_steps_list(select=index)
        self._update_buttons()
        self._build_controls()
        self._recompute()

    def delete_step(self):
        if self.original_image is None:
            return
        active = self.active
        selection = self.step_list.curselection()
        if active and not active["is_new"]:
            index = active["index"]
        elif selection:
            index = selection[0]
        else:
            index = None
        had_active = active is not None
        self._finish_active(False)
        if index is None or index >= len(self.steps):
            if had_active:
                self._update_buttons()
                self._show_message("Perubahan dibatalkan.")
                self._recompute()
            return
        self._push_undo()
        del self.steps[index]
        self._cache = None
        self._refresh_steps_list()
        self._update_buttons()
        self._show_message("Langkah dihapus.")
        self._recompute()

    @staticmethod
    def _summary(params):
        return ", ".join(str(value) for value in params.values() if value != "")

    def _refresh_steps_list(self, select=None):
        self.step_list.delete(0, tk.END)
        for index, step in enumerate(self.steps, 1):
            detail = self._summary(step["params"])
            self.step_list.insert(tk.END, f"{index}. {step['name']}"
                                  + (f"  [{detail}]" if detail else ""))
        if select is not None and select < len(self.steps):
            self.step_list.selection_set(select)

    # ------------------------------------------------------------------
    # Undo / Redo / Reset
    # ------------------------------------------------------------------
    def _push_undo(self):
        self.undo_stack.append(copy.deepcopy(self.steps))
        del self.undo_stack[:-MAX_UNDO]
        self.redo_stack.clear()

    def _restore(self, steps, message):
        self.steps = steps
        self._cache = None
        self._refresh_steps_list()
        self._update_buttons()
        self._show_message(message)
        self._recompute()

    def undo(self):
        if self.original_image is None:
            return
        if self.active:              # Undo saat mengedit = batalkan draft
            self.cancel_active()
            return
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.steps))
        self._restore(self.undo_stack.pop(), "Langkah terakhir dibatalkan.")

    def redo(self):
        if self.original_image is None:
            return
        self._finish_active(False)
        if not self.redo_stack:
            self._update_buttons()
            self._recompute()
            return
        self.undo_stack.append(copy.deepcopy(self.steps))
        self._restore(self.redo_stack.pop(), "Langkah diulang.")

    def reset_image(self):
        if self.original_image is None:
            messagebox.showinfo("Reset", "Belum ada gambar yang dibuka.")
            return
        self._finish_active(False)
        if self.steps:
            self._push_undo()        # reset pun bisa di-Undo
        self._restore([], "Gambar dikembalikan ke kondisi asli.")

    # ------------------------------------------------------------------
    # Tampilan before / after (dengan zoom & pan)
    # ------------------------------------------------------------------
    def _to_rgb(self, image):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2RGB)

    def _zoom_crop(self, rgb, box_w, box_h):
        """Ambil bagian gambar yang terlihat pada zoom/pan saat ini, sudah diubah ukurannya."""
        ih, iw = rgb.shape[:2]
        base = min(box_w / iw, box_h / ih, 1.0)
        eff = base * self.zoom
        self._fit_size = (iw * base, ih * base)
        vw, vh = min(float(iw), box_w / eff), min(float(ih), box_h / eff)
        left = min(max(self.center[0] * iw - vw / 2, 0), iw - vw)
        top = min(max(self.center[1] * ih - vh / 2, 0), ih - vh)
        l, t = int(left), int(top)
        r = min(iw, max(l + 1, int(round(left + vw))))
        b = min(ih, max(t + 1, int(round(top + vh))))
        region = rgb[t:b, l:r]
        size = (max(1, int(round((r - l) * eff))), max(1, int(round((b - t) * eff))))
        if size == (r - l, b - t):
            return region
        interp = (cv2.INTER_AREA if eff < 1 else
                  cv2.INTER_NEAREST if eff >= 3 else cv2.INTER_LINEAR)
        return cv2.resize(region, size, interpolation=interp)

    def _put_image(self, key, rgb, center_x, center_y):
        self._photos[key] = ImageTk.PhotoImage(Image.fromarray(np.ascontiguousarray(rgb)))
        self.canvas.create_image(center_x, center_y, image=self._photos[key])

    def _badge(self, x, y, text, color, anchor="nw"):
        item = self.canvas.create_text(x, y, text=text, anchor=anchor,
                                       fill="#101216", font=(FONT, 10, "bold"))
        x1, y1, x2, y2 = self.canvas.bbox(item)
        box = self.canvas.create_rectangle(x1 - 8, y1 - 3, x2 + 8, y2 + 3,
                                           fill=color, outline="")
        self.canvas.tag_lower(box, item)

    def _difference_map(self):
        before = to_bgr(self.original_image)
        after = to_bgr(self.current_image)
        diff = cv2.absdiff(before, after)
        return cv2.applyColorMap(diff.max(axis=2), cv2.COLORMAP_INFERNO)

    def _render_views(self):
        self.canvas.delete("all")
        self._split_box = None
        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 150)

        if self.original_image is None:
            self.canvas.create_text(width / 2, height / 2, fill=MUTED, font=(FONT, 13),
                                    text="Belum ada gambar. Klik 'Buka Gambar'.")
            return

        mode = self.view_mode.get()
        top = 28  # ruang untuk label
        box_w, box_h = width - 8, height - top - 4
        center_y = top + (height - top) // 2
        same_size = self.original_image.shape[:2] == self.current_image.shape[:2]

        if self.zoom > 1:
            self.canvas.create_text(width - 8, height - 6, anchor="se", fill=MUTED,
                                    font=(FONT, 9), text=f"Zoom {int(self.zoom * 100)}%")

        if mode == "single":
            image = self.original_image if self._peek else self.current_image
            self._put_image("single", self._zoom_crop(self._to_rgb(image), box_w, box_h),
                            width // 2, center_y)
            hint = "geser untuk pan" if self.zoom > 1 else "tahan klik untuk lihat asli"
            self._badge(width // 2, 6,
                        "BEFORE (asli)" if self._peek else f"HASIL  |  {hint}",
                        BEFORE if self._peek else AFTER, anchor="n")
            return

        if mode == "side":
            half = width // 2
            for i, (key, image, text, color) in enumerate((
                ("before", self.original_image, "BEFORE (asli)", BEFORE),
                ("after", self.current_image, "AFTER (hasil)", AFTER),
            )):
                shown = self._zoom_crop(self._to_rgb(image), half - 12, box_h)
                center_x = i * half + half // 2
                self._put_image(key, shown, center_x, center_y)
                self._badge(center_x, 6, text, color, anchor="n")
            return

        if not same_size:
            message = ("Peta beda tidak tersedia: ukuran gambar berubah." if mode == "diff"
                       else "Ukuran gambar berubah. Pakai 'Berdampingan'.")
            self.canvas.create_text(width / 2, height / 2, fill=MUTED, font=(FONT, 11),
                                    text=message)
            return

        if mode == "diff":
            shown = self._zoom_crop(self._to_rgb(self._difference_map()), box_w, box_h)
            self._put_image("diff", shown, width // 2, center_y)
            self._badge(width // 2, 6, "PETA BEDA: makin terang, makin berubah",
                        ACCENT, anchor="n")
            return

        # Mode geser: kiri garis = before, kanan garis = after
        before = self._zoom_crop(self._to_rgb(self.original_image), box_w, box_h)
        after = self._zoom_crop(self._to_rgb(self.current_image), box_w, box_h)
        h, w = before.shape[:2]
        cut_x = int(w * self.split)
        mixed = after.copy()
        mixed[:, :cut_x] = before[:, :cut_x]
        self._put_image("slide", mixed, width // 2, center_y)
        left = width // 2 - w // 2
        self._split_box = (left, w)
        line_x = left + cut_x
        self.canvas.create_line(line_x, center_y - h // 2, line_x, center_y + h // 2,
                                fill="white", width=2)
        self.canvas.create_oval(line_x - 9, center_y - 9, line_x + 9, center_y + 9,
                                fill="white", outline=ACCENT, width=3)
        self._badge(12, 6, "BEFORE", BEFORE)
        self._badge(width - 12, 6, "AFTER", AFTER, anchor="ne")

    # Interaksi kanvas
    def _move_split(self, event):
        if self.view_mode.get() != "slide" or self._split_box is None:
            return
        left, w = self._split_box
        self.split = min(max((event.x - left) / w, 0.0), 1.0)
        self._render_views()

    def _on_press(self, event):
        if self.original_image is None:
            return
        if self.view_mode.get() == "slide":
            self._move_split(event)
        elif self.zoom > 1:
            self._pan_last = (event.x, event.y)
        else:
            self._set_peek(True)

    def _on_drag(self, event):
        if self.view_mode.get() == "slide":
            self._move_split(event)
        elif self.zoom > 1 and self._pan_last:
            dx, dy = event.x - self._pan_last[0], event.y - self._pan_last[1]
            self._pan_last = (event.x, event.y)
            fit_w, fit_h = self._fit_size
            self.center[0] = min(max(self.center[0] - dx / (fit_w * self.zoom), 0.0), 1.0)
            self.center[1] = min(max(self.center[1] - dy / (fit_h * self.zoom), 0.0), 1.0)
            self._render_views()

    def _on_release(self, event):
        self._pan_last = None
        self._set_peek(False)

    def _set_peek(self, value):
        if self._peek == value:
            return
        self._peek = value
        if self.view_mode.get() == "single":
            self._render_views()

    def _on_wheel(self, event):
        if self.original_image is None:
            return
        up = getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4
        zoom = self.zoom * 1.25 if up else self.zoom / 1.25
        self.zoom = round(min(MAX_ZOOM, max(1.0, zoom)), 3)
        if self.zoom <= 1.001:
            self.reset_zoom()
        else:
            self._render_views()

    def reset_zoom(self):
        self.zoom = 1.0
        self.center = [0.5, 0.5]
        self._render_views()

    def _toggle_data(self):
        if self.show_data.get():
            self.data_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0), before=self.view_row)
        else:
            self.data_frame.pack_forget()

    # ------------------------------------------------------------------
    # Data: histogram dan statistik
    # ------------------------------------------------------------------
    def _draw_histograms(self):
        panels = ((self.ax_before, self.original_image, "Histogram BEFORE", BEFORE),
                  (self.ax_after, self.current_image, "Histogram AFTER", AFTER))
        peaks = []
        for ax, image, title, color in panels:
            ax.clear()
            ax.set_facecolor(CARD)
            ax.set_title(title, fontsize=8, color=color, loc="left",
                         fontweight="bold", pad=3)
            ax.tick_params(labelsize=7, colors=MUTED)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            if image is None:
                continue
            if image.ndim == 2:
                channels = [(image, "#d5d8e0")]
            else:
                b, g, r = cv2.split(to_bgr(image))
                channels = [(r, "#ff5c6c"), (g, "#3ddc84"), (b, "#5b9dff")]
            peak = 0.0
            for data, line_color in channels:
                percent = np.bincount(data.ravel(), minlength=256) * 100.0 / data.size
                ax.fill_between(np.arange(256), percent, color=line_color, alpha=0.28)
                ax.plot(percent, color=line_color, linewidth=0.8)
                peak = max(peak, percent.max())
            peaks.append(peak)

        # Skala Y sama untuk before & after; puncak yang terlalu tinggi dipotong.
        self.ax_after.set_xlim(0, 255)
        if peaks:
            top = min(max(peaks), peaks[0] * 3) * 1.1
            self.ax_after.set_ylim(0, top)
            if max(peaks) > top:
                self.ax_after.text(0.5, 0.92, "puncak tinggi dipotong", fontsize=7,
                                   color=MUTED, ha="center", va="top",
                                   transform=self.ax_after.transAxes)
        self.ax_after.set_xlabel("Intensitas (0 = gelap, 255 = terang)  |  sumbu Y: % piksel",
                                 fontsize=7, color=MUTED)
        self.hist_canvas.draw_idle()

    def _statistics(self, image):
        gray = to_gray(image)
        return {
            "mean": float(gray.mean()),
            "median": float(np.median(gray)),
            "std": float(gray.std()),
            "min": int(gray.min()),
            "max": int(gray.max()),
            "dark": float((gray < 64).mean() * 100),
            "light": float((gray >= 192).mean() * 100),
        }

    def _update_table(self):
        before = self._statistics(self.original_image)
        after = self._statistics(self.current_image)
        rows = (
            ("Kecerahan (rata-rata)", "mean", "{:.1f}"),
            ("Nilai tengah (median)", "median", "{:.1f}"),
            ("Kontras (std. deviasi)", "std", "{:.1f}"),
            ("Piksel paling gelap", "min", "{:d}"),
            ("Piksel paling terang", "max", "{:d}"),
            ("Area gelap (<64) %", "dark", "{:.1f}"),
            ("Area terang (>=192) %", "light", "{:.1f}"),
        )
        self.table.delete(*self.table.get_children())
        for name, key, fmt in rows:
            delta = after[key] - before[key]
            if abs(delta) < 0.05:
                change = "tetap"
            else:
                change = ("+" if delta > 0 else "") + fmt.format(delta)
            self.table.insert("", tk.END, values=(
                name, fmt.format(before[key]), fmt.format(after[key]), change))

        def describe(key, up, down):
            delta = after[key] - before[key]
            if abs(delta) < 1:
                return None
            return up if delta > 0 else down

        notes = [n for n in (describe("mean", "lebih terang", "lebih gelap"),
                             describe("std", "kontras naik", "kontras turun")) if n]
        if self.original_image.shape[:2] == self.current_image.shape[:2]:
            changed = cv2.absdiff(to_bgr(self.original_image),
                                  to_bgr(self.current_image)).max(axis=2)
            change_text = f"{(changed > 0).mean() * 100:.1f}% piksel berubah."
        else:
            change_text = "Ukuran/orientasi berubah."
        height, width = self.current_image.shape[:2]
        mode = "grayscale" if self.current_image.ndim == 2 else "berwarna"
        note = " (dari preview)" if self.scale > 1.001 else ""
        self.summary_label.configure(text=(
            f"Kesimpulan: {', '.join(notes) or 'kecerahan & kontras hampir sama'}. "
            f"{change_text}\n"
            f"Ukuran {int(round(width * self.scale))} x {int(round(height * self.scale))} px, "
            f"hasil {mode}. Statistik dihitung dari grayscale{note}."))

    def _refresh_data(self):
        self._pending_data = None
        if self.current_image is None:
            return
        self._draw_histograms()
        self._update_table()

    def _refresh_result(self):
        if self.current_image is None:
            return
        names = [step["name"] for step in self.steps]
        if self.active:
            if self.active["is_new"]:
                names.append(f"{self.active['name']} (belum diterapkan)")
            else:
                names[self.active["index"]] += " (diedit)"
        self.history_label.configure(text=" \u2192 ".join(["Asli"] + names))
        self._render_views()
        # Tunda sedikit agar slider tetap lancar saat digeser
        if self._pending_data is not None:
            self.root.after_cancel(self._pending_data)
        self._pending_data = self.root.after(80, self._refresh_data)

    # ------------------------------------------------------------------
    # File
    # ------------------------------------------------------------------
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        # imdecode + fromfile: aman untuk path non-ASCII; IMREAD_COLOR menerapkan
        # orientasi EXIF dan mengubah gambar 16-bit/alpha menjadi BGR 8-bit.
        try:
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            image = None
        if image is None:
            messagebox.showerror("Gagal membuka gambar", "File tidak dapat dibaca sebagai gambar.")
            return

        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > PREVIEW_MAX:
            factor = PREVIEW_MAX / longest
            preview = cv2.resize(
                image, (max(1, round(width * factor)), max(1, round(height * factor))),
                interpolation=cv2.INTER_AREA)
        else:
            preview = image

        self.filename = path
        self.full_image = image
        self.original_image = preview
        self.current_image = preview
        self.scale = longest / max(preview.shape[:2])
        self.active = None
        self.steps = []
        self.undo_stack = []
        self.redo_stack = []
        self._cache = None
        self.zoom, self.center = 1.0, [0.5, 0.5]

        info = f"{os.path.basename(path)}  ({width} x {height} px"
        if preview is not image:
            info += f", preview {preview.shape[1]} x {preview.shape[0]}"
        self.file_label.configure(text=info + ")")
        self._refresh_steps_list()
        self._update_buttons()
        self._show_message("Gambar berhasil dibuka. Pilih fitur untuk mulai mengedit.")
        self._recompute()

    def save_file(self):
        if not self._require_image():
            return
        self._close_active(True, "Langkah diterapkan sebelum disimpan.")

        default_name = "hasil_edit.png"
        if self.filename:
            stem = os.path.splitext(os.path.basename(self.filename))[0]
            default_name = f"{stem}_edited.png"

        path = filedialog.asksaveasfilename(
            title="Simpan hasil gambar",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif *.tiff"),
            ],
        )
        if not path:
            return

        # Semua langkah diulang pada resolusi penuh, bukan preview.
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        success = False
        try:
            result = self.full_image
            for step in self.steps:
                result = self._apply(result, step["name"], step["params"], self.scale)
            ext = os.path.splitext(path)[1].lower() or ".png"
            flags = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
            ok, buffer = cv2.imencode(ext, result, flags)
            if ok:
                buffer.tofile(path)
                success = True
        except (OSError, cv2.error, ValueError, MemoryError):
            success = False
        finally:
            self.root.configure(cursor="")

        if success:
            messagebox.showinfo("Berhasil", f"Gambar disimpan di:\n{path}")
        else:
            messagebox.showerror("Gagal menyimpan", "Gambar tidak berhasil disimpan.")


if __name__ == "__main__":
    root = tk.Tk()
    app = MiniPhotoshop(root)
    root.mainloop()