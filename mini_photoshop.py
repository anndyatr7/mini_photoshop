import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MiniPhotoshop:
    """Aplikasi pengolahan citra sederhana dengan preview otomatis dan edit berantai."""

    def __init__(self, root):
        self.root = root
        self.root.title("Mini Photoshop - Pengolahan Citra")
        self.root.geometry("1280x850")
        self.root.minsize(950, 700)

        self.filename = None
        self.original_image = None
        self.current_image = None
        self.feature_base_image = None
        self.current_feature = None

        self._build_menu()
        self._build_layout()
        self._show_empty_state()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open File", command=self.open_file)
        file_menu.add_command(label="Save Result", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Reset Image", command=self.reset_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_layout(self):
        image_frame = tk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

        left_frame = tk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        right_frame = tk.Frame(image_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(left_frame, text="Gambar Asli", font=("Arial", 11, "bold")).pack()
        tk.Label(right_frame, text="Hasil Pengolahan", font=("Arial", 11, "bold")).pack()

        self.panel_img_left = tk.Label(left_frame, text="Belum ada gambar", bg="lightgray")
        self.panel_img_left.pack(fill=tk.BOTH, expand=True, pady=5)
        self.panel_img_right = tk.Label(right_frame, text="Hasil akan ditampilkan di sini", bg="lightgray")
        self.panel_img_right.pack(fill=tk.BOTH, expand=True, pady=5)

        histogram_frame = tk.Frame(self.root)
        histogram_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        self.hist_left_frame = tk.Frame(histogram_frame)
        self.hist_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.hist_right_frame = tk.Frame(histogram_frame)
        self.hist_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(self.hist_left_frame, text="Histogram Gambar Asli",
                 font=("Arial", 9, "bold")).pack()
        tk.Label(self.hist_right_frame, text="Histogram Hasil",
                 font=("Arial", 9, "bold")).pack()

        self.hist_left_canvas = None
        self.hist_right_canvas = None

        control_frame = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=8, pady=8)

        features = [
            ("Thresholding", self.apply_thresholding),
            ("Equalization", self.apply_equalization),
            ("Sketch", self.apply_sketch),
            ("Blur", self.apply_blur),
            ("Grayscale", self.apply_grayscale),
            ("Brightness", self.apply_brightness),
            ("Negative", self.apply_negative),
            ("Mirroring", self.apply_mirroring),
            ("Sharpening", self.apply_sharpening),
            ("Edge Detection", self.apply_edge_detection),
        ]

        for index, (label, command) in enumerate(features):
            row = index % 5
            column = index // 5
            tk.Button(
                button_frame, text=label, width=15, command=command
            ).grid(row=row, column=column, padx=4, pady=4)

        tk.Button(
            button_frame, text="RESET", width=15,
            bg="#f0d5d5", command=self.reset_image
        ).grid(row=5, column=0, padx=4, pady=4)

        self.frame_dynamic = tk.Frame(
            control_frame, bd=1, relief=tk.SUNKEN, padx=8, pady=6
        )
        self.frame_dynamic.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8
        )
        self._show_feature_message("Pilih fitur pengolahan citra di sebelah kiri.")

    def _show_empty_state(self):
        self.panel_img_left.configure(text="Belum ada gambar", image="")
        self.panel_img_right.configure(text="Belum ada gambar", image="")

    def _show_feature_message(self, text):
        self._clear_dynamic_frame()
        tk.Label(self.frame_dynamic, text=text, justify=tk.LEFT).pack(
            anchor=tk.W, padx=5, pady=12
        )

    def _clear_dynamic_frame(self):
        for widget in self.frame_dynamic.winfo_children():
            widget.destroy()

    def _require_image(self):
        if self.current_image is None:
            messagebox.showwarning("Belum ada gambar", "Silakan buka gambar terlebih dahulu.")
            return False
        return True

    # ------------------------------------------------------------------
    # Image state and display
    # ------------------------------------------------------------------
    @staticmethod
    def _copy_image(image):
        return None if image is None else image.copy()

    def _begin_feature(self, feature_name):
        """Simpan hasil terakhir sebagai dasar fitur yang baru dipilih."""
        if not self._require_image():
            return False
        self.current_feature = feature_name
        self.feature_base_image = self.current_image.copy()
        return True

    def _apply_from_feature_base(self, operation):
        """Setiap perubahan parameter dihitung dari snapshot, bukan hasil slider sebelumnya."""
        if self.feature_base_image is None:
            return
        try:
            result = operation(self.feature_base_image.copy())
            if result is None:
                return
            self.current_image = result
            self._refresh_result()
        except (cv2.error, ValueError, TypeError) as error:
            messagebox.showerror("Kesalahan pemrosesan", str(error))

    @staticmethod
    def _to_gray(image):
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _to_bgr(image):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _fit_image(self, image, max_width=560, max_height=330):
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA) if scale < 1 else image

    def _display_image(self, panel, image):
        display = self._fit_image(image)
        if display.ndim == 2:
            rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
        elif display.shape[2] == 4:
            rgb = cv2.cvtColor(display, cv2.COLOR_BGRA2RGBA)
        else:
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        panel.configure(image=photo, text="")
        panel.image = photo

    def _refresh_result(self):
        if self.current_image is None:
            return
        self._display_image(self.panel_img_right, self.current_image)
        self._draw_histogram(self.hist_right_frame, self.current_image, "right")

    def _draw_histogram(self, parent, image, side):
        old_canvas = self.hist_left_canvas if side == "left" else self.hist_right_canvas
        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()

        fig = Figure(figsize=(5.2, 1.45), dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensitas piksel", fontsize=7)
        ax.set_ylabel("Jumlah", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)

        if image.ndim == 2:
            ax.hist(image.ravel(), bins=256, range=(0, 256),
                    color="black", alpha=0.8)
        else:
            b, g, r = cv2.split(self._to_bgr(image))
            ax.hist(r.ravel(), bins=256, range=(0, 256),
                    color="red", alpha=0.38, label="Red")
            ax.hist(g.ravel(), bins=256, range=(0, 256),
                    color="green", alpha=0.38, label="Green")
            ax.hist(b.ravel(), bins=256, range=(0, 256),
                    color="blue", alpha=0.38, label="Blue")
            ax.legend(loc="upper right", fontsize=6, frameon=False)

        fig.tight_layout(pad=1.0)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if side == "left":
            self.hist_left_canvas = canvas
        else:
            self.hist_right_canvas = canvas

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            messagebox.showerror("Gagal membuka gambar", "File tidak dapat dibaca sebagai gambar.")
            return

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        self.filename = path
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.feature_base_image = image.copy()
        self.current_feature = None

        self._display_image(self.panel_img_left, self.original_image)
        self._display_image(self.panel_img_right, self.current_image)
        self._draw_histogram(self.hist_left_frame, self.original_image, "left")
        self._draw_histogram(self.hist_right_frame, self.current_image, "right")
        self._show_feature_message(
            "Gambar berhasil dibuka. Pilih fitur untuk mulai mengedit."
        )

    def save_file(self):
        if not self._require_image():
            return

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

        image_to_save = self.current_image
        if image_to_save.ndim == 3 and image_to_save.shape[2] == 3:
            image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_BGR2RGB)
            image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(path, image_to_save)
        if success:
            messagebox.showinfo("Berhasil", f"Gambar disimpan di:\n{path}")
        else:
            messagebox.showerror("Gagal menyimpan", "Gambar tidak berhasil disimpan.")

    def reset_image(self):
        if self.original_image is None:
            messagebox.showinfo("Reset", "Belum ada gambar yang dibuka.")
            return

        self.current_image = self.original_image.copy()
        self.feature_base_image = self.original_image.copy()
        self.current_feature = None

        self._display_image(self.panel_img_left, self.original_image)
        self._refresh_result()
        self._show_feature_message("Gambar dikembalikan ke kondisi asli.")

    # ------------------------------------------------------------------
    # Feature controls
    # ------------------------------------------------------------------
    def apply_thresholding(self):
        if not self._begin_feature("Thresholding"):
            return
        self._clear_dynamic_frame()
        tk.Label(self.frame_dynamic, text="Nilai Threshold (0–255):").pack(anchor=tk.W)
        value_label = tk.Label(self.frame_dynamic, text="128")
        value_label.pack(anchor=tk.W)

        def update(value):
            threshold = int(float(value))
            value_label.config(text=str(threshold))

            def operation(image):
                gray = self._to_gray(image)
                _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                return result

            self._apply_from_feature_base(operation)

        slider = tk.Scale(
            self.frame_dynamic, from_=0, to=255, orient=tk.HORIZONTAL,
            command=update, resolution=1
        )
        slider.set(128)
        slider.pack(fill=tk.X)

    def apply_equalization(self):
        if not self._begin_feature("Equalization"):
            return

        def operation(image):
            if image.ndim == 2:
                return cv2.equalizeHist(image)
            bgr = self._to_bgr(image)
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        self._apply_from_feature_base(operation)
        self._show_feature_message("Histogram equalization diterapkan otomatis.")

    def apply_sketch(self):
        if not self._begin_feature("Sketch"):
            return

        def operation(image):
            gray = self._to_gray(image)
            inverted = cv2.bitwise_not(gray)
            blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
            inverted_blur = cv2.bitwise_not(blurred)
            return cv2.divide(gray, inverted_blur, scale=256.0)

        self._apply_from_feature_base(operation)
        self._show_feature_message("Efek sketch diterapkan otomatis.")

    def apply_blur(self):
        if not self._begin_feature("Blur"):
            return
        self._clear_dynamic_frame()
        tk.Label(self.frame_dynamic, text="Pilih jenis blur:").pack(anchor=tk.W)

        modes = {
            "Gaussian Blur": "gaussian",
            "Averaging Blur": "averaging",
            "Median Blur": "median",
            "Bilateral Blur": "bilateral",
            "Standard Blur (2D)": "standard",
        }
        # Tidak ada metode yang dipilih otomatis. Gambar baru berubah
        # setelah pengguna memilih salah satu metode blur.
        selected = tk.StringVar(value="")

        def update():
            mode = selected.get()
            if not mode:
                return

            def operation(image):
                if mode == "gaussian":
                    return cv2.GaussianBlur(image, (5, 5), 0)
                if mode == "averaging":
                    return cv2.blur(image, (5, 5))
                if mode == "median":
                    return cv2.medianBlur(image, 5)
                if mode == "bilateral":
                    return cv2.bilateralFilter(self._to_bgr(image), 9, 75, 75)
                kernel = np.ones((5, 5), np.float32) / 25
                return cv2.filter2D(image, -1, kernel)

            self._apply_from_feature_base(operation)

        for label, value in modes.items():
            tk.Radiobutton(
                self.frame_dynamic, text=label, variable=selected,
                value=value, command=update
            ).pack(anchor=tk.W, padx=8)

    def apply_grayscale(self):
        if not self._begin_feature("Grayscale"):
            return
        self._apply_from_feature_base(self._to_gray)
        self._show_feature_message("Konversi grayscale diterapkan otomatis.")

    def apply_brightness(self):
        if not self._begin_feature("Brightness"):
            return
        self._clear_dynamic_frame()
        tk.Label(self.frame_dynamic, text="Brightness (-255 sampai 255):").pack(anchor=tk.W)
        value_label = tk.Label(self.frame_dynamic, text="0")
        value_label.pack(anchor=tk.W)

        def update(value):
            amount = int(float(value))
            value_label.config(text=str(amount))

            def operation(image):
                result = image.astype(np.int16) + amount
                return np.clip(result, 0, 255).astype(np.uint8)

            self._apply_from_feature_base(operation)

        slider = tk.Scale(
            self.frame_dynamic, from_=-255, to=255, orient=tk.HORIZONTAL,
            command=update, resolution=1
        )
        slider.set(0)
        slider.pack(fill=tk.X)

    def apply_negative(self):
        if not self._begin_feature("Negative"):
            return
        self._apply_from_feature_base(cv2.bitwise_not)
        self._show_feature_message("Negative diterapkan otomatis.")

    
    
    def apply_mirroring(self):
        if not self._begin_feature("Mirroring"):
            return

        self._clear_dynamic_frame()

        tk.Label(
            self.frame_dynamic,
            text="Pilih Jenis Mirroring:"
        ).pack(anchor=tk.W, padx=5, pady=5)

        def process_mirror(flip_code):
            # Ambil gambar hasil pengolahan terakhir
            current_image = self.current_image.copy()

            # Lakukan mirroring pada gambar aktif
            flip_img = cv2.flip(current_image, flip_code)

            # Simpan hasil sebagai gambar aktif terbaru
            self.current_image = flip_img

            # Perbarui tampilan gambar dan histogram
            self._refresh_result()

        tk.Button(
            self.frame_dynamic,
            text="Flip Horizontal",
            command=lambda: process_mirror(1)
        ).pack(anchor=tk.W, padx=10, pady=3)

        tk.Button(
            self.frame_dynamic,
            text="Flip Vertikal",
            command=lambda: process_mirror(0)
        ).pack(anchor=tk.W, padx=10, pady=3)

        tk.Button(
            self.frame_dynamic,
            text="Flip Keduanya",
            command=lambda: process_mirror(-1)
        ).pack(anchor=tk.W, padx=10, pady=3)


    def apply_sharpening(self):
        if not self._begin_feature("Sharpening"):
            return
        self._clear_dynamic_frame()
        tk.Label(
            self.frame_dynamic,
            text="Intensitas Sharpening (0–5):"
        ).pack(anchor=tk.W)
        value_label = tk.Label(self.frame_dynamic, text="1.0")
        value_label.pack(anchor=tk.W)

        def update(value):
            amount = float(value)
            value_label.config(text=f"{amount:.1f}")

            def operation(image):
                # Amount 0 menghasilkan gambar dasar; semakin besar
                # nilainya, semakin kuat penajaman tepi.
                kernel = np.array([
                    [0, -1, 0],
                    [-1, 4 + amount, -1],
                    [0, -1, 0],
                ], dtype=np.float32)
                return cv2.filter2D(image, -1, kernel)

            self._apply_from_feature_base(operation)

        slider = tk.Scale(
            self.frame_dynamic,
            from_=0.0,
            to=5.0,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            command=update,
        )
        slider.set(1.0)
        slider.pack(fill=tk.X)

    def apply_edge_detection(self):
        if not self._begin_feature("Edge Detection"):
            return
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1],
        ], dtype=np.float32)

        def operation(image):
            gray = self._to_gray(image)
            return cv2.filter2D(gray, cv2.CV_16S, kernel).clip(0, 255).astype(np.uint8)

        self._apply_from_feature_base(operation)
        self._show_feature_message("Edge detection diterapkan otomatis.")


if __name__ == "__main__":
    root = tk.Tk()
    app = MiniPhotoshop(root)
    root.mainloop()
