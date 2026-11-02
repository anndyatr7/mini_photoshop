import os
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class MiniPhotoshop:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Photoshop - Pengolahan Citra")
        self.root.geometry("1280x800")
        
        self.filename = ""
        self.image = None
        self.matrix_img_left = None
        self.matrix_img_right = None
        
        # Buat direktori penyimpanan hasil jika belum ada
        if not os.path.exists("result_image"):
            os.makedirs("result_image")

        # Layout Utama GUI
        self.create_widgets()

    def create_widgets(self):
        # Menu Bar
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open File", command=self.open_file)
        filemenu.add_command(label="Save File", command=self.save_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        # Panel Gambar (Kiri & Kanan)
        # Panel Gambar (Kiri & Kanan) - Hapus parameter width & height agar menyesuaikan gambar
        frame_img = tk.Frame(self.root)
        frame_img.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.panel_img_left = tk.Label(frame_img, text="No Image Loaded", bg="lightgray")
        self.panel_img_left.pack(side=tk.LEFT, padx=10, expand=True)

        self.panel_img_right = tk.Label(frame_img, text="Processed Image", bg="lightgray")
        self.panel_img_right.pack(side=tk.RIGHT, padx=10, expand=True)

        # Panel Histogram (Kiri & Kanan) - Hapus parameter width & height
        frame_hist = tk.Frame(self.root)
        frame_hist.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)

        self.panel_hist_left = tk.Label(frame_hist, bg="white")
        self.panel_hist_left.pack(side=tk.LEFT, padx=10, expand=True)

        self.panel_hist_right = tk.Label(frame_hist, bg="white")
        self.panel_hist_right.pack(side=tk.RIGHT, padx=10, expand=True)

        # Panel Kontrol & Tombol Fitur
        frame_control = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
        frame_control.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Tombol-tombol Fitur Pengolahan Citra
        btn_frame = tk.Frame(frame_control)
        btn_frame.pack(side=tk.LEFT, padx=10, pady=10)

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
            ("Edge Detection", self.apply_edge_detection)
        ]

        for i, (text, cmd) in enumerate(features):
            row = i % 5
            col = i // 5
            b = tk.Button(btn_frame, text=text, width=15, command=cmd)
            b.grid(row=col, column=row, padx=5, pady=5)

        # Area Interaksi Dinamis / Parameter Tambahan di Kanan Bawah
        self.frame_dynamic = tk.Frame(frame_control, bd=1, relief=tk.SUNKEN)
        self.frame_dynamic.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(self.frame_dynamic, text="Pilih salah satu fitur di sebelah kiri").pack(padx=20, pady=20)

    def open_file(self):
        filetypes = ([
            ('image files', ('*.jpeg', '*.JPEG', '*.jpg', '*.JPG', '*.png'))
        ])
        self.filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes
        )
        if self.filename:
            showinfo(title='Selected File', message=self.filename)
            self.show_img_left()
            self.show_hist_left()

    def show_img_left(self):
        self.matrix_img_left = cv2.imread(self.filename)
        self.image = self.matrix_img_left.copy() # Simpan citra asli untuk diproses
        
        # Resize untuk tampilan GUI
        display_img = cv2.resize(self.matrix_img_left, (533, 300), interpolation=cv2.INTER_CUBIC)
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        img = ImageTk.PhotoImage(image=Image.fromarray(display_img_rgb))
        self.panel_img_left.configure(image=img)
        self.panel_img_left.image = img

    def show_hist_left(self):
        plt.figure(figsize=(6, 2))
        x = self.matrix_img_left
        b, g, r = cv2.split(x)
        plt.hist(b.flatten(), 255, [0, 256], color='blue', alpha=0.8)
        plt.hist(g.flatten(), 255, [0, 256], color='green', alpha=0.8)
        plt.hist(r.flatten(), 255, [0, 256], color='red', alpha=0.8)
        plt.xlim([0, 255])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('hist_left.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        matrix_img_hist = cv2.imread('hist_left.png')
        matrix_img_hist = cv2.resize(matrix_img_hist, (320, 120), interpolation=cv2.INTER_CUBIC)
        matrix_img_hist_rgb = cv2.cvtColor(matrix_img_hist, cv2.COLOR_BGR2RGB)
        
        img = ImageTk.PhotoImage(image=Image.fromarray(matrix_img_hist_rgb))
        self.panel_hist_left.configure(image=img)
        self.panel_hist_left.image = img

    def update_right_panel(self, processed_matrix):
        self.matrix_img_right = processed_matrix
        
        # Tampilkan gambar hasil di panel kanan
        display_img = cv2.resize(processed_matrix, (533, 300), interpolation=cv2.INTER_CUBIC)
        if len(display_img.shape) == 2: # Jika grayscale/binary
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        else:
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
        img = ImageTk.PhotoImage(image=Image.fromarray(display_img_rgb))
        self.panel_img_right.configure(image=img)
        self.panel_img_right.image = img

        # Buat histogram kanan
        plt.figure(figsize=(6, 2))
        if len(processed_matrix.shape) == 2:
            plt.hist(processed_matrix.flatten(), 255, [0, 256], color='black', alpha=0.8)
        else:
            b, g, r = cv2.split(processed_matrix)
            plt.hist(b.flatten(), 255, [0, 256], color='blue', alpha=0.8)
            plt.hist(g.flatten(), 255, [0, 256], color='green', alpha=0.8)
            plt.hist(r.flatten(), 255, [0, 256], color='red', alpha=0.8)
        plt.xlim([0, 255])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('hist_right.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        matrix_hist_r = cv2.imread('hist_right.png')
        matrix_hist_r = cv2.resize(matrix_hist_r, (320, 120), interpolation=cv2.INTER_CUBIC)
        matrix_hist_r_rgb = cv2.cvtColor(matrix_hist_r, cv2.COLOR_BGR2RGB)
        
        img_h = ImageTk.PhotoImage(image=Image.fromarray(matrix_hist_r_rgb))
        self.panel_hist_right.configure(image=img_h)
        self.panel_hist_right.image = img_h

    def save_file(self):
        if self.matrix_img_right is not None:
            save_path = os.path.join("result_image", os.path.basename(self.filename))
            # Konversi kembali ke BGR jika formatnya RGB/Grayscale sebelum disimpan
            if len(self.matrix_img_right.shape) == 2:
                save_img = self.matrix_img_right
            else:
                save_img = cv2.cvtColor(self.matrix_img_right, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            showinfo(title='Image Saved', message=f"Disimpan di: {save_path}")
        else:
            showinfo(title='Error', message="Belum ada gambar yang diproses!")

    def clear_dynamic_frame(self):
        for widget in self.frame_dynamic.winfo_children():
            widget.destroy()

    # --- IMPLEMENTASI 11 FITUR UTAMA SESUAI LAPORAN ---

    def apply_thresholding(self):
        self.clear_dynamic_frame()
        tk.Label(self.frame_dynamic, text="Atur Nilai Threshold (Th):").pack(anchor=tk.W, padx=5)
        
        slider = tk.Scale(self.frame_dynamic, from_=0, to=255, orient=tk.HORIZONTAL)
        slider.set(128)
        slider.pack(fill=tk.X, padx=5)

        def process():
            th_val = slider.get()
            img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, img_gray_th = cv2.threshold(img_gray, th_val, 255, cv2.THRESH_BINARY)
            self.update_right_panel(img_gray_th)

        tk.Button(self.frame_dynamic, text="Process", command=process).pack(pady=5)

    def apply_equalization(self):
        img_yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.update_right_panel(hist_eq)

    def apply_sketch(self):
        grey_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        invert_img = cv2.bitwise_not(grey_img)
        blur_img = cv2.GaussianBlur(invert_img, (21, 21), 0)
        inverted_blur_img = cv2.bitwise_not(blur_img)
        sketch_img = cv2.divide(grey_img, inverted_blur_img, scale=256.0)
        self.update_right_panel(sketch_img)

    def apply_blur(self):
        self.clear_dynamic_frame()
        tk.Label(self.frame_dynamic, text="Pilih Jenis Blur:").pack(anchor=tk.W, padx=5)
        
        v = tk.StringVar(value="gaussian")
        blur_types = [
            ("Gaussian Blur", "gaussian"),
            ("Averaging Blur", "averaging"),
            ("Median Blurring", "median"),
            ("Bilateral Blurring", "bilateral"),
            ("Standard Blur (2D)", "standard")
        ]
        
        for text, mode in blur_types:
            tk.Radiobutton(self.frame_dynamic, text=text, variable=v, value=mode).pack(anchor=tk.W, padx=10)

        def process():
            mode = v.get()
            if mode == "gaussian":
                res = cv2.GaussianBlur(self.image, (5, 5), 0)
            elif mode == "averaging":
                res = cv2.blur(self.image, (5, 5))
            elif mode == "median":
                res = cv2.medianBlur(self.image, 5)
            elif mode == "bilateral":
                res = cv2.bilateralFilter(self.image, 9, 75, 75)
            elif mode == "standard":
                kernel = np.ones((5, 5), np.float32) / 25
                res = cv2.filter2D(self.image, -1, kernel)
            self.update_right_panel(res)

        tk.Button(self.frame_dynamic, text="Process", command=process).pack(pady=5)

    def apply_grayscale(self):
        # Menggunakan rumus standar / konversi openCV
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.update_right_panel(gray_img)

    def apply_brightness(self):
        self.clear_dynamic_frame()
        tk.Label(self.frame_dynamic, text="Atur Nilai Brightness (-255 s.d 255):").pack(anchor=tk.W, padx=5)
        
        slider = tk.Scale(self.frame_dynamic, from_=-255, to=255, orient=tk.HORIZONTAL)
        slider.set(40)
        slider.pack(fill=tk.X, padx=5)

        def process():
            val = slider.get()
            img_temp = self.image.astype('int16') + val
            img_temp = np.where(img_temp > 255, 255, np.where(img_temp < 0, 0, img_temp))
            bright_img = img_temp.astype('uint8')
            self.update_right_panel(bright_img)

        tk.Button(self.frame_dynamic, text="Process", command=process).pack(pady=5)

    def apply_negative(self):
        neg_img = cv2.bitwise_not(self.image)
        self.update_right_panel(neg_img)

    def apply_mirroring(self):
        self.clear_dynamic_frame()
        tk.Label(self.frame_dynamic, text="Pilih Jenis Mirror:").pack(anchor=tk.W, padx=5)
        
        v = tk.IntVar(value=1)
        tk.Radiobutton(self.frame_dynamic, text="Flip Horizontal (Sumbu Y)", variable=v, value=1).pack(anchor=tk.W, padx=10)
        tk.Radiobutton(self.frame_dynamic, text="Flip Vertikal (Sumbu X)", variable=v, value=0).pack(anchor=tk.W, padx=10)
        tk.Radiobutton(self.frame_dynamic, text="Flip Keduanya (-1)", variable=v, value=-1).pack(anchor=tk.W, padx=10)

        def process():
            key = v.get()
            flip_img = cv2.flip(self.image, key)
            self.update_right_panel(flip_img)

        tk.Button(self.frame_dynamic, text="Process", command=process).pack(pady=5)

    def apply_sharpening(self):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharp_img = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
        self.update_right_panel(sharp_img)

    def apply_edge_detection(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edge_img = cv2.filter2D(src=gray_img, ddepth=-1, kernel=kernel)
        self.update_right_panel(edge_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = MiniPhotoshop(root)
    root.mainloop()