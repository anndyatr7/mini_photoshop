# 🎨 Mini Photoshop - Aplikasi Pengolahan Citra (Digital Image Processing)

Aplikasi desktop GUI berbasis Python untuk melakukan berbagai macam operasi pengolahan citra digital secara interaktif. Proyek ini dikembangkan sebagai tugas akhir mata kuliah Pengolahan Citra.

---

## 🚀 Fitur Utama
Aplikasi ini memiliki 11 fitur pengolahan citra utama yang berjalan secara *real-time* beserta visualisasi histogram warna:
1. **Open & Save File**: Mendukung format gambar populer seperti `.jpg`, `.jpeg`, dan `.png`.
2. **Histogram**: Visualisasi grafik distribusi warna (Red, Green, Blue) citra.
3. **Thresholding**: Mengubah citra keabuan menjadi citra biner (hitam putih) dengan slider batas nilai (*Th*).
4. **Equalization**: Perataan histogram untuk memperbaiki kontras distribusi intensitas piksel citra.
5. **Sketch**: Efek konversi citra menyerupai sketsa pensil menggunakan teknik inversi dan *Gaussian Blur*.
6. **Blur**: Berbagai metode *low-pass filter* (Gaussian, Averaging, Median, Bilateral, dan Standard 2D) untuk mengurangi *noise*.
7. **Grayscale**: Konversi citra warna RGB ke keabuan menggunakan standar NTSC.
8. **Brightness**: Pengaturan tingkat kecerahan citra secara dinamis.
9. **Negative**: Membalikkan nilai intensitas warna citra (*reverse*).
10. **Mirroring**: Pencerminan citra secara horizontal, vertikal, atau keduanya.
11. **Sharpening & Edge Detection**: Penajaman tepi objek serta ekstraksi garis tepi menggunakan kernel matriks konvolusi 2D.

---

## 🛠️ Kebutuhan Sistem & Teknologi
Proyek ini dibangun menggunakan bahasa **Python** dan beberapa pustaka (*library*) pengolahan data/citra:
* **Python 3.x**
* **OpenCV (`cv2`)** - Pemrosesan matriks citra dan komputer visi.
* **Tkinter** - Antarmuka grafis (GUI).
* **NumPy** - Operasi matriks dan array.
* **Matplotlib** - Pembuatan grafik histogram.
* **Pillow (`PIL`)** - Manipulasi dan konversi format gambar.

---

## ⚙️ Cara Instalasi dan Menjalankan

1. **Clone repository ini (atau download sebagai ZIP):**
   ```bash
   git clone [https://github.com/username-kamu/nama-repository.git](https://github.com/username-kamu/nama-repository.git)
   cd nama-repository
