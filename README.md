# 🎨 Mini Photoshop - Aplikasi Pengolahan Citra (Digital Image Processing)

Aplikasi desktop GUI berbasis Python untuk melakukan berbagai macam operasi pengolahan citra digital secara interaktif[cite: 5, 7]. Proyek ini dikembangkan sebagai tugas akhir mata kuliah Pengolahan Citra[cite: 4].

---

## 🚀 Fitur Utama
Aplikasi ini memiliki 11 fitur pengolahan citra utama yang berjalan secara *real-time* beserta visualisasi histogram warna[cite: 7, 25]:
1. **Open & Save File**: Mendukung format gambar populer seperti `.jpg`, `.jpeg`, dan `.png`[cite: 7, 8].
2. **Histogram**: Visualisasi grafik distribusi warna (Red, Green, Blue) citra.
3. **Thresholding**: Mengubah citra keabuan menjadi citra biner (hitam putih) dengan slider batas nilai (*Th*)[cite: 11, 12].
4. **Equalization**: Perataan histogram untuk memperbaiki kontras distribusi intensitas piksel citra[cite: 12, 13].
5. **Sketch**: Efek konversi citra menyerupai sketsa pensil menggunakan teknik inversi dan *Gaussian Blur*[cite: 14].
6. **Blur**: Berbagai metode *low-pass filter* (Gaussian, Averaging, Median, Bilateral, dan Standard 2D) untuk mengurangi *noise*[cite: 14, 15, 16, 17, 18].
7. **Grayscale**: Konversi citra warna RGB ke keabuan menggunakan standar NTSC[cite: 18].
8. **Brightness**: Pengaturan tingkat kecerahan citra secara dinamis[cite: 19].
9. **Negative**: Membalikkan nilai intensitas warna citra (*reverse*)[cite: 20].
10. **Mirroring**: Pencerminan citra secara horizontal, vertikal, atau keduanya[cite: 21].
11. **Sharpening & Edge Detection**: Penajaman tepi objek serta ekstraksi garis tepi menggunakan kernel matriks konvolusi 2D[cite: 22, 23, 24].

---

## 🛠️ Kebutuhan Sistem & Teknologi
Proyek ini dibangun menggunakan bahasa **Python**[cite: 5] dan beberapa pustaka (*library*) pengolahan data/citra:
* **Python 3.x**
* **OpenCV (`cv2`)** - Pemrosesan matriks citra dan komputer visi[cite: 6].
* **Tkinter** - Antarmuka grafis (GUI)[cite: 5].
* **NumPy** - Operasi matriks dan array[cite: 19].
* **Matplotlib** - Pembuatan grafik histogram[cite: 6].
* **Pillow (`PIL`)** - Manipulasi dan konversi format gambar[cite: 7].

---

## ⚙️ Cara Instalasi dan Menjalankan

1. **Clone repository ini (atau download sebagai ZIP):**
   ```bash
   git clone [https://github.com/username-kamu/nama-repository.git](https://github.com/username-kamu/nama-repository.git)
   cd nama-repository
