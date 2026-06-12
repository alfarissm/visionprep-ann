# Desain: Tool Preprocessing Gambar — UAS INF322 Image Processing

Tanggal: 2026-06-10

## Tujuan
Aplikasi desktop (Windows 11, Python + Tkinter) untuk preprocessing gambar .jpg
secara semi-otomatis, menghasilkan dataset siap latih untuk model ANN.
Hanya memakai NumPy dasar + Matplotlib dasar (sesuai ketentuan soal).

## Arsitektur
```
project_uas/
  main.py                  # GUI Tkinter: pilih folder, pilih tahap, klik Start
  processing/
    imgio.py               # baca/simpan .jpg, parsing nama file & kelas
    grayscale.py           # tahap 1: RGB -> grayscale (rata-rata R,G,B)
    resizing.py            # tahap 2: average pooling / max pooling, ukuran bebas
    binarization.py        # tahap 3: inversi + threshold -> hitam/putih
    dataset.py             # tahap 4: gambar biner -> inputs .npy + labels one-hot .npy
    randomize.py           # tahap 5: acak baris inputs & labels bersamaan
    ann.py                 # pembuktian: training + testing ANN 1 hidden layer (NumPy murni)
```

## Alur data (semua file di folder yang sama, dipilih user)
```
nama (i).jpg  --grayscale-->  nama (i)_gs.jpg
              --resizing-->   nama (i)_rs.jpg
              --binarisasi--> nama (i)_bin.jpg
              --dataset-->    inputs_<n>_<kolom>.npy, labels_<n>_<k+1>.npy, classes.txt
              --randomize-->  random_inputs_*.npy, random_labels_*.npy
              --ANN-->        bobot w/b .npy + log akurasi per epoch
```

## Konvensi (mengikuti materi dosen M10-M14)
- Dataset: kolom 0 = nomor sampel, kolom 1..N = piksel (flatten row-major).
- Label: one-hot, kolom 0 = nomor sampel.
- Kelas dideteksi otomatis dari prefix nama file sebelum angka:
  `lingkaran (3).jpg` -> kelas "lingkaran"; `Hangul7.jpg` -> kelas "Hangul".
- Binarisasi: inversi (255-x) lalu threshold (default 100) -> latar hitam, objek putih.
- ANN: 1 hidden layer, aktivasi sigmoid, backpropagation, learning rate 0.001.
  Input dinormalisasi 0-1 agar sigmoid tidak saturasi.

## GUI
- Pilih folder gambar (Browse).
- Radiobutton tahap: Grayscale / Resizing / Binarization / Create Dataset /
  Randomize / Train ANN. Klik tahap lalu klik "Start" (sesuai soal).
- Parameter: metode pooling (Average/Max), Row, Col, threshold,
  jumlah hidden neuron, epochs.
- Log proses di Text area; proses jalan di thread supaya GUI tidak beku.

## Pengujian
- Uji pipeline penuh dengan subset gambar Hangul dosen (3 kelas x 5 gambar).
- Bukti validitas dataset: akurasi training ANN naik mendekati 100%.
- Uji kotak hitam per tombol untuk laporan.
