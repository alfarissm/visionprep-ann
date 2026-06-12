"""Pembuktian dataset: training + testing ANN 1 hidden layer (NumPy murni).

Arsitektur dan algoritma mengikuti materi M10-M14:
- 1 hidden layer, aktivasi sigmoid, backpropagation, learning rate 0.001.
- Input dinormalisasi 0-1 (piksel/255) agar sigmoid tidak saturasi.
- Bobot hasil training disimpan sebagai ann_*.npy di folder dataset.
"""

import os

import numpy as np

from . import grayscale, resizing, binarization, imgio


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def load_classes(folder):
    """Baca ukuran gambar dan daftar kelas dari classes.txt."""
    path = os.path.join(folder, "classes.txt")
    if not os.path.exists(path):
        return None, None, None
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    rows, cols = (int(x) for x in lines[0].split())
    return rows, cols, lines[1:]


def find_random_files(folder):
    inputs_name, labels_name = None, None
    for filename in sorted(os.listdir(folder)):
        if filename.startswith("random_inputs_") and filename.endswith(".npy"):
            inputs_name = filename
        if filename.startswith("random_labels_") and filename.endswith(".npy"):
            labels_name = filename
    return inputs_name, labels_name


def train(folder, hidden_neurons=20, epochs=100, learn_rate=0.001, log=print):
    """Latih ANN dengan dataset acak di folder. Mengembalikan akurasi akhir."""
    inputs_name, labels_name = find_random_files(folder)
    if inputs_name is None or labels_name is None:
        log("Dataset random_*.npy tidak ditemukan. Jalankan Randomize dulu.")
        return None

    # Kolom 0 berisi nomor sampel, tidak ikut diproses.
    inputs = np.load(os.path.join(folder, inputs_name))[:, 1:].astype(float)
    labels = np.load(os.path.join(folder, labels_name))[:, 1:]
    inputs = inputs / 255.0  # normalisasi 0-1

    m, i_n = inputs.shape
    o_n = labels.shape[1]
    h_n = hidden_neurons
    log(f"Data latih: {m} sampel, {i_n} input neuron, "
        f"{h_n} hidden neuron, {o_n} output neuron.")

    w_i_h = np.random.uniform(-0.5, 0.5, (h_n, i_n))
    w_h_o = np.random.uniform(-0.5, 0.5, (o_n, h_n))
    b_i_h = np.zeros((h_n, 1))
    b_h_o = np.zeros((o_n, 1))

    accuracy = 0.0
    for epoch in range(1, epochs + 1):
        nr_correct = 0
        for inp, label in zip(inputs, labels):
            inp = inp.reshape(-1, 1)
            label = label.reshape(-1, 1)

            # Forward propagation input -> hidden -> output
            h = sigmoid(b_i_h + w_i_h @ inp)
            o = sigmoid(b_h_o + w_h_o @ h)

            nr_correct += int(np.argmax(o) == np.argmax(label))

            # Backpropagation output -> hidden
            delta_o = o - label
            w_h_o += -learn_rate * delta_o @ h.T
            b_h_o += -learn_rate * delta_o

            # Backpropagation hidden -> input
            delta_h = w_h_o.T @ delta_o * (h * (1 - h))
            w_i_h += -learn_rate * delta_h @ inp.T
            b_i_h += -learn_rate * delta_h

        accuracy = round(nr_correct / m * 100, 2)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            log(f"  epoch {epoch}: akurasi {accuracy}%")

    np.save(os.path.join(folder, "ann_w_i_h.npy"), w_i_h)
    np.save(os.path.join(folder, "ann_b_i_h.npy"), b_i_h)
    np.save(os.path.join(folder, "ann_w_h_o.npy"), w_h_o)
    np.save(os.path.join(folder, "ann_b_h_o.npy"), b_h_o)
    log(f"Training selesai. Akurasi akhir: {accuracy}%. "
        "Bobot disimpan sebagai ann_*.npy.")
    return accuracy


def predict(folder, sample_index, log=print):
    """Uji insample: prediksi satu sampel dari dataset acak.

    Mengembalikan (gambar 2D, label asli, label prediksi) untuk ditampilkan.
    """
    inputs_name, labels_name = find_random_files(folder)
    if inputs_name is None:
        log("Dataset random_*.npy tidak ditemukan.")
        return None
    rows, cols, classes = load_classes(folder)
    if classes is None:
        log("classes.txt tidak ditemukan. Jalankan Create Dataset dulu.")
        return None
    weight_path = os.path.join(folder, "ann_w_i_h.npy")
    if not os.path.exists(weight_path):
        log("Bobot ANN belum ada. Jalankan Train ANN dulu.")
        return None

    inputs = np.load(os.path.join(folder, inputs_name))[:, 1:].astype(float)
    labels = np.load(os.path.join(folder, labels_name))[:, 1:]
    m = inputs.shape[0]
    if not 0 <= sample_index < m:
        log(f"Nomor sampel harus 0 - {m - 1}.")
        return None

    w_i_h = np.load(os.path.join(folder, "ann_w_i_h.npy"))
    b_i_h = np.load(os.path.join(folder, "ann_b_i_h.npy"))
    w_h_o = np.load(os.path.join(folder, "ann_w_h_o.npy"))
    b_h_o = np.load(os.path.join(folder, "ann_b_h_o.npy"))

    inp = (inputs[sample_index] / 255.0).reshape(-1, 1)
    h = sigmoid(b_i_h + w_i_h @ inp)
    o = sigmoid(b_h_o + w_h_o @ h)

    actual = classes[int(np.argmax(labels[sample_index]))]
    predicted = classes[int(np.argmax(o))]
    log(f"Sampel {sample_index}: label asli = {actual}, "
        f"prediksi ANN = {predicted}.")
    picture = inputs[sample_index].reshape(rows, cols)
    return picture, actual, predicted


def predict_external(folder, image_path, method, threshold, invert, log=print):
    """Uji outsample: prediksi satu gambar .jpg dari luar dataset.

    Gambar mentah diproses penuh (grayscale luminance -> resize pooling ke
    ukuran dataset -> binarisasi) memakai parameter yang sama, lalu diprediksi
    memakai bobot ANN hasil training. Mengembalikan (gambar biner 2D, prediksi).
    """
    rows, cols, classes = load_classes(folder)
    if classes is None:
        log("classes.txt tidak ditemukan. Jalankan Create Dataset dulu.")
        return None
    if not os.path.exists(os.path.join(folder, "ann_w_i_h.npy")):
        log("Bobot ANN belum ada. Jalankan Train ANN dulu.")
        return None

    # Preprocessing gambar uji dengan pipeline & ukuran yang sama spt dataset.
    pic = imgio.read_image(image_path)
    gray = grayscale.to_grayscale(pic)
    small = resizing.resize_pooling(gray, rows, cols, method)
    binary = binarization.binarize(small, threshold, invert)

    w_i_h = np.load(os.path.join(folder, "ann_w_i_h.npy"))
    b_i_h = np.load(os.path.join(folder, "ann_b_i_h.npy"))
    w_h_o = np.load(os.path.join(folder, "ann_w_h_o.npy"))
    b_h_o = np.load(os.path.join(folder, "ann_b_h_o.npy"))

    inp = (binary.reshape(-1, 1).astype(float)) / 255.0
    h = sigmoid(b_i_h + w_i_h @ inp)
    o = sigmoid(b_h_o + w_h_o @ h)

    predicted = classes[int(np.argmax(o))]
    name = os.path.basename(image_path)
    log(f"Gambar uji '{name}' (luar dataset) -> prediksi ANN = {predicted}.")
    return binary, None, predicted
