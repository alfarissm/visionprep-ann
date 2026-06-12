"""Pembuktian dataset: ANN 1 hidden layer (NumPy murni).

Aktivasi sigmoid, backpropagation, learning rate 0.001. Input dinormalisasi
0-1 agar sigmoid tidak saturasi. Mengikuti algoritma materi M10-M14.
"""

import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def ann_train(inputs, labels, hidden, epochs, learn_rate=0.001, log=print):
    """Latih ANN. inputs sudah 0-1, labels one-hot. Mengembalikan bobot."""
    m, i_n = inputs.shape
    o_n = labels.shape[1]
    log(f"Data latih: {m} sampel, {i_n} input neuron, "
        f"{hidden} hidden neuron, {o_n} output neuron.")
    w_i_h = np.random.uniform(-0.5, 0.5, (hidden, i_n))
    w_h_o = np.random.uniform(-0.5, 0.5, (o_n, hidden))
    b_i_h = np.zeros((hidden, 1))
    b_h_o = np.zeros((o_n, 1))
    accuracy = 0.0
    for epoch in range(1, epochs + 1):
        correct = 0
        for inp, label in zip(inputs, labels):
            inp = inp.reshape(-1, 1)
            label = label.reshape(-1, 1)
            # Forward propagation
            h = sigmoid(b_i_h + w_i_h @ inp)
            o = sigmoid(b_h_o + w_h_o @ h)
            correct += int(np.argmax(o) == np.argmax(label))
            # Backpropagation
            delta_o = o - label
            w_h_o += -learn_rate * delta_o @ h.T
            b_h_o += -learn_rate * delta_o
            delta_h = w_h_o.T @ delta_o * (h * (1 - h))
            w_i_h += -learn_rate * delta_h @ inp.T
            b_i_h += -learn_rate * delta_h
        accuracy = round(correct / m * 100, 2)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            log(f"  epoch {epoch}: akurasi {accuracy}%")
    log(f"Training selesai. Akurasi akhir: {accuracy}%.")
    return (w_i_h, b_i_h, w_h_o, b_h_o)


def ann_predict(weights, vector01):
    """Forward propagation satu sampel (vektor 0-1). Mengembalikan output ANN."""
    w_i_h, b_i_h, w_h_o, b_h_o = weights
    inp = vector01.reshape(-1, 1)
    h = sigmoid(b_i_h + w_i_h @ inp)
    return sigmoid(b_h_o + w_h_o @ h)
