"""Uji pipeline lengkap tanpa GUI, memakai folder test_images."""

import os

from processing import grayscale, resizing, binarization, dataset, randomize, ann

FOLDER = os.path.join(os.path.dirname(__file__), "test_images")

print("=== 1. Grayscale ===")
grayscale.process_folder(FOLDER)
print("\n=== 2. Resizing (average pooling, 20x20) ===")
resizing.process_folder(FOLDER, 20, 20, resizing.AVERAGE)
print("\n=== 3. Binarization (threshold 100, inversi) ===")
binarization.process_folder(FOLDER, 100, invert=True)
print("\n=== 4. Create Dataset ===")
dataset.process_folder(FOLDER)
print("\n=== 5. Randomize ===")
randomize.process_folder(FOLDER)
print("\n=== 6. Train ANN ===")
ann.train(FOLDER, hidden_neurons=20, epochs=100)
print("\n=== 7. Test insample (sampel 0 dan 7) ===")
ann.predict(FOLDER, 0)
ann.predict(FOLDER, 7)
