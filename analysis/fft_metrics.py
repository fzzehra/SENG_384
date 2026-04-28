import csv
import os
import cv2
import numpy as np

# 🔥 FIX (EN ÖNEMLİ KISIM)
import matplotlib
matplotlib.use("Agg")  # GUI kapat

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image could not be loaded: {image_path}")

    return image


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_to_match(image1, image2):
    h, w = image1.shape[:2]
    resized_image2 = cv2.resize(image2, (w, h))
    return image1, resized_image2


def compute_fft(gray_image):
    fft_result = np.fft.fft2(gray_image)
    fft_shifted = np.fft.fftshift(fft_result)
    return fft_shifted


def magnitude_spectrum(fft_shifted):
    magnitude = np.abs(fft_shifted)
    spectrum = np.log1p(magnitude)
    return spectrum


def compute_energy(fft_shifted):
    energy = np.sum(np.abs(fft_shifted) ** 2)
    return float(energy)


def compute_frequency_bands(fft_shifted, low_radius_ratio=0.15):
    rows, cols = fft_shifted.shape
    center_row, center_col = rows // 2, cols // 2

    radius = int(min(rows, cols) * low_radius_ratio)

    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)

    low_mask = distance_from_center <= radius
    high_mask = ~low_mask

    magnitude_squared = np.abs(fft_shifted) ** 2

    low_energy = np.sum(magnitude_squared[low_mask])
    high_energy = np.sum(magnitude_squared[high_mask])

    energy_ratio = float(high_energy / low_energy) if low_energy != 0 else float("inf")

    return float(low_energy), float(high_energy), energy_ratio


def compute_mse(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    return float(np.mean((image1 - image2) ** 2))


def compute_psnr(image1, image2):
    mse_value = compute_mse(image1, image2)

    if mse_value == 0:
        return float("inf")

    return float(10 * np.log10((255.0 ** 2) / mse_value))


def compute_ssim(image1, image2):
    return float(ssim(image1, image2))


def save_spectrum_image(spectrum, output_path, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(spectrum, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # 🔥 sadece dosyaya kaydet
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()  # 🔥 önemli (memory leak + crash önler)


def export_results(results_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")


def export_results_csv(results_dict, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "metric", "value"])

        for category, subdict in results_dict.items():
            for key, value in subdict.items():
                writer.writerow([category, key, value])


def analyze_images(original_path, transformed_path):

    original = load_image(original_path)
    transformed = load_image(transformed_path)

    original, transformed = resize_to_match(original, transformed)

    original_gray = to_grayscale(original)
    transformed_gray = to_grayscale(transformed)

    original_fft = compute_fft(original_gray)
    transformed_fft = compute_fft(transformed_gray)

    original_spectrum = magnitude_spectrum(original_fft)
    transformed_spectrum = magnitude_spectrum(transformed_fft)

    original_energy = compute_energy(original_fft)
    transformed_energy = compute_energy(transformed_fft)

    original_low, original_high, original_ratio = compute_frequency_bands(original_fft)
    transformed_low, transformed_high, transformed_ratio = compute_frequency_bands(transformed_fft)

    mse_value = compute_mse(original_gray, transformed_gray)
    psnr_value = compute_psnr(original_gray, transformed_gray)
    ssim_value = compute_ssim(original_gray, transformed_gray)

    os.makedirs("static/results", exist_ok=True)

    save_spectrum_image(original_spectrum, "static/results/original_spectrum.png", "Original Spectrum")
    save_spectrum_image(transformed_spectrum, "static/results/transformed_spectrum.png", "Transformed Spectrum")

    return {
        "metrics": {
            "mse": mse_value,
            "psnr": psnr_value,
            "ssim": ssim_value
        },
        "energy": {
            "original": original_energy,
            "transformed": transformed_energy,
            "original_ratio": original_ratio,
            "transformed_ratio": transformed_ratio
        }
    }