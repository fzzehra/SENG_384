import csv
import os
import cv2
import numpy as np
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
    """
    Resize image2 to match image1's width and height.
    """
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
    """
    Total spectral energy = sum of squared magnitudes.
    """
    energy = np.sum(np.abs(fft_shifted) ** 2)
    return float(energy)

def compute_frequency_bands(fft_shifted, low_radius_ratio=0.15):
    """
    Split FFT energy into low-frequency and high-frequency regions.

    Parameters:
        fft_shifted: centered FFT result
        low_radius_ratio: ratio of image size used to define low-frequency region

    Returns:
        low_energy, high_energy, energy_ratio
    """
    rows, cols = fft_shifted.shape
    center_row, center_col = rows // 2, cols // 2

    # Low-frequency region radius based on smaller image dimension
    radius = int(min(rows, cols) * low_radius_ratio)

    # Create mask for low-frequency region
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)

    low_mask = distance_from_center <= radius
    high_mask = ~low_mask

    magnitude_squared = np.abs(fft_shifted) ** 2

    low_energy = np.sum(magnitude_squared[low_mask])
    high_energy = np.sum(magnitude_squared[high_mask])

    if low_energy == 0:
        energy_ratio = float("inf")
    else:
        energy_ratio = float(high_energy / low_energy)

    return float(low_energy), float(high_energy), energy_ratio

def compute_mse(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    mse_value = np.mean((image1 - image2) ** 2)
    return float(mse_value)


def compute_psnr(image1, image2):
    mse_value = compute_mse(image1, image2)

    if mse_value == 0:
        return float("inf")

    max_pixel = 255.0
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse_value)
    return float(psnr_value)


def compute_ssim(image1, image2):
    ssim_value = ssim(image1, image2)
    return float(ssim_value)


def save_spectrum_image(spectrum, output_path, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(spectrum, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


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
   
    if original_path == transformed_path:
        print("Warning: comparing same images")
    
    original = load_image(original_path)
    transformed = load_image(transformed_path)

    original, transformed = resize_to_match(original, transformed)

    original_gray = to_grayscale(original)
    transformed_gray = to_grayscale(transformed)

    # FFT
    original_fft = compute_fft(original_gray)
    transformed_fft = compute_fft(transformed_gray)

    # Spectrum
    original_spectrum = magnitude_spectrum(original_fft)
    transformed_spectrum = magnitude_spectrum(transformed_fft)

    # Energy
    original_energy = compute_energy(original_fft)
    transformed_energy = compute_energy(transformed_fft)

    # Frequency bands
    original_low, original_high, original_ratio = compute_frequency_bands(original_fft)
    transformed_low, transformed_high, transformed_ratio = compute_frequency_bands(transformed_fft)

    # Metrics
    mse_value = compute_mse(original_gray, transformed_gray)
    psnr_value = compute_psnr(original_gray, transformed_gray)
    ssim_value = compute_ssim(original_gray, transformed_gray)

    # Save spectrum images
    os.makedirs("results", exist_ok=True)

    save_spectrum_image(original_spectrum, "results/original_spectrum.png", "Original Spectrum")
    save_spectrum_image(transformed_spectrum, "results/transformed_spectrum.png", "Transformed Spectrum")

    # Result dictionary
    results = {
        "metrics": {
            "mse": mse_value,
            "psnr": psnr_value,
            "ssim": ssim_value
        },
        "energy": {
            "original": original_energy,
            "transformed": transformed_energy,
            "original_low": original_low,
            "original_high": original_high,
            "transformed_low": transformed_low,
            "transformed_high": transformed_high,
            "original_ratio": original_ratio,
            "transformed_ratio": transformed_ratio,
            "ratio_difference": transformed_ratio - original_ratio
        },
        "outputs": {
            "original_spectrum": "results/original_spectrum.png",
            "transformed_spectrum": "results/transformed_spectrum.png"
        }
    }

    return results

def main():
    
    results = analyze_images("images/original.jpg", "images/transformed.jpg")

    export_results(results, "results/analysis_results.txt")
    export_results_csv(results, "results/analysis_results.csv")

    print("Analysis completed successfully.")

    for key, value in results.items():
        print(f"{key}: {value}")

    
if __name__ == "__main__":
    main()