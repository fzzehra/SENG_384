import cv2
import numpy as np


def apply_fourier_phase_effect(image, intensity=0.5):
    intensity = float(np.clip(intensity, 0.0, 1.0))

    if intensity == 0:
        return image

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y_float = y.astype(np.float32)

    f = np.fft.fft2(y_float)
    f_shift = np.fft.fftshift(f)

    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)

    h, w = phase.shape

    rng = np.random.default_rng()
    noise = rng.normal(0, 1, phase.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=3)

    # Daha görünür phase bozulması
    new_phase = phase + noise * (intensity * 8.0)

    modified = magnitude * np.exp(1j * new_phase)

    inverse_shift = np.fft.ifftshift(modified)
    reconstructed = np.fft.ifft2(inverse_shift)
    reconstructed = np.real(reconstructed)

    reconstructed = cv2.normalize(
        reconstructed,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Daha güçlü blend
    blend_amount = np.clip(0.35 + intensity * 0.65, 0, 1)

    blended_y = cv2.addWeighted(
        y,
        1.0 - blend_amount,
        reconstructed,
        blend_amount,
        0
    )

    result_ycrcb = cv2.merge([blended_y, cr, cb])
    result = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)

    return result