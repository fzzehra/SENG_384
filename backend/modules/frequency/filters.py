import cv2
import numpy as np


def _frequency_filter_channel(channel, radius_ratio, mode):
    f_shift = np.fft.fftshift(np.fft.fft2(channel.astype(np.float32)))

    rows, cols = channel.shape
    center_row, center_col = rows // 2, cols // 2
    radius = max(1, int(min(rows, cols) * radius_ratio))

    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)

    if mode == "low":
        mask = distance <= radius
    else:
        mask = distance > radius

    filtered = f_shift * mask
    result = np.fft.ifft2(np.fft.ifftshift(filtered))

    return np.real(result)


def apply_frequency_filter(image, mode="low", intensity=0.5):
    intensity = float(np.clip(intensity, 0.0, 1.0))

    if mode == "low":
        radius_ratio = 0.30 - intensity * 0.27
    else:
        radius_ratio = 0.03 + intensity * 0.27

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    filtered_y = _frequency_filter_channel(y, radius_ratio, mode)
    filtered_y = cv2.normalize(filtered_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    result = cv2.cvtColor(cv2.merge([filtered_y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return result
