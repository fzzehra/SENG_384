import cv2
import numpy as np


def apply_moustache(image, landmarks, intensity=0.7, hair_len=12, color=(15, 15, 15)):
    print(">>> YENI MOUSTACHE KODU CALISTI")

    if landmarks is None or len(landmarks) < 292:
        return image

    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)

    coords = []
    for lm in landmarks:
        x, y = lm[0], lm[1]
        if x <= 1.0 and y <= 1.0:
            coords.append((int(x * w), int(y * h)))
        else:
            coords.append((int(x), int(y)))

    def pt(i):
        return np.array(coords[i], dtype=np.int32)

    left_mouth = pt(61)
    right_mouth = pt(291)
    upper_lip = pt(13)
    nose_bottom = pt(2)

    mouth_width = int(np.linalg.norm(right_mouth - left_mouth))
    cx = int((left_mouth[0] + right_mouth[0]) / 2)
    cy = int((nose_bottom[1] + upper_lip[1]) / 2) + 3

    wing_offset = int(mouth_width * 0.20)
    wing_w = max(20, int(mouth_width * 0.22))
    wing_h = max(6, int(hair_len * 0.35))

    left_center = (cx - wing_offset, cy)
    right_center = (cx + wing_offset, cy)

    # Sol parça
    cv2.ellipse(mask, left_center, (wing_w, wing_h), -8, 200, 360, 255, -1)

    # Sağ parça
    cv2.ellipse(mask, right_center, (wing_w, wing_h), 8, 180, 340, 255, -1)

    # Ortayı kapat ama çok kalın yapma
    bridge_w = max(8, int(mouth_width * 0.05))
    bridge_h = max(3, int(wing_h * 0.55))
    cv2.ellipse(mask, (cx, cy), (bridge_w, bridge_h), 0, 0, 360, 255, -1)

    # Yukarı taşmayı kes
    upper_cut = max(0, nose_bottom[1] + 2)
    mask[:upper_cut, :] = 0

    # Aşağı taşmayı kes
    lower_cut = min(h, upper_lip[1] + wing_h + 2)
    mask[lower_cut:, :] = 0

    # Boşlukları kapat
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Çok hafif yay
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    # Yumuşat
    mask = cv2.GaussianBlur(mask, (17, 9), 0)

    intensity = float(max(0.0, min(1.0, intensity)))
    alpha = (mask.astype(np.float32) / 255.0) * (0.35 + 0.45 * intensity)

    b, g, r = color
    for c, val in enumerate((b, g, r)):
        output[:, :, c] = output[:, :, c] * (1.0 - alpha) + val * alpha

    return np.clip(output, 0, 255).astype(np.uint8)