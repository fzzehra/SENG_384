from typing import List, Tuple
import cv2
import numpy as np

LandmarkList = List[Tuple[int, int]]

def apply_aging_effect(image, intensity=0.5, landmarks=None):
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]
    output = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # Saç rengi adayları: koyu saç + kahverengi/sarımsı saç
    dark_hair = cv2.inRange(val, 0, 130)
    dark_sat = cv2.inRange(sat, 25, 255)
    dark_hair = cv2.bitwise_and(dark_hair, dark_sat)

    brown_blonde_hair = cv2.inRange(hue, 5, 35)
    brown_sat = cv2.inRange(sat, 45, 255)
    brown_val = cv2.inRange(val, 40, 230)
    brown_hair = cv2.bitwise_and(brown_blonde_hair, brown_sat)
    brown_hair = cv2.bitwise_and(brown_hair, brown_val)

    hair_color = cv2.bitwise_or(dark_hair, brown_hair)

    # Mavi arka plan gibi alanları ele
    blue_bg = cv2.inRange(hue, 85, 140)
    hair_color[blue_bg > 0] = 0

    hair_region = np.zeros((h, w), dtype=np.uint8)
    face_mask = np.zeros((h, w), dtype=np.uint8)

    if landmarks is not None and len(landmarks) >= 468:
        top = landmarks[10]
        bottom = landmarks[152]
        left = landmarks[234]
        right = landmarks[454]

        # Yüz maskesi
        face_center = ((left[0] + right[0]) // 2, (top[1] + bottom[1]) // 2)
        face_axes = (
            int(abs(right[0] - left[0]) * 0.58),
            int(abs(bottom[1] - top[1]) * 0.62)
        )
        cv2.ellipse(face_mask, face_center, face_axes, 0, 0, 360, 255, -1)

        # Saç bölgesi: yüzün üstü + yanları + aşağı inen uzun saç alanı
        x1 = max(0, int(left[0] - w * 0.35))
        x2 = min(w, int(right[0] + w * 0.35))
        y1 = max(0, int(top[1] - h * 0.25))
        y2 = min(h, int(bottom[1] + h * 0.55))

        cv2.ellipse(
            hair_region,
            ((x1 + x2) // 2, (y1 + y2) // 2),
            ((x2 - x1) // 2, (y2 - y1) // 2),
            0,
            0,
            360,
            255,
            -1
        )
    else:
        cv2.rectangle(
            hair_region,
            (int(w * 0.05), 0),
            (int(w * 0.95), int(h * 0.85)),
            255,
            -1
        )

    # Saç bölgesinden yüzü çıkar
    face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
    hair_region[face_mask > 80] = 0

    hair_mask = cv2.bitwise_and(hair_color, hair_region)

    # Maskeyi temizle
    kernel = np.ones((5, 5), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.GaussianBlur(hair_mask, (9, 9), 0)

    mask = hair_mask.astype(np.float32) / 255.0
    mask = np.clip(mask, 0.0, 0.65)
    mask = np.repeat(mask[:, :, None], 3, axis=2)

    # Saçı gri/beyaz tona yaklaştır
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    white_hair = cv2.addWeighted(
        gray_bgr,
        0.55,
        np.full_like(image, 220),
        0.45,
        0
    )

    strength = 0.85 * intensity
    result = output * (1 - mask * strength) + white_hair * (mask * strength)

    return result.astype(np.uint8)