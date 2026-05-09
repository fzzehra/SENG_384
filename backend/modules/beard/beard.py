import cv2
import numpy as np
import random
import math


def apply_beard_effect(image, landmarks, intensity=0.7, hair_len=8, color=(10, 10, 10)):
    if landmarks is None:
        return image

    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)

    # Landmark'ları piksel koordinatına çevir
    coords = []

    for lm in landmarks:
        lx, ly = lm[0], lm[1]

        # MediaPipe normalize değer veriyorsa
        if lx <= 1.0 and ly <= 1.0:
            coords.append((int(lx * w), int(ly * h)))
        else:
            coords.append((int(lx), int(ly)))

    def get_pts(idxs):
        return np.array([coords[i] for i in idxs], np.int32)

    # 1. BÖLGELER

    CHIN_ZONE = [
        172, 136, 150, 149, 176, 148, 152,
        377, 400, 378, 379, 365, 397
    ]

    CHEEK_L = [234, 93, 132, 58, 172]
    CHEEK_R = [454, 323, 361, 288, 397]

    MUSTACHE_ZONE = [
        61, 185, 40, 39, 37, 0,
        267, 269, 270, 409, 291,
        375, 321, 405, 314, 17,
        84, 181, 91, 146
    ]

    # Dudak koruma
    LIPS = [
        61, 185, 40, 39, 37, 0,
        267, 269, 270, 409, 291,
        375, 321, 405, 314, 17,
        84, 181, 91, 146
    ]

    # 2. MASKELEME

    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask, [get_pts(CHIN_ZONE)], 255)
    cv2.fillPoly(mask, [get_pts(CHEEK_L)], 255)
    cv2.fillPoly(mask, [get_pts(CHEEK_R)], 255)
    cv2.fillPoly(mask, [get_pts(MUSTACHE_ZONE)], 255)

    # Dudakları çıkar
    lip_m = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(lip_m, [get_pts(LIPS)], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lip_m = cv2.dilate(lip_m, kernel, iterations=2)

    mask = cv2.subtract(mask, lip_m)

    # Blur
    mask_f = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0

    # 3. KIL ÇİZİMİ

    hair_layer = np.zeros_like(image, dtype=np.float32)

    ys, xs = np.where(mask > 25)

    density = 0.45 + intensity * 0.55

    for i in range(len(xs)):

        if random.random() > density:
            continue

        x, y = xs[i], ys[i]

        local_strength = mask_f[y, x]

        if random.random() > local_strength:
            continue

        # Doğal açı
        angle = math.radians(random.uniform(82, 98))

        # Doğal uzunluk
        length = (
            hair_len *
            random.uniform(0.45, 1.15) *
            local_strength
        )

        x2 = int(x + length * math.cos(angle))
        y2 = int(y + length * math.sin(angle))

        # Renk varyasyonu
        v = random.randint(-18, 12)

        c = (
            int(np.clip(color[0] + v, 0, 255)),
            int(np.clip(color[1] + v, 0, 255)),
            int(np.clip(color[2] + v, 0, 255))
        )

        thickness = 1

        cv2.line(
            hair_layer,
            (x, y),
            (x2, y2),
            c,
            thickness,
            cv2.LINE_AA
        )

    # Katmanı birleştir
    output = cv2.addWeighted(output, 1.0, hair_layer, intensity, 0)

    return np.clip(output, 0, 255).astype(np.uint8)