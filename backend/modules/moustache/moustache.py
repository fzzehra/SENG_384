import cv2
import numpy as np
import random
import math

def apply_moustache(image, landmarks, intensity=0.7, hair_len=12, color=(8, 8, 8)):
    if landmarks is None or len(landmarks) < 468:
        return image

    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)

    coords = []
    for lm in landmarks:
        lx, ly = lm[0], lm[1]
        if lx <= 1.0 and ly <= 1.0:
            coords.append((int(lx * w), int(ly * h)))
        else:
            coords.append((int(lx), int(ly)))

    def pt(i):
        return coords[i]

    def get_pts(idxs):
        return np.array([coords[i] for i in idxs], np.int32)

    left_mouth = pt(61)
    right_mouth = pt(291)
    nose_base = pt(164)
    cupid = pt(0)

    mouth_w = abs(right_mouth[0] - left_mouth[0])

    y_top = int(nose_base[1] + h * 0.005)
    y_bottom = int(cupid[1] - h * 0.005)

    if y_bottom <= y_top:
        y_bottom = y_top + int(h * 0.035)

    x_left = int(left_mouth[0] - mouth_w * 0.08)
    x_right = int(right_mouth[0] + mouth_w * 0.08)
    center_x = int((left_mouth[0] + right_mouth[0]) / 2)

    mask = np.zeros((h, w), dtype=np.uint8)

    left_poly = np.array([
        [x_left, int(y_bottom)],
        [int(center_x - mouth_w * 0.04), int(y_bottom)],
        [int(center_x - mouth_w * 0.03), int(y_top)],
        [int(left_mouth[0] + mouth_w * 0.08), int(y_top)],
    ], dtype=np.int32)

    right_poly = np.array([
        [int(center_x + mouth_w * 0.04), int(y_bottom)],
        [x_right, int(y_bottom)],
        [int(right_mouth[0] - mouth_w * 0.08), int(y_top)],
        [int(center_x + mouth_w * 0.03), int(y_top)],
    ], dtype=np.int32)

    cv2.fillPoly(mask, [left_poly], 255)
    cv2.fillPoly(mask, [right_poly], 255)

    lips = [
        61, 185, 40, 39, 37, 0,
        267, 269, 270, 409, 291,
        375, 321, 405, 314, 17,
        84, 181, 91, 146
    ]

    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [get_pts(lips)], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lip_mask = cv2.dilate(lip_mask, kernel, iterations=1)
    mask = cv2.subtract(mask, lip_mask)

    # ✅ Noise maskı kaldırıldı — kılları siliyordu
    # Sadece Gaussian blur ile yumuşat
    mask_f = cv2.GaussianBlur(mask, (13, 13), 0).astype(np.float32) / 255.0

    hair_layer = np.zeros_like(image, dtype=np.float32)
    ys, xs = np.where(mask > 15)

    density = 0.85
    # ✅ Kıl uzunluğu kısaltıldı
    dynamic_len = hair_len * (0.8 + intensity * 0.6)

    for i in range(len(xs)):
        if random.random() > density:
            continue

        x, y = xs[i], ys[i]
        local_strength = mask_f[y, x]

        if random.random() > local_strength:
            continue

        if x < center_x:
            # ✅ Açı aralığı genişletildi → dağınık görünüm
            angle = math.radians(random.uniform(120, 200))
        else:
            angle = math.radians(random.uniform(-20, 60))

        # ✅ Uzunluk varyasyonu artırıldı
        length = dynamic_len * random.uniform(0.3, 1.0) * local_strength

        x2 = int(x + length * math.cos(angle))
        y2 = int(y + length * math.sin(angle))

        v = random.randint(-20, 15)
        c = (
            int(np.clip(color[0] + v, 0, 255)),
            int(np.clip(color[1] + v, 0, 255)),
            int(np.clip(color[2] + v, 0, 255))
        )

        # ✅ Bazı kıllar 2px kalınlık → derinlik hissi
        thickness = 1 if random.random() > 0.25 else 2
        cv2.line(hair_layer, (x, y), (x2, y2), c, thickness, cv2.LINE_AA)

    # Shadow biraz azaltıldı (daha hafif altyapı)
    shadow_alpha = (mask_f * intensity * 0.40)[:, :, None]
    output = output * (1.0 - shadow_alpha)

    final = output + (hair_layer * 4.0)

    return np.clip(final, 0, 255).astype(np.uint8)