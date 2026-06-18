import cv2
import numpy as np


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    color_hex = (color_hex or "#241815").strip().lstrip("#")
    if len(color_hex) != 6:
        color_hex = "241815"
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return (b, g, r)


def _pt(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    p = landmarks[idx]

    if isinstance(p, dict):
        x, y = p["x"], p["y"]
    elif hasattr(p, "x") and hasattr(p, "y"):
        x, y = p.x, p.y
    else:
        x, y = p[0], p[1]

    if x <= 1.5 and y <= 1.5:
        x, y = x * w, y * h

    return np.array([x, y], dtype=np.float32)


def apply_beard_effect(image, landmarks, intensity=0.8, color_hex="#241815", **kwargs):
    out = image.copy()
    h, w = out.shape[:2]

    intensity = float(np.clip(intensity, 0.0, 1.0))
    base_bgr = np.array(_hex_to_bgr(color_hex), dtype=np.float32)

    left_cheek = _pt(landmarks, 58, w, h)
    right_cheek = _pt(landmarks, 288, w, h)
    left_jaw = _pt(landmarks, 172, w, h)
    left_low = _pt(landmarks, 150, w, h)
    chin = _pt(landmarks, 152, w, h)
    right_low = _pt(landmarks, 379, w, h)
    right_jaw = _pt(landmarks, 397, w, h)

    mouth_left = _pt(landmarks, 61, w, h)
    mouth_right = _pt(landmarks, 291, w, h)
    upper_lip = _pt(landmarks, 13, w, h)
    lower_lip = _pt(landmarks, 17, w, h)

    face_width = float(np.linalg.norm(right_cheek - left_cheek))
    beard_height = max(20.0, chin[1] - lower_lip[1])

    if face_width < 20 or beard_height < 6:
        return out

    beard_poly = np.array([
        [left_cheek[0],  mouth_left[1] + 6],
        [left_jaw[0],    mouth_left[1] + 10],
        [left_low[0],    chin[1] - 10],
        [chin[0],        chin[1] + 8],
        [right_low[0],   chin[1] - 10],
        [right_jaw[0],   mouth_right[1] + 10],
        [right_cheek[0], mouth_right[1] + 6],
        [right_cheek[0], (mouth_right[1] + chin[1]) / 2],
        [left_cheek[0],  (mouth_left[1] + chin[1]) / 2],
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [beard_poly], 255)

    # Çene altını biraz daha doldur
    cx = int(chin[0])
    cy = int((lower_lip[1] + chin[1]) / 2 + 8)
    beard_w = int(face_width * 0.40)
    beard_h = int(max(18, beard_height * 0.90))

    cv2.ellipse(
        mask,
        (cx, cy),
        (beard_w, beard_h),
        0, 0, 180, 255, -1
    )

    # Yanları biraz güçlendir
    cv2.ellipse(
        mask,
        (int(cx - face_width * 0.23), int(cy - beard_h * 0.10)),
        (int(face_width * 0.14), int(beard_h * 0.55)),
        -12, 250, 90, 220, -1
    )
    cv2.ellipse(
        mask,
        (int(cx + face_width * 0.23), int(cy - beard_h * 0.10)),
        (int(face_width * 0.14), int(beard_h * 0.55)),
        12, 90, 290, 220, -1
    )

    # Ağız kısmını boş bırak
    mouth_cx = int((mouth_left[0] + mouth_right[0]) / 2)
    mouth_cy = int((upper_lip[1] + lower_lip[1]) / 2)
    mouth_w = int(np.linalg.norm(mouth_right - mouth_left) * 0.45)
    mouth_h = int(max(8, abs(lower_lip[1] - upper_lip[1]) * 1.5))

    cv2.ellipse(
        mask,
        (mouth_cx, mouth_cy),
        (mouth_w, mouth_h),
        0, 0, 360, 0, -1
    )

    # Soul patch
    cv2.ellipse(
        mask,
        (int(chin[0]), int(lower_lip[1] + beard_height * 0.20)),
        (max(5, int(face_width * 0.06)), max(6, int(beard_height * 0.16))),
        0, 0, 360, 180, -1
    )

    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    alpha = (mask.astype(np.float32) / 255.0) * (0.48 + 0.30 * intensity)

    out_f = out.astype(np.float32)
    for c in range(3):
        out_f[:, :, c] = out_f[:, :, c] * (1.0 - alpha) + base_bgr[c] * alpha

    # Kıl dokusu
    ys, xs = np.where(mask > 20)
    if len(xs) > 0:
        rng = np.random.default_rng(123)
        count = min(3200, max(1400, len(xs) // 2))
        pick = rng.choice(len(xs), size=count, replace=(len(xs) < count))

        hair_layer = out_f.copy()
        center_x = float(chin[0])
        min_len = max(4, int(beard_height * 0.10))
        max_len = max(8, int(beard_height * (0.22 + 0.12 * intensity)))

        for idx in pick:
            x = int(xs[idx])
            y = int(ys[idx])

            dist = abs(x - center_x) / max(1.0, face_width * 0.5)

            if x < center_x - face_width * 0.08:
                angle = np.deg2rad(rng.normal(-10, 8))
            elif x > center_x + face_width * 0.08:
                angle = np.deg2rad(rng.normal(10, 8))
            else:
                angle = np.deg2rad(rng.normal(0, 6))

            ln = int(rng.integers(min_len, max_len + 1))

            x2 = int(np.clip(x + np.sin(angle) * ln * 0.45, 0, w - 1))
            y2 = int(np.clip(y + np.cos(angle) * ln, 0, h - 1))

            col = (
                int(rng.integers(10, 24)),
                int(rng.integers(12, 26)),
                int(rng.integers(14, 30)),
            )
            cv2.line(hair_layer, (x, y), (x2, y2), col, 1, cv2.LINE_AA)

        hair_alpha = (mask.astype(np.float32) / 255.0) * (0.20 + 0.18 * intensity)
        for c in range(3):
            out_f[:, :, c] = out_f[:, :, c] * (1.0 - hair_alpha) + hair_layer[:, :, c] * hair_alpha

    return np.clip(out_f, 0, 255).astype(np.uint8)