import cv2
import numpy as np


UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]


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


def apply_moustache(image, landmarks, intensity=0.8, color_hex="#241815", **kwargs):
    out = image.copy()
    h, w = out.shape[:2]

    intensity = float(np.clip(intensity, 0.0, 1.0))
    base_bgr = np.array(_hex_to_bgr(color_hex), dtype=np.float32)

    lip_left = _pt(landmarks, 61, w, h)
    lip_right = _pt(landmarks, 291, w, h)
    nose_left = _pt(landmarks, 98, w, h)
    nose_mid = _pt(landmarks, 2, w, h)
    nose_right = _pt(landmarks, 327, w, h)

    mouth_width = float(np.linalg.norm(lip_right - lip_left))
    if mouth_width < 8:
        return out

    top_lift = max(2, int(mouth_width * 0.05))
    fill_down = max(2, int(mouth_width * 0.04))

    upper_lip_pts = np.array([_pt(landmarks, i, w, h) for i in UPPER_LIP_OUTER], dtype=np.int32)

    top_pts = np.array([
        lip_left + np.array([0, -top_lift]),
        nose_left,
        nose_mid + np.array([0, -max(1, top_lift // 3)]),
        nose_right,
        lip_right + np.array([0, -top_lift]),
    ], dtype=np.int32)

    bottom_pts = upper_lip_pts[::-1].copy()
    bottom_pts[:, 1] += fill_down

    poly = np.vstack([top_pts, bottom_pts])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    # Dudağın üstünü daha iyi doldursun
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    k = max(5, (int(mouth_width) // 10) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    alpha = (mask.astype(np.float32) / 255.0) * (0.50 + 0.28 * intensity)

    out_f = out.astype(np.float32)
    for c in range(3):
        out_f[:, :, c] = out_f[:, :, c] * (1.0 - alpha) + base_bgr[c] * alpha

    # Kıl dokusu
    ys, xs = np.where(mask > 25)
    if len(xs) > 0:
        rng = np.random.default_rng(42)
        count = min(2200, max(900, len(xs) // 3))
        pick = rng.choice(len(xs), size=count, replace=(len(xs) < count))

        cx = float((lip_left[0] + lip_right[0]) * 0.5)
        min_len = max(3, int(mouth_width * 0.025))
        max_len = max(6, int(mouth_width * (0.06 + 0.04 * intensity)))

        hair_layer = out_f.copy()

        for idx in pick:
            x = int(xs[idx])
            y = int(ys[idx])

            side = -1 if x < cx else 1
            dist = abs(x - cx) / max(1.0, mouth_width * 0.5)

            angle = np.deg2rad(rng.normal(side * 16, 10))
            ln = int(rng.integers(min_len, max_len + 1))

            x2 = int(np.clip(x + np.cos(angle) * ln, 0, w - 1))
            y2 = int(np.clip(y + np.sin(angle) * ln * 0.45 + 1, 0, h - 1))

            col = (
                int(rng.integers(12, 28)),
                int(rng.integers(14, 30)),
                int(rng.integers(16, 34)),
            )
            cv2.line(hair_layer, (x, y), (x2, y2), col, 1, cv2.LINE_AA)

        hair_alpha = (mask.astype(np.float32) / 255.0) * (0.22 + 0.16 * intensity)
        for c in range(3):
            out_f[:, :, c] = out_f[:, :, c] * (1.0 - hair_alpha) + hair_layer[:, :, c] * hair_alpha

    return np.clip(out_f, 0, 255).astype(np.uint8)