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


def apply_moustache(image, landmarks, intensity=0.8, color_hex="#241815", **kwargs):
    out = image.copy()
    h, w = out.shape[:2]

    intensity = float(np.clip(intensity, 0.0, 1.0))
    base_bgr = np.array(_hex_to_bgr(color_hex), dtype=np.float32)

    lip_left   = _pt(landmarks, 61,  w, h)
    lip_right  = _pt(landmarks, 291, w, h)
    upper_lip  = _pt(landmarks, 13,  w, h)   # üst dudak Cupid's bow ortası
    nose_left  = _pt(landmarks, 98,  w, h)   # burun sol kanat tabanı
    nose_right = _pt(landmarks, 327, w, h)   # burun sağ kanat tabanı

    mouth_width = float(np.linalg.norm(lip_right - lip_left))
    if mouth_width < 8:
        return out

    # ── Boyutlar ──────────────────────────────────────────────────────────
    mou_h    = int(mouth_width * 0.11)
    side_ext = int(mouth_width * 0.16)

    left_x = int(lip_left[0])  - side_ext
    right_x = int(lip_right[0]) + side_ext
    cx_m   = (left_x + right_x) // 2

    # Bıyık burun tabanı ile üst dudak arasında (philtrum ortası)
    nose_base_y   = int((nose_left[1] + nose_right[1]) / 2)
    philtrum_mid  = int((nose_base_y + upper_lip[1]) / 2)

    top_y  = philtrum_mid - mou_h
    bot_y  = min(philtrum_mid + mou_h // 2, int(upper_lip[1]) - 3)  # dudağa girmez

    # Dış uçlar (kenarbastıkça aşağı kıvrılır)
    tip_y = min(philtrum_mid + int(mou_h * 0.55), int(upper_lip[1]) - 3)

    # ── Polygon: ortada yukarı kavisli, uçlarda aşağı kıvrık ─────────────
    poly = np.array([
        [left_x,            tip_y],                    # sol dış uç (aşağı kıvrık)
        [int(lip_left[0]),  top_y + mou_h // 2],       # sol iç üst
        [cx_m,              top_y],                    # üst orta (en yüksek)
        [int(lip_right[0]), top_y + mou_h // 2],       # sağ iç üst
        [right_x,           tip_y],                    # sağ dış uç (aşağı kıvrık)
        [right_x,           bot_y],                    # sağ alt
        [cx_m,              bot_y + mou_h // 4],       # alt orta
        [left_x,            bot_y],                    # sol alt
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    cv2.dilate(mask, np.ones((3, 3), np.uint8), dst=mask, iterations=1)
    mask = cv2.GaussianBlur(mask, (23, 23), 0)

    out_f = out.astype(np.float32)
    mf    = mask.astype(np.float32) / 255.0

    # ── Sakalla aynı: medium base tonu ───────────────────────────────────
    alpha = mf * (0.36 + 0.24 * intensity)
    for c in range(3):
        out_f[:, :, c] = out_f[:, :, c] * (1.0 - alpha) + base_bgr[c] * alpha

    ys, xs = np.where(mask > 15)
    if len(xs) > 0:
        rng  = np.random.default_rng(42)
        cx_f = float(cx_m)

        # ── Stipple (sakal dibi noktaları) ───────────────────────────────
        sn     = min(5000, max(1500, int(len(xs) * 0.80)))
        sp     = rng.choice(len(xs), size=sn, replace=(len(xs) < sn))
        slayer = out_f.copy()
        for idx in sp:
            x = int(xs[idx]); y = int(ys[idx])
            d = int(rng.integers(5, 22))
            cv2.circle(slayer, (x, y), 1, (d, d + 2, d + 8), -1, cv2.LINE_AA)
        sa = mf * (0.20 + 0.13 * intensity)
        for c in range(3):
            out_f[:, :, c] = out_f[:, :, c] * (1.0 - sa) + slayer[:, :, c] * sa

        # ── Kıl çizgileri — yoğunluk slider'ına göre ─────────────────────
        hn = min(5000, max(800, int(len(xs) * (0.15 + 0.65 * intensity))))
        hp = rng.choice(len(xs), size=hn, replace=(len(xs) < hn))
        hlayer  = out_f.copy()
        min_len = max(4,  int(mouth_width * 0.032))
        max_len = max(10, int(mouth_width * (0.085 + 0.045 * intensity)))

        for idx in hp:
            x = int(xs[idx]); y = int(ys[idx])
            side  = -1 if x < cx_f else 1
            dist_ratio = abs(x - cx_f) / max(1.0, (right_x - left_x) * 0.5)
            angle = np.deg2rad(rng.normal(side * (8 + 14 * dist_ratio), 7))
            ln    = int(rng.integers(min_len, max_len + 1))
            x2 = int(np.clip(x + np.cos(angle) * ln,        0, w - 1))
            y2 = int(np.clip(y + np.sin(angle) * ln * 0.22, 0, h - 1))
            d  = int(rng.integers(6, 26))
            thick = 1 if rng.random() > 0.18 else 2
            cv2.line(hlayer, (x, y), (x2, y2), (d, d + 3, d + 9), thick, cv2.LINE_AA)

        ha = mf * (0.26 + 0.18 * intensity)
        for c in range(3):
            out_f[:, :, c] = out_f[:, :, c] * (1.0 - ha) + hlayer[:, :, c] * ha

    return np.clip(out_f, 0, 255).astype(np.uint8)
