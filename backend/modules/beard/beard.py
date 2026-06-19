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

    left_cheek  = _pt(landmarks, 58,  w, h)
    right_cheek = _pt(landmarks, 288, w, h)
    left_jaw    = _pt(landmarks, 172, w, h)
    left_low    = _pt(landmarks, 150, w, h)
    chin        = _pt(landmarks, 152, w, h)
    right_low   = _pt(landmarks, 379, w, h)
    right_jaw   = _pt(landmarks, 397, w, h)
    mouth_left  = _pt(landmarks, 61,  w, h)
    mouth_right = _pt(landmarks, 291, w, h)
    upper_lip   = _pt(landmarks, 13,  w, h)
    lower_lip   = _pt(landmarks, 17,  w, h)

    face_width   = float(np.linalg.norm(right_cheek - left_cheek))
    beard_height = max(20.0, chin[1] - lower_lip[1])

    if face_width < 20 or beard_height < 6:
        return out

    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(chin[0])

    cheek_top_y = int(mouth_left[1] - beard_height * 0.55)

    # ── Sol yanak: tepeden çene seviyesine kadar uzun elips ───────────────
    lc_cx = int(left_cheek[0])
    lc_cy = int((cheek_top_y + lower_lip[1]) / 2)
    lc_h  = int((lower_lip[1] - cheek_top_y) / 2 + beard_height * 0.55)
    lc_w  = int(face_width * 0.20)
    cv2.ellipse(mask, (lc_cx, lc_cy), (lc_w, lc_h), 0, 0, 360, 225, -1)

    # ── Sağ yanak: simetrik ───────────────────────────────────────────────
    rc_cx = int(right_cheek[0])
    cv2.ellipse(mask, (rc_cx, lc_cy), (lc_w, lc_h), 0, 0, 360, 225, -1)

    # ── Çene bandı: alt dudağa yakın, çene ucuna kadar ──────────────────
    cy      = int(lower_lip[1] + beard_height * 0.08)   # alt dudağa çok yakın
    beard_w = int(face_width * 0.56)
    beard_h_down = int(max(10, chin[1] - cy + 3))
    cv2.ellipse(mask, (cx, cy), (beard_w, beard_h_down), 0, 0, 180, 255, -1)
    # Üst yarı: yanakları bağlar
    cv2.ellipse(mask, (cx, cy),
                (int(face_width * 0.52), int(beard_height * 0.90)),
                0, 180, 360, 245, -1)

    # ── Çene ucu dolgusu (daha büyük) ────────────────────────────────────
    cv2.ellipse(mask,
                (cx, int(chin[1] - beard_height * 0.30)),
                (int(face_width * 0.38), int(beard_height * 0.58)),
                0, 0, 360, 255, -1)

    # ── Soul patch (alt dudak altı) ───────────────────────────────────────
    cv2.ellipse(mask,
                (cx, int(lower_lip[1] + beard_height * 0.18)),
                (max(5, int(face_width * 0.06)), max(6, int(beard_height * 0.18))),
                0, 0, 360, 210, -1)

    # ── Ağız ve üst dudak temizliği ───────────────────────────────────────
    mx      = int((mouth_left[0] + mouth_right[0]) / 2)
    mw      = int(np.linalg.norm(mouth_right - mouth_left) * 0.58)
    lip_mid = int((upper_lip[1] + lower_lip[1]) / 2)
    lip_h   = int(max(12, abs(lower_lip[1] - upper_lip[1]) * 1.6 + 6))
    cv2.ellipse(mask, (mx, lip_mid), (mw, lip_h), 0, 0, 360, 0, -1)
    cv2.ellipse(mask, (mx, int(upper_lip[1])),
                (mw, int(beard_height * 0.35)), 0, 180, 360, 0, -1)

    # ── Çene altı sert kesim (yüz sınırı) ────────────────────────────────
    chin_cut = int(chin[1]) + 4
    if chin_cut < h:
        mask[chin_cut:, :] = 0

    mask = cv2.GaussianBlur(mask, (39, 39), 0)

    # ── Temel renk karışımı ───────────────────────────────────────────────
    alpha = (mask.astype(np.float32) / 255.0) * (0.38 + 0.26 * intensity)
    out_f = out.astype(np.float32)
    for c in range(3):
        out_f[:, :, c] = out_f[:, :, c] * (1.0 - alpha) + base_bgr[c] * alpha

    # ── Kıl dokusu ───────────────────────────────────────────────────────
    ys, xs = np.where(mask > 15)
    if len(xs) > 0:
        rng = np.random.default_rng(42)

        # Nokta bazlı stubble (sakal dibi)
        stipple_count = min(8000, max(3000, int(len(xs) * 0.9)))
        stipple_pick  = rng.choice(len(xs), size=stipple_count, replace=(len(xs) < stipple_count))
        stipple_layer = out_f.copy()
        for idx in stipple_pick:
            x = int(xs[idx])
            y = int(ys[idx])
            d = int(rng.integers(5, 22))
            col = (d + 1, d + 3, d + 9)
            cv2.circle(stipple_layer, (x, y), 1, col, -1, cv2.LINE_AA)

        stipple_alpha = (mask.astype(np.float32) / 255.0) * (0.18 + 0.14 * intensity)
        for c in range(3):
            out_f[:, :, c] = out_f[:, :, c] * (1.0 - stipple_alpha) + stipple_layer[:, :, c] * stipple_alpha

        # Kıl çizgileri
        hair_count = min(6000, max(2500, int(len(xs) * 0.65)))
        hair_pick  = rng.choice(len(xs), size=hair_count, replace=(len(xs) < hair_count))
        hair_layer = out_f.copy()
        cx_f    = float(chin[0])
        min_len = max(3, int(beard_height * 0.06))
        max_len = max(7, int(beard_height * (0.13 + 0.09 * intensity)))

        for idx in hair_pick:
            x = int(xs[idx])
            y = int(ys[idx])

            # Yön: kenarda dışa, ortada aşağı
            if x < cx_f - face_width * 0.12:
                angle = np.deg2rad(rng.normal(-22, 11))
            elif x > cx_f + face_width * 0.12:
                angle = np.deg2rad(rng.normal(22, 11))
            else:
                angle = np.deg2rad(rng.normal(0, 8))

            ln = int(rng.integers(min_len, max_len + 1))
            x2 = int(np.clip(x + np.sin(angle) * ln * 0.35, 0, w - 1))
            y2 = int(np.clip(y + np.cos(angle) * ln, 0, h - 1))

            d = int(rng.integers(5, 24))
            col = (d + 2, d + 4, d + 10)
            thick = 1 if rng.random() > 0.22 else 2
            cv2.line(hair_layer, (x, y), (x2, y2), col, thick, cv2.LINE_AA)

        hair_alpha = (mask.astype(np.float32) / 255.0) * (0.28 + 0.16 * intensity)
        for c in range(3):
            out_f[:, :, c] = out_f[:, :, c] * (1.0 - hair_alpha) + hair_layer[:, :, c] * hair_alpha

    return np.clip(out_f, 0, 255).astype(np.uint8)
