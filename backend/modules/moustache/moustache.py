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
    m01 = mask.astype(np.float32) / 255.0

    # ── Sakalla aynı: çok hafif gölge/tint (ten dokusu korunur) ──────────
    tint_a = m01 * (0.06 + 0.16 * intensity)
    tint = base_bgr / 255.0
    for c in range(3):
        mult = 1.0 - tint_a * (1.0 - np.clip(tint[c] * 2.2 + 0.25, 0.0, 1.0))
        out_f[:, :, c] *= mult

    # ── Sakalla aynı: tek tek kıllar ──────────────────────────────────────
    ys, xs = np.where(mask > 20)
    if len(xs) > 0:
        rng = np.random.default_rng(42)

        # Yoğunluk: intensity arttıkça kıl sayısı artar
        area = float(len(xs))
        hair_per_px = 0.04 + 0.22 * intensity
        n_hair = int(min(45000, max(300, area * hair_per_px)))

        # Maske değerine göre ağırlıklı seçim: kenarlarda seyrek, merkezde sık
        weights = m01[ys, xs]
        weights = weights / weights.sum()
        pick = rng.choice(len(xs), size=n_hair, replace=True, p=weights)
        px = xs[pick].astype(np.float32) + rng.uniform(-0.5, 0.5, n_hair)
        py = ys[pick].astype(np.float32) + rng.uniform(-0.5, 0.5, n_hair)

        # Kıl boyu sakaldakiyle aynı ölçekte (sakal yüksekliği ≈ ağız genişliğinin yarısı)
        ref_h = max(20.0, mouth_width * 0.5)
        min_len = max(2.0, ref_h * 0.035)
        max_len = max(min_len + 1.0, ref_h * (0.07 + 0.07 * intensity))
        lengths = rng.uniform(min_len, max_len, n_hair)

        # Yön: aşağı doğru, kenarlarda hafif dışa
        half_w = max(1.0, (right_x - left_x) * 0.5)
        side = np.clip((px - cx_m) / half_w, -1.0, 1.0)
        angles = rng.normal(0.0, 0.28, n_hair) + side * 0.35

        hair_col = np.zeros((h, w, 3), dtype=np.uint8)
        hair_alpha = np.zeros((h, w), dtype=np.uint8)

        for i in range(n_hair):
            x0, y0 = float(px[i]), float(py[i])
            ln = float(lengths[i])
            a = float(angles[i])
            dx, dy = np.sin(a), np.cos(a)

            # hafif kavis
            bend = rng.normal(0, 0.12) * ln
            xm = x0 + dx * ln * 0.5 + bend
            ym = y0 + dy * ln * 0.5
            x1 = x0 + dx * ln + bend * 1.6
            y1 = y0 + dy * ln

            pts = np.array([[x0, y0], [xm, ym], [x1, y1]], dtype=np.float32)
            pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

            # renk: baz renk etrafında doğal varyasyon
            v = rng.uniform(0.55, 1.25)
            col = np.clip(base_bgr * v + rng.uniform(-4, 10), 0, 255)
            col = tuple(int(c) for c in col)

            op = int(rng.uniform(110, 230) * (0.5 + 0.5 * m01[min(h - 1, max(0, int(y0))), min(w - 1, max(0, int(x0)))]))
            cv2.polylines(hair_col, [pts], False, col, 1, cv2.LINE_AA)
            cv2.polylines(hair_alpha, [pts], False, op, 1, cv2.LINE_AA)

        hair_alpha_f = cv2.GaussianBlur(hair_alpha, (3, 3), 0).astype(np.float32) / 255.0
        hair_alpha_f *= m01  # bıyık bölgesi dışına çıkmasın
        hair_alpha_f *= (0.70 + 0.30 * intensity)

        col_f = hair_col.astype(np.float32)
        a3 = hair_alpha_f[:, :, None]
        out_f = out_f * (1.0 - a3) + col_f * a3

    return np.clip(out_f, 0, 255).astype(np.uint8)