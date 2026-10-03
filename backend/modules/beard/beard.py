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


# MediaPipe FaceMesh yüz oval indeksleri (sıralı çevre)
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Çene hattı: görüntüde soldan sağa
_JAW_PATH = [132, 58, 172, 136, 150, 149, 176, 148, 152,
             377, 400, 378, 379, 365, 397, 288, 361]


def apply_beard_effect(image, landmarks, intensity=0.8, color_hex="#241815", **kwargs):
    out = image.copy()
    h, w = out.shape[:2]

    intensity = float(np.clip(intensity, 0.0, 1.0))
    base_bgr = np.array(_hex_to_bgr(color_hex), dtype=np.float32)

    def P(i):
        return _pt(landmarks, i, w, h)

    left_cheek = P(93)
    right_cheek = P(323)
    chin = P(152)
    mouth_left = P(61)
    mouth_right = P(291)
    upper_lip = P(13)
    lower_lip = P(17)

    face_width = float(np.linalg.norm(P(454) - P(234)))
    beard_height = max(20.0, float(chin[1] - lower_lip[1]))

    if face_width < 20 or beard_height < 6:
        return out

    # ───────────────────────── 1) Sakal bölgesi maskesi ─────────────────────────
    # Çene hattını takip eden çokgen: sakal yüzün dışına taşamaz.
    cheek_top_y = mouth_left[1] - beard_height * 0.35

    jaw_pts = [P(i) for i in _JAW_PATH]
    lt = np.array([left_cheek[0] + face_width * 0.03, cheek_top_y])
    rt = np.array([right_cheek[0] - face_width * 0.03, cheek_top_y])
    mouth_r_o = np.array([mouth_right[0] + face_width * 0.10, mouth_right[1] - beard_height * 0.10])
    mouth_l_o = np.array([mouth_left[0] - face_width * 0.10, mouth_left[1] - beard_height * 0.10])

    poly = np.array([lt] + jaw_pts + [rt, mouth_r_o, mouth_l_o], dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

    # Ağız ve üst dudak temizliği
    mx = int((mouth_left[0] + mouth_right[0]) / 2)
    mw = int(np.linalg.norm(mouth_right - mouth_left) * 0.62)
    lip_mid = int((upper_lip[1] + lower_lip[1]) / 2)
    lip_h = int(max(10, abs(lower_lip[1] - upper_lip[1]) * 1.5 + 6))
    cv2.ellipse(mask, (mx, lip_mid), (mw, lip_h), 0, 0, 360, 0, -1)
    cv2.ellipse(mask, (mx, int(upper_lip[1])), (mw, int(beard_height * 0.30)), 0, 180, 360, 0, -1)

    # Yüz ovali içine kırp (hafif içeri çekilmiş) → yanak dışına taşma yok
    face_poly = np.array([P(i) for i in _FACE_OVAL], dtype=np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_poly], 255)
    erode_k = max(3, int(face_width * 0.025))
    face_mask = cv2.erode(face_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)))

    blur_k = int(face_width * 0.06) // 2 * 2 + 1
    blur_k = max(5, blur_k)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask = (mask.astype(np.float32) * (face_mask.astype(np.float32) / 255.0))
    mask = np.clip(mask, 0, 255)
    m01 = mask / 255.0

    out_f = out.astype(np.float32)

    # ───────────────────────── 2) Çok hafif gölge/tint (boya gibi durmaz) ─────────
    # Multiply tarzı: ten dokusu korunur, sadece hafif sakal gölgesi verir.
    tint_a = m01 * (0.06 + 0.16 * intensity)
    tint = base_bgr / 255.0
    for c in range(3):
        mult = 1.0 - tint_a * (1.0 - np.clip(tint[c] * 2.2 + 0.25, 0.0, 1.0))
        out_f[:, :, c] *= mult

    # ───────────────────────── 3) Tek tek kıllar ─────────────────────────────────
    ys, xs = np.where(mask > 20)
    if len(xs) > 0:
        rng = np.random.default_rng(42)

        # Yoğunluk: intensity arttıkça kıl sayısı artar
        area = float(len(xs))
        hair_per_px = 0.04 + 0.22 * intensity
        n_hair = int(min(45000, max(1500, area * hair_per_px)))

        # Maske değerine göre ağırlıklı seçim: kenarlarda seyrek, merkezde sık
        weights = m01[ys, xs]
        weights = weights / weights.sum()
        pick = rng.choice(len(xs), size=n_hair, replace=True, p=weights)
        px = xs[pick].astype(np.float32) + rng.uniform(-0.5, 0.5, n_hair)
        py = ys[pick].astype(np.float32) + rng.uniform(-0.5, 0.5, n_hair)

        # Uzunluk: yoğunlukla hafif artar ama kısa kalır
        min_len = max(2.0, beard_height * 0.035)
        max_len = max(min_len + 1.0, beard_height * (0.07 + 0.07 * intensity))
        lengths = rng.uniform(min_len, max_len, n_hair)

        # Yön: genel olarak aşağı, yanaklarda çeneye doğru hafif içe
        cx = float(chin[0])
        side = np.clip((cx - px) / (face_width * 0.5), -1.0, 1.0)
        angles = rng.normal(0.0, 0.28, n_hair) + side * 0.45  # radyan, dikeyden sapma

        # Katmanlar: renk + alfa
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
        hair_alpha_f *= m01  # sakal bölgesi dışına çıkmasın
        hair_alpha_f *= (0.70 + 0.30 * intensity)

        # Kıl rengi: AA çizgilerin siyah arka planını karıştırmamak için renk katmanını normalize et
        col_f = hair_col.astype(np.float32)
        a3 = hair_alpha_f[:, :, None]
        out_f = out_f * (1.0 - a3) + col_f * a3

    return np.clip(out_f, 0, 255).astype(np.uint8)