from typing import List, Tuple
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------
def _lm(landmarks, idx):
    p = landmarks[idx]
    return (int(round(p[0])), int(round(p[1])))

def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def _perlin_like_noise(h, w, scale=4.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude, frequency = 1.0, 1.0
    for _ in range(4):
        sh, sw = max(1, int(h/(scale*frequency))), max(1, int(w/(scale*frequency)))
        small = rng.random((sh, sw)).astype(np.float32) * 2 - 1
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += big * amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

# ---------------------------------------------------------------------------
# YÜZ MASKESİ (Daha Keskin Sınırlar)
# ---------------------------------------------------------------------------
def _build_face_mask(h, w, landmarks):
    mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks is None: return mask
    # Yüz hattını belirleyen landmarklar
    face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    pts = np.array([_lm(landmarks, i) for i in face_contour])
    cv2.fillPoly(mask, [pts], 255)
    return mask

# ---------------------------------------------------------------------------
# KIRIŞIKLIK HARİTASI (Texture & Displacement Yaklaşımı)
# ---------------------------------------------------------------------------
def _make_wrinkle_map(h, w, landmarks, face_scale, seed=42):
    rng = np.random.default_rng(seed)
    canvas = np.zeros((h, w), dtype=np.float32)
    if landmarks is None: return canvas

    S = face_scale / 250.0
    top, bottom = _lm(landmarks, 10), _lm(landmarks, 152)
    left, right = _lm(landmarks, 234), _lm(landmarks, 454)
    face_w, face_h = _dist(left, right), _dist(top, bottom)

    def _stamp_wrinkle_natural(p1, p2, width_px, strength):
        length = max(1, int(_dist(p1, p2)))
        t_vals = np.linspace(0, 1, length + 1)
        # Çizgi uçlarını çok yumuşak bitir (bant görüntüsünü engeller)
        fade = (np.sin(np.pi * t_vals) ** 1.2) * strength
        
        dx, dy = (p2[0]-p1[0])/length, (p2[1]-p1[1])/length
        nx, ny = -dy, dx
        
        r_range = np.arange(-int(width_px * 2), int(width_px * 2) + 1)
        gauss = np.exp(-(r_range ** 2) / (2 * (width_px/1.5) ** 2))

        for i, (bx, by, f) in enumerate(zip(p1[0]+(p2[0]-p1[0])*t_vals, p1[1]+(p2[1]-p1[1])*t_vals, fade)):
            px_arr = np.round(bx + nx * r_range).astype(int)
            py_arr = np.round(by + ny * r_range).astype(int)
            valid = (px_arr >= 0) & (px_arr < w) & (py_arr >= 0) & (py_arr < h)
            np.maximum.at(canvas, (py_arr[valid], px_arr[valid]), gauss[valid] * f)

    w_px = max(0.8, 1.2 * S)

    # 1. Alın (Daha seyrek ve kavisli)
    n_forehead = rng.integers(4, 7)
    for i in range(n_forehead):
        t = (i + 1) / (n_forehead + 1)
        cy = int(top[1] + t * face_h * 0.18)
        p1 = (int(left[0] + face_w * 0.2), cy + rng.integers(-5, 5))
        p2 = (int(right[0] - face_w * 0.2), cy + rng.integers(-5, 5))
        _stamp_wrinkle_natural(p1, p2, w_px, rng.uniform(0.3, 0.5))

    # göz çevresi (karga ayakları + alt göz çizgileri)
    for eye_idx, sign in [(33, -1), (263, 1)]:

        ex, ey = _lm(landmarks, eye_idx)

        # karga ayakları
        for k in range(4):

            angle = np.radians(-35 + k * 25)
            length = face_w * rng.uniform(0.06, 0.11)

            _stamp_wrinkle_natural(
                (ex, ey),
                (
                    int(ex + sign * np.cos(angle) * length),
                    int(ey + np.sin(angle) * length)
                ),
                w_px * 0.8,
                rng.uniform(0.55, 0.82)
            )

    # alt göz kapağı ince çizgiler
    for k in range(2):

        p1 = (
            int(ex - sign * face_w * 0.08),
            int(ey + face_h * 0.03)
        )

        p2 = (
            int(p1[0] + sign * face_w * 0.16),
            int(ey + face_h * 0.05)
        )

        _stamp_wrinkle_natural(
            p1,
            p2,
            w_px * 0.6,
            rng.uniform(0.35, 0.5)
        )
    # 3. Nasolabial (Burun-Ağız kenarı)
    # nazolabial çizgiler (burun kenarı -> ağız)
    for n_idx, m_idx in [(129, 61), (358, 291)]:

        _stamp_wrinkle_natural(
            _lm(landmarks, n_idx),
            _lm(landmarks, m_idx),
            w_px * 2.0,
            rng.uniform(0.75, 0.95)
        )

        # marionette çizgileri (ağız köşesi aşağı)
    for m_idx, jaw_idx in [(61, 152), (291, 152)]:

        mouth = _lm(landmarks, m_idx)
        chin = _lm(landmarks, jaw_idx)

        p2 = (
            mouth[0] + rng.integers(-4, 4),
            int(mouth[1] + (chin[1] - mouth[1]) * 0.6)
        )

        _stamp_wrinkle_natural(
            mouth,
            p2,
            w_px * 1.4,
            rng.uniform(0.55, 0.75)
        ) 
        _stamp_wrinkle_natural(_lm(landmarks, n_idx), _lm(landmarks, m_idx), w_px * 1.2, 0.45)

    # Gürültü ile doku modülasyonu (Çizgiyi parçalar, deri gözenek etkisi verir)
    noise = _perlin_like_noise(h, w, scale=2.0, seed=seed)
    canvas = canvas * (0.7 + 0.3 * noise)
    
    canvas = cv2.GaussianBlur(canvas, (7, 7), 1.8)
    canvas = np.clip(canvas, 0, 0.55)
    return canvas

# ---------------------------------------------------------------------------
# UYGULAMA (Shadow/Highlight Displacement)
# ---------------------------------------------------------------------------
def _apply_wrinkles(image, face_mask, intensity, landmarks):
    h, w = image.shape[:2]
    face_scale = _dist(_lm(landmarks, 234), _lm(landmarks, 454))
    
    # Yüz maskesini yumuşat (kenar geçişleri için)
    face_soft = cv2.GaussianBlur(face_mask.astype(np.float32)/255.0, (31, 31), 0)
    
    wrinkle_map = _make_wrinkle_map(h, w, landmarks, face_scale) * face_soft
    
    # Lab renk uzayında sadece L (parlaklık) kanalına müdahale
    # Bu, deri rengini (griye dönmeden) korur
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Daha doğal: sert çizgi yerine yumuşak gölge
    soft_wrinkle = cv2.GaussianBlur(wrinkle_map, (9, 9), 2.0)

    lab[:, :, 0] -= (soft_wrinkle * intensity * 62.0)
    lab[:, :, 0] += (cv2.Laplacian(soft_wrinkle, cv2.CV_32F).clip(0, 1) * intensity * 14.0)

    res = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return res

# ---------------------------------------------------------------------------
# SAÇ (Artifact Korumalı)
# ---------------------------------------------------------------------------
def _gray_hair(image, landmarks, intensity):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # Koyu saç + kahverengi saç adayları
    dark_hair = cv2.bitwise_and(
        cv2.inRange(val, 0, 135),
        cv2.inRange(sat, 25, 255)
    )

    brown_hair = cv2.bitwise_and(
        cv2.inRange(hue, 5, 35),
        cv2.inRange(sat, 45, 255)
    )
    brown_hair = cv2.bitwise_and(brown_hair, cv2.inRange(val, 35, 210))

    hair_color = cv2.bitwise_or(dark_hair, brown_hair)

    # Mavi/gri arka planı ve cilt tonlarını ele
    blue_bg = cv2.inRange(hue, 85, 145)
    low_sat = cv2.inRange(sat, 0, 20)
    hair_color[blue_bg > 0] = 0
    hair_color[low_sat > 0] = 0

    hair_region = np.zeros((h, w), dtype=np.uint8)
    face_exclude = np.zeros((h, w), dtype=np.uint8)

    top = _lm(landmarks, 10)
    chin = _lm(landmarks, 152)
    left = _lm(landmarks, 234)
    right = _lm(landmarks, 454)

    face_w = _dist(left, right)
    face_h = _dist(top, chin)
    cx = (left[0] + right[0]) // 2

    # Saç bölgesi: yüzün üstü ve yanları, boyuna inmez
    x1 = max(0, int(left[0] - face_w * 0.45))
    x2 = min(w, int(right[0] + face_w * 0.45))
    y1 = max(0, int(top[1] - face_h * 0.55))
    y2 = min(h, int(chin[1] - face_h * 0.12))

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

    # Yüzü tamamen çıkar: kaş, göz, dudak, burun boyanmasın
    cv2.ellipse(
        face_exclude,
        (cx, int(top[1] + face_h * 0.42)),
        (int(face_w * 0.58), int(face_h * 0.68)),
        0,
        0,
        360,
        255,
        -1
    )

    face_exclude = cv2.dilate(face_exclude, np.ones((13, 13), np.uint8), iterations=2)

    hair_mask = cv2.bitwise_and(hair_color, hair_region)
    hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(face_exclude))

    kernel = np.ones((5, 5), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.GaussianBlur(hair_mask, (11, 11), 0)

    alpha = hair_mask.astype(np.float32) / 255.0
    alpha = np.clip(alpha * intensity * 0.95, 0, 0.75)
    alpha = alpha[:, :, None]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    silver = np.full_like(image, (185, 190, 195), dtype=np.float32)
    target = gray_bgr * 0.60 + silver * 0.40

    result = image.astype(np.float32) * (1 - alpha) + target * alpha

    return np.clip(result, 0, 255).astype(np.uint8)
# ---------------------------------------------------------------------------
# ANA FLOW
# ---------------------------------------------------------------------------
def apply_aging_effect(image, intensity=0.5, landmarks=None):
    if landmarks is None: return image
    
    img = image.copy()
    # 1. Saç (Omuzlara sızma engellendi)
    #img = _gray_hair(img, landmarks, intensity)
    
    # 2. Yüz Maskesi
    f_mask = _build_face_mask(image.shape[0], image.shape[1], landmarks)
    
    # 3. Kırışıklıklar (Gri bant değil, gölge-ışık bükülmesi)
    img = _apply_wrinkles(img, f_mask, intensity * 1.35, landmarks)
    img = _apply_sagging(img, landmarks, intensity)
    
    return img
def _apply_sagging(image, landmarks, intensity):
    if landmarks is None or len(landmarks) < 468:
        return image

    h, w = image.shape[:2]
    output = image.copy()

    def p(idx):
        x, y = landmarks[idx]
        return int(x), int(y)

    left_face = p(234)
    right_face = p(454)
    chin = p(152)
    nose = p(1)

    face_w = abs(right_face[0] - left_face[0])
    face_h = abs(chin[1] - nose[1])

    # Sarkma maskesi: alt yanak + çene/gıdı bölgesi
    mask = np.zeros((h, w), dtype=np.float32)

    center_x = (left_face[0] + right_face[0]) // 2
    center_y = int(nose[1] + face_h * 0.10)

    cv2.ellipse(
        mask,
        (center_x, center_y),
        (int(face_w * 0.55), int(face_h * 0.12)),
        0,
        0,
        360,
        1.0,
        -1
    )

    # ağız üstünü etkilemesin
    mask[:int(nose[1] + face_h * 0.02), :] = 0

    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    # displacement map
    yy, xx = np.indices((h, w), dtype=np.float32)

    sag_amount = intensity * 1.2

    map_x = xx.copy()
    map_y = yy - (mask * sag_amount)

    sagged = cv2.remap(
        output,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # gıdı için çene altını hafif karart
    shadow = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        shadow,
        (center_x, int(chin[1] + face_h * 0.15)),
        (int(face_w * 0.35), int(face_h * 0.18)),
        0,
        0,
        360,
        1.0,
        -1
    )
    shadow = cv2.GaussianBlur(shadow, (41, 41), 0)

    shadow_3 = shadow[:, :, None] * intensity * 0.08
    sagged = sagged.astype(np.float32) * (1 - shadow_3)

    return np.clip(sagged, 0, 255).astype(np.uint8)