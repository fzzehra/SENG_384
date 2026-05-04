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
    n_forehead = rng.integers(3, 5)
    for i in range(n_forehead):
        t = (i + 1) / (n_forehead + 1)
        cy = int(top[1] + t * face_h * 0.18)
        p1 = (int(left[0] + face_w * 0.2), cy + rng.integers(-5, 5))
        p2 = (int(right[0] - face_w * 0.2), cy + rng.integers(-5, 5))
        _stamp_wrinkle_natural(p1, p2, w_px, rng.uniform(0.3, 0.5))

    # 2. Göz Kenarı (Karga Ayakları - Daha ince)
    for eye_idx, sign in [(33, -1), (263, 1)]:
        ex, ey = _lm(landmarks, eye_idx)
        for k in range(3):
            angle = np.radians(-30 + k * 30)
            length = face_w * rng.uniform(0.05, 0.08)
            _stamp_wrinkle_natural((ex, ey), (int(ex + sign*np.cos(angle)*length), int(ey + np.sin(angle)*length)), w_px * 0.7, 0.4)

    # 3. Nasolabial (Burun-Ağız kenarı)
    for n_idx, m_idx in [(129, 61), (358, 291)]:
        _stamp_wrinkle_natural(_lm(landmarks, n_idx), _lm(landmarks, m_idx), w_px * 1.2, 0.45)

    # Gürültü ile doku modülasyonu (Çizgiyi parçalar, deri gözenek etkisi verir)
    noise = _perlin_like_noise(h, w, scale=2.0, seed=seed)
    canvas = canvas * (0.7 + 0.3 * noise)
    
    return cv2.GaussianBlur(canvas, (3, 3), 0)

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
    
    # Kırışıklık "Çukur" (Gölge) Etkisi
    lab[:, :, 0] -= (wrinkle_map * intensity * 65.0) 
    # Kırışıklık "Tümsek" (Highlight) Etkisi - Işığın vurduğu kenar
    lab[:, :, 0] += (cv2.Laplacian(wrinkle_map, cv2.CV_32F).clip(0,1) * intensity * 20.0)

    res = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return res

# ---------------------------------------------------------------------------
# SAÇ (Artifact Korumalı)
# ---------------------------------------------------------------------------
def _gray_hair(image, landmarks, intensity):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)

    # Saç maskesi ve Yüz dışlama
    hair_mask = cv2.inRange(val, 0, 160) & cv2.inRange(sat, 10, 250)
    
    # Landmark tabanlı kafa bölgesi (Boyuna inmeyi engeller)
    top = _lm(landmarks, 10)
    exclude_mask = np.zeros((h, w), dtype=np.uint8)
    # Yüzü ve omuz altını koru
    cv2.circle(exclude_mask, (_lm(landmarks, 1)[0], _lm(landmarks, 1)[1]), int(_dist(_lm(landmarks, 234), _lm(landmarks, 454))*1.2), 255, -1)
    
    final_hair_mask = cv2.bitwise_and(hair_mask, exclude_mask)
    final_hair_mask = cv2.GaussianBlur(final_hair_mask.astype(np.float32)/255.0, (15, 15), 0)

    # Saçın dokusunu bozmadan grileştir (Desaturate)
    gray_img = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    
    alpha = (final_hair_mask * intensity * 0.8)[:, :, None]
    # Hafif gümüş parlaklığı ekle
    blended = image.astype(np.float32)*(1-alpha) + (gray_img.astype(np.float32) * 1.1)*alpha
    
    return np.clip(blended, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# ANA FLOW
# ---------------------------------------------------------------------------
def apply_aging_effect(image, intensity=0.5, landmarks=None):
    if landmarks is None: return image
    
    img = image.copy()
    # 1. Saç (Omuzlara sızma engellendi)
    img = _gray_hair(img, landmarks, intensity)
    
    # 2. Yüz Maskesi
    f_mask = _build_face_mask(image.shape[0], image.shape[1], landmarks)
    
    # 3. Kırışıklıklar (Gri bant değil, gölge-ışık bükülmesi)
    img = _apply_wrinkles(img, f_mask, intensity, landmarks)
    
    return img