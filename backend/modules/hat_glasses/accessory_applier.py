from typing import List, Tuple
import cv2
import numpy as np

LandmarkList = List[Tuple[int, int]]


# ─────────────────────────────────────────────
# 1. SAÇ MASKESI — landmark + renk hibrit
# ─────────────────────────────────────────────
def _build_hair_mask(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    h, w = image.shape[:2]
    hair_region = np.zeros((h, w), dtype=np.uint8)
    face_mask   = np.zeros((h, w), dtype=np.uint8)

    if landmarks is not None and len(landmarks) >= 100:
        top    = landmarks[10]
        bottom = landmarks[152]
        left   = landmarks[234]
        right  = landmarks[454]

        # Yüz maskesi (saçtan çıkaracağız)
        face_cx = (left[0] + right[0]) // 2
        face_cy = (top[1] + bottom[1]) // 2
        face_ax = int(abs(right[0] - left[0]) * 0.56)
        face_ay = int(abs(bottom[1] - top[1]) * 0.60)
        cv2.ellipse(face_mask, (face_cx, face_cy), (face_ax, face_ay),
                    0, 0, 360, 255, -1)

        # Saç bölgesi: yüzün üstü + yanlar (daha dar tutuyoruz)
        hx1 = max(0, int(left[0]  - w * 0.22))
        hx2 = min(w, int(right[0] + w * 0.22))
        hy1 = max(0, int(top[1]   - h * 0.30))
        hy2 = min(h, int(top[1]   + h * 0.15))   # sadece üst saç bandı

        cv2.ellipse(
            hair_region,
            ((hx1 + hx2) // 2, (hy1 + hy2) // 2),
            ((hx2 - hx1) // 2, (hy2 - hy1) // 2),
            0, 0, 360, 255, -1
        )

        # Yan saç bantları (kulak üstü)
        for side_x, side_cx in [(left[0], int(left[0] - w * 0.08)),
                                  (right[0], int(right[0] + w * 0.08))]:
            cv2.ellipse(
                hair_region,
                (side_cx, int((top[1] + bottom[1]) * 0.45)),
                (int(w * 0.10), int(h * 0.20)),
                0, 0, 360, 255, -1
            )
    else:
        cv2.rectangle(hair_region,
                      (int(w * 0.1), 0),
                      (int(w * 0.9), int(h * 0.45)), 255, -1)

    # Yüzü saç bölgesinden çıkar
    face_blur = cv2.GaussianBlur(face_mask, (31, 31), 0)
    hair_region[face_blur > 60] = 0

    # ── Renk maskesi (karışık saç için geniş aralık) ──
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # Koyu/siyah saç
    dark  = cv2.inRange(val, 0, 110)
    dark  = cv2.bitwise_and(dark, cv2.inRange(sat, 15, 255))

    # Kahverengi saç
    brown = cv2.inRange(hue, 5, 30)
    brown = cv2.bitwise_and(brown, cv2.inRange(sat, 35, 255))
    brown = cv2.bitwise_and(brown, cv2.inRange(val, 30, 210))

    # Koyu sarı / kumral
    blonde = cv2.inRange(hue, 15, 40)
    blonde = cv2.bitwise_and(blonde, cv2.inRange(sat, 40, 200))
    blonde = cv2.bitwise_and(blonde, cv2.inRange(val, 60, 220))

    # Mavi/gri arka planı çıkar
    bg_blue = cv2.inRange(hue, 85, 140)
    bg_gray = cv2.inRange(sat, 0, 20)

    color_mask = cv2.bitwise_or(dark, brown)
    color_mask = cv2.bitwise_or(color_mask, blonde)
    color_mask[bg_blue > 0] = 0
    color_mask[bg_gray > 0] = 0

    # Renk maskesi + bölge maskesi birleştir
    hair_mask = cv2.bitwise_and(color_mask, hair_region)

    # Temizle
    k = np.ones((7, 7), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN,  k)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, k)
    hair_mask = cv2.GaussianBlur(hair_mask, (15, 15), 0)

    # Eğer renk maskesi çok az yakaladıysa sadece bölgeyi kullan (fallback)
    if cv2.countNonZero(hair_mask) < (image.shape[0] * image.shape[1] * 0.005):
        hair_mask = cv2.GaussianBlur(hair_region, (15, 15), 0)

    return hair_mask


# ─────────────────────────────────────────────
# 2. SAÇ AĞARTMA
# ─────────────────────────────────────────────
def _apply_hair_graying(image: np.ndarray, hair_mask: np.ndarray,
                         intensity: float) -> np.ndarray:
    mask_f = np.clip(hair_mask.astype(np.float32) / 255.0, 0.0, 0.75)
    mask_f = np.repeat(mask_f[:, :, None], 3, axis=2)

    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Saf beyaz yerine gri-beyaz karışımı (daha doğal)
    white = np.full_like(image, 210)
    white_hair = cv2.addWeighted(gray_bgr, 0.6, white, 0.4, 0)

    strength = np.clip(intensity * 0.95, 0.0, 0.95)
    result = image * (1 - mask_f * strength) + white_hair * (mask_f * strength)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 3. CİLT YAŞLANDIRMA
# ─────────────────────────────────────────────
def _build_face_mask(image: np.ndarray, landmarks) -> np.ndarray:
    h, w = image.shape[:2]
    face_mask = np.zeros((h, w), dtype=np.uint8)

    if landmarks is not None and len(landmarks) >= 100:
        top    = landmarks[10]
        bottom = landmarks[152]
        left   = landmarks[234]
        right  = landmarks[454]

        cx = (left[0] + right[0]) // 2
        cy = (top[1] + bottom[1]) // 2
        ax = int(abs(right[0] - left[0]) * 0.52)
        ay = int(abs(bottom[1] - top[1]) * 0.58)
        cv2.ellipse(face_mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    else:
        cx, cy = w // 2, int(h * 0.42)
        cv2.ellipse(face_mask, (cx, cy), (int(w * 0.35), int(h * 0.42)),
                    0, 0, 360, 255, -1)

    return cv2.GaussianBlur(face_mask, (21, 21), 0)


def _apply_skin_aging(image: np.ndarray, face_mask: np.ndarray,
                       intensity: float) -> np.ndarray:
    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)
    mask_f = face_mask.astype(np.float32) / 255.0

    # ── a) Cilt tonu: sarı/mat ──
    tone_shift = np.zeros_like(output)
    tone_shift[:, :, 0] -= 4   * intensity   # B azalt
    tone_shift[:, :, 1] -= 3   * intensity   # G hafif azalt
    tone_shift[:, :, 2] += 8   * intensity   # R artır (sarımsı)
    output += tone_shift * mask_f[:, :, None]

    # ── b) Kontrast düşür (mat görünüm) ──
    output = output * (1 - 0.08 * intensity * mask_f[:, :, None]) + \
             128 * (0.08 * intensity * mask_f[:, :, None])

    # ── c) Kırışıklık dokusu (noise tabanlı) ──
    wrinkle_strength = intensity * 18
    noise = np.random.randn(h, w).astype(np.float32) * wrinkle_strength

    # Sadece belirli frekanslarda (ince çizgiler gibi)
    noise_blur   = cv2.GaussianBlur(noise, (0, 0), 1.2)
    noise_detail = noise - noise_blur   # yüksek frekans = ince çizgi
    noise_detail = cv2.GaussianBlur(noise_detail, (3, 3), 0)

    # Göz, alın, ağız çevresi ağırlıklı
    wrinkle_map = np.zeros((h, w), dtype=np.float32)
    if landmarks is not None:
        pass  # landmark bazlı bölge sonraki adımda

    for c in range(3):
        output[:, :, c] += noise_detail * mask_f * 0.6

    # ── d) Hafif bulanıklık (cilt elastikiyeti kaybı) ──
    blurred = cv2.GaussianBlur(output.astype(np.uint8), (0, 0), 1.5)
    blend   = intensity * 0.25
    output  = output * (1 - blend) + blurred.astype(np.float32) * blend

    return np.clip(output, 0, 255).astype(np.uint8)


def _apply_wrinkles_landmark(image: np.ndarray, landmarks,
                              intensity: float) -> np.ndarray:
    """Landmark bazlı kırışıklık bölgeleri: alın, göz kenarı, ağız çevresi."""
    if landmarks is None or len(landmarks) < 100:
        return image

    h, w = image.shape[:2]
    wrinkle_mask = np.zeros((h, w), dtype=np.float32)

    def add_region(cx, cy, rx, ry, weight=1.0):
        region = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(region, (int(cx), int(cy)), (int(rx), int(ry)),
                    0, 0, 360, 255, -1)
        region_f = cv2.GaussianBlur(region, (0, 0), rx // 3 + 1)
        wrinkle_mask[:] += region_f.astype(np.float32) / 255.0 * weight

    # Alın
    top = landmarks[10]
    add_region(top[0], top[1] + h * 0.04, w * 0.18, h * 0.04, 0.8)

    # Sol göz kenarı (kaz ayağı)
    le = landmarks[33]
    add_region(le[0] - w * 0.03, le[1], w * 0.04, h * 0.025, 1.0)

    # Sağ göz kenarı
    re = landmarks[263]
    add_region(re[0] + w * 0.03, re[1], w * 0.04, h * 0.025, 1.0)

    # Ağız kenarları (nasolabial)
    lm = landmarks[61]
    rm = landmarks[291]
    add_region(lm[0] - w * 0.02, lm[1] + h * 0.02, w * 0.03, h * 0.03, 0.7)
    add_region(rm[0] + w * 0.02, rm[1] + h * 0.02, w * 0.03, h * 0.03, 0.7)

    wrinkle_mask = np.clip(wrinkle_mask, 0, 1)

    # Kırışıklık dokusu
    noise = np.random.randn(h, w).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.8)

    output = image.astype(np.float32)
    strength = intensity * 14
    for c in range(3):
        output[:, :, c] += noise * wrinkle_mask * strength * (0.6 if c < 2 else 0.3)

    return np.clip(output, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 4. ANA FONKSİYON
# ─────────────────────────────────────────────
def apply_aging_effect(image: np.ndarray,
                        intensity: float = 0.5,
                        landmarks=None) -> np.ndarray:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    output = image.copy()

    # Saç ağartma
    hair_mask = _build_hair_mask(output, landmarks, intensity)
    output    = _apply_hair_graying(output, hair_mask, intensity)

    # Cilt yaşlandırma
    face_mask = _build_face_mask(output, landmarks)
    output    = _apply_skin_aging(output, face_mask, intensity)

    # Landmark bazlı kırışıklıklar
    output = _apply_wrinkles_landmark(output, landmarks, intensity)

    return output