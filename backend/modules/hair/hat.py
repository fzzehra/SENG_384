"""
hat.py — Şapka aksesuarını yüz landmark'larına göre yerleştirme modülü.

Kullanım:
    from hat import place_hat
    result = place_hat(image, landmarks)
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

# Tip alias
LandmarkList = List[Tuple[int, int]]

# ─────────────────────────────────────────────
# MediaPipe landmark indeksleri (şapka için)
# ─────────────────────────────────────────────
HAT_LANDMARKS = {
    "forehead_center": 10,   # alın orta — şapka alt kenarı hizalama
    "forehead_left":   109,  # alın sol uç
    "forehead_right":  338,  # alın sağ uç
    "temple_left":     234,  # şakak sol  — şapka genişliği
    "temple_right":    454,  # şakak sağ  — şapka genişliği
    "hairline_left":   299,  # saç çizgisi sol
    "hairline_right":  70,   # saç çizgisi sağ
}


# ─────────────────────────────────────────────
# Yardımcı: landmark sözlüğü çıkar
# ─────────────────────────────────────────────
def _get_hat_points(landmarks: LandmarkList) -> dict:
    """HAT_LANDMARKS indekslerini gerçek koordinatlara çevirir."""
    return {
        name: landmarks[idx]
        for name, idx in HAT_LANDMARKS.items()
        if idx < len(landmarks)
    }


# ─────────────────────────────────────────────
# Yardımcı: baş eğim açısı
# ─────────────────────────────────────────────
def _compute_head_angle(pts: dict) -> float:
    """
    Şakak noktalarından baş eğim açısını hesaplar (derece).
    Pozitif → sağa eğik, negatif → sola eğik.
    """
    left  = np.array(pts["temple_left"])
    right = np.array(pts["temple_right"])
    delta = right - left
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


# ─────────────────────────────────────────────
# Yardımcı: görüntü döndürme (alpha korunur)
# ─────────────────────────────────────────────
def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Merkezi etrafında döndür, siyah kenarlar oluşmaz (expand=True)."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    flags = cv2.INTER_LINEAR
    border = cv2.BORDER_CONSTANT

    if image.shape[2] == 4:
        return cv2.warpAffine(image, M, (new_w, new_h),
                              flags=flags, borderMode=border,
                              borderValue=(0, 0, 0, 0))
    return cv2.warpAffine(image, M, (new_w, new_h),
                          flags=flags, borderMode=border)


# ─────────────────────────────────────────────
# Yardımcı: alpha blending ile üst üste bindirme
# ─────────────────────────────────────────────
def _overlay_png(
    background: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """
    RGBA overlay'i BGR background üzerine alpha blending ile yazar.
    x, y → overlay'in sol-üst köşesi background'daki hedef koordinatı.
    """
    out = background.copy()
    oh, ow = overlay.shape[:2]
    bh, bw = background.shape[:2]

    # Sınır kırpma
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, bw), min(y + oh, bh)

    if x1 >= x2 or y1 >= y2:
        return out  # Görüntü dışında, bir şey yapma

    # Overlay'in görünür dilimi
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    roi     = out[y1:y2, x1:x2].astype(np.float32)
    hat_roi = overlay[oy1:oy2, ox1:ox2]

    if hat_roi.shape[2] == 4:
        alpha = hat_roi[:, :, 3:4].astype(np.float32) / 255.0
        bgr   = hat_roi[:, :, :3].astype(np.float32)
    else:
        alpha = np.ones((hat_roi.shape[0], hat_roi.shape[1], 1), dtype=np.float32)
        bgr   = hat_roi.astype(np.float32)

    blended = bgr * alpha + roi * (1.0 - alpha)
    out[y1:y2, x1:x2] = blended.astype(np.uint8)
    return out


# ─────────────────────────────────────────────
# Ana fonksiyon
# ─────────────────────────────────────────────
def place_hat(
    image: np.ndarray,
    landmarks: LandmarkList,
    hat_image: np.ndarray,
    width_scale: float = 1.35,
    vertical_offset: float = 0.85,
) -> np.ndarray:
    """
    Şapkayı yüz landmark'larına göre yerleştirir.

    Parameters
    ----------
    image          : BGR veya BGRA kaynak görüntü.
    landmarks      : detect_landmarks() çıktısı, (x, y) tuple listesi.
    hat_image      : Şapka görüntüsü (BGRA önerilir, şeffaflık için).
    width_scale    : Şapka genişliği = iki şakak arası mesafe × bu katsayı.
                     Varsayılan 1.35 — şakakların biraz dışına taşar.
    vertical_offset: 0.0 → şapka alt kenarı alın ortasında,
                     1.0 → şapka tamamen alnın üstünde.
                     Varsayılan 0.85 (şapka hafifçe alına iner).

    Returns
    -------
    BGR görüntü — şapka yerleştirilmiş.
    """
    if image is None:
        raise ValueError("image cannot be None.")
    if not landmarks:
        raise ValueError("landmarks is empty — run detect_landmarks() first.")
    if hat_image is None:
        raise ValueError("hat_image cannot be None.")

    pts = _get_hat_points(landmarks)

    # ── 1. Şapka genişliği: şakaklar arası mesafe ──────────────────────────
    left_temple  = np.array(pts["temple_left"])
    right_temple = np.array(pts["temple_right"])
    face_width   = int(np.linalg.norm(right_temple - left_temple))
    target_w     = int(face_width * width_scale)

    # En az 1 piksel
    target_w = max(target_w, 1)

    # ── 2. Şapka yüksekliği: oranı koru ───────────────────────────────────
    hat_h_orig, hat_w_orig = hat_image.shape[:2]
    aspect      = hat_h_orig / max(hat_w_orig, 1)
    target_h    = int(target_w * aspect)
    target_h    = max(target_h, 1)

    # ── 3. Ölçekle ─────────────────────────────────────────────────────────
    hat_resized = cv2.resize(hat_image, (target_w, target_h),
                             interpolation=cv2.INTER_AREA)

    # ── 4. Baş açısına göre döndür ─────────────────────────────────────────
    angle = _compute_head_angle(pts)
    hat_rotated = _rotate_image(hat_resized, angle)

    # ── 5. Yerleşim koordinatları ──────────────────────────────────────────
    # Yatay: şakak merkezine hizala
    center_x = int((left_temple[0] + right_temple[0]) / 2)
    hat_place_x = center_x - hat_rotated.shape[1] // 2

    # Dikey: alın noktasından yukarı çık
    forehead_y = pts["forehead_center"][1]
    hat_place_y = int(forehead_y - hat_rotated.shape[0] * vertical_offset)

    # ── 6. Bindirme ────────────────────────────────────────────────────────
    result = _overlay_png(image, hat_rotated, hat_place_x, hat_place_y)
    return result


# ─────────────────────────────────────────────
# Opsiyonel: landmark noktalarını debug görsel
# ─────────────────────────────────────────────
def debug_hat_landmarks(
    image: np.ndarray,
    landmarks: LandmarkList,
) -> np.ndarray:
    """
    Şapka için kullanılan landmark noktalarını görüntü üzerine çizer.
    Geliştirme/test aşamasında indeks doğrulamak için kullanın.
    """
    pts = _get_hat_points(landmarks)
    out = image.copy()

    colors = {
        "forehead_center": (0, 255, 255),
        "forehead_left":   (255, 128, 0),
        "forehead_right":  (255, 128, 0),
        "temple_left":     (0, 0, 255),
        "temple_right":    (0, 0, 255),
        "hairline_left":   (0, 255, 0),
        "hairline_right":  (0, 255, 0),
    }

    for name, (x, y) in pts.items():
        color = colors.get(name, (255, 255, 255))
        cv2.circle(out, (x, y), 5, color, -1)
        cv2.putText(out, name, (x + 6, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                    cv2.LINE_AA)

    # Şakaklar arası yatay çizgi (genişlik referansı)
    if "temple_left" in pts and "temple_right" in pts:
        cv2.line(out, pts["temple_left"], pts["temple_right"],
                 (0, 0, 255), 1, cv2.LINE_AA)

    return out