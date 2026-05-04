import os
import cv2
import numpy as np


GLASSES_CONFIG = {
    "glasses1.png": {"scale": 1.7, "x_offset": 0, "y_offset": 0},
    "glasses2.png": {"scale": 1.65, "x_offset": 0, "y_offset": 0},
    "glasses3.png": {"scale": 1.7, "x_offset": 0, "y_offset": 0},
    "glasses4.png": {"scale": 1.7, "x_offset": 0, "y_offset": 0},
}


def _crop_transparent(img):
    """Şeffaf kenarları kırp, sadece içerik kalsın."""
    if img.shape[2] != 4:
        return img
    alpha = img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


def _overlay(base, overlay, cx, cy):
    """
    overlay görüntüsünü base üzerine yapıştır.
    cx, cy → overlay'in base üzerindeki merkez koordinatı.
    """
    oh, ow = overlay.shape[:2]
    bh, bw = base.shape[:2]

    # Sol üst köşe
    x1 = cx - ow // 2
    y1 = cy - oh // 2

    # Kırpma sınırları
    ox1 = max(0, -x1)
    oy1 = max(0, -y1)
    ox2 = ow - max(0, (x1 + ow) - bw)
    oy2 = oh - max(0, (y1 + oh) - bh)

    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = bx1 + (ox2 - ox1)
    by2 = by1 + (oy2 - oy1)

    if ox1 >= ox2 or oy1 >= oy2:
        return base

    src = overlay[oy1:oy2, ox1:ox2]
    dst = base[by1:by2, bx1:bx2]

    if overlay.shape[2] == 4:
        alpha = src[:, :, 3:4].astype(np.float32) / 255.0
        blended = (src[:, :, :3].astype(np.float32) * alpha +
                   dst.astype(np.float32) * (1 - alpha))
        base[by1:by2, bx1:bx2] = blended.astype(np.uint8)
    else:
        base[by1:by2, bx1:bx2] = src[:, :, :3]

    return base


def place_glasses(image, landmarks, glasses_path):
    """
    Gözlüğü yüze yerleştir.

    Parameters
    ----------
    image         : BGR görüntü
    landmarks     : detect_landmarks() çıktısı
    glasses_path  : Gözlük PNG dosyasının tam yolu

    Returns
    -------
    Gözlük yerleştirilmiş BGR görüntü
    """
    if landmarks is None or len(landmarks) < 100:
        print("GLASSES: landmark yok")
        return image

    glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
    if glasses_img is None:
        print(f"GLASSES: dosya okunamadı → {glasses_path}")
        return image

    if glasses_img.shape[2] != 4:
        print("GLASSES: alpha kanalı yok, BGRA'ya çeviriliyor")
        glasses_img = cv2.cvtColor(glasses_img, cv2.COLOR_BGR2BGRA)

    output = image.copy()

    # Landmark noktaları
    l_eye  = landmarks[33]   # sol göz dış köşe
    r_eye  = landmarks[263]  # sağ göz dış köşe
    l_eye_inner = landmarks[133]  # sol göz iç köşe
    r_eye_inner = landmarks[362]  # sağ göz iç köşe

    # Göz merkezi
    eye_cx = (l_eye[0] + r_eye[0]) // 2
    eye_cy = (l_eye[1] + r_eye[1]) // 2

    # Yüz genişliği
    face_width = abs(r_eye[0] - l_eye[0])

    # Config
    glasses_name = os.path.basename(glasses_path)
    cfg = GLASSES_CONFIG.get(glasses_name, {"scale": 0.85, "y_offset": 0})

    print(f"GLASSES: {glasses_name}, face_width={face_width}, scale={cfg['scale']}")

    # 1. Şeffaf kenarları kırp
    glasses_img = _crop_transparent(glasses_img)

    # 2. Hedef genişlik = iki göz dış köşesi arası × scale
    target_w = int(face_width * cfg["scale"])
    # /0.6 çünkü l_eye(33) - r_eye(263) arası yüz genişliğinin ~%60'ı

    # 3. Oranı koru
    h0, w0 = glasses_img.shape[:2]
    target_h = int(h0 * target_w / w0)

    glasses_resized = cv2.resize(glasses_img, (target_w, target_h),
                                 interpolation=cv2.INTER_AREA)

    # 4. Baş açısı
    angle = np.degrees(np.arctan2(
        r_eye[1] - l_eye[1],
        r_eye[0] - l_eye[0]
    ))
    if abs(angle) > 0.5:
        h_r, w_r = glasses_resized.shape[:2]
        M = cv2.getRotationMatrix2D((w_r // 2, h_r // 2), angle, 1.0)
        
        # Yeni canvas boyutunu hesapla (kesilmesin)
        cos_a = abs(np.cos(np.radians(angle)))
        sin_a = abs(np.sin(np.radians(angle)))
        new_w = int(h_r * sin_a + w_r * cos_a)
        new_h = int(h_r * cos_a + w_r * sin_a)
        
        # Merkezi yeni canvas'a göre ayarla
        M[0, 2] += (new_w - w_r) / 2
        M[1, 2] += (new_h - h_r) / 2
        
        glasses_resized = cv2.warpAffine(
            glasses_resized, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
    )

    # 5. Yerleştir — if bloğunun DIŞINDA olmalı
    final_cx = eye_cx + cfg.get("x_offset", 0)
    final_cy = eye_cy + cfg.get("y_offset", 0)

    print(f"GLASSES: target_w={target_w}, target_h={target_h}, center=({final_cx},{final_cy})")

    output = _overlay(output, glasses_resized, int(final_cx), int(final_cy))
    return output