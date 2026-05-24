# ==============================================================================
# backend/modules/hair/hair.py
# ==============================================================================
import os
import cv2
import numpy as np

# Özel peruklar için el ile ince ayar yapmak isterseniz buraya ekleyebilirsiniz.
# Biçim: "dosya_adi.png": { "left_ratio": (X_oran, Y_oran), "right_ratio": (X_oran, Y_oran), "top_ratio": (X_oran, Y_oran) }
HAIRSTYLE_ANCHOR_PRESETS = {
    "saç.png": {
        "left_ratio": (0.24, 0.45),
        "right_ratio": (0.76, 0.45),
        "top_ratio": (0.50, 0.22)
    },
    "wavy.png": {
        "left_ratio": (0.26, 0.48),
        "right_ratio": (0.74, 0.48),
        "top_ratio": (0.50, 0.30)
    },
    "wavy2.png": {
        "left_ratio": (0.20, 0.42),
        "right_ratio": (0.80, 0.42),
        "top_ratio": (0.50, 0.15)
    }
}

def _hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _face_mask(image, landmarks):
    h, w = image.shape[:2]
    face_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    pts = np.array([landmarks[i] for i in face_idx], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_hair_color(image, landmarks, color_hex='#000000', intensity=0.5):
    h, w = image.shape[:2]
    color_bgr = _hex_to_bgr(color_hex)

    top_head     = landmarks[10]
    chin         = landmarks[152]
    left_temple  = landmarks[234]
    right_temple = landmarks[454]

    face_w = abs(right_temple[0] - left_temple[0])
    face_h = abs(chin[1] - top_head[1])
    cx = (left_temple[0] + right_temple[0]) // 2
    cy = (top_head[1] + chin[1]) // 2

    limit_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(limit_mask, (cx, cy - face_h // 8),
                (int(face_w * 0.7), int(face_h * 0.72)), 0, 0, 360, 255, -1)
    above_rect_y = max(0, top_head[1] - int(face_h * 0.5))
    cv2.rectangle(limit_mask,
                  (cx - int(face_w * 0.65), above_rect_y),
                  (cx + int(face_w * 0.65), top_head[1] + int(face_h * 0.1)),
                  255, -1)
    limit_mask = cv2.GaussianBlur(limit_mask, (41, 41), 0)

    face_m = _face_mask(image, landmarks)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hair_color_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([180, 255, 120]))

    hair_mask = cv2.bitwise_and(hair_color_mask, limit_mask)
    hair_mask = cv2.subtract(hair_mask, face_m)
    hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 0)

    color_layer = np.full_like(image, color_bgr, dtype=np.float32)
    alpha = (hair_mask.astype(np.float32) / 255.0 * intensity)[:, :, None]

    result = image.astype(np.float32) * (1 - alpha) + color_layer * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def detect_wig_anchors(overlay_img, filename=None):
    """
    Peruk görselinin şakak ve saç çizgisi referans noktalarını otomatik olarak algılar.
    Özellikle saçın iç kısmındaki 'yüz boşluğu/delik' sınırlarını hassas şekilde tarar
    böylece saçın yüzü kapatması engellenmiş olur.
    """
    h_wig, w_wig = overlay_img.shape[:2]

    # 1. Preset kontrolü (Dosya adıyla kayıtlı hassas oranlar varsa öncelikli kullan)
    if filename and filename in HAIRSTYLE_ANCHOR_PRESETS:
        preset = HAIRSTYLE_ANCHOR_PRESETS[filename]
        l_x, l_y = preset["left_ratio"]
        r_x, r_y = preset["right_ratio"]
        t_x, t_y = preset["top_ratio"]
        return (
            (int(w_wig * l_x), int(h_wig * l_y)),
            (int(w_wig * r_x), int(h_wig * r_y)),
            (int(w_wig * t_x), int(h_wig * t_y))
        )

    # 2. Saydamlık (Alpha) kanalı ayrıştırma
    if overlay_img.shape[2] == 4:
        alpha = overlay_img[:, :, 3]
    else:
        gray = cv2.cvtColor(overlay_img[:, :, :3], cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Saç piksel sınırlarını belirleme
    y_indices, x_indices = np.where(alpha > 30)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return (
            (int(w_wig * 0.25), int(h_wig * 0.45)),
            (int(w_wig * 0.75), int(h_wig * 0.45)),
            (int(w_wig * 0.50), int(h_wig * 0.30))
        )

    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()

    # Saç çizgisi Y-koordinatını bulma (Dikey merkez bandı taraması)
    center_x = (xmin + xmax) // 2
    center_start = int(w_wig * 0.45)
    center_end = int(w_wig * 0.55)
    
    y_hairline = ymin + int((ymax - ymin) * 0.30)  # Başlangıç tahmini
    for y in range(ymin + int((ymax - ymin) * 0.05), ymax - int((ymax - ymin) * 0.15)):
        row_alpha = alpha[y, center_start:center_end]
        if np.mean(row_alpha) < 50:  # Alın açıklığı başladıysa
            y_hairline = y
            break

    # Şakak seviyesi dikey Y-koordinatı (saç çizgisinin hemen altı)
    y_temple = min(ymax - 5, y_hairline + int((ymax - ymin) * 0.05))

    # --- HASSAS İÇ YÜZ BOŞLUĞU TARAMASI (NEW) ---
    # Merkezden sola doğru tarayarak saçın sol iç kenarını bulur
    x_left_inner = xmin + int((xmax - xmin) * 0.22)  # Güvenli fallback
    for x in range(center_x, xmin, -1):
        if alpha[y_temple, x] > 100:  # Saç dokusuna çarptık
            x_left_inner = x
            break

    # Merkezden sağa doğru tarayarak saçın sağ iç kenarını bulur
    x_right_inner = xmin + int((xmax - xmin) * 0.78)  # Güvenli fallback
    for x in range(center_x, xmax):
        if alpha[y_temple, x] > 100:  # Saç dokusuna çarptık
            x_right_inner = x
            break

    # Koruma Marjı: Algılanan iç boşluk çok darsa görselin aşırı esnememesi için koruma uygula
    min_gap = int(w_wig * 0.18)
    if (x_right_inner - x_left_inner) < min_gap:
        x_left_inner = center_x - int(w_wig * 0.16)
        x_right_inner = center_x + int(w_wig * 0.16)

    # Tepe noktası alnın biraz üstünde konumlanır
    y_top = ymin + int((y_hairline - ymin) * 0.30)

    P_wig_left = (x_left_inner, y_temple)
    P_wig_right = (x_right_inner, y_temple)
    P_wig_top = (center_x, y_top)

    print(f"[HAIR SCAN] Filename: {filename}")
    print(f"[HAIR SCAN] BBox: x=[{xmin},{xmax}], y=[{ymin},{ymax}]")
    print(f"[HAIR SCAN] Inner Face Gap: [{x_left_inner} - {x_right_inner}] Width: {x_right_inner - x_left_inner}")

    return P_wig_left, P_wig_right, P_wig_top


def apply_hair_overlay(image, landmarks, overlay_path, intensity=1.0, scale_factor=1.0, x_offset=0, y_offset=0, wig_color=None, wig_color_intensity=0.0):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return image

    h, w = image.shape[:2]

    # 1. Arka plan temizliği (Fake dama tahtası ve beyaz zemin ayıklama)
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
    else:
        b, g, r = cv2.split(overlay[:, :, :3])
        a = np.full(b.shape, 255, dtype=np.uint8)

    gray = cv2.cvtColor(overlay[:, :, :3], cv2.COLOR_BGR2GRAY)
    diff_rg = np.abs(r.astype(int) - g.astype(int))
    diff_gb = np.abs(g.astype(int) - b.astype(int))
    is_gray = (diff_rg < 15) & (diff_gb < 15)
    is_light = gray > 170
    
    bg_mask = np.where(is_gray & is_light, 0, 255).astype(np.uint8)
    bg_mask = cv2.GaussianBlur(bg_mask, (3, 3), 0)
    _, bg_mask = cv2.threshold(bg_mask, 127, 255, cv2.THRESH_BINARY)
    a = cv2.bitwise_and(a, bg_mask)
    overlay = cv2.merge([b, g, r, a])

    # 2. Kaynak referans noktalarını (iç yüz boşluğuna göre) otomatik bul
    P_wig_left, P_wig_right, P_wig_top = detect_wig_anchors(overlay, os.path.basename(overlay_path))

    # 3. Yüz üzerindeki hedef referans noktalarını belirle
    P_face_left = np.array(landmarks[234], dtype=np.float32)   # Sol Şakak
    P_face_right = np.array(landmarks[454], dtype=np.float32)  # Sağ Şakak
    top_head = np.array(landmarks[10], dtype=np.float32)       # Kaş Üstü/Alın Orta
    chin = np.array(landmarks[152], dtype=np.float32)          # Çene

    # Dikey yüz yüksekliğini kullanarak gerçek saç çizgisini (hairline) yukarı doğru ekstrapole et
    face_h = np.linalg.norm(chin - top_head)
    face_axis = top_head - chin
    face_axis_norm = face_axis / (np.linalg.norm(face_axis) + 1e-6)
    P_face_top = top_head + face_axis_norm * (face_h * 0.08)

    # 4. Yüz merkezini ve yaw (kafa dönüşü) sapmasını hesapla
    face_center = (P_face_left + P_face_right) / 2.0
    if len(landmarks) > 1:
        nose_tip = np.array(landmarks[1], dtype=np.float32)
        yaw_offset = (nose_tip[0] - face_center[0]) * 0.3
        face_center[0] += yaw_offset

    # 5. Kullanıcı slider ayarlarını (scale ve offset) hedef noktalara yedir
    # 1.05 katsayısı saç şakak çizgisinin kafadan hafif dışarıda, doğal durması içindir
    target_left = face_center + (P_face_left - face_center) * scale_factor * 1.05
    target_right = face_center + (P_face_right - face_center) * scale_factor * 1.05
    target_top = face_center + (P_face_top - face_center) * scale_factor

    offset_vector = np.array([x_offset, y_offset], dtype=np.float32)
    target_left += offset_vector
    target_right += offset_vector
    target_top += offset_vector

    # 6. 2B Affin Dönüşüm matrisini hesapla
    src_pts = np.float32([P_wig_left, P_wig_right, P_wig_top])
    dst_pts = np.float32([target_left, target_right, target_top])
    M = cv2.getAffineTransform(src_pts, dst_pts)

    # 7. Renk değişimi (İsteğe bağlı)
    if wig_color and wig_color_intensity > 0:
        color_bgr = _hex_to_bgr(wig_color)
        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        color_layer = np.full_like(overlay_rgb, color_bgr, dtype=np.float32)
        blended_rgb = overlay_rgb * (1.0 - wig_color_intensity) + color_layer * wig_color_intensity
        overlay[:, :, :3] = np.clip(blended_rgb, 0, 255).astype(np.uint8)

    # 8. Saçı tüm resmi kapsayacak şekilde tek seferde Affin Warp ile döndür/ölçekle
    warped_overlay = cv2.warpAffine(
        overlay, M, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0, 0)
    )

    # 9. Katmanları karıştır (Soft edges oluşturmak için adaptif Gaussian Blur kullanılmıştır)
    crop_rgb = warped_overlay[:, :, :3]
    crop_alpha = warped_overlay[:, :, 3:4].astype(np.float32) / 255.0

    blur_size = max(5, int(face_h * 0.03))
    if blur_size % 2 == 0: 
        blur_size += 1
    crop_alpha = cv2.GaussianBlur(crop_alpha, (blur_size, blur_size), 0)
    if len(crop_alpha.shape) == 2:
        crop_alpha = crop_alpha[:, :, np.newaxis]

    crop_alpha = np.clip(crop_alpha * intensity, 0.0, 1.0)

    # Nihai görseli birleştir
    result = image.astype(np.float32)
    result = result * (1.0 - crop_alpha) + crop_rgb.astype(np.float32) * crop_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


# Eski rotasyon fonksiyonu diğer modüllerle olası uyumluluk için korunmuştur.
def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    if image.shape[2] == 4:
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


EYEBROW_LANDMARKS = {
    "left":  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
}

def apply_eyebrow_color(image, landmarks, color_hex='#000000', intensity=0.5):
    h, w = image.shape[:2]
    color_bgr = _hex_to_bgr(color_hex)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for side, indices in EYEBROW_LANDMARKS.items():
        side_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(side_mask, [hull], 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        side_mask = cv2.dilate(side_mask, k, iterations=1)
        mask = cv2.bitwise_or(mask, side_mask)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark_weight = np.clip((0.5 - gray) / 0.3, 0.0, 1.0)
    combined = (mask.astype(np.float32) / 255.0) * dark_weight
    alpha = np.clip(combined * intensity * 0.9, 0.0, 0.9)
    color_layer = np.full_like(image, color_bgr, dtype=np.float32)
    result = image.astype(np.float32) * (1 - alpha[:, :, None]) + color_layer * alpha[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)