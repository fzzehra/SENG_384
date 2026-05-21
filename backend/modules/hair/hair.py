#hair.py 
import os
import cv2
import numpy as np

# Peruk PNG'lerinin saç-çizgisi anchor oranları (üstten yüzde).
# Yeni bir peruk eklendiğinde buraya kaydını yap; yoksa varsayılan kullanılır.
#HAIRSTYLE_ANCHORS = {
    # "long_wavy.png": 0.18,
    # "buzzcut.png":   0.30,
    # "afro.png":      0.10,
#}
#DEFAULT_HAIRLINE_ANCHOR = 0.20

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

    # Kafa elipsi: sadece baş bölgesini kapsar, omuz/gövde dışarıda
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


def apply_hair_overlay(image, landmarks, overlay_path, intensity=1.0, scale_factor=1.0, x_offset=0, y_offset=0, wig_color=None, wig_color_intensity=0.0):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return image

    # === DEBUG (geçici) ===
    print(f"[HAIR DEBUG] PNG: {os.path.basename(overlay_path)}")
    print(f"[HAIR DEBUG] PNG size: {overlay.shape[1]}x{overlay.shape[0]}")
    
    # Remove baked-in background (white or checkerboard) if needed
    b, g, r = cv2.split(overlay[:, :, :3])
    gray = cv2.cvtColor(overlay[:, :, :3], cv2.COLOR_BGR2GRAY)
    
    # Fake transparency checkerboards are made of white and light gray squares.
    # They are grayscale (R ~= G ~= B) and light (value > 170).
    diff_rg = np.abs(r.astype(int) - g.astype(int))
    diff_gb = np.abs(g.astype(int) - b.astype(int))
    is_gray = (diff_rg < 15) & (diff_gb < 15)
    is_light = gray > 170
    
    bg_mask = np.where(is_gray & is_light, 0, 255).astype(np.uint8)
    
    # Smooth the mask slightly to avoid jagged edges on the hair
    bg_mask = cv2.GaussianBlur(bg_mask, (3, 3), 0)
    _, bg_mask = cv2.threshold(bg_mask, 127, 255, cv2.THRESH_BINARY)
    
    if overlay.shape[2] == 4:
        # Combine existing alpha with our new background mask
        overlay[:, :, 3] = cv2.bitwise_and(overlay[:, :, 3], bg_mask)
    else:
        # Create new alpha from mask
        overlay = cv2.merge([b, g, r, bg_mask])


    # === PERUĞUN İÇ BOŞLUĞUNU ÖLÇ ===
    # PNG'nin orta kısmında yüzün geçeceği transparan delik var.
    # Bu deliğin genişliğini ölçüp ona göre ölçekleme yapacağız.
    inner_width_ratio = 1.0  # varsayılan: iç delik yoksa dış genişlik kullanılır
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3]
        oh, ow = alpha.shape

        # Dikey ortadan al — saçın "yüz seviyesi" bandı
        # Üstten %40 ile %75 arası genelde yüzün durduğu yer
        mid_band = alpha[int(oh * 0.40):int(oh * 0.75), :]

        # Bu bandın her satırındaki transparan piksel sayısının ortalaması
        # = ortadaki deliğin yatay genişliği
        transparent_per_row = np.sum(mid_band < 50, axis=1)  # her satırda kaç transparan piksel
        if len(transparent_per_row) > 0:
            avg_hole_width = float(np.mean(transparent_per_row))
            inner_width_ratio = avg_hole_width / float(ow)

        print(f"[HAIR DEBUG] inner hole ratio: {inner_width_ratio:.2f} (delik PNG'nin %{int(inner_width_ratio*100)}'i)")

    h, w = image.shape[:2]

    # === ANCHOR LANDMARK'LAR ===
    # MediaPipe Face Mesh referansı:
    #   10  = forehead top (kaş üstü, hairline DEĞİL)
    #   152 = chin
    #   234 / 454 = sol/sağ temple (kulak hizası)
    #   127 / 356 = sol/sağ yüz dış kenarı (temple'dan biraz daha geniş)
    #    33 /  263 = sol/sağ göz dış köşesi
    top_head     = np.array(landmarks[10],  dtype=np.float32)
    chin         = np.array(landmarks[152], dtype=np.float32)
    left_temple  = np.array(landmarks[234], dtype=np.float32)
    right_temple = np.array(landmarks[454], dtype=np.float32)
    left_eye     = np.array(landmarks[33],  dtype=np.float32)
    right_eye    = np.array(landmarks[263], dtype=np.float32)

    # === 1) ROTASYON: gözler arası açı (temple yerine, daha güvenilir) ===
    eye_delta = right_eye - left_eye
    angle_deg = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))

    # === 2) YÜZ GENİŞLİĞİ (rotasyondan bağımsız, gerçek mesafe) ===
    # Sadece x farkı değil, Euclid mesafe — eğik kafalarda doğru çalışır
    face_w = float(np.linalg.norm(right_temple - left_temple))
    face_h = float(np.linalg.norm(chin - top_head))

    # === 3) YÜZ MERKEZİ (rotasyon-aware) ===
    # Temple orta noktası + biraz nose ağırlığı, ama rotasyona göre düzelt
    face_center = (left_temple + right_temple) / 2.0
    if len(landmarks) > 1:
        nose_tip = np.array(landmarks[1], dtype=np.float32)
        face_center = 0.75 * face_center + 0.25 * nose_tip
    cx = float(face_center[0])

    # === 4) HAIRLINE TAHMİNİ ===
    # Kritik nokta: landmarks[10] "alın üstü" değil, kaşların ~2-3cm üstü.
    # Gerçek saç çizgisi yüz yüksekliğinin yaklaşık %25-30'u kadar yukarıda.
    # top_head'den chin'e olan vektörün TERS yönünde, face_h * 0.30 kadar ekstrapole ediyoruz.
    face_axis = top_head - chin                      # çeneden alına vektör
    face_axis_norm = face_axis / (np.linalg.norm(face_axis) + 1e-6)
    hairline_point = top_head + face_axis_norm * (face_h * 0.10)
    hairline_y = float(hairline_point[1])
    # cx'i de hairline noktasının x'iyle harmanla — eğik kafalarda hizalamayı düzeltir
    cx = 0.5 * cx + 0.5 * float(hairline_point[0])
    # Yaw tahmini: burun-çene ekseninin yüz merkez ekseninden sapması
    nose_to_chin = np.array(landmarks[152]) - np.array(landmarks[1])
    face_midline = (left_temple + right_temple) / 2.0 - chin
    # Eğer kafa yana dönükse peruğu o yöne kaydır
    yaw_offset_x = (nose_tip[0] - face_center[0]) * 0.3
    cx += yaw_offset_x

    # === 5) PERUK GENİŞLİĞİ ===
    # Peruk yüzden biraz geniş olmalı (saç yüz konturundan dışarı taşar).
    # 1.45 katsayısı çoğu PNG için iyi; ama PNG'nin kendi en/boy oranını da hesaba kat.
    # === ASIL ÖLÇEKLEME: İç delik yüze eşit olacak şekilde ölçekle ===
    # Hedef: peruğun iç deliği ≈ face_w × 1.05 (yüzden %5 geniş, hafif boşluk için)
    # Yani: target_w × inner_width_ratio = face_w × 1.05
    # Buradan: target_w = (face_w × 1.05) / inner_width_ratio
    if inner_width_ratio > 0.15:  # makul bir delik varsa
        target_w = int((face_w * 1.05 * scale_factor) / inner_width_ratio)
    else:
        # Delik yok veya çok küçük → düz dış ölçekleme
        target_w = int(face_w * 1.15 * scale_factor)
    target_w = max(target_w, 1)

    print(f"[HAIR DEBUG] target_w computed: {target_w} (face_w={face_w:.0f}, ratio={inner_width_ratio:.2f})")
    print(f"[HAIR DEBUG] face_w={face_w:.1f}, target_w={target_w}, scale_factor={scale_factor}")
    print(f"[HAIR DEBUG] hairline_y={hairline_y:.1f}, top_head.y={top_head[1]:.1f}, chin.y={chin[1]:.1f}")
    scale = target_w / float(overlay.shape[1])
    target_h = max(int(overlay.shape[0] * scale), 1)

    overlay_resized = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)

    overlay_rotated = _rotate_image(overlay_resized, angle_deg)

    x_start = int(cx - overlay_rotated.shape[1] // 2 + x_offset)
    HAIRLINE_ANCHOR_RATIO = 0.20
    y_start = int(hairline_y - overlay_rotated.shape[0] * HAIRLINE_ANCHOR_RATIO + y_offset)

    #overlay_filename = os.path.basename(overlay_path)
#HAIRLINE_ANCHOR_RATIO = HAIRSTYLE_ANCHORS.get(overlay_filename, DEFAULT_HAIRLINE_ANCHOR)
#y_start = int(hairline_y - overlay_rotated.shape[0] * HAIRLINE_ANCHOR_RATIO) + y_offset

    x1 = max(0, x_start)
    y1 = max(0, y_start)
    x2 = min(w, x_start + overlay_rotated.shape[1])
    y2 = min(h, y_start + overlay_rotated.shape[0])

    if x1 >= x2 or y1 >= y2:
        return image

    ox1 = x1 - x_start
    oy1 = y1 - y_start

    crop = overlay_rotated[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)]

    if crop.size == 0:
        return image

    if wig_color and wig_color_intensity > 0:
        bgr = _hex_to_bgr(wig_color)
        color_layer = np.full_like(crop[:, :, :3], bgr, dtype=np.float32)
        crop_rgb = crop[:, :, :3].astype(np.float32)
        # Blend the color into the wig
        blended = crop_rgb * (1.0 - wig_color_intensity) + color_layer * wig_color_intensity
        crop[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)

    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    # Resim çözünürlüğüne göre adaptif blur
    blur_size = max(5, int(face_w * 0.04))
    if blur_size % 2 == 0: blur_size += 1  # tek sayı olmalı
    alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]
    alpha = np.clip(alpha * intensity, 0.0, 1.0)

    result = image.astype(np.float32)
    result[y1:y2, x1:x2] = (
        result[y1:y2, x1:x2] * (1.0 - alpha) +
        crop[:, :, :3].astype(np.float32) * alpha
    )

    return np.clip(result, 0, 255).astype(np.uint8)
EYEBROW_LANDMARKS = {
    "left":  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
}

def apply_eyebrow_color(image, landmarks, color_hex='#000000', intensity=0.5):
    h, w = image.shape[:2]
    color_bgr = _hex_to_bgr(color_hex)

    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Her kaşı ayrı ayrı çiz — birbirine karışmasın
    for side, indices in EYEBROW_LANDMARKS.items():
        side_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(side_mask, [hull], 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        side_mask = cv2.dilate(side_mask, k, iterations=1)
        mask = cv2.bitwise_or(mask, side_mask)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Sadece koyu piksellere uygula
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark_weight = np.clip((0.5 - gray) / 0.3, 0.0, 1.0)

    combined = (mask.astype(np.float32) / 255.0) * dark_weight
    alpha = np.clip(combined * intensity * 0.9, 0.0, 0.9)

    color_layer = np.full_like(image, color_bgr, dtype=np.float32)
    result = image.astype(np.float32) * (1 - alpha[:, :, None]) + color_layer * alpha[:, :, None]

    return np.clip(result, 0, 255).astype(np.uint8)