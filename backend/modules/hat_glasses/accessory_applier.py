import cv2
import numpy as np
from typing import List, Tuple

def overlay_transparent(background, overlay, x, y, size=None):
    if size:
        overlay = cv2.resize(overlay, size)

    h, w = overlay.shape[:2]
    
    # --- TAŞMA KONTROLÜ (EKLEME) ---
    # Başlangıç noktası resmin dışındaysa kırp
    x_start, y_start = max(0, x), max(0, y)
    x_end, y_end = min(background.shape[1], x + w), min(background.shape[0], y + h)

    # Eğer aksesuar tamamen resmin dışındaysa direkt orijinal resmi döndür
    if x_start >= x_end or y_start >= y_end:
        return background

    # Bindirilecek parcanın overlay üzerindeki sınırlarını belirle
    overlay_x_start, overlay_y_start = x_start - x, y_start - y
    overlay_x_end, overlay_y_end = overlay_x_start + (x_end - x_start), overlay_y_start + (y_end - y_start)

    crop_overlay = overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
    crop_background = background[y_start:y_end, x_start:x_end]
    # ------------------------------

    overlay_img = crop_overlay[:, :, :3]
    mask = crop_overlay[:, :, 3] / 255.0
    mask_3d = np.repeat(mask[:, :, None], 3, axis=2)

    # Vektörel çarpım (Daha hızlıdır)
    blended = (1.0 - mask_3d) * crop_background + mask_3d * overlay_img
    background[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)

    return background

def apply_accessories(image, landmarks, hat_path=None, glasses_path=None):
    """
    Landmark noktalarına göre şapka ve gözlük ekler.
    """
    if landmarks is None or len(landmarks) < 468:
        return image

    output = image.copy()
    h, w = image.shape[:2]

    # --- GÖZLÜK YERLEŞİMİ ---
    if glasses_path:
        glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        if glasses_img is not None:
            # Göz landmarkları: 33 (Sol dış), 263 (Sağ dış)
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            
            # Gözlük genişliği iki göz arası mesafeye göre (biraz pay ekleyerek)
            eye_width = int(abs(right_eye[0] - left_eye[0]) * 1.8)
            # Gözlük yüksekliği (oranı koruyarak)
            aspect_ratio = glasses_img.shape[1] / glasses_img.shape[0]
            eye_height = int(eye_width / aspect_ratio)
            
            # Yerleşim noktası (Gözlerin ortası)
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            
            top_left_x = center_x - (eye_width // 2)
            top_left_y = center_y - (eye_height // 2)
            
            output = overlay_transparent(output, glasses_img, top_left_x, top_left_y, (eye_width, eye_height))

    # --- ŞAPKA YERLEŞİMİ ---
    if hat_path:
        hat_img = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
        if hat_img is not None:
            # Alın üstü (10) ve yüz genişliği (234, 454)
            top_head = landmarks[10]
            left_face = landmarks[234]
            right_face = landmarks[454]
            
            face_width = int(abs(right_face[0] - left_face[0]) * 2.2)
            aspect_ratio = hat_img.shape[1] / hat_img.shape[0]
            hat_height = int(face_width / aspect_ratio)
            
            # Yerleşim noktası (Alnın biraz yukarısı)
            hat_x = top_head[0] - (face_width // 2)
            hat_y = top_head[1] - int(hat_height * 0.8) # Şapkanın tipine göre 0.8 ayarlanabilir
            
            output = overlay_transparent(output, hat_img, hat_x, hat_y, (face_width, hat_height))

    return output