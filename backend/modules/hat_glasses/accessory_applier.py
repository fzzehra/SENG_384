import cv2
import numpy as np
from typing import List, Tuple

def overlay_transparent(background, overlay, x, y, size=None):
    bg_img = background.copy()

    if overlay is None or overlay.shape[2] < 4:
        return bg_img

    if size:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)

    h, w = overlay.shape[:2]
    bg_h, bg_w = bg_img.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return bg_img

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    overlay_crop = overlay[oy1:oy2, ox1:ox2]

    overlay_rgb = overlay_crop[:, :, :3].astype(float)
    alpha = overlay_crop[:, :, 3].astype(float) / 255.0
    alpha = alpha[:, :, None]

    bg_region = bg_img[y1:y2, x1:x2].astype(float)

    blended = (1 - alpha) * bg_region + alpha * overlay_rgb
    bg_img[y1:y2, x1:x2] = blended.astype(np.uint8)

    return bg_img

def get_content_bbox(img_rgba):
    """Şeffaf olmayan piksellerin bounding box'ını döner."""
    alpha = img_rgba[:, :, 3]
    cols = np.any(alpha > 10, axis=0)
    rows = np.any(alpha > 10, axis=1)
    if not cols.any() or not rows.any():
        return 0, 0, img_rgba.shape[1], img_rgba.shape[0]
    x1, x2 = np.where(cols)[0][[0, -1]]
    y1, y2 = np.where(rows)[0][[0, -1]]
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

def apply_accessories(image, landmarks, hat_path=None, glasses_path=None):
    """
    Landmark noktalarına göre şapka ve gözlük ekler.
    """
    if landmarks is None or len(landmarks) < 100:
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
            eye_width = int(abs(right_eye[0] - left_eye[0]) * 1.55)
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
    # --- ŞAPKA YERLEŞİMİ ---
    if hat_path:
        hat_img = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
        if hat_img is not None:
            top_head = landmarks[10]
            left_face = landmarks[234]
            right_face = landmarks[454]

            face_width = abs(right_face[0] - left_face[0])

            # Şeffaf alanı çıkar, gerçek içerik genişliğini bul
            cx, cy, cw, ch = get_content_bbox(hat_img)
            content_ratio = hat_img.shape[1] / max(cw, 1)  # canvas/içerik oranı

            # Hedef genişlik: yüz genişliğine göre
            hat_width = int(face_width * 1.6)
            # Canvas boyutunu içerik oranına göre ölçekle
            hat_width_canvas = int(hat_width * content_ratio)

            aspect_ratio = hat_img.shape[1] / max(hat_img.shape[0], 1)
            hat_height_canvas = int(hat_width_canvas / aspect_ratio)

            center_x = (left_face[0] + right_face[0]) // 2
            hat_x = center_x - (hat_width_canvas // 2)

            # İçerik alanının alt kısmını alın üstüne hizala
            content_bottom_ratio = (cy + ch) / max(hat_img.shape[0], 1)
            hat_y = int(top_head[1] - hat_height_canvas * content_bottom_ratio)
            hat_y -= int(h * 0.02)

            output = overlay_transparent(output, hat_img, hat_x, hat_y, 
                                        (hat_width_canvas, hat_height_canvas))
    return output