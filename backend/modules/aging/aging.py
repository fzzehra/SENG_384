from typing import List, Tuple
import cv2
import numpy as np

LandmarkList = List[Tuple[int, int]]

def apply_aging_effect(image, intensity=0.5, landmarks=None):
    if landmarks is None or len(landmarks) < 468:
        return image

    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]
    output = image.copy()

    # --- 1. YÜZ MASKESİ (Hassaslaştırılmış) ---
    # Alın dahil tüm yüzü kapsayan genişletilmiş nokta listesi
    face_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    face_points = np.array([landmarks[i] for i in face_idx], dtype=np.int32)
    
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_points], 255)
    
    # Alındaki keskinliği gidermek için devasa bir blur (Yumuşak geçiş şart)
    face_mask_blurred = cv2.GaussianBlur(face_mask, (101, 101), 0)
    f_mask_3d = (face_mask_blurred.astype(np.float32) / 255.0)[:, :, None]

    # --- 2. SAÇ MASKESİ (Dışarı Taşmayı Önleme) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Ten rengini dışarıda bırakmak için S ve V değerlerini daralttık
    hair_color_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([180, 255, 120]))
    
    # Saçın arka plana taşmasını önlemek için yüzün 1.5 katı büyüklüğünde bir alan belirle
    hair_limit_mask = np.zeros((h, w), dtype=np.uint8)
    center_x = landmarks[1][0] # Burun ucu civarı merkez
    center_y = landmarks[1][1]
    cv2.circle(hair_limit_mask, (center_x, center_y), int(w * 0.6), 255, -1)
    
    # Sadece kafa çevresindeki koyu alanları saç kabul et
    final_hair_mask = cv2.bitwise_and(hair_color_mask, hair_limit_mask)
    # Yüzü saç maskesinden tamamen temizle (Yüzün grileşmemesi için)
    final_hair_mask = cv2.subtract(final_hair_mask, face_mask)
    
    final_hair_mask = cv2.GaussianBlur(final_hair_mask, (31, 31), 0)
    h_mask_3d = (final_hair_mask.astype(np.float32) / 255.0)[:, :, None]

    # --- 3. EFEKTLERİ UYGULA ---
    
    # A) Yüz Kırıştırma (Doku Keskinleştirme)
    # Blur miktarını artırarak sadece büyük kırışıklıkları hedefle
    face_blur = cv2.GaussianBlur(output, (0, 0), 2)
    wrinkled_face = cv2.addWeighted(output, 1.5 + (intensity * 0.5), face_blur, -0.5 - (intensity * 0.5), 0)
    # Yüzü hafif solgunlaştır
    wrinkled_face = cv2.convertScaleAbs(wrinkled_face, alpha=1.0, beta=-5 * intensity)

    # B) Saç Grileştirme
    gray_img = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    # Daha doğal kırçıllı saç için 230 yerine image parlaklığına bağlı ton kullanıyoruz
    white_hair_tone = cv2.addWeighted(gray_img, 0.6, np.full_like(image, 210), 0.4, 0)

    # Birleştirme (Alpha Blending)
    # Önce yüzü yaşlandır
    output = (output * (1 - f_mask_3d) + wrinkled_face * f_mask_3d).astype(np.uint8)
    # Sonra saçları grileştir
    output = (output * (1 - h_mask_3d * intensity) + white_hair_tone * (h_mask_3d * intensity)).astype(np.uint8)

    return output