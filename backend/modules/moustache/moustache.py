import cv2
import numpy as np
import random
import math

def apply_moustache(image, landmarks, intensity=0.7, hair_len=7, color=(15, 15, 15)):
    if landmarks is None:
        return image

    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)

    # Landmark koordinatlarını hazırla
    coords = []
    for lm in landmarks:
        lx, ly = lm[0], lm[1]
        if lx <= 1.0 and ly <= 1.0:
            coords.append((int(lx * w), int(ly * h)))
        else:
            coords.append((int(lx), int(ly)))

    def get_pts(idxs):
        return np.array([coords[i] for i in idxs], np.int32)

    # --- 1. BIYIK BÖLGESİ TANIMLAMA ---
    # MediaPipe Face Mesh indekslerine göre üst dudak ve burun altı bölgesi
    MUSTACHE_ZONE = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, # Üst sınır (dudak üstü)
        306, 291, 427, 434, 430, 431, 262, 428, 199, 194, 208, 171, 32, 211, 214, 61 # Alt/Yan sınırlar
    ]
    
    # Dudakların tam üzerine kıl gelmemesi için koruma bölgesi
    LIPS_INNER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [get_pts(MUSTACHE_ZONE)], 255)
    
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [get_pts(LIPS_INNER)], 255)
    
    # Dudak bölgesini temizle ve biraz daralt
    mask = cv2.subtract(mask, lip_mask)

    # Yumuşatma ve kenar geçişleri
    mask_f = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0

    # --- 2. ÇİZİM KATMANI ---
    hair_layer = np.zeros_like(image, dtype=np.float32)
    ys, xs = np.where(mask > 50)

    # Yoğunluk ayarı
    density = 0.4 + intensity * 0.45
    nose_tip_x = coords[1][0] # Burun ucu referansı

    for i in range(len(xs)):
        if random.random() > density:
            continue

        x, y = xs[i], ys[i]
        local_strength = mask_f[y, x]

        if random.random() > local_strength:
            continue

        # --- 3. DOĞAL YÖNLENDİRME ---
        # Bıyık telleri merkezden dışa ve hafif aşağı doğru uzar
        center_x = coords[1][0] # Burun merkezi
        if x < center_x:
            # Sol taraf: Sola ve aşağı (110-160 derece arası)
            angle = math.radians(random.uniform(100, 150))
        else:
            # Sağ taraf: Sağa ve aşağı (30-80 derece arası)
            angle = math.radians(random.uniform(30, 80))

        # Dinamik uzunluk
        length = hair_len * (0.6 + intensity) * random.uniform(0.5, 1.1) * local_strength

        x2 = int(x + length * math.cos(angle))
        y2 = int(y + length * math.sin(angle))

        # Renk çeşitliliği (doğal görünüm için hafif ton farkları)
        v = random.randint(-15, 15)
        c = (
            int(np.clip(color[0] + v, 0, 255)),
            int(np.clip(color[1] + v, 0, 255)),
            int(np.clip(color[2] + v, 0, 255))
        )

        cv2.line(hair_layer, (x, y), (x2, y2), c, 1, cv2.LINE_AA)

    # --- 4. BİRLEŞTİRME ---
    # Hafif bir gölge efekti (deriyle bütünleşmesi için)
    shadow_alpha = (mask_f * intensity * 0.25)[:, :, None]
    output = output * (1.0 - shadow_alpha)

    # Kılları ekle (biraz keskinlik için 1.3 ile çarptık)
    final = output + (hair_layer * 1.3)

    return np.clip(final, 0, 255).astype(np.uint8)