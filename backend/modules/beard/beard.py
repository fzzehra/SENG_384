import cv2
import numpy as np
import random
import math

def apply_beard_effect(image, landmarks, intensity=0.7, hair_len=9, color=(4, 4, 4)):
    if landmarks is None:
        return image

    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)

    coords = []
    for lm in landmarks:
        lx, ly = lm[0], lm[1]
        if lx <= 1.0 and ly <= 1.0:
            coords.append((int(lx * w), int(ly * h)))
        else:
            coords.append((int(lx), int(ly)))

    def get_pts(idxs):
        return np.array([coords[i] for i in idxs], np.int32)

    # 1. BÖLGELER

    # çene + jawline
    CHIN_ZONE = [
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
        377, 400, 378, 379, 365, 397, 288, 361, 323, 454
    ]

    # sol yanak: favoriden ağız köşesine doğru inen klasik sakal çizgisi
    CHEEK_L = [
        234, 227, 137, 177, 215, 138, 135, 169, 170,
        140, 171, 208, 32, 194, 211, 61, 58, 172
    ]

    # sağ yanak
    CHEEK_R = [
        454, 447, 366, 401, 435, 367, 364, 394, 395,
        369, 396, 428, 262, 418, 431, 291, 288, 397
    ]

    # bıyık: buruna taşmadan üst dudak çevresi
    MUSTACHE_ZONE = [
        61, 185, 40, 39, 37, 0,
        267, 269, 270, 409, 291,
        164, 165, 167, 391, 393
    ]

    LIPS = [
        61, 185, 40, 39, 37, 0,
        267, 269, 270, 409, 291,
        375, 321, 405, 314, 17,
        84, 181, 91, 146
    ]

    mask = np.zeros((h, w), dtype=np.uint8)

    chin_mask = np.zeros((h, w), dtype=np.uint8)
    cheek_mask = np.zeros((h, w), dtype=np.uint8)
    mustache_mask = np.zeros((h, w), dtype=np.uint8)
    # Favori / kulak önü uzantısı
    sideburn_mask = np.zeros((h, w), dtype=np.uint8)

    LEFT_SIDEBURN = [234, 127, 162, 21, 54, 103, 67, 109, 227, 137]
    RIGHT_SIDEBURN = [454, 356, 389, 251, 284, 332, 297, 338, 447, 366]

    cv2.fillPoly(sideburn_mask, [get_pts(LEFT_SIDEBURN)], 255)
    cv2.fillPoly(sideburn_mask, [get_pts(RIGHT_SIDEBURN)], 255)

    # Çok yukarı taşmasın
    eye_y = min(coords[33][1], coords[263][1])
    sideburn_mask[:eye_y, :] = 0

    mask = cv2.bitwise_or(mask, sideburn_mask)

    cv2.fillPoly(chin_mask, [get_pts(CHIN_ZONE)], 255)
    cv2.fillPoly(cheek_mask, [get_pts(CHEEK_L)], 255)
    cv2.fillPoly(cheek_mask, [get_pts(CHEEK_R)], 255)
    cv2.fillPoly(mustache_mask, [get_pts(MUSTACHE_ZONE)], 255)

    # Yanak üst sınırı: elmacık kemiği seviyesine çıkmasın
    nose_y = coords[2][1]
    # Sert kesmek yerine sadece çok üst kısmı temizle
    cheek_mask[:nose_y - int(h * 0.04), :] = 0

    mask = cv2.bitwise_or(chin_mask, cheek_mask)
    mask = cv2.bitwise_or(mask, mustache_mask)
    lip_m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_m, [get_pts(LIPS)], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lip_m = cv2.dilate(lip_m, kernel, iterations=2)

    mask = cv2.subtract(mask, lip_m)
    # Sakalın üst sınırı: burun ortasının üstüne çıkmasın
    nose_y = coords[2][1]
    mask[:nose_y, :] = 0

    # Yanaklarda elmacık kemiğine taşmayı azalt
    left_eye_y = coords[33][1]
    right_eye_y = coords[263][1]
    eye_line_y = min(left_eye_y, right_eye_y)

    mask[:eye_line_y + int(h * 0.13), :] = 0
    # burun üstüne taşmayı engelle
    nose_y = coords[2][1]

    mask[:nose_y, :] = 0

    # Üst sakal çizgisini dağınık yap
    noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 9)
    noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edge_softener = cv2.inRange(noise, 80, 255)
    mask = cv2.bitwise_and(mask, edge_softener)

    mask_f = cv2.GaussianBlur(mask, (35, 35), 0).astype(np.float32) / 255.0

    hair_layer = np.zeros_like(image, dtype=np.float32)

    ys, xs = np.where(mask > 25)

    density = 0.38 + intensity * 0.42

    for i in range(len(xs)):
        if random.random() > density:
            continue

        x, y = xs[i], ys[i]
        local_strength = mask_f[y, x]

        if random.random() > local_strength:
            continue

        angle = math.radians(random.uniform(70, 110))

        dynamic_len = hair_len * (0.7 + intensity * 1.4)

        length = (
            dynamic_len *
            random.uniform(0.45, 1.25) *
            local_strength
        )

        x2 = int(x + length * math.cos(angle))
        y2 = int(y + length * math.sin(angle))

        v = random.randint(-18, 12)

        c = (
            int(np.clip(color[0] + v, 0, 255)),
            int(np.clip(color[1] + v, 0, 255)),
            int(np.clip(color[2] + v, 0, 255))
        )

        cv2.line(
            hair_layer,
            (x, y),
            (x2, y2),
            c,
            1,
            cv2.LINE_AA
        )

    shadow_alpha = (mask_f * intensity * 0.20)[:, :, None]
    output = output * (1.0 - shadow_alpha)

    final = output + (hair_layer * 1.25)

    return np.clip(final, 0, 255).astype(np.uint8)