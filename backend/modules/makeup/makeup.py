import cv2
import numpy as np


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def blend_mask(image, mask, color_bgr, intensity=0.5):
    intensity = max(0.0, min(1.0, float(intensity)))

    output = image.copy().astype(np.float32)
    color_layer = np.full_like(output, color_bgr, dtype=np.float32)

    alpha = (mask.astype(np.float32) / 255.0) * intensity

    for c in range(3):
        output[:, :, c] = output[:, :, c] * (1 - alpha) + color_layer[:, :, c] * alpha

    return np.clip(output, 0, 255).astype(np.uint8)


def get_point_xy(landmarks, idx, w, h):
    point = landmarks[idx]

    if hasattr(point, "x") and hasattr(point, "y"):
        return [int(point.x * w), int(point.y * h)]

    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
        return [int(point[0]), int(point[1])]

    raise ValueError(f"Unsupported landmark format at index {idx}: {point}")


def get_points(landmarks, indices, w, h):
    return np.array(
        [get_point_xy(landmarks, i, w, h) for i in indices],
        dtype=np.int32
    )


def apply_lip_color(image, landmarks, color_hex="#d96b86", intensity=0.5):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    upper_lip_idx = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78
    ]

    lower_lip_idx = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
    ]

    upper_pts = get_points(landmarks, upper_lip_idx, w, h)
    lower_pts = get_points(landmarks, lower_lip_idx, w, h)

    cv2.fillPoly(mask, [upper_pts], 255)
    cv2.fillPoly(mask, [lower_pts], 255)

    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    return blend_mask(image, mask, hex_to_bgr(color_hex), intensity)

def apply_blush(image, landmarks, color_hex="#f35c7a", intensity=0.32):
    import numpy as np
    import cv2

    h, w = image.shape[:2]
    def p(idx): return np.array(get_point_xy(landmarks, idx, w, h), dtype=np.float32)

    face_width = np.linalg.norm(p(454) - p(234))
    face_height = np.linalg.norm(p(10) - p(152))

    # --- 1. MERKEZ: TAM ELMACIK (yukarı + içe) ---
    # sol yanak (izleyicinin sağı): 123 = göz altı, 101 = buruna yakın iç yanak
    left_center = p(123)*0.50 + p(101)*0.25 + p(50)*0.15 + p(205)*0.10
    right_center = p(352)*0.50 + p(330)*0.25 + p(280)*0.15 + p(425)*0.10

    # gözün hemen altına otursun (eskiden fazla aşağı itiyorduk)
    left_eye_bottom = p(145)[1]
    right_eye_bottom = p(374)[1]
    left_center[1] = max(left_center[1], left_eye_bottom + face_height*0.035)
    right_center[1] = max(right_center[1], right_eye_bottom + face_height*0.035)

    def build_round(center):
        mask = np.zeros((h, w), dtype=np.float32)
        r = face_width * 0.125 # kontur için küçülttük

        cv2.circle(mask, (int(center[0]), int(center[1])), int(r), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=r*0.55, sigmaY=r*0.5)
        return mask

    left_mask = build_round(left_center)
    right_mask = build_round(right_center)
    combined = np.clip(left_mask + right_mask, 0, 1)

    # --- 2. GÖZ KALKANI ---
    left_eye_c = (p(33) + p(133)) * 0.5
    right_eye_c = (p(362) + p(263)) * 0.5
    eye_mask = np.zeros((h,w), np.uint8)
    eye_r = int(face_width * 0.12)
    cv2.circle(eye_mask, (int(left_eye_c[0]), int(left_eye_c[1])), eye_r, 255, -1)
    cv2.circle(eye_mask, (int(right_eye_c[0]), int(right_eye_c[1])), eye_r, 255, -1)
    eye_f = cv2.GaussianBlur(eye_mask, (0,0), sigmaX=eye_r*0.5).astype(np.float32)/255.0
    combined *= (1.0 - eye_f * 0.95)

    # --- 3. BURUN + AĞIZ ---
    nose_center = p(1)
    nose_shield = np.zeros((h,w), np.uint8)
    cv2.ellipse(nose_shield, (int(nose_center[0]), int(nose_center[1])),
                (int(face_width*0.09), int(face_height*0.11)), 0,0,360,255,-1)
    nose_f = cv2.GaussianBlur(nose_shield, (0,0), sigmaX=face_width*0.06).astype(np.float32)/255.0
    combined *= (1.0 - nose_f*0.7)

    # --- 4. YÜZ SINIRI ---
    face_outline_idx = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
    face_poly = get_points(landmarks, face_outline_idx, w, h)
    face_mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(face_mask, [face_poly], 255)
    face_eroded = cv2.erode(face_mask, np.ones((7,7), np.uint8), iterations=2)
    face_soft = cv2.GaussianBlur(face_eroded, (0,0), sigmaX=face_width*0.025).astype(np.float32)/255.0
    combined *= face_soft

    combined = cv2.GaussianBlur(combined, (0,0), sigmaX=face_width*0.02)
    combined = np.power(combined, 0.95)

    mask_uint8 = (np.clip(combined,0,1) * 255).astype(np.uint8)

    if intensity > 1.5:  # 0-100 aralığı
            slider_norm = intensity / 100.0
    else:                # 0-1 aralığı
            slider_norm = intensity
        
    slider_norm = np.clip(slider_norm, 0, 1)
    # 1.7 kuvvet = başta çok yumuşak, sonda yavaş artar
    # 0.58 = maksimum opaklık (100 bile %58)
    final_intensity = (slider_norm ** 1.7) * 0.58
    return blend_mask(image, mask_uint8, hex_to_bgr(color_hex), np.clip(intensity,0,1))

def apply_eyeshadow(image, landmarks, color_hex="#b565a7", intensity=0.35):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    left_lid_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133]
    right_lid_idx = [263, 466, 388, 387, 386, 385, 384, 398, 362]

    left_base = get_points(landmarks, left_lid_idx, w, h)
    right_base = get_points(landmarks, right_lid_idx, w, h)

    lift = int(h * 0.018)

    left_top = left_base.copy()
    left_top[:, 1] -= lift

    right_top = right_base.copy()
    right_top[:, 1] -= lift

    left_shadow = np.vstack([left_top, left_base[::-1]])
    right_shadow = np.vstack([right_top, right_base[::-1]])

    cv2.fillPoly(mask, [left_shadow.astype(np.int32)], 255)
    cv2.fillPoly(mask, [right_shadow.astype(np.int32)], 255)

    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return blend_mask(image, mask, hex_to_bgr(color_hex), intensity)

def apply_eyeliner(image, landmarks, color_hex="#0a9e5c", intensity=0.85, wing=True, thickness=0.5):
    import numpy as np
    import cv2

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def pt(i):
        return np.array(get_point_xy(landmarks, i, w, h), dtype=np.float32)

    # --- RESİMDEKİ İNCELİK ---
    base_px = 1 + thickness * 0.6 # iç kısım ~1.3px
    max_px = 2 + thickness * 1.2 # dış kısım ~2.6px

    left_lash_idx = [133, 173, 157, 158, 159, 160, 161, 246, 33]
    right_lash_idx = [362, 398, 384, 385, 386, 387, 388, 466, 263]

    def draw_eye_liner(lash_idx):
        pts = np.array([pt(i) for i in lash_idx], dtype=np.float32)
        # iç köşenin %20'sini atla - resimdeki gibi
        start = int(len(pts) * 0.22)
        pts = pts[start:]
        # kirpik dibinin 1px üstüne al
        pts[:, 1] -= h * 0.0022

        n = len(pts)
        for i in range(n - 1):
            prog = i / (n - 1)
            thick = int(np.clip(base_px + prog * (max_px - base_px), 1, 4))
            p1 = pts[i].astype(np.int32)
            p2 = pts[i+1].astype(np.int32)
            cv2.line(mask, tuple(p1), tuple(p2), 255, thick, cv2.LINE_AA)
        return pts

    left_pts = draw_eye_liner(left_lash_idx)
    right_pts = draw_eye_liner(right_lash_idx)

    if wing:
        def draw_wing(pts):
            outer = pts[-1]
            ref = pts[-3]
            eye_w = np.linalg.norm(pts[-1] - pts[0])
            wing_len = eye_w * 0.38 # resimdeki uzunluk

            dir_vec = (outer - ref) / (np.linalg.norm(outer - ref) + 1e-6)
            lift = np.array([0, -eye_w * 0.26]) # yukarı kalkış
            tip = outer + dir_vec * wing_len + lift

            # tek çizgi, 3px -> 1px incelen
            steps = 14
            for i in range(steps):
                t0 = i / steps
                t1 = (i+1) / steps
                p1 = outer + (tip - outer) * t0
                p2 = outer + (tip - outer) * t1
                thick = max(1, int(3.2 * (1 - t1) + 0.8))
                cv2.line(mask, tuple(p1.astype(int)), tuple(p2.astype(int)), 255, thick, cv2.LINE_AA)

        draw_wing(left_pts)
        draw_wing(right_pts)

    # göz içine taşmayı kes
    left_eye_open_idx = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
    right_eye_open_idx = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
    eye_open = np.zeros((h,w), np.uint8)
    cv2.fillPoly(eye_open, [np.array([pt(i) for i in left_eye_open_idx], np.int32)], 255)
    cv2.fillPoly(eye_open, [np.array([pt(i) for i in right_eye_open_idx], np.int32)], 255)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(eye_open))

    # SADECE anti-alias için minik blur (senin kodda 2 büyük blur vardı)
    mask = cv2.GaussianBlur(mask, (3,3), 0)

    # blend
    mask_f = mask.astype(np.float32) / 255.0
    color_bgr = hex_to_bgr(color_hex)
    intensity = np.clip(float(intensity), 0, 1)

    out = image.astype(np.float32)
    for c in range(3):
        out[:,:,c] = out[:,:,c]*(1 - mask_f*intensity) + color_bgr[c]*mask_f*intensity

    return np.clip(out, 0, 255).astype(np.uint8)

def apply_eye_color(image, landmarks, color_hex="#4a90e2", intensity=0.5):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def pt(i):
        return np.array(get_point_xy(landmarks, i, w, h), dtype=np.int32)

    left_eye_idx = [33, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    right_eye_idx = [263, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

    left_eye_poly = np.array([pt(i) for i in left_eye_idx], dtype=np.int32)
    right_eye_poly = np.array([pt(i) for i in right_eye_idx], dtype=np.int32)

    eye_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(eye_mask, [left_eye_poly], 255)
    cv2.fillPoly(eye_mask, [right_eye_poly], 255)

    total_landmarks = len(landmarks)

    if total_landmarks > 477:
        left_iris_pts = np.array([pt(i) for i in range(468, 473)], dtype=np.int32)
        right_iris_pts = np.array([pt(i) for i in range(473, 478)], dtype=np.int32)

        left_center = np.mean(left_iris_pts, axis=0).astype(np.int32)
        right_center = np.mean(right_iris_pts, axis=0).astype(np.int32)

        left_radius = max(2, int(np.mean(np.linalg.norm(left_iris_pts - left_center, axis=1)) * 1.25))
        right_radius = max(2, int(np.mean(np.linalg.norm(right_iris_pts - right_center, axis=1)) * 1.25))

    else:
        left_corner_1 = pt(33)
        left_corner_2 = pt(133)
        right_corner_1 = pt(362)
        right_corner_2 = pt(263)

        left_center = ((left_corner_1 + left_corner_2) / 2).astype(np.int32)
        right_center = ((right_corner_1 + right_corner_2) / 2).astype(np.int32)

        left_radius = max(2, int(np.linalg.norm(left_corner_1 - left_corner_2) * 0.18))
        right_radius = max(2, int(np.linalg.norm(right_corner_1 - right_corner_2) * 0.18))

    iris_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_mask, tuple(left_center), left_radius, 255, -1)
    cv2.circle(iris_mask, tuple(right_center), right_radius, 255, -1)

    mask = cv2.bitwise_and(iris_mask, eye_mask)

    blur_size = max(7, int(min(h, w) * 0.01))
    if blur_size % 2 == 0:
        blur_size += 1

    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    final_intensity = max(0.0, min(1.0, intensity)) * 0.65

    return blend_mask(image, mask, hex_to_bgr(color_hex), final_intensity)


def apply_makeup_pipeline(image, landmarks, makeup_type, color_hex="#d96b86", intensity=0.5):
    if makeup_type == "lip_color":
        print("MAKEUP: applying lip color")
        return apply_lip_color(image, landmarks, color_hex=color_hex, intensity=intensity)

    elif makeup_type == "lipstick":
        print("MAKEUP: applying lipstick")
        return apply_lip_color(image, landmarks, color_hex=color_hex, intensity=intensity)

    elif makeup_type == "blush":
        print("MAKEUP: applying blush")
        return apply_blush(image, landmarks, color_hex=color_hex, intensity=intensity)

    elif makeup_type == "eyeshadow":
        print("MAKEUP: applying eyeshadow")
        return apply_eyeshadow(image, landmarks, color_hex=color_hex, intensity=intensity)

    elif makeup_type == "eye_color":
        print("MAKEUP: applying eye color")
        return apply_eye_color(image, landmarks, color_hex=color_hex, intensity=intensity)

    elif makeup_type == "eyeliner":
        print("MAKEUP: applying eyeliner")
        return apply_eyeliner(image, landmarks, color_hex=color_hex, intensity=intensity)

    elif makeup_type == "eyebrow_slit":
        print("MAKEUP: applying eyebrow slit")
        return apply_eyebrow_slit(
            image,
            landmarks,
            side="right",
            intensity=intensity
        )

  
    elif makeup_type == "eyeliner_no_wing":
        print("MAKEUP: applying eyeliner (no wing)")
        return apply_eyeliner(image, landmarks, color_hex=color_hex, intensity=intensity, wing=False)
    
    
    print(f"MAKEUP: unknown makeup_type: {makeup_type}")
    return image

def get_point_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    lx, ly = lm[0], lm[1]
    if lx <= 1.0 and ly <= 1.0:
        return int(lx * w), int(ly * h)
    return int(lx), int(ly)

def apply_eyebrow_slit(image, landmarks, side="right", intensity=0.95):
    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)

    def pt(i):
        return np.array(get_point_xy(landmarks, i, w, h), dtype=np.int32)

    # 1. KAŞ LANDMARKLARI - MEDIAPIPE 468
    if side == "left":
        # Sol kaş: dıştan içe
        brow_indices = [70, 63, 105, 66, 107]
        outer_idx = 70 # kaş dışı
        inner_idx = 107 # kaş başı
        direction = -1 # / yönü
    else:
        # Sağ kaş: içten dışa 
        brow_indices = [336, 296, 334, 293, 300]
        outer_idx = 300 # kaş dışı
        inner_idx = 336 # kaş başı
        direction = 1 # \ yönü

    brow_pts = np.array([pt(i) for i in brow_indices], dtype=np.int32)
    outer = pt(outer_idx)
    inner = pt(inner_idx)

    # 2. ÇİZİK POZİSYONU - KAŞIN DIŞ 2/3 KISMINDA
    # Referansta tam ortadan değil, dışa yakın
    center = (outer * 0.65 + inner * 0.35).astype(np.int32)

    # 3. ÇİZİK BOYUTU - İNCE VE KISA
    brow_w = abs(outer[0] - inner[0])
    slit_len = int(brow_w * 0.35) # %35 - senin kodda %55'ti çok uzun
    slit_thickness = max(2, int(brow_w * 0.08 * intensity)) # %8 - senin %18'di çok kalın

    # 4. AÇI - REFERANSTAKİ GİBİ 55-60 DERECE
    angle = np.deg2rad(58 * direction) # senin 68'di çok dik

    dx = int(np.cos(angle) * slit_len / 2)
    dy = int(np.sin(angle) * slit_len / 2)

    p1 = (int(center[0] - dx), int(center[1] - dy))
    p2 = (int(center[0] + dx), int(center[1] + dy))

    # 5. MASKE - TRAPEZ ŞEKLİNDE
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Üst geniş, alt dar trapez çizik
    top_w = slit_thickness + 2
    bot_w = slit_thickness
    
    # Çizgiyi trapez polygon olarak çiz
    perp_angle = angle + np.pi/2
    tdx = int(np.cos(perp_angle) * top_w / 2)
    tdy = int(np.sin(perp_angle) * top_w / 2)
    bdx = int(np.cos(perp_angle) * bot_w / 2)
    bdy = int(np.sin(perp_angle) * bot_w / 2)
    
    pts = np.array([
        [p1[0] - tdx, p1[1] - tdy],
        [p1[0] + tdx, p1[1] + tdy],
        [p2[0] + bdx, p2[1] + bdy],
        [p2[0] - bdx, p2[1] - bdy]
    ], np.int32)
    
    cv2.fillPoly(mask, [pts], 255)

    # 6. SADECE KAŞ BÖLGESİYLE SINIRLA
    brow_region = np.zeros((h, w), dtype=np.uint8)
    
    # Kaş polygonunu genişlet
    expanded_top = brow_pts.copy()
    expanded_top[:, 1] -= int(h * 0.012) # üstten 1.2%
    
    expanded_bot = brow_pts.copy()
    expanded_bot[:, 1] += int(h * 0.018) # alttan 1.8%
    
    poly = np.vstack([expanded_top, expanded_bot[::-1]])
    cv2.fillPoly(brow_region, [poly.astype(np.int32)], 255)
    
    # Dilate çok az - senin 9x9 çok geniş
    brow_region = cv2.dilate(brow_region, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.bitwise_and(mask, brow_region)

    # 7. BLUR - KENARLAR SOFT OLSUN
    # Referansta keskin değil, soft geçiş var
    mask = cv2.GaussianBlur(mask, (7, 7), 0) # senin 5x5'ti

    # 8. TEN RENGİ - KAŞIN HEMEN ALTINDAN AL
    sample_x = int(np.clip(center[0], 0, w-1))
    sample_y = int(np.clip(center[1] + h * 0.025, 0, h-1)) # %2.5 aşağı
    
    # 5x5 alan ortalaması al - tek piksel noise olur
    y1, y2 = max(0, sample_y-2), min(h, sample_y+3)
    x1, x2 = max(0, sample_x-2), min(w, sample_x+3)
    skin_color = np.mean(output[y1:y2, x1:x2], axis=(0,1))

    # 9. UYGULA - SOFT BLEND
    skin_layer = np.full_like(output, skin_color)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    alpha = np.clip(alpha * intensity, 0, 1)
    
    # Çok hafif multiply - ten rengi biraz koyulaşsın
    output_dark = output * 0.96
    result = output_dark * (1 - alpha) + skin_layer * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


