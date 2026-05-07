import cv2
import numpy as np


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


def _get_hair_mask(image, landmarks):
    h, w = image.shape[:2]

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

    return hair_mask


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


def apply_hair_overlay(image, landmarks, overlay_path, intensity=1.0):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.shape[2] != 4:
        return image

    h, w = image.shape[:2]

    top_head     = landmarks[10]
    chin         = landmarks[152]
    left_temple  = landmarks[234]
    right_temple = landmarks[454]

    face_w = abs(right_temple[0] - left_temple[0])
    face_h = abs(chin[1] - top_head[1])
    cx = (left_temple[0] + right_temple[0]) // 2

    hair_anchor_y = top_head[1] - int(face_h * 0.18)

    target_w = int(face_w * 1.45)
    scale = target_w / overlay.shape[1]
    target_h = int(overlay.shape[0] * scale)

    overlay_resized = cv2.resize(overlay, (target_w, target_h))

    x_start = cx - target_w // 2
    y_start = hair_anchor_y - int(target_h * 0.22)

    x1 = max(0, x_start)
    y1 = max(0, y_start)
    x2 = min(w, x_start + target_w)
    y2 = min(h, y_start + target_h)

    if x1 >= x2 or y1 >= y2:
        return image

    ox1 = x1 - x_start
    oy1 = y1 - y_start

    crop = overlay_resized[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)]

    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
    alpha = alpha * intensity

    result = image.astype(np.float32)
    result[y1:y2, x1:x2] = (
        result[y1:y2, x1:x2] * (1 - alpha) +
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

    for side, indices in EYEBROW_LANDMARKS.items():
        pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    color_layer = np.full_like(image, color_bgr, dtype=np.float32)
    alpha = (mask.astype(np.float32) / 255.0 * intensity)[:, :, None]

    result = image.astype(np.float32) * (1 - alpha) + color_layer * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_hair_texture(image, landmarks, texture_type='curly', intensity=0.5):
    """
    Fotoğraftaki kişinin kendi saçına texture değişimi uygular.
    texture_type: 'curly' (kıvırcık) veya 'straight' (düz)
    intensity: 0.0-1.0 (0=orijinal, 1=maksimum etki)
    """
    h, w = image.shape[:2]
    hair_mask = _get_hair_mask(image, landmarks)

    if texture_type == 'curly':
        amplitude = intensity * 14.0
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        dx = (amplitude * np.sin(y_coords * 0.18 + x_coords * 0.04) +
              amplitude * 0.4 * np.sin(y_coords * 0.35 + 1.2))
        dy = (amplitude * np.sin(x_coords * 0.18 + y_coords * 0.04 + 1.57) +
              amplitude * 0.4 * np.sin(x_coords * 0.35 + 0.8))

        map_x = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)

        transformed = cv2.remap(image, map_x, map_y,
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    elif texture_type == 'straight':
        ksize = max(3, int(intensity * 28) | 1)
        kernel = np.zeros((ksize, 1), np.float32)
        kernel[:, 0] = 1.0 / ksize
        transformed = cv2.filter2D(image, -1, kernel)

    else:
        return image

    alpha = (hair_mask.astype(np.float32) / 255.0)[:, :, None]
    result = image.astype(np.float32) * (1 - alpha) + transformed.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_hair_style(image, landmarks, style='loose', intensity=0.5):
    h, w = image.shape[:2]
    hair_mask = _get_hair_mask(image, landmarks)

    top_head     = landmarks[10]
    chin         = landmarks[152]
    left_temple  = landmarks[234]
    right_temple = landmarks[454]

    face_w = abs(right_temple[0] - left_temple[0])
    face_h = abs(chin[1] - top_head[1])
    cx = (left_temple[0] + right_temple[0]) // 2

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    if style == 'loose':
        # Saçı aşağı uzat — saçın alt kısmını piksel kaydırarak uzatır
        shift = int(intensity * 35)

        # Saç maskesini aşağıya genişlet
        kernel = np.zeros((shift * 2 + 1, 1), np.uint8)
        kernel[shift:, 0] = 1
        extended_mask = cv2.dilate(hair_mask, kernel, iterations=1)

        # Saç rengini yukarıdan aşağıya kopyala (piksel kaydırma)
        map_x = x_coords.astype(np.float32)
        map_y = np.clip(y_coords - shift, 0, h - 1).astype(np.float32)
        shifted = cv2.remap(image, map_x, map_y,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Sadece genişlemiş saç bölgesinde uygula, yüz maskesini çıkar
        face_m = _face_mask(image, landmarks)
        extended_mask = cv2.subtract(extended_mask, face_m)
        extended_mask = cv2.GaussianBlur(extended_mask, (15, 15), 0)

        alpha = (extended_mask.astype(np.float32) / 255.0 * intensity)[:, :, None]
        result = image.astype(np.float32) * (1 - alpha) + shifted.astype(np.float32) * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

    elif style == 'bun':
        # Topuz noktası: baş landmarks'ının belirgin üstü
        crown_x = float(cx)
        crown_y = float(top_head[1]) - face_h * 0.38

        # Warp sadece çeneden yukarıdaki bölgeye uygulanacak
        upper_limit_y = int(top_head[1] + face_h * 0.15)
        upper_region = np.zeros((h, w), dtype=np.float32)
        upper_region[:upper_limit_y, :] = 1.0

        dx = x_coords - crown_x
        dy = y_coords - crown_y
        pull = intensity * 0.30

        map_x = np.clip(x_coords + dx * pull, 0, w - 1).astype(np.float32)
        map_y = np.clip(y_coords + dy * pull, 0, h - 1).astype(np.float32)

        warped = cv2.remap(image, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Warp: sadece saç maskesi + üst bölge kesişiminde
        combined = hair_mask.astype(np.float32) / 255.0 * upper_region
        alpha = combined[:, :, None]
        result = image.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha

        # Topuz yuvarlağı — kafanın net üstünde
        bun_mask = np.zeros((h, w), dtype=np.uint8)
        bun_r = max(int(face_w * 0.18 * (0.5 + intensity * 0.5)), 10)
        cv2.circle(bun_mask, (int(crown_x), int(crown_y)), bun_r, 255, -1)
        bun_mask = cv2.GaussianBlur(bun_mask, (21, 21), 0)

        valid_pixels = image[hair_mask > 80]
        bun_color = np.median(valid_pixels, axis=0).astype(np.float32) if len(valid_pixels) > 0 else np.array([50, 35, 25], dtype=np.float32)

        bun_alpha = (bun_mask.astype(np.float32) / 255.0 * intensity)[:, :, None]
        bun_layer = np.full_like(image, bun_color, dtype=np.float32)
        result = result * (1 - bun_alpha) + bun_layer * bun_alpha

        return np.clip(result, 0, 255).astype(np.uint8)

    elif style == 'ponytail':
        pt_x = float(cx) + face_w * 0.55
        pt_y = float(top_head[1]) + face_h * 0.25

        upper_limit_y = int(top_head[1] + face_h * 0.15)
        upper_region = np.zeros((h, w), dtype=np.float32)
        upper_region[:upper_limit_y, :] = 1.0

        dx = x_coords - pt_x
        dy = y_coords - pt_y
        pull = intensity * 0.28

        map_x = np.clip(x_coords + dx * pull, 0, w - 1).astype(np.float32)
        map_y = np.clip(y_coords + dy * pull, 0, h - 1).astype(np.float32)

        warped = cv2.remap(image, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        combined = hair_mask.astype(np.float32) / 255.0 * upper_region
        alpha = combined[:, :, None]
        result = image.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

    return image