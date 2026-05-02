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


def apply_hair_overlay(image, landmarks, overlay_path, intensity=1.0):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.shape[2] != 4:
        return image

    h, w = image.shape[:2]

    # -------------------------
    # FACE KEY POINTS
    # -------------------------
    top_head     = landmarks[10]
    chin         = landmarks[152]
    left_temple  = landmarks[234]
    right_temple = landmarks[454]

    face_w = abs(right_temple[0] - left_temple[0])
    face_h = abs(chin[1] - top_head[1])
    cx = (left_temple[0] + right_temple[0]) // 2

    # -------------------------
    # 🔥 FIX 1: HAIRLINE ALIGNMENT
    # -------------------------
    hair_anchor_y = top_head[1] - int(face_h * 0.18)

    # -------------------------
    # 🔥 FIX 2: PROPORTIONAL SCALE
    # -------------------------
    target_w = int(face_w * 1.45)
    scale = target_w / overlay.shape[1]
    target_h = int(overlay.shape[0] * scale)

    overlay_resized = cv2.resize(overlay, (target_w, target_h))

    # -------------------------
    # 🔥 FIX 3: CENTERING (critical)
    # -------------------------
    x_start = cx - target_w // 2
    y_start = hair_anchor_y - int(target_h * 0.22)

    # -------------------------
    # bounds
    # -------------------------
    x1 = max(0, x_start)
    y1 = max(0, y_start)
    x2 = min(w, x_start + target_w)
    y2 = min(h, y_start + target_h)

    if x1 >= x2 or y1 >= y2:
        return image

    ox1 = x1 - x_start
    oy1 = y1 - y_start

    crop = overlay_resized[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)]

    # -------------------------
    # 🔥 FIX 4: SAFE ALPHA
    # -------------------------
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
    alpha = alpha * intensity

    # -------------------------
    # BLEND
    # -------------------------
    result = image.astype(np.float32)

    result[y1:y2, x1:x2] = (
        result[y1:y2, x1:x2] * (1 - alpha) +
        crop[:, :, :3].astype(np.float32) * alpha
    )

    return np.clip(result, 0, 255).astype(np.uint8)