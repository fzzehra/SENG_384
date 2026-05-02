import cv2
import numpy as np


def _hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def apply_hair_color(image, landmarks, color_hex='#000000', intensity=0.5):
    h, w = image.shape[:2]
    color_bgr = _hex_to_bgr(color_hex)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hair_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([180, 255, 120]))

    limit_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = landmarks[1][0], landmarks[1][1]
    cv2.circle(limit_mask, (cx, cy), int(w * 0.6), 255, -1)

    face_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    face_pts = np.array([landmarks[i] for i in face_idx], dtype=np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_pts], 255)

    hair_mask = cv2.bitwise_and(hair_mask, limit_mask)
    hair_mask = cv2.subtract(hair_mask, face_mask)
    hair_mask = cv2.GaussianBlur(hair_mask, (31, 31), 0)

    color_layer = np.full_like(image, color_bgr, dtype=np.float32)
    alpha = (hair_mask.astype(np.float32) / 255.0 * intensity)[:, :, None]

    result = image.astype(np.float32) * (1 - alpha) + color_layer * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_hair_overlay(image, landmarks, overlay_path, intensity=1.0):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.shape[2] != 4:
        return image

    h, w = image.shape[:2]

    left_temple  = landmarks[234]
    right_temple = landmarks[454]
    top_head     = landmarks[10]

    face_width = abs(right_temple[0] - left_temple[0])
    target_w = int(face_width * 1.4)
    if target_w <= 0:
        return image

    scale = target_w / overlay.shape[1]
    target_h = int(overlay.shape[0] * scale)

    overlay_resized = cv2.resize(overlay, (target_w, target_h))

    cx = (left_temple[0] + right_temple[0]) // 2
    x_start = cx - target_w // 2
    y_start = top_head[1] - int(target_h * 0.85)

    x1 = max(x_start, 0)
    y1 = max(y_start, 0)
    x2 = min(x_start + target_w, w)
    y2 = min(y_start + target_h, h)

    if x1 >= x2 or y1 >= y2:
        return image

    ox1 = x1 - x_start
    oy1 = y1 - y_start
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    crop = overlay_resized[oy1:oy2, ox1:ox2]
    alpha = (crop[:, :, 3:4].astype(np.float32) / 255.0) * intensity

    result = image.copy().astype(np.float32)
    result[y1:y2, x1:x2] = (
        result[y1:y2, x1:x2] * (1 - alpha) +
        crop[:, :, :3].astype(np.float32) * alpha
    )
    return np.clip(result, 0, 255).astype(np.uint8)