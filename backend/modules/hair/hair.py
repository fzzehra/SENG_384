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


def apply_hair_overlay(image, landmarks, overlay_path, intensity=1.0, scale_factor=1.0, x_offset=0, y_offset=0):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return image
    
    # Remove baked-in background (white or checkerboard) if needed
    b, g, r = cv2.split(overlay[:, :, :3])
    # Assume pixels where all BGR channels are > 200 are background (covers white and typical gray checkerboard)
    bg_mask = np.where((b > 200) & (g > 200) & (r > 200), 0, 255).astype(np.uint8)
    
    if overlay.shape[2] == 4:
        # Combine existing alpha with our new background mask
        overlay[:, :, 3] = cv2.bitwise_and(overlay[:, :, 3], bg_mask)
    else:
        # Create new alpha from mask
        overlay = cv2.merge([b, g, r, bg_mask])

    h, w = image.shape[:2]

    top_head = landmarks[10]
    chin = landmarks[152]
    left_temple = landmarks[234]
    right_temple = landmarks[454]

    face_w = abs(right_temple[0] - left_temple[0])
    face_h = abs(chin[1] - top_head[1])
    cx = (left_temple[0] + right_temple[0]) // 2

    hairline_y = top_head[1]
    if len(landmarks) > 299:
        left_hairline = landmarks[299]
        right_hairline = landmarks[70]
        hairline_y = int((left_hairline[1] + right_hairline[1]) / 2)

    target_w = max(int(face_w * 1.45 * scale_factor), 1)
    scale = target_w / float(overlay.shape[1])
    target_h = max(int(overlay.shape[0] * scale), 1)

    overlay_resized = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)

    angle = 0.0
    if len(landmarks) > 454:
        delta = np.array(right_temple) - np.array(left_temple)
        angle = float(np.degrees(np.arctan2(delta[1], delta[0])))

    overlay_rotated = _rotate_image(overlay_resized, angle)

    # Apply offsets here
    x_start = cx - overlay_rotated.shape[1] // 2 + x_offset
    y_start = hairline_y - int(overlay_rotated.shape[0] * 0.15) + y_offset

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

    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
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
    
    for side, indices in EYEBROW_LANDMARKS.items():
        pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
    
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    color_layer = np.full_like(image, color_bgr, dtype=np.float32)
    alpha = (mask.astype(np.float32) / 255.0 * intensity)[:, :, None]
    
    result = image.astype(np.float32) * (1 - alpha) + color_layer * alpha
    return np.clip(result, 0, 255).astype(np.uint8)

   