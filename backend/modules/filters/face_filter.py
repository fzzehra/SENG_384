import os
import cv2
import numpy as np

# Her maskeye özel oran ayarları
MASK_CONFIG = {
    "butterfly_mask.png": {"width_scale": 2.2, "height_scale": 0.52, "y_offset": 0.05, "alpha": 0.9},
    "crystal_mask.png":   {"width_scale": 1.7, "height_scale": 1.4, "y_offset": 0.08, "alpha": 0.95},
    "pearl_mask.png":     {"width_scale": 2.2, "height_scale": 0.65, "y_offset": 0.185, "alpha": 0.95},
    "peacock_mask.png":   {"width_scale": 2.6, "height_scale": 0.65, "y_offset": 0.15, "alpha": 0.9},
    "dragon.png":         {"width_scale": 1, "height_scale": 1, "y_offset": 0, "alpha": 0.9},
}

def apply_face_filter(image, landmarks, mask_name="butterfly_mask.png", intensity=1.0):
    h, w = image.shape[:2]

    left_eye_inner  = landmarks[133]
    right_eye_inner = landmarks[362]
    left_eye_outer  = landmarks[33]
    right_eye_outer = landmarks[263]

    eye_span = right_eye_outer[0] - left_eye_outer[0]

    cfg = MASK_CONFIG.get(mask_name)
    target_w = int(eye_span * cfg["width_scale"])
    target_h = int(target_w * cfg["height_scale"])

    mask_path = os.path.join("static", "filters", "assets", mask_name)
    if not os.path.exists(mask_path):
        mask_path = os.path.join("backend", "modules", "filters", "assets", mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("Mask not found:", mask_path)
        return image

    if target_w < 10 or target_h < 10:
        return image

    mask = cv2.resize(mask, (target_w, target_h))

    center_x = int((left_eye_outer[0] + right_eye_outer[0]) / 2)
    center_y = int((left_eye_inner[1] + right_eye_inner[1]) / 2)

    x = center_x - target_w // 2
    y = center_y - target_h // 2 - int(eye_span * cfg["y_offset"])

    overlay_h, overlay_w = mask.shape[:2]

    x1 = max(0, x);               y1 = max(0, y)
    x2 = min(w, x + overlay_w);   y2 = min(h, y + overlay_h)
    if x1 >= x2 or y1 >= y2:
        return image

    mx1 = x1 - x;  my1 = y1 - y
    mx2 = mx1 + (x2 - x1)
    my2 = my1 + (y2 - y1)

    crop  = mask[my1:my2, mx1:mx2]
    alpha = (crop[:, :, 3:4] / 255.0) * float(intensity) * cfg["alpha"]

    roi         = image[y1:y2, x1:x2].astype(np.float32)
    overlay_rgb = crop[:, :, :3].astype(np.float32)

    image[y1:y2, x1:x2] = np.clip(
        roi * (1.0 - alpha) + overlay_rgb * alpha, 0, 255
    ).astype(np.uint8)

    print(f"{mask_name} filter applied successfully.")
    return image