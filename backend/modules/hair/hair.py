# ==============================================================================
# backend/modules/hair/hair.py
# ==============================================================================
import os
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

HAIRSTYLE_ANCHOR_PRESETS = {
    "hair.png": {
        "left_ratio": (0.24, 0.45),
        "right_ratio": (0.76, 0.45),
        "top_ratio": (0.50, 0.22)
    },
    "kısa.png": {
        "left_ratio": (0.26, 0.48),
        "right_ratio": (0.74, 0.48),
        "top_ratio": (0.50, 0.30)
    },
    "orange.png": {
        "left_ratio": (0.20, 0.42),
        "right_ratio": (0.80, 0.42),
        "top_ratio": (0.50, 0.15)
    }
}

def _get_face_metrics(landmarks):
    lm = lambda i: np.array(landmarks[i], dtype=np.float32)
    left_eye_outer = lm(33)
    right_eye_outer = lm(263)
    left_temple = lm(234)
    right_temple = lm(454)
    top_head = lm(10)
    chin = lm(152)
    nose_tip = lm(1) if len(landmarks) > 1 else (left_temple + right_temple) / 2

    face_width = np.linalg.norm(right_temple - left_temple)
    face_height = np.linalg.norm(chin - top_head)
    face_center = (left_temple + right_temple) / 2.0
    face_center[0] += (nose_tip[0] - face_center[0]) * 0.3

    eye_angle = np.degrees(np.arctan2(
        right_eye_outer[1] - left_eye_outer[1],
        right_eye_outer[0] - left_eye_outer[0]
    ))

    face_axis = top_head - chin
    face_axis_n = face_axis / (np.linalg.norm(face_axis) + 1e-6)
    forehead_pt = top_head + face_axis_n * (face_height * 0.25)

    return {
        "face_width": face_width,
        "face_height": face_height,
        "face_center": face_center,
        "eye_angle": eye_angle,
        "top_head": top_head,
        "chin": chin,
        "left_temple": left_temple,
        "right_temple": right_temple,
        "forehead_pt": forehead_pt
    }

def _adjust_wig_gap(P_left, P_right, P_top, w_wig, face_width, scale_factor):
    cx = w_wig // 2
    current_gap = P_right[0] - P_left[0]
    desired_gap_face = face_width * 0.88
    desired_gap_wig = desired_gap_face / max(scale_factor, 0.1)

    min_gap = w_wig * 0.18
    desired_gap_wig = np.clip(desired_gap_wig, min_gap, w_wig * 0.6)

    new_left_x = int(cx - desired_gap_wig / 2)
    new_right_x = int(cx + desired_gap_wig / 2)

    P_left_adj = (new_left_x, P_left[1])
    P_right_adj = (new_right_x, P_right[1])
    return P_left_adj, P_right_adj, P_top

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


def detect_wig_anchors(overlay_img, filename=None):
    h_wig, w_wig = overlay_img.shape[:2]

    if filename and filename in HAIRSTYLE_ANCHOR_PRESETS:
        preset = HAIRSTYLE_ANCHOR_PRESETS[filename]
        l_x, l_y = preset["left_ratio"]
        r_x, r_y = preset["right_ratio"]
        t_x, t_y = preset["top_ratio"]
        return (
            (int(w_wig * l_x), int(h_wig * l_y)),
            (int(w_wig * r_x), int(h_wig * r_y)),
            (int(w_wig * t_x), int(h_wig * t_y))
        )

    if overlay_img.shape[2] == 4:
        alpha = overlay_img[:, :, 3]
    else:
        gray = cv2.cvtColor(overlay_img[:, :, :3], cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    y_indices, x_indices = np.where(alpha > 30)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return (
            (int(w_wig * 0.25), int(h_wig * 0.45)),
            (int(w_wig * 0.75), int(h_wig * 0.45)),
            (int(w_wig * 0.50), int(h_wig * 0.30))
        )

    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()

    center_x = (xmin + xmax) // 2
    center_start = int(w_wig * 0.45)
    center_end = int(w_wig * 0.55)

    y_hairline = ymin + int((ymax - ymin) * 0.30)
    for y in range(ymin + int((ymax - ymin) * 0.05), ymax - int((ymax - ymin) * 0.15)):
        row_alpha = alpha[y, center_start:center_end]
        if np.mean(row_alpha) < 50:
            y_hairline = y
            break

    y_temple = min(ymax - 5, y_hairline + int((ymax - ymin) * 0.05))

    x_left_inner = xmin + int((xmax - xmin) * 0.22)
    for x in range(center_x, xmin, -1):
        if alpha[y_temple, x] > 100:
            x_left_inner = x
            break

    x_right_inner = xmin + int((xmax - xmin) * 0.78)
    for x in range(center_x, xmax):
        if alpha[y_temple, x] > 100:
            x_right_inner = x
            break

    min_gap = int(w_wig * 0.18)
    if (x_right_inner - x_left_inner) < min_gap:
        x_left_inner = center_x - int(w_wig * 0.16)
        x_right_inner = center_x + int(w_wig * 0.16)

    y_top = ymin + int((y_hairline - ymin) * 0.30)

    P_wig_left = (x_left_inner, y_temple)
    P_wig_right = (x_right_inner, y_temple)
    P_wig_top = (center_x, y_top)

    return P_wig_left, P_wig_right, P_wig_top


def _build_tps_control_points(landmarks, overlay, face_center, scale_factor, x_offset, y_offset):
    h_wig, w_wig = overlay.shape[:2]

    P_wig_left, P_wig_right, P_wig_top = detect_wig_anchors(overlay, None)

    lm = lambda i: np.array(landmarks[i], dtype=np.float32)

    P_face_left  = lm(234)
    P_face_right = lm(454)
    top_head     = lm(10)
    chin         = lm(152)

    face_h      = np.linalg.norm(chin - top_head)
    face_axis   = top_head - chin
    face_axis_n = face_axis / (np.linalg.norm(face_axis) + 1e-6)
    P_face_top  = top_head + face_axis_n * (face_h * 0.32)
    offset = np.array([x_offset, y_offset], dtype=np.float32)

    def target(face_pt):
        return face_center + (face_pt - face_center) * scale_factor + offset

    wig_cx = w_wig // 2

    src = np.float32([
        P_wig_left,
        P_wig_right,
        P_wig_top,
        (wig_cx, 0),
        (0, h_wig // 2),
        (w_wig, h_wig // 2),
        (wig_cx - w_wig * 0.25, P_wig_top[1]),
        (wig_cx + w_wig * 0.25, P_wig_top[1]),
    ])

    face_w = np.linalg.norm(P_face_right - P_face_left)
    dst = np.float32([
        target(P_face_left),
        target(P_face_right),
        target(P_face_top),
        target(top_head + face_axis_n * (face_h * 0.35)),
        target(P_face_left  + np.array([-face_w * 0.10, 0])),
        target(P_face_right + np.array([ face_w * 0.10, 0])),
        target(top_head + face_axis_n * (face_h * 0.12) + np.array([-face_w * 0.25, 0])),
        target(top_head + face_axis_n * (face_h * 0.12) + np.array([ face_w * 0.25, 0])),
    ])

    return src, dst


def _warp_overlay_tps(overlay, src_pts, dst_pts, out_w, out_h):
    rbf_x = RBFInterpolator(dst_pts, src_pts[:, 0], kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(dst_pts, src_pts[:, 1], kernel='thin_plate_spline')

    yy, xx = np.mgrid[0:out_h, 0:out_w]
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    map_x = rbf_x(grid).reshape(out_h, out_w).astype(np.float32)
    map_y = rbf_y(grid).reshape(out_h, out_w).astype(np.float32)

    warped = cv2.remap(
        overlay, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return warped


def _adapt_color(overlay_bgr, image_bgr, hair_mask):
    if hair_mask is None or hair_mask.sum() == 0:
        return overlay_bgr

    face_pixels = image_bgr[hair_mask > 30]
    if len(face_pixels) == 0:
        return overlay_bgr

    src_mean = face_pixels.mean(axis=0)

    wig_pixels_mask = overlay_bgr.mean(axis=2) > 10
    wig_pixels = overlay_bgr[wig_pixels_mask]
    if len(wig_pixels) == 0:
        return overlay_bgr

    wig_mean = wig_pixels.mean(axis=0)
    diff = (src_mean - wig_mean) * 0.3
    adapted = np.clip(overlay_bgr.astype(np.float32) + diff, 0, 255).astype(np.uint8)
    return adapted


def apply_hair_overlay(image, landmarks, overlay_path, intensity=1.0, scale_factor=1.0, x_offset=0, y_offset=0, wig_color=None, wig_color_intensity=0.0, auto=True):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return image

    h, w = image.shape[:2]
    metrics = _get_face_metrics(landmarks)

    # 1. Renk ve alpha temizliği
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
    else:
        b, g, r = cv2.split(overlay[:, :, :3])
        a = np.full(b.shape, 255, dtype=np.uint8)

    gray_wig = cv2.cvtColor(overlay[:, :, :3], cv2.COLOR_BGR2GRAY)
    diff_rg = np.abs(r.astype(int) - g.astype(int))
    diff_gb = np.abs(g.astype(int) - b.astype(int))
    is_gray = (diff_rg < 15) & (diff_gb < 15)
    is_light = gray_wig > 170
    bg_mask = np.where(is_gray & is_light, 0, 255).astype(np.uint8)
    bg_mask = cv2.GaussianBlur(bg_mask, (3, 3), 0)
    _, bg_mask = cv2.threshold(bg_mask, 127, 255, cv2.THRESH_BINARY)
    a = cv2.bitwise_and(a, bg_mask)
    overlay = cv2.merge([b, g, r, a])

    # 2. Göz açısına göre peruk döndür
    overlay = _rotate_image(overlay, -metrics["eye_angle"])

    # 3. Peruk anchor noktaları
    P_wig_left, P_wig_right, P_wig_top = detect_wig_anchors(overlay, os.path.basename(overlay_path))
    h_wig, w_wig = overlay.shape[:2]

    # 4. Otomatik scale
    if auto:
        # Peruk gerçek genişliği alpha bbox'tan
        alpha = overlay[:, :, 3]
        ys, xs = np.where(alpha > 30)
        wig_bbox_w = xs.max() - xs.min() if len(xs) > 0 else w_wig
        desired_hair_w = metrics["face_width"] * 1.65
        auto_scale = desired_hair_w / max(wig_bbox_w, 1.0)
        # Kullanıcı slider'ı sadece fine-tune
        scale_factor = np.clip(auto_scale * scale_factor, 0.7, 2.2)
    else:
        scale_factor = max(0.5, min(3.0, scale_factor))

    # 5. Inner gap ayarı
    if auto:
        P_wig_left, P_wig_right, P_wig_top = _adjust_wig_gap(
            P_wig_left, P_wig_right, P_wig_top, w_wig,
            metrics["face_width"], scale_factor
        )

    # 6. Hedef noktalar
    fc = metrics["face_center"]
    offset = np.array([x_offset, y_offset], dtype=np.float32)

    # Sol/sağ temple noktaları yüze oturacak
    t_left = fc + (metrics["left_temple"] - fc) * scale_factor + offset
    t_right = fc + (metrics["right_temple"] - fc) * scale_factor + offset
    # Tepe noktası alnın üstüne
    t_top = fc + (metrics["forehead_pt"] - fc) * scale_factor + offset

    # 7. Affine warp
    M = cv2.getAffineTransform(
        np.float32([P_wig_left, P_wig_right, P_wig_top]),
        np.float32([t_left, t_right, t_top])
    )
    warped_overlay = cv2.warpAffine(overlay, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

    # 8. İsteğe bağlı peruk rengi
    if wig_color and wig_color_intensity > 0:
        color_bgr = _hex_to_bgr(wig_color)
        rgb = warped_overlay[:, :, :3].astype(np.float32)
        layer = np.full_like(rgb, color_bgr, dtype=np.float32)
        rgb = rgb * (1.0 - wig_color_intensity) + layer * wig_color_intensity
        warped_overlay[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)

    # 9. Segmentasyon maskesi
    hair_mask = None
    try:
        from backend.modules.hair.segmentation import get_hair_mask
        hair_mask = get_hair_mask(image)
    except Exception:
        pass

    # 10. Renk adaptasyonu
    warped_rgb = warped_overlay[:, :, :3].copy()
    warped_rgb = _adapt_color(warped_rgb, image, hair_mask)

    # 11. Alpha ve yüz arkası
    crop_alpha = warped_overlay[:, :, 3].astype(np.float32) / 255.0
    blur_size = max(5, int(metrics["face_height"] * 0.03))
    if blur_size % 2 == 0: blur_size += 1
    crop_alpha = cv2.GaussianBlur(crop_alpha, (blur_size, blur_size), 0)

    if hair_mask is not None:
        hair_f = hair_mask.astype(np.float32) / 255.0
        hair_f = cv2.GaussianBlur(hair_f, (21, 21), 0)
        crop_alpha = crop_alpha * (0.6 + 0.4 * hair_f)

    # Yüz maskesi, alın altında saç geride kalsın
    face_idx = [10,338,297,332,284,251,389,356,454,323,361,288,
                397,365,379,378,400,377,152,148,176,149,150,136,
                172,58,132,93,234,127,162,21,54,103,67,109]
    face_pts = np.array([landmarks[i] for i in face_idx], dtype=np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_pts], 255)
    face_mask_f = cv2.GaussianBlur(face_mask.astype(np.float32)/255.0, (31,31), 0)

    hairline_y = int(metrics["top_head"][1])
    y_coords = np.arange(h, dtype=np.float32)
    gradient = np.clip((y_coords - hairline_y) / (metrics["face_height"] * 0.18), 0.0, 1.0)
    gradient_2d = np.tile(gradient[:, None], (1, w))
    hair_hide = face_mask_f * gradient_2d
    crop_alpha = crop_alpha * (1.0 - hair_hide)

    crop_alpha = np.clip(crop_alpha * intensity, 0.0, 1.0)[:, :, None]
    result = image.astype(np.float32) * (1.0 - crop_alpha) + warped_rgb.astype(np.float32) * crop_alpha
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


EYEBROW_LANDMARKS = {
    "left":  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
}

def apply_eyebrow_color(image, landmarks, color_hex='#000000', intensity=0.5):
    h, w = image.shape[:2]
    color_bgr = _hex_to_bgr(color_hex)
    mask = np.zeros((h, w), dtype=np.uint8)

    for side, indices in EYEBROW_LANDMARKS.items():
        side_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(side_mask, [hull], 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        side_mask = cv2.dilate(side_mask, k, iterations=1)
        mask = cv2.bitwise_or(mask, side_mask)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark_weight = np.clip((0.5 - gray) / 0.3, 0.0, 1.0)
    combined = (mask.astype(np.float32) / 255.0) * dark_weight
    alpha = np.clip(combined * intensity * 0.9, 0.0, 0.9)
    color_layer = np.full_like(image, color_bgr, dtype=np.float32)
    result = image.astype(np.float32) * (1 - alpha[:, :, None]) + color_layer * alpha[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)