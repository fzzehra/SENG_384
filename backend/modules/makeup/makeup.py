import cv2
import numpy as np
def _create_soft_mask(shape, points, blur_ksize=15):
    mask = np.zeros(shape[:2], dtype=np.uint8)

    points = np.array(points, dtype=np.int32)
    if len(points) < 3:
        return mask

    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    return mask


def _blend_color(image, mask, color, alpha):
    """
    image: BGR image
    mask: single channel uint8 mask
    color: BGR tuple
    alpha: 0.0 - 1.0
    """
    alpha = max(0.0, min(1.0, alpha))

    mask_f = (mask.astype(np.float32) / 255.0) * alpha
    mask_f = np.expand_dims(mask_f, axis=2)

    color_layer = np.zeros_like(image, dtype=np.float32)
    color_layer[:] = color

    image_f = image.astype(np.float32)

    result = image_f * (1.0 - mask_f) + color_layer * mask_f
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_lipstick(image, landmarks, color=(0, 0, 255), intensity=0.5):
    """
    Apply red lipstick.
    color is BGR.
    """
    pts = np.array(landmarks, dtype=np.int32)

    # Outer lips
    outer_lip_indices = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409
    ]

    # Inner mouth opening
    inner_lip_indices = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        415, 310, 311, 312, 13, 82, 81, 80, 191
    ]

    outer_pts = pts[outer_lip_indices]
    inner_pts = pts[inner_lip_indices]

    outer_mask = _create_soft_mask(image.shape, outer_pts, blur_ksize=11)
    inner_mask = _create_soft_mask(image.shape, inner_pts, blur_ksize=9)

    # Remove mouth opening so teeth/tongue area is not painted
    lipstick_mask = cv2.subtract(outer_mask, inner_mask)

    result = _blend_color(image, lipstick_mask, color=color, alpha=0.75 * intensity)
    return result


def apply_eyeshadow(image, landmarks, color=(180, 0, 180), intensity=0.5):
    """
    Apply purple eyeshadow above the eyes.
    """
    pts = np.array(landmarks, dtype=np.int32)

    # Left eye upper contour + eyebrow
    left_eye_upper = [33, 246, 161, 160, 159, 158, 157, 173, 133]
    left_brow = [70, 63, 105, 66, 107]

    # Right eye upper contour + eyebrow
    right_eye_upper = [263, 466, 388, 387, 386, 385, 384, 398, 362]
    right_brow = [336, 296, 334, 293, 300]

    # Full eye contours — gözün içini maske dışında bırakmak için
    left_eye_full = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_full = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

    def build_shadow_region(eye_indices, brow_indices):
        eye_pts = pts[eye_indices]
        brow_pts = pts[brow_indices]

        eye_center_y = np.mean(eye_pts[:, 1])

        adjusted_brow = brow_pts.copy()
        adjusted_brow[:, 1] = (0.65 * adjusted_brow[:, 1] + 0.35 * eye_center_y).astype(np.int32)

        region = np.vstack([eye_pts, adjusted_brow[::-1]])
        return region

    left_region = build_shadow_region(left_eye_upper, left_brow)
    right_region = build_shadow_region(right_eye_upper, right_brow)

    left_mask = _create_soft_mask(image.shape, left_region, blur_ksize=21)
    right_mask = _create_soft_mask(image.shape, right_region, blur_ksize=21)

    shadow_mask = cv2.max(left_mask, right_mask)

    # Göz açıklığını (iris/beyaz kısım) shadow mask'ten çıkar
    left_eye_opening = _create_soft_mask(image.shape, pts[left_eye_full], blur_ksize=5)
    right_eye_opening = _create_soft_mask(image.shape, pts[right_eye_full], blur_ksize=5)
    eye_opening_mask = cv2.max(left_eye_opening, right_eye_opening)

    shadow_mask = cv2.subtract(shadow_mask, eye_opening_mask)

    result = _blend_color(image, shadow_mask, color=color, alpha=0.60 * intensity)
    return result


def apply_makeup_pipeline(image, landmarks, makeup_type, intensity=0.5):
    if makeup_type == "lipstick":
        print("MAKEUP: applying lipstick")
        return apply_lipstick(image, landmarks, color=(0, 0, 255), intensity=intensity)

    elif makeup_type == "eyeshadow":
        print("MAKEUP: applying eyeshadow")
        return apply_eyeshadow(image, landmarks, color=(180, 0, 180), intensity=intensity)
def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def blend_mask(image, mask, color_bgr, intensity=0.5):
    output = image.copy().astype(np.float32)
    color_layer = np.full_like(output, color_bgr, dtype=np.float32)
    alpha = (mask.astype(np.float32) / 255.0) * intensity

    for c in range(3):
        output[:, :, c] = output[:, :, c] * (1 - alpha) + color_layer[:, :, c] * alpha

    return np.clip(output, 0, 255).astype(np.uint8)


def get_point_xy(landmarks, idx, w, h):
    point = landmarks[idx]

    # MediaPipe landmark objesi ise
    if hasattr(point, "x") and hasattr(point, "y"):
        return [int(point.x * w), int(point.y * h)]

    # (x, y) tuple/list ise
    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
        return [int(point[0]), int(point[1])]

    raise ValueError(f"Unsupported landmark format at index {idx}: {point}")
def get_points(landmarks, indices, w, h):
    return np.array([get_point_xy(landmarks, i, w, h) for i in indices], dtype=np.int32)


def get_center(landmarks, indices, w, h):
    pts = get_points(landmarks, indices, w, h)
    center = np.mean(pts, axis=0).astype(int)
    return tuple(center)


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


def apply_blush(image, landmarks, color_hex="#f4a7b9", intensity=0.35):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def p(idx):
        return np.array(get_point_xy(landmarks, idx, w, h), dtype=np.float32)

    # Ana referans noktaları
    nose = p(1)
    left_anchor = p(50)
    right_anchor = p(280)
    left_face_edge = p(234)
    right_face_edge = p(454)

    # Allığı biraz içeri alıyoruz ki saça taşmasın
    left_center = (left_anchor * 0.72 + nose * 0.28).astype(np.int32)
    right_center = (right_anchor * 0.72 + nose * 0.28).astype(np.int32)

    # Biraz aşağı indiriyoruz, daha doğal dursun
    shift_down = int(h * 0.015)
    left_center[1] += shift_down
    right_center[1] += shift_down

    # Yüz genişliğine göre boyut hesaplama
    face_width = np.linalg.norm(right_face_edge - left_face_edge)
    axis_x = max(24, int(face_width * 0.095))   # yatayda biraz büyük
    axis_y = max(16, int(face_width * 0.070))   # dikeyde de yumuşak allık

    # Sol ve sağ yanağa simetrik ellipse
    cv2.ellipse(mask, tuple(left_center), (axis_x, axis_y), -18, 0, 360, 255, -1)
    cv2.ellipse(mask, tuple(right_center), (axis_x, axis_y), 18, 0, 360, 255, -1)

    # Yüz maskesi oluşturup allığı yüz içinde tutuyoruz
    face_outline_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454,
        323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132,
        93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    face_poly = np.array(
        [get_point_xy(landmarks, i, w, h) for i in face_outline_idx],
        dtype=np.int32
    )

    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_poly], 255)

    # Allığı yüzün dışına taşırma
    mask = cv2.bitwise_and(mask, face_mask)

    # Yumuşak geçiş
    blur_size = max(41, int(min(h, w) * 0.08))
    if blur_size % 2 == 0:
        blur_size += 1

    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Biraz daha doğal dursun ama belli olsun
    final_intensity = max(0.0, min(1.0, intensity)) * 0.75

    return blend_mask(image, mask, hex_to_bgr(color_hex), final_intensity)

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
def apply_eye_color(image, landmarks, color_hex="#4a90e2", intensity=0.5):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def pt(i):
        return np.array(get_point_xy(landmarks, i, w, h), dtype=np.int32)

    # Göz çevresi polygonları
    left_eye_idx = [33, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    right_eye_idx = [263, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

    left_eye_poly = np.array([pt(i) for i in left_eye_idx], dtype=np.int32)
    right_eye_poly = np.array([pt(i) for i in right_eye_idx], dtype=np.int32)

    eye_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(eye_mask, [left_eye_poly], 255)
    cv2.fillPoly(eye_mask, [right_eye_poly], 255)

    total_landmarks = len(landmarks)

    # Eğer iris landmarkları varsa onları kullan
    if total_landmarks > 477:
        left_iris_pts = np.array([pt(i) for i in range(468, 473)], dtype=np.int32)
        right_iris_pts = np.array([pt(i) for i in range(473, 478)], dtype=np.int32)

        left_center = np.mean(left_iris_pts, axis=0).astype(np.int32)
        right_center = np.mean(right_iris_pts, axis=0).astype(np.int32)

        left_radius = max(2, int(np.mean(np.linalg.norm(left_iris_pts - left_center, axis=1)) * 1.25))
        right_radius = max(2, int(np.mean(np.linalg.norm(right_iris_pts - right_center, axis=1)) * 1.25))

    else:
        # Fallback: iris landmark yoksa yaklaşık merkez/radius hesapla
        left_corner_1 = pt(33)
        left_corner_2 = pt(133)
        right_corner_1 = pt(362)
        right_corner_2 = pt(263)

        left_center = ((left_corner_1 + left_corner_2) / 2).astype(np.int32)
        right_center = ((right_corner_1 + right_corner_2) / 2).astype(np.int32)

        left_radius = max(2, int(np.linalg.norm(left_corner_1 - left_corner_2) * 0.18))
        right_radius = max(2, int(np.linalg.norm(right_corner_1 - right_corner_2) * 0.18))

    # Sadece iris kısmını boya
    iris_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_mask, tuple(left_center), left_radius, 255, -1)
    cv2.circle(iris_mask, tuple(right_center), right_radius, 255, -1)

    # Göz dışına taşmasın
    mask = cv2.bitwise_and(iris_mask, eye_mask)

    # Yumuşatma
    blur_size = max(7, int(min(h, w) * 0.01))
    if blur_size % 2 == 0:
        blur_size += 1

    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Biraz doğal dursun
    final_intensity = max(0.0, min(1.0, intensity)) * 0.65

    return blend_mask(image, mask, hex_to_bgr(color_hex), final_intensity)


def apply_makeup_pipeline(image, landmarks, makeup_type, color_hex="#d96b86", intensity=0.5):
    if makeup_type == "lip_color":
        return apply_lip_color(image, landmarks, color_hex=color_hex, intensity=intensity)
    elif makeup_type == "blush":
        return apply_blush(image, landmarks, color_hex=color_hex, intensity=intensity)
    elif makeup_type == "eyeshadow":
        return apply_eyeshadow(image, landmarks, color_hex=color_hex, intensity=intensity)

    return image