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


def apply_blush(image, landmarks, color_hex="#f4a7b9", intensity=0.35):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    def p(idx):
        return np.array(get_point_xy(landmarks, idx, w, h), dtype=np.float32)

    nose = p(1)
    left_anchor = p(50)
    right_anchor = p(280)
    left_face_edge = p(234)
    right_face_edge = p(454)

    left_center = (left_anchor * 0.72 + nose * 0.28).astype(np.int32)
    right_center = (right_anchor * 0.72 + nose * 0.28).astype(np.int32)

    shift_down = int(h * 0.015)
    left_center[1] += shift_down
    right_center[1] += shift_down

    face_width = np.linalg.norm(right_face_edge - left_face_edge)
    axis_x = max(24, int(face_width * 0.095))
    axis_y = max(16, int(face_width * 0.070))

    cv2.ellipse(mask, tuple(left_center), (axis_x, axis_y), -18, 0, 360, 255, -1)
    cv2.ellipse(mask, tuple(right_center), (axis_x, axis_y), 18, 0, 360, 255, -1)

    face_outline_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454,
        323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132,
        93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    face_poly = get_points(landmarks, face_outline_idx, w, h)

    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_poly], 255)

    mask = cv2.bitwise_and(mask, face_mask)

    blur_size = max(41, int(min(h, w) * 0.08))
    if blur_size % 2 == 0:
        blur_size += 1

    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

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

    print(f"MAKEUP: unknown makeup_type: {makeup_type}")
    return image