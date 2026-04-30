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

    # Stronger visible blending
    result = _blend_color(image, lipstick_mask, color=color, alpha=0.75 * intensity)

    # Slight richness boost
    result = cv2.addWeighted(image, 1.0 - (0.12 * intensity), result, 0.12 * intensity + 1.0, 0)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_eyeshadow(image, landmarks, color=(180, 0, 180), intensity=0.5):
    """
    Apply purple eyeshadow above the eyes.
    """
    pts = np.array(landmarks, dtype=np.int32)

    # Left eye / eyebrow region
    left_eye_upper = [33, 246, 161, 160, 159, 158, 157, 173, 133]
    left_brow = [70, 63, 105, 66, 107]

    # Right eye / eyebrow region
    right_eye_upper = [263, 466, 388, 387, 386, 385, 384, 398, 362]
    right_brow = [336, 296, 334, 293, 300]

    def build_shadow_region(eye_indices, brow_indices):
        eye_pts = pts[eye_indices]
        brow_pts = pts[brow_indices]

        eye_center_y = np.mean(eye_pts[:, 1])

        # Move eyebrow points a bit downward toward eyelid area
        adjusted_brow = brow_pts.copy()
        adjusted_brow[:, 1] = (0.65 * adjusted_brow[:, 1] + 0.35 * eye_center_y).astype(np.int32)

        region = np.vstack([eye_pts, adjusted_brow[::-1]])
        return region

    left_region = build_shadow_region(left_eye_upper, left_brow)
    right_region = build_shadow_region(right_eye_upper, right_brow)

    left_mask = _create_soft_mask(image.shape, left_region, blur_ksize=21)
    right_mask = _create_soft_mask(image.shape, right_region, blur_ksize=21)

    shadow_mask = cv2.max(left_mask, right_mask)

    # Stronger visible purple
    result = _blend_color(image, shadow_mask, color=color, alpha=0.60 * intensity)

    # Slight smooth enhancement
    result = cv2.addWeighted(image, 1.0 - (0.08 * intensity), result, 0.08 * intensity + 1.0, 0)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_makeup_pipeline(image, landmarks, makeup_type, intensity=0.5):
    if makeup_type == "lipstick":
        print("MAKEUP: applying lipstick")
        return apply_lipstick(image, landmarks, color=(0, 0, 255), intensity=intensity)

    elif makeup_type == "eyeshadow":
        print("MAKEUP: applying eyeshadow")
        return apply_eyeshadow(image, landmarks, color=(180, 0, 180), intensity=intensity)

    return image