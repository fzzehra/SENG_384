import os
import cv2
import numpy as np


def get_point_xy(landmarks, idx, w, h):
    point = landmarks[idx]

    if hasattr(point, "x") and hasattr(point, "y"):
        return [int(point.x * w), int(point.y * h)]

    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
        return [int(point[0]), int(point[1])]

    raise ValueError(f"Unsupported landmark format at index {idx}: {point}")


def overlay_png(base, overlay, x, y, scale=1.0, angle=0):
    if overlay is None:
        return base

    h, w = base.shape[:2]
    oh, ow = overlay.shape[:2]

    new_w = max(1, int(ow * scale))
    new_h = max(1, int(oh * scale))

    overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    center = (new_w // 2, new_h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    overlay = cv2.warpAffine(
        overlay,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    x1 = int(x - new_w // 2)
    y1 = int(y - new_h // 2)
    x2 = x1 + new_w
    y2 = y1 + new_h

    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return base

    ox1 = max(0, -x1)
    oy1 = max(0, -y1)
    ox2 = new_w - max(0, x2 - w)
    oy2 = new_h - max(0, y2 - h)

    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = bx1 + (ox2 - ox1)
    by2 = by1 + (oy2 - oy1)

    ov = overlay[oy1:oy2, ox1:ox2]

    if ov.shape[2] == 4:
        alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
        rgb = ov[:, :, :3].astype(np.float32)
    else:
        alpha = np.ones((*ov.shape[:2], 1), dtype=np.float32)
        rgb = ov.astype(np.float32)

    roi = base[by1:by2, bx1:bx2].astype(np.float32)
    blended = roi * (1 - alpha) + rgb * alpha
    base[by1:by2, bx1:bx2] = np.clip(blended, 0, 255).astype(np.uint8)

    return base


def apply_piercing(image, landmarks, piercing_type, item_path):
    h, w = image.shape[:2]
    output = image.copy()

    png = cv2.imread(item_path, cv2.IMREAD_UNCHANGED)
    if png is None:
        print("PIERCING PNG NOT FOUND:", item_path)
        return output

    def pt(i):
        return np.array(get_point_xy(landmarks, i, w, h), dtype=np.float32)

    face_w = np.linalg.norm(pt(454) - pt(234))

    if piercing_type == "eyebrow_piercing":
        inner = pt(336)
        outer = pt(300)
        center = outer * 0.72 + inner * 0.28

        angle = np.degrees(np.arctan2(outer[1] - inner[1], outer[0] - inner[0]))
        scale = face_w * 0.0018

        return overlay_png(output, png, center[0], center[1], scale=scale, angle=angle)

    if piercing_type == "septum_piercing":
        nose = pt(2)
        left_nostril = pt(97)
        right_nostril = pt(326) if len(landmarks) > 326 else pt(98)

        nose_w = np.linalg.norm(right_nostril - left_nostril)
        center = nose.copy()
        center[1] += nose_w * 0.18

        scale = nose_w * 0.010

        return overlay_png(output, png, center[0], center[1], scale=scale, angle=0)

    if piercing_type == "lip_piercing":
        lower_lip = pt(17)
        mouth_left = pt(61)
        mouth_right = pt(291)

        mouth_w = np.linalg.norm(mouth_right - mouth_left)

        left_center = lower_lip.copy()
        right_center = lower_lip.copy()

        left_center[0] = mouth_left[0] + mouth_w * 0.18
        right_center[0] = mouth_right[0] - mouth_w * 0.18

        left_center[1] = lower_lip[1] + mouth_w * 0.04
        right_center[1] = lower_lip[1] + mouth_w * 0.04

        scale = mouth_w * 0.0048

        output = overlay_png(
            output,
            png,
            left_center[0],
            left_center[1],
            scale=scale,
            angle=0
        )

        output = overlay_png(
            output,
            png,
            right_center[0],
            right_center[1],
            scale=scale,
            angle=0
        )

        return output

    return output