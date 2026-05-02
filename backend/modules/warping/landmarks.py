from typing import Sequence

import numpy as np

from .types import Point


FEATURE_GROUPS = {
    "smile": {
        "corners": [61, 291],
        "upper_lip": [13, 312, 82],
        "lower_lip": [14, 317, 87],
    },
    "eyebrow_raise": {
        "left_brow": [70, 63, 105, 66, 107],
        "right_brow": [336, 296, 334, 293, 300],
    },
    "lip_widen": {
        "corners": [61, 291],
        "upper_lip": [0, 37, 267, 13, 82, 312],
        "lower_lip": [17, 84, 314, 14, 87, 317],
    },
}


def _clip_point(point: np.ndarray, width: int, height: int) -> np.ndarray:
    point[0] = np.clip(point[0], 0, width - 1)
    point[1] = np.clip(point[1], 0, height - 1)
    return point


def modify_landmarks(
    landmarks: Sequence[Point],
    image_shape,
    expression: str,
    intensity: float = 0.5,
) -> np.ndarray:
    if expression not in FEATURE_GROUPS:
        raise ValueError(f"Unsupported expression: {expression}")

    h, w = image_shape[:2]
    intensity = float(np.clip(intensity, 0.0, 1.0))
    pts = np.array(landmarks, dtype=np.float32).copy()

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("landmarks must have shape (N, 2)")

    if expression == "smile":
        left_corner = 61
        right_corner = 291

        dx = 12.0 * intensity
        dy = 7.0 * intensity

        pts[left_corner] += np.array([-dx, -dy], dtype=np.float32)
        pts[right_corner] += np.array([dx, -dy], dtype=np.float32)

        left_neighbors = [78, 95, 191, 185, 40, 80, 88, 178]
        right_neighbors = [308, 324, 415, 409, 270, 310, 318, 402]

        for idx in left_neighbors:
            pts[idx] += np.array([-dx * 0.45, -dy * 0.45], dtype=np.float32)

        for idx in right_neighbors:
            pts[idx] += np.array([dx * 0.45, -dy * 0.45], dtype=np.float32)

        upper_lip = [13, 312, 82, 0, 37, 267]
        for idx in upper_lip:
            pts[idx] += np.array([0.0, -3.0 * intensity], dtype=np.float32)

        lower_lip = [14, 317, 87, 17, 84, 314]
        for idx in lower_lip:
            pts[idx] += np.array([0.0, 2.0 * intensity], dtype=np.float32)

        around_mouth = [62, 292, 76, 306, 184, 408, 57, 287]
        for idx in around_mouth:
            pts[idx] += np.array([0.0, -2.5 * intensity], dtype=np.float32)

    elif expression == "eyebrow_raise":
        brow_points = (
            FEATURE_GROUPS[expression]["left_brow"]
            + FEATURE_GROUPS[expression]["right_brow"]
        )

        for idx in brow_points:
            pts[idx] += np.array([0.0, -15.0 * intensity], dtype=np.float32)

    elif expression == "lip_widen":
        # Dudak Kalınlaştırma (Vertical Thickening) - Ağzı açmadan hacim verme
        thickness = 12.0 * intensity
        mouth_center_x = np.mean(pts[[61, 291], 0])

        # Dış hatlar (Hacmi asıl veren kısımlar)
        upper_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lower_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # İç hatlar (Ağız birleşim çizgisi - Az hareket etmeli ki yırtılma olmasın)
        upper_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        lower_inner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

        # Üst dış hattı yukarı taşı
        for idx in upper_outer:
            dist_to_center = abs(pts[idx][0] - mouth_center_x)
            weight = max(0.3, 1 - dist_to_center / 130)
            pts[idx] += np.array([0.0, -thickness * weight], dtype=np.float32)

        # Üst iç hattı çok az yukarı taşı (Yırtılmayı önler)
        for idx in upper_inner:
            dist_to_center = abs(pts[idx][0] - mouth_center_x)
            weight = max(0.1, 1 - dist_to_center / 130)
            pts[idx] += np.array([0.0, -thickness * 0.3 * weight], dtype=np.float32)

        # Alt dış hattı aşağı taşı
        for idx in lower_outer:
            dist_to_center = abs(pts[idx][0] - mouth_center_x)
            weight = max(0.3, 1 - dist_to_center / 130)
            pts[idx] += np.array([0.0, thickness * weight], dtype=np.float32)

        # Alt iç hattı çok az aşağı taşı
        for idx in lower_inner:
            dist_to_center = abs(pts[idx][0] - mouth_center_x)
            weight = max(0.1, 1 - dist_to_center / 130)
            pts[idx] += np.array([0.0, thickness * 0.3 * weight], dtype=np.float32)

        # Çevre dokuları (Burun altı ve çene) genişçe esnet
        surround_up = [164, 2, 94, 327, 48, 278, 57, 287]
        for idx in surround_up:
            pts[idx] += np.array([0.0, -thickness * 0.2], dtype=np.float32)

        surround_down = [18, 200, 199, 175, 152, 377, 400, 410]
        for idx in surround_down:
            pts[idx] += np.array([0.0, thickness * 0.2], dtype=np.float32)

    for i in range(len(pts)):
        pts[i] = _clip_point(pts[i], w, h)

    return pts