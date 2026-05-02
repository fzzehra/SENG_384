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
        "upper_lip": [0, 37, 267],
        "lower_lip": [17, 84, 314],
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
        dx = 15.0 * intensity

        for idx in [61, 78]:
            pts[idx] += np.array([-dx, 0.0], dtype=np.float32)

        for idx in [185, 146, 191, 95, 57]:
            pts[idx] += np.array([-dx * 0.6, 0.0], dtype=np.float32)

        for idx in [291, 308]:
            pts[idx] += np.array([dx, 0.0], dtype=np.float32)

        for idx in [409, 375, 415, 324, 287]:
            pts[idx] += np.array([dx * 0.6, 0.0], dtype=np.float32)

        for idx in FEATURE_GROUPS[expression]["upper_lip"]:
            pts[idx] += np.array([0.0, -4.0 * intensity], dtype=np.float32)

        for idx in FEATURE_GROUPS[expression]["lower_lip"]:
            pts[idx] += np.array([0.0, 4.0 * intensity], dtype=np.float32)

    for i in range(len(pts)):
        pts[i] = _clip_point(pts[i], w, h)

    return pts