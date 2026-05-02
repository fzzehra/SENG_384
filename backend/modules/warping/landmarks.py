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
        corners = [61, 291]
        upper_lip = [13, 312, 82, 11, 302, 72]
        lower_lip = [14, 317, 87, 12, 307, 77]

        dx = 25.0 * intensity
        dy = 14.0 * intensity

        pts[corners[0]] += np.array([-dx, -dy], dtype=np.float32)
        pts[corners[1]] += np.array([dx, -dy], dtype=np.float32)

        for idx in upper_lip:
            dist_to_center = abs(pts[idx][0] - np.mean(pts[corners, 0]))
            pts[idx] += np.array([0.0, -8.0 * intensity * (1 - dist_to_center / 150)], dtype=np.float32)

        for idx in lower_lip:
            dist_to_center = abs(pts[idx][0] - np.mean(pts[corners, 0]))
            pts[idx] += np.array([0.0, 5.0 * intensity * (1 - dist_to_center / 150)], dtype=np.float32)

        around_mouth = [62, 292, 76, 306, 184, 408]
        for idx in around_mouth:
            pts[idx] += np.array([0.0, -4.0 * intensity], dtype=np.float32)

    elif expression == "eyebrow_raise":
        for idx in FEATURE_GROUPS[expression]["left_brow"] + FEATURE_GROUPS[expression]["right_brow"]:
            pts[idx] += np.array([0.0, -15.0 * intensity], dtype=np.float32)

    elif expression == "lip_widen":
        dx = 15.0 * intensity
        
        # Left corner and inner corner
        for idx in [61, 78]: 
            pts[idx] += np.array([-dx, 0.0], dtype=np.float32)
        # Left nearby points (upper/lower lip + skin) to smooth the stretch
        for idx in [185, 146, 191, 95, 57]: 
            pts[idx] += np.array([-dx * 0.6, 0.0], dtype=np.float32)
            
        # Right corner and inner corner
        for idx in [291, 308]: 
            pts[idx] += np.array([dx, 0.0], dtype=np.float32)
        # Right nearby points
        for idx in [409, 375, 415, 324, 287]: 
            pts[idx] += np.array([dx * 0.6, 0.0], dtype=np.float32)

        for idx in FEATURE_GROUPS[expression]["upper_lip"]:
            pts[idx] += np.array([0.0, -4.0 * intensity], dtype=np.float32)

        for idx in FEATURE_GROUPS[expression]["lower_lip"]:
            pts[idx] += np.array([0.0, 4.0 * intensity], dtype=np.float32)

    for i in range(len(pts)):
        pts[i] = _clip_point(pts[i], w, h)

    return pts