from typing import Sequence
import cv2
import numpy as np
from .types import Point

def apply_face_slimming_smooth(
    image: np.ndarray,
    landmarks: Sequence[Point],
    intensity: float = 0.5,
) -> np.ndarray:
    h, w = image.shape[:2]
    intensity = float(np.clip(intensity, -1.0, 1.0))
    pts = np.array(landmarks, dtype=np.float32)

    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    left_indices  = [234, 93, 132, 58, 172, 136, 150, 149]
    right_indices = [454, 323, 361, 288, 397, 365, 379, 378]

    radius   = 85.0
    strength = 14.0 * abs(intensity)
    sign = 1 if intensity >= 0 else -1

    for left_idx, right_idx in zip(left_indices, right_indices):
        if left_idx >= len(pts) or right_idx >= len(pts):
            continue

        for point, direction in [(pts[left_idx], sign), (pts[right_idx], -sign)]:
            px, py = point
            dx   = map_x - px
            dy   = map_y - py
            dist = np.sqrt(dx * dx + dy * dy)
            mask    = dist < radius
            falloff = np.clip(1 - (dist / radius) ** 2, 0, 1)
            pull    = strength * falloff * mask
            map_x  += direction * pull

    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)
