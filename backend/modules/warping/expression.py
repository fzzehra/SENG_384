from typing import Sequence

import numpy as np

from .landmarks import modify_landmarks
from .slimming import apply_face_slimming_smooth
from .triangulation import delaunay_triangulation
from .transforms import warp_triangles
from .types import Point


def apply_expression(
    image: np.ndarray,
    landmarks: Sequence[Point],
    expression: str,
    intensity: float = 0.5,
):
    if image is None or image.size == 0:
        raise ValueError("image is empty")

    src_landmarks = np.array(landmarks, dtype=np.float32)

    if len(src_landmarks) < 3:
        raise ValueError("At least 3 landmarks are required.")

    if expression == "face_slimming":
        warped = apply_face_slimming_smooth(
            image=image,
            landmarks=src_landmarks,
            intensity=intensity,
        )
        return warped, src_landmarks, []

    dst_landmarks = modify_landmarks(
        src_landmarks,
        image.shape,
        expression,
        intensity,
    )

    triangles = delaunay_triangulation(image.shape, src_landmarks)
    warped    = warp_triangles(image, src_landmarks, dst_landmarks, triangles)

    return warped, dst_landmarks, triangles