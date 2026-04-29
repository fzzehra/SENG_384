from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

from .types import Point, Triangle


def _bounding_rect(points: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    return int(x), int(y), int(w), int(h)


def _apply_affine_transform(
    src: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    warp_mat = cv2.getAffineTransform(
        src_tri.astype(np.float32),
        dst_tri.astype(np.float32),
    )
    return cv2.warpAffine(
        src,
        warp_mat,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def warp_triangles(
    image: np.ndarray,
    src_landmarks: Sequence[Point],
    dst_landmarks: Sequence[Point],
    triangles: Iterable[Triangle],
) -> np.ndarray:
    src_pts = np.array(src_landmarks, dtype=np.float32)
    dst_pts = np.array(dst_landmarks, dtype=np.float32)

    output = np.zeros_like(image)
    accum_mask = np.zeros(image.shape[:2], dtype=np.float32)

    for tri in triangles:
        src_tri = src_pts[list(tri)]
        dst_tri = dst_pts[list(tri)]

        sx, sy, sw, sh = _bounding_rect(src_tri)
        dx, dy, dw, dh = _bounding_rect(dst_tri)

        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            continue

        src_patch = image[sy:sy + sh, sx:sx + sw]

        if src_patch.size == 0:
            continue

        src_tri_local = src_tri - np.array([sx, sy], dtype=np.float32)
        dst_tri_local = dst_tri - np.array([dx, dy], dtype=np.float32)

        warped_patch = _apply_affine_transform(
            src_patch,
            src_tri_local,
            dst_tri_local,
            (dw, dh),
        )

        mask = np.zeros((dh, dw), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_local), 1.0, lineType=cv2.LINE_AA)

        y1, y2 = dy, dy + dh
        x1, x2 = dx, dx + dw

        if y1 < 0 or x1 < 0 or y2 > image.shape[0] or x2 > image.shape[1]:
            continue

        if image.ndim == 3:
            mask_3 = mask[..., None]
            output[y1:y2, x1:x2] = (
                output[y1:y2, x1:x2] * (1.0 - mask_3)
                + warped_patch * mask_3
            )
        else:
            output[y1:y2, x1:x2] = (
                output[y1:y2, x1:x2] * (1.0 - mask)
                + warped_patch * mask
            )

        accum_mask[y1:y2, x1:x2] = np.maximum(accum_mask[y1:y2, x1:x2], mask)

    if image.ndim == 3:
        output = output + image * (1.0 - accum_mask[..., None])
    else:
        output = output + image * (1.0 - accum_mask)

    return np.clip(output, 0, 255).astype(image.dtype)