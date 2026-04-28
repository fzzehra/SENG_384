import cv2
import numpy as np
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Triangle = Tuple[int, int, int]


class WarpingError(Exception):
    """Raised when warping cannot be completed safely."""


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
    "face_slimming": {
        "left_cheek_jaw": [172, 136, 150, 149],
        "right_cheek_jaw": [397, 365, 379, 378],
        "chin": [152],
    },
}


def _clip_point(point: np.ndarray, width: int, height: int) -> np.ndarray:
    point[0] = np.clip(point[0], 0, width - 1)
    point[1] = np.clip(point[1], 0, height - 1)
    return point


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
        dst_tri.astype(np.float32)
    )
    return cv2.warpAffine(
        src,
        warp_mat,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


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
        corners = FEATURE_GROUPS[expression]["corners"]
        upper_lip = FEATURE_GROUPS[expression]["upper_lip"]
        lower_lip = FEATURE_GROUPS[expression]["lower_lip"]

        dx = 35.0 * intensity
        dy = 28.0 * intensity

        pts[corners[0]] += np.array([-dx, -dy], dtype=np.float32)
        pts[corners[1]] += np.array([dx, -dy], dtype=np.float32)

        for idx in upper_lip:
            pts[idx] += np.array([0.0, -10.0 * intensity], dtype=np.float32)

        for idx in lower_lip:
            pts[idx] += np.array([0.0, 8.0 * intensity], dtype=np.float32)

    elif expression == "eyebrow_raise":
        for idx in FEATURE_GROUPS[expression]["left_brow"] + FEATURE_GROUPS[expression]["right_brow"]:
            pts[idx] += np.array([0.0, -15.0 * intensity], dtype=np.float32)

    elif expression == "lip_widen":
        corners = FEATURE_GROUPS[expression]["corners"]
        pts[corners[0]] += np.array([-22.0 * intensity, 0.0], dtype=np.float32)
        pts[corners[1]] += np.array([22.0 * intensity, 0.0], dtype=np.float32)

        for idx in FEATURE_GROUPS[expression]["upper_lip"]:
            pts[idx] += np.array([0.0, -5.0 * intensity], dtype=np.float32)

        for idx in FEATURE_GROUPS[expression]["lower_lip"]:
            pts[idx] += np.array([0.0, 5.0 * intensity], dtype=np.float32)

    elif expression == "face_slimming":
        center_x = np.mean(pts[:, 0])

        for idx in FEATURE_GROUPS[expression]["left_cheek_jaw"]:
            pts[idx][0] += (center_x - pts[idx][0]) * 0.25 * intensity

        for idx in FEATURE_GROUPS[expression]["right_cheek_jaw"]:
            pts[idx][0] += (center_x - pts[idx][0]) * 0.25 * intensity

        for idx in FEATURE_GROUPS[expression]["chin"]:
            pts[idx][1] -= 6.0 * intensity

    for i in range(len(pts)):
        pts[i] = _clip_point(pts[i], w, h)

    return pts


def delaunay_triangulation(image_shape, landmarks: Sequence[Point]) -> List[Triangle]:
    h, w = image_shape[:2]
    rect = (0, 0, int(w), int(h))
    subdiv = cv2.Subdiv2D(rect)

    pts = np.array(landmarks, dtype=np.float32)

    for x, y in pts:
        px = float(np.clip(x, 0, w - 1))
        py = float(np.clip(y, 0, h - 1))
        subdiv.insert((px, py))

    triangle_list = subdiv.getTriangleList()
    triangles: List[Triangle] = []
    seen = set()

    def find_index(point: np.ndarray) -> Optional[int]:
        distances = np.linalg.norm(pts - point, axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] < 1.5:
            return idx
        return None

    for t in triangle_list:
        p1 = np.array([t[0], t[1]], dtype=np.float32)
        p2 = np.array([t[2], t[3]], dtype=np.float32)
        p3 = np.array([t[4], t[5]], dtype=np.float32)

        if not (
            0 <= p1[0] < w and 0 <= p1[1] < h and
            0 <= p2[0] < w and 0 <= p2[1] < h and
            0 <= p3[0] < w and 0 <= p3[1] < h
        ):
            continue

        i1 = find_index(p1)
        i2 = find_index(p2)
        i3 = find_index(p3)

        if None in (i1, i2, i3):
            continue

        tri = tuple(sorted((i1, i2, i3)))

        if len(set(tri)) == 3 and tri not in seen:
            seen.add(tri)
            triangles.append(tri)

    if not triangles:
        raise WarpingError("No Delaunay triangles could be created.")

    return triangles


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

        accum_mask[y1:y2, x1:x2] = np.maximum(
            accum_mask[y1:y2, x1:x2],
            mask
        )

    if image.ndim == 3:
        accum_mask_3 = accum_mask[..., None]
        output = output + image * (1.0 - accum_mask_3)
    else:
        output = output + image * (1.0 - accum_mask)

    return np.clip(output, 0, 255).astype(image.dtype)


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

    dst_landmarks = modify_landmarks(
        src_landmarks,
        image.shape,
        expression,
        intensity
    )

    triangles = delaunay_triangulation(image.shape, src_landmarks)
    warped = warp_triangles(image, src_landmarks, dst_landmarks, triangles)

    return warped, dst_landmarks, triangles