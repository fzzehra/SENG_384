from typing import List, Optional, Sequence

import cv2
import numpy as np

from .types import Point, Triangle, WarpingError


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