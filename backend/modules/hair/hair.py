# backend/modules/hair/hair.py

import cv2
import numpy as np

_FACE_CONTOUR = [10,338,297,332,284,251,389,356,454,323,361,288,
                 397,365,379,378,400,377,152,148,176,149,150,136,
                 172,58,132,93,234,127,162,21,54,103,67,109]


def _hex_to_bgr(hex_color: str):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)



def _face_mask(shape, landmarks):
    h, w = shape[:2]
    pts = np.array([landmarks[i] for i in _FACE_CONTOUR], dtype=np.int32)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m


# ---------------------------------------------------------------------------
# SAÇ RENKLENDİRME
# ---------------------------------------------------------------------------

def apply_hair_color(image, landmarks, color_hex='#000000', intensity=0.5):
    h, w = image.shape[:2]
    cbgr = _hex_to_bgr(color_hex)

    # Use segmentation model for complete hair detection
    hair_mask = None
    try:
        from .segmentation import get_hair_mask
        hair_mask = get_hair_mask(image)
    except Exception as e:
        print(f"[hair_color] segmentation failed: {e}, using fallback")
        hair_mask = None

    if hair_mask is not None and hair_mask.sum() > 0:
        fm = _face_mask(image.shape, landmarks)
        hair_mask = cv2.subtract(hair_mask, fm)
        # Çene altını kapat
        chin_y = landmarks[152][1]
        cutoff_y = min(h, chin_y + int(h * 0.03))
        hair_mask[cutoff_y:, :] = 0
        hair_mask = cv2.GaussianBlur(hair_mask, (15, 15), 0)
    else:
        # Fallback: HSV-based detection
        top  = landmarks[10];  chin = landmarks[152]
        lt   = landmarks[234]; rt   = landmarks[454]
        fw   = abs(rt[0]-lt[0]); fh = abs(chin[1]-top[1])
        cx   = (lt[0]+rt[0])//2
        lim  = np.zeros((h, w), dtype=np.uint8)
        cy   = (top[1]+chin[1])//2
        cv2.ellipse(lim, (cx, cy-fh//8), (int(fw*.70), int(fh*.72)), 0, 0, 360, 255, -1)
        ay   = max(0, top[1]-int(fh*.5))
        cv2.rectangle(lim, (cx-int(fw*.65), ay), (cx+int(fw*.65), top[1]+int(fh*.1)), 255, -1)
        lim  = cv2.GaussianBlur(lim, (41, 41), 0)
        fm   = _face_mask(image.shape, landmarks)
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dm   = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([180, 255, 120]))
        hair_mask = cv2.GaussianBlur(cv2.subtract(cv2.bitwise_and(dm, lim), fm), (21, 21), 0)

    # "Color" blend mode: keep luminosity, replace hue+saturation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(
        np.uint8([[list(cbgr)]]), cv2.COLOR_BGR2LAB
    ).astype(np.float32)[0, 0]

    alpha = hair_mask.astype(np.float32) / 255.0 * (intensity ** 2)
    lab[:, :, 1] = lab[:, :, 1] * (1 - alpha) + target_lab[1] * alpha
    lab[:, :, 2] = lab[:, :, 2] * (1 - alpha) + target_lab[2] * alpha

    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# KAŞ RENKLENDİRME
# ---------------------------------------------------------------------------

EYEBROW_LANDMARKS = {
    "left":  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
}


def apply_eyebrow_color(image, landmarks, color_hex='#000000', intensity=0.5):
    h, w = image.shape[:2]
    cbgr = _hex_to_bgr(color_hex)
    mask = np.zeros((h, w), dtype=np.uint8)
    for _, idx in EYEBROW_LANDMARKS.items():
        sm  = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([landmarks[i] for i in idx], dtype=np.int32)
        cv2.fillPoly(sm, [cv2.convexHull(pts)], 255)
        mask = cv2.bitwise_or(mask,
               cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))))
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark = np.clip((0.5-gray) / 0.3, 0.0, 1.0)
    a    = np.clip(mask.astype(np.float32) / 255.0 * dark * intensity * .9, 0.0, 0.9)
    cl   = np.full_like(image, cbgr, dtype=np.float32)
    return np.clip(image.astype(np.float32) * (1-a[:, :, None]) + cl * a[:, :, None], 0, 255).astype(np.uint8)
