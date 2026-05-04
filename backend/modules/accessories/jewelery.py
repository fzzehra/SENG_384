import os
import cv2
import numpy as np


def _safe_asset_name(name):
    if not name:
        return ""

    name = os.path.basename(name)

    if not name.lower().endswith(".png"):
        return ""

    return name


def _load_rgba(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Accessory asset not found: {path}")

    overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if overlay is None:
        raise ValueError(f"Accessory image could not be read: {path}")

    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)

    if overlay.shape[2] == 3:
        alpha = np.full(overlay.shape[:2], 255, dtype=np.uint8)
        overlay = np.dstack([overlay, alpha])

    return overlay


def _resize_keep_aspect(rgba, target_width):
    h, w = rgba.shape[:2]

    if w <= 0 or h <= 0:
        return rgba

    scale = target_width / float(w)
    target_height = max(1, int(h * scale))
    target_width = max(1, int(target_width))

    return cv2.resize(rgba, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _rotate_rgba(rgba, angle_deg):
    if abs(angle_deg) < 0.5:
        return rgba

    h, w = rgba.shape[:2]
    center = (w / 2, h / 2)

    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        rgba,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated


def _overlay_rgba_on_bgr(base_bgr, overlay_rgba, center_xy, opacity=1.0):
    output = base_bgr.copy()

    h, w = output.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    cx, cy = center_xy
    cx = int(cx)
    cy = int(cy)

    x1 = cx - ow // 2
    y1 = cy - oh // 2
    x2 = x1 + ow
    y2 = y1 + oh

    # Clip to image bounds
    ox1 = max(0, -x1)
    oy1 = max(0, -y1)
    ox2 = ow - max(0, x2 - w)
    oy2 = oh - max(0, y2 - h)

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)

    if x1c >= x2c or y1c >= y2c:
        return output

    overlay_crop = overlay_rgba[oy1:oy2, ox1:ox2]

    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)
    overlay_alpha = overlay_crop[:, :, 3].astype(np.float32) / 255.0
    overlay_alpha = np.clip(overlay_alpha * opacity, 0.0, 1.0)
    overlay_alpha = overlay_alpha[:, :, None]

    roi = output[y1c:y2c, x1c:x2c].astype(np.float32)

    blended = roi * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha
    output[y1c:y2c, x1c:x2c] = np.clip(blended, 0, 255).astype(np.uint8)

    return output


def _dist(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def _face_metrics(landmarks):
    if not landmarks or len(landmarks) < 468:
        raise ValueError("Not enough landmarks for jewelry placement.")

    left_face = np.array(landmarks[234], dtype=np.float32)
    right_face = np.array(landmarks[454], dtype=np.float32)
    chin = np.array(landmarks[152], dtype=np.float32)
    nose = np.array(landmarks[1], dtype=np.float32)

    face_width = max(60.0, _dist(left_face, right_face))

    # Approximate head tilt by cheek line angle
    dx = right_face[0] - left_face[0]
    dy = right_face[1] - left_face[1]
    tilt_deg = np.degrees(np.arctan2(dy, dx))

    return {
        "left_face": left_face,
        "right_face": right_face,
        "chin": chin,
        "nose": nose,
        "face_width": face_width,
        "tilt_deg": tilt_deg,
    }


def apply_earring(image, landmarks, asset_path, intensity=1.0):
    metrics = _face_metrics(landmarks)

    rgba = _load_rgba(asset_path)

    face_width = metrics["face_width"]
    tilt = metrics["tilt_deg"]

    # Earring size relative to face width
    target_width = int(face_width * 0.15)
    earring = _resize_keep_aspect(rgba, target_width)

    # Approximate ear-lobe positions from face contour landmarks.
    # Works best on frontal faces where ears are not fully covered.
    left_face = metrics["left_face"]
    right_face = metrics["right_face"]
    chin = metrics["chin"]

    y_anchor = int(left_face[1] * 0.45 + chin[1] * 0.55)

    left_center = (
        int(left_face[0] - face_width * 0.06),
        y_anchor
    )

    right_center = (
        int(right_face[0] + face_width * 0.06),
        y_anchor
    )

    left_earring = _rotate_rgba(earring, tilt)
    right_earring = _rotate_rgba(earring, tilt)

    output = image.copy()

    output = _overlay_rgba_on_bgr(
        output,
        left_earring,
        left_center,
        opacity=intensity
    )

    output = _overlay_rgba_on_bgr(
        output,
        right_earring,
        right_center,
        opacity=intensity
    )

    return output


def apply_necklace(image, landmarks, asset_path, intensity=1.0):
    metrics = _face_metrics(landmarks)

    rgba = _load_rgba(asset_path)

    face_width = metrics["face_width"]
    chin = metrics["chin"]
    nose = metrics["nose"]

    # Necklace width should be wider than face
    target_width = int(face_width * 1.35)
    necklace = _resize_keep_aspect(rgba, target_width)

    # Place below chin, centered around face axis
    center_x = int((nose[0] + chin[0]) / 2)
    center_y = int(chin[1] + face_width * 0.35)

    output = _overlay_rgba_on_bgr(
        image,
        necklace,
        (center_x, center_y),
        opacity=intensity
    )

    return output


def apply_jewelry_pipeline(image, landmarks, jewelry_type, item_name, intensity=1.0):
    item_name = _safe_asset_name(item_name)

    if not item_name:
        print("JEWELRY: no valid item selected")
        return image

    if jewelry_type == "earring":
        asset_path = os.path.join("static", "accessories", "earrings", item_name)
        print("JEWELRY: applying earring", asset_path)
        return apply_earring(image, landmarks, asset_path, intensity=intensity)

    if jewelry_type == "necklace":
        asset_path = os.path.join("static", "accessories", "necklaces", item_name)
        print("JEWELRY: applying necklace", asset_path)
        return apply_necklace(image, landmarks, asset_path, intensity=intensity)

    return image