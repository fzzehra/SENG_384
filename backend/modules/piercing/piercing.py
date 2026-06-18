import os
import cv2
import numpy as np

def get_point_xy(landmarks, idx, w, h):
    point = landmarks[idx]
    if hasattr(point, "x") and hasattr(point, "y"):
        return [int(point.x * w), int(point.y * h)]
    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
        return [int(point[0]), int(point[1])]
    raise ValueError(f"Unsupported landmark format at index {idx}: {point}")

def overlay_png(base, overlay, x, y, scale=1.0, angle=0):
    if overlay is None:
        return base

    h, w = base.shape[:2]
    oh, ow = overlay.shape[:2]

    new_w = max(1, int(ow * scale))
    new_h = max(1, int(oh * scale))
    overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if angle!= 0:
        center = (new_w // 2, new_h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        bound_w = int((new_h * sin) + (new_w * cos))
        bound_h = int((new_h * cos) + (new_w * sin))
        
        matrix[0, 2] += (bound_w / 2) - center[0]
        matrix[1, 2] += (bound_h / 2) - center[1]
        
        overlay = cv2.warpAffine(
            overlay, matrix, (bound_w, bound_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        new_w, new_h = bound_w, bound_h
    
    x1 = int(round(x - new_w / 2))
    y1 = int(round(y - new_h / 2))
    x2 = x1 + new_w
    y2 = y1 + new_h

    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return base

    ox1 = max(0, -x1)
    oy1 = max(0, -y1)
    ox2 = new_w - max(0, x2 - w)
    oy2 = new_h - max(0, y2 - h)

    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = bx1 + (ox2 - ox1)
    by2 = by1 + (oy2 - oy1)

    if ox2 <= ox1 or oy2 <= oy1:
        return base

    ov = overlay[oy1:oy2, ox1:ox2]

    if ov.shape[2] == 4:
        alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
        if alpha.max() < 0.01:
            return base
        rgb = ov[:, :, :3].astype(np.float32)
    else:
        alpha = np.ones((*ov.shape[:2], 1), dtype=np.float32)
        rgb = ov.astype(np.float32)

    roi = base[by1:by2, bx1:bx2].astype(np.float32)
    blended = roi * (1 - alpha) + rgb * alpha
    base[by1:by2, bx1:bx2] = np.clip(blended, 0, 255).astype(np.uint8)

    return base

def apply_piercing(image, landmarks, piercing_type, item_path):
    h, w = image.shape[:2]
    output = image.copy()

    png = cv2.imread(item_path, cv2.IMREAD_UNCHANGED)
    if png is None:
        print(f"[PIERCING ERROR] PNG NOT FOUND: {item_path}")
        return output

    def pt(i):
        return np.array(get_point_xy(landmarks, i, w, h), dtype=np.float32)

    left_face = pt(234)
    right_face = pt(454)
    face_w = np.linalg.norm(right_face - left_face)

    if piercing_type == "eyebrow_piercing":
        brow_inner = pt(336)
        brow_outer = pt(300)
        brow_peak  = pt(296)

        brow_w = np.linalg.norm(brow_outer - brow_inner)
        brow_height = brow_w * 0.25

        # --- AÇI ---
        dx_brow = brow_outer[0] - brow_inner[0]
        dy_brow = brow_outer[1] - brow_inner[1]
        brow_angle = np.degrees(np.arctan2(dy_brow, dx_brow))
        angle = brow_angle + 0

        # --- POZİSYON ---
        center_x = brow_outer[0] * 0.92 + brow_inner[0] * 0.08
        center_y = brow_peak[1] + brow_height * 0.6

        center = np.array([center_x, center_y], dtype=np.float32)

        # --- BOYUT ---
        TARGET_PNG_W = 120
        resize_factor = TARGET_PNG_W / png.shape[1]
        new_h = int(png.shape[0] * resize_factor)
        bgr = cv2.resize(png[:,:,:3], (TARGET_PNG_W, new_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(png[:,:,3], (TARGET_PNG_W, new_h), interpolation=cv2.INTER_AREA)
        png = cv2.merge([bgr[:,:,0], bgr[:,:,1], bgr[:,:,2], alpha])

        target_width = brow_w * 0.65
        scale = target_width / png.shape[1]

        print(f"[EYEBROW] Center: ({center_x:.1f}, {center_y:.1f}), Scale: {scale:.3f}, Angle: {angle:.1f}°")

        output = overlay_png(output, png, center[0], center[1], scale=scale, angle=angle)

              
        return output
      
   
    
    if piercing_type == "septum_piercing":
        left_nostril = pt(97)
        right_nostril = pt(326)
        nose_tip = pt(4)
        columella = pt(2)
        
        nose_w = np.linalg.norm(right_nostril - left_nostril)
        
        center = (left_nostril + right_nostril) / 2
        center[1] = columella[1] + face_w * 0.0
        
        scale = nose_w / png.shape[1] * 2.0
        
        output = overlay_png(output, png, center[0], center[1], scale=scale, angle=0)
        
        # MASKE: üst yay + iki yan giriş noktalarını gizle
        mask = np.zeros((h, w), dtype=np.uint8)

        png_rendered_w = png.shape[1] * scale
        png_rendered_h = png.shape[0] * scale

        # Üst orta — daire
        mask_radius = int(nose_w * 0.38)
        top_x = int(center[0])
        top_y = int(center[1] - png_rendered_h * 0.45)
        cv2.circle(mask, (top_x, top_y), mask_radius, 255, -1)

        # Sol giriş — burun deliği sol
        left_x = int(center[0] - png_rendered_w * 0.32)
        left_y = int(center[1] - png_rendered_h * 0.15)
        cv2.circle(mask, (left_x, left_y), int(nose_w * 0.30), 255, -1)

        # Sağ giriş — burun deliği sağ
        right_x = int(center[0] + png_rendered_w * 0.32)
        right_y = int(center[1] - png_rendered_h * 0.15)
        cv2.circle(mask, (right_x, right_y), int(nose_w * 0.30), 255, -1)

        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=nose_w * 0.06)

        mask_3d = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        output = (output.astype(np.float32) * (1 - mask_3d) + image.astype(np.float32) * mask_3d)
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output
    
    if piercing_type == "lip_piercing":
        lip_left = pt(61)
        lip_right = pt(291)
        lip_bottom_inner = pt(14)   # alt dudak üst çizgisi
        lip_bottom_outer = pt(17)   # alt dudak alt çizgisi
        
        mouth_w = np.linalg.norm(lip_right - lip_left)
        center_x_base = (lip_left[0] + lip_right[0]) / 2
        offset_x = mouth_w * 0.30

        # --- Her iki taraf için giriş noktaları ---
        sides = [
            {
                "top_x": center_x_base - offset_x,           # SOL
                "flip": False
            },
            {
                "top_x": center_x_base + offset_x,           # SAĞ
                "flip": True   # PNG yatay çevrilecek
            }
        ]

        for side in sides:
            top_x = side["top_x"]
            top_y = lip_bottom_inner[1]

            bottom_x = top_x + (mouth_w * 0.015 * (-1 if side["flip"] else 1))
            bottom_y = lip_bottom_outer[1]

            center_x = (top_x + bottom_x) / 2
            center_y = (top_y + bottom_y) / 2

            dx = bottom_x - top_x
            dy = bottom_y - top_y
            angle = -np.degrees(np.arctan2(dy, dx)) + 90

            pierce_length = np.linalg.norm([dx, dy])
            scale = pierce_length / png.shape[0] * 1.1

            # Sağ taraf için PNG'yi yatay çevir
            png_side = cv2.flip(png, 1) if side["flip"] else png

            output = overlay_png(output, png_side, center_x, center_y, scale=scale, angle=angle)

            # Giriş maskeleme
            mask = np.zeros((h, w), dtype=np.uint8)
            entry_radius = int(mouth_w * 0.022)
            cv2.circle(mask, (int(top_x), int(top_y)), entry_radius, 255, -1)
            cv2.circle(mask, (int(bottom_x), int(bottom_y)), entry_radius, 255, -1)

            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mouth_w * 0.007)
            mask_3d = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
            output = (output.astype(np.float32) * (1 - mask_3d) + image.astype(np.float32) * mask_3d)
            output = np.clip(output, 0, 255).astype(np.uint8)

        return output

    return output