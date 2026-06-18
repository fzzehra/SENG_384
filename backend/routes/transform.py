#transform.py

from backend.modules.accessories.jewelery import apply_jewelry_pipeline
from backend.modules.makeup.makeup import apply_lip_color, apply_blush, apply_eyeshadow, apply_eye_color
import os
import cv2
import numpy as np
from flask import Blueprint, request, current_app
from PIL import Image
from ultralytics import YOLO

from backend.modules.utils.helpers import error_response, success_response
from backend.modules.landmark.landmark import process_landmark_pipeline
from backend.modules.warping import apply_expression
from backend.modules.makeup.makeup import apply_makeup_pipeline
from backend.modules.aging.aging import apply_aging_effect
from backend.modules.hat_glasses.glasses import place_glasses
from backend.modules.hair.hat import place_hat
from backend.modules.hair.hair import apply_hair_color, apply_eyebrow_color
from backend.modules.beard.beard import apply_beard_effect
from backend.modules.moustache.moustache import apply_moustache
from backend.modules.makeup.makeup import apply_makeup_pipeline
from backend.modules.piercing.piercing import apply_piercing

pose_model = YOLO("yolov8n-pose.pt")
transform_bp = Blueprint("transform", __name__)
print("LOADED TRANSFORM FILE:", __file__)

TRANSFORM_MAP = {
    "smile": "smile",
    "eyebrow": "eyebrow_raise",
    "lip_widening": "lip_widen",
    "face_widening": "face_widening",

    "lip_color": "lip_color",
    "blush": "blush",
    "eyeshadow": "eyeshadow",

    # Eski isimlerle uyumluluk için bırakıldı.
    "lip_widen": "lip_widen",
    "slim_face": "face_slimming",
    "face_slimming": "face_slimming",
    "lip_widen": "lip_widen",
    "face_widening": "face_slimming",
}


def apply_deaging_effect(image, intensity=0.5):
    intensity = float(max(0.0, min(1.0, intensity)))
    smooth = cv2.bilateralFilter(image, 9, 75, 75)
    deaged = cv2.addWeighted(image, 1 - intensity, smooth, intensity, 0)
    return deaged


def draw_landmarks_on_image(image, landmarks):
    output = image.copy()
    h, w = output.shape[:2]
    for point in landmarks:
        try:
            x, y = point
            x = int(x)
            y = int(y)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
        except Exception:
            continue
    return output

def trim_transparent(pil_img):
    pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)

    if len(arr.shape) < 3 or arr.shape[2] < 4:
        return pil_img

    alpha = arr[:, :, 3]
    coords = np.argwhere(alpha > 0)

    if coords.size == 0:
        return pil_img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return pil_img.crop((x0, y0, x1, y1))

def apply_png_overlay(base_img, overlay_img, x, y, scale=1.0):
    if overlay_img is None:
        return base_img

    if scale != 1.0:
        overlay_img = cv2.resize(
            overlay_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    h, w = overlay_img.shape[:2]
    base_h, base_w = base_img.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(base_w, x + w)
    y2 = min(base_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return base_img

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay_img[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    roi = base_img[y1:y2, x1:x2]

    if overlay_crop.shape[2] == 4:
        overlay_rgb = overlay_crop[:, :, :3]
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]   # EN ÖNEMLİ SATIR
    else:
        overlay_rgb = overlay_crop[:, :, :3]
        alpha = np.ones((overlay_rgb.shape[0], overlay_rgb.shape[1], 1), dtype=np.float32)

    # Note: If base_img is BGR and overlay is RGB, might need a swap here.
    # But usually OpenCV imread(..., IMREAD_UNCHANGED) returns BGRA.
    # Let's keep it as the user provided.
    blended = (roi * (1 - alpha) + overlay_rgb * alpha).astype(np.uint8)
    base_img[y1:y2, x1:x2] = blended

    return base_img


def overlay_rgba(background_bgr, overlay_path, center_xy, target_width, angle_deg=0.0, opacity=1.0):
    if not os.path.exists(overlay_path):
        print("Overlay file not found:", overlay_path)
        return background_bgr

    try:
        overlay_img = Image.open(overlay_path).convert("RGBA")
    except Exception as e:
        print("Overlay image could not be opened:", e)
        return background_bgr

    overlay_img = trim_transparent(overlay_img)

    ow, oh = overlay_img.size
    if ow <= 0 or oh <= 0:
        return background_bgr

    target_width = max(1, int(target_width))
    scale = target_width / float(ow)

    new_w = max(1, int(ow * scale))
    new_h = max(1, int(oh * scale))

    overlay_img = overlay_img.resize((new_w, new_h), Image.LANCZOS)

    if abs(angle_deg) > 0.1:
        overlay_img = overlay_img.rotate(angle_deg, expand=True, resample=Image.BICUBIC)

    overlay_rgba_np = np.array(overlay_img)

    opacity = float(max(0.0, min(1.0, opacity)))

    bg_h, bg_w = background_bgr.shape[:2]
    oh2, ow2 = overlay_rgba_np.shape[:2]

    cx, cy = center_xy
    x = int(cx - ow2 / 2)
    y = int(cy - oh2 / 2)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + ow2)
    y2 = min(bg_h, y + oh2)

    if x1 >= x2 or y1 >= y2:
        return background_bgr

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    crop = overlay_rgba_np[oy1:oy2, ox1:ox2]

    alpha = crop[:, :, 3:4] / 255.0
    alpha = alpha * opacity

    overlay_bgr = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2BGR)

    background_bgr[y1:y2, x1:x2] = (
        alpha * overlay_bgr +
        (1.0 - alpha) * background_bgr[y1:y2, x1:x2]
    ).astype(np.uint8)

    return background_bgr


def resolve_accessory_path(t_type, item_name):
    category = "earrings" if t_type == "earring" else "necklaces"
    folder = os.path.join("static", "accessories", category)

    candidates = [
        os.path.join(folder, item_name),
        os.path.join(folder, f"{item_name}.png"),
        os.path.join(folder, f"{item_name}.jpg"),
        os.path.join(folder, f"{item_name}.jpeg"),
        os.path.join(folder, f"{item_name}.webp"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    print("Accessory file candidates not found:")
    for candidate in candidates:
        print(" -", candidate)

    return None


JEWELRY_ITEM_SCALE = {
    "earring": {
        "_default": 1.0,
        "hoop_gold": 1.0,
        "pearl_drop": 1.1,
        "silver_earring": 1.0,
    },
    "necklace": {
        "_default": 1.0,
        "gold_necklace": 1.0,
        "pearl_necklace": 1.0,
        "simple_chain": 1.05,
    },
}


def _jewelry_item_scale(t_type, item_path):
    key = os.path.splitext(os.path.basename(item_path or ""))[0]
    table = JEWELRY_ITEM_SCALE.get(t_type, {})
    return float(table.get(key, table.get("_default", 1.0)))


def _trim_bgra(overlay):
    if overlay is None or overlay.ndim < 3 or overlay.shape[2] < 4:
        return overlay
    alpha = overlay[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return overlay
    return overlay[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def overlay_with_anchor(base_bgr, overlay, target_xy, target_width,
                        anchor=(0.5, 0.0), angle_deg=0.0, opacity=1.0):
    if overlay is None:
        return base_bgr

    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
    if overlay.shape[2] == 3:
        alpha = np.full(overlay.shape[:2], 255, dtype=np.uint8)
        overlay = np.dstack([overlay, alpha])

    overlay = _trim_bgra(overlay)
    oh, ow = overlay.shape[:2]
    if ow <= 0 or oh <= 0:
        return base_bgr

    scale = max(1, int(round(target_width))) / float(ow)
    new_w = max(1, int(round(ow * scale)))
    new_h = max(1, int(round(oh * scale)))
    overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ax, ay = anchor
    anchor_x = ax * new_w
    anchor_y = ay * new_h

    if abs(angle_deg) > 0.5:
        center = (new_w / 2.0, new_h / 2.0)
        rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos = abs(rot[0, 0])
        sin = abs(rot[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        rot[0, 2] += rot_w / 2.0 - center[0]
        rot[1, 2] += rot_h / 2.0 - center[1]
        overlay = cv2.warpAffine(
            overlay, rot, (rot_w, rot_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        rx = rot[0, 0] * anchor_x + rot[0, 1] * anchor_y + rot[0, 2]
        ry = rot[1, 0] * anchor_x + rot[1, 1] * anchor_y + rot[1, 2]
        anchor_x, anchor_y = rx, ry
        new_w, new_h = rot_w, rot_h

    tx, ty = target_xy
    x = int(round(tx - anchor_x))
    y = int(round(ty - anchor_y))

    bh, bw = base_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + new_w)
    y2 = min(bh, y + new_h)
    if x1 >= x2 or y1 >= y2:
        return base_bgr

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    crop = overlay[oy1:oy2, ox1:ox2]
    alpha = (crop[:, :, 3:4].astype(np.float32) / 255.0) * float(max(0.0, min(1.0, opacity)))
    rgb = crop[:, :, :3].astype(np.float32)
    roi = base_bgr[y1:y2, x1:x2].astype(np.float32)
    base_bgr[y1:y2, x1:x2] = np.clip(roi * (1.0 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)
    return base_bgr


def _ext_point(ext, key):
    value = ext.get(key) if ext else None
    if isinstance(value, tuple) and len(value) == 2:
        return np.array(value, dtype=np.float32)
    return None


def _facemesh_refs(landmarks, ext):
    if not landmarks or len(landmarks) < 468:
        return None
    left = np.array(landmarks[234], dtype=np.float32)
    right = np.array(landmarks[454], dtype=np.float32)
    chin = np.array(landmarks[152], dtype=np.float32)
    nose = np.array(landmarks[1], dtype=np.float32)
    face_width = max(40.0, float(np.linalg.norm(right - left)))
    tilt = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
    return {
        "left": left,
        "right": right,
        "chin": chin,
        "nose": nose,
        "face_width": face_width,
        "tilt": tilt,
        "ear_left": _ext_point(ext, "left_ear"),
        "ear_right": _ext_point(ext, "right_ear"),
        "shoulder_left": _ext_point(ext, "left_shoulder"),
        "shoulder_right": _ext_point(ext, "right_shoulder"),
    }


def _yolo_keypoints(image):
    results = pose_model(image, verbose=False)
    if (
        not results
        or not hasattr(results[0], "keypoints")
        or results[0].keypoints is None
        or len(results[0].keypoints.xy) == 0
    ):
        return None
    kp = results[0].keypoints.xy[0].cpu().numpy()
    if len(kp) < 7:
        return None
    return kp


def _place_earrings(output, overlay, refs, kp, item_scale, intensity):
    face_width = refs["face_width"] if refs else None
    tilt = refs["tilt"] if refs else 0.0

    ear_a = ear_b = None
    source = None

    if kp is not None and kp[3][0] > 0 and kp[4][0] > 0:
        ear_a, ear_b = kp[3], kp[4]
        source = "YOLO-kulak"
        if face_width is None:
            face_width = float(np.linalg.norm(np.array(ear_b) - np.array(ear_a)))
    elif refs and refs["ear_left"] is not None and refs["ear_right"] is not None:
        ear_a, ear_b = refs["ear_left"], refs["ear_right"]
        source = "FaceMesh-kulak(tahmini)"

    if ear_a is None or ear_b is None or not face_width:
        print("JEWELRY: küpe için referans bulunamadı, atlandı")
        return output

    pts = sorted(
        [np.array(ear_a, dtype=np.float32), np.array(ear_b, dtype=np.float32)],
        key=lambda p: float(p[0])
    )
    img_left, img_right = pts[0], pts[1]

    drop = 0.075 * face_width
    left_lobe = (float(img_left[0]), float(img_left[1]) + drop)
    right_lobe = (float(img_right[0]), float(img_right[1]) + drop)

    target_w = max(8, int(face_width * 0.11 * item_scale))
    print("JEWELRY: küpe kaynak=%s face_width=%.1f target_w=%d tilt=%.1f" %
          (source, face_width, target_w, tilt))

    output = overlay_with_anchor(output, overlay, left_lobe, target_w,
                                 anchor=(0.5, 0.06), angle_deg=-tilt, opacity=intensity)
    output = overlay_with_anchor(output, overlay, right_lobe, target_w,
                                 anchor=(0.5, 0.06), angle_deg=-tilt, opacity=intensity)
    return output


def _place_necklace(output, overlay, refs, kp, item_scale, intensity):
    span = mid = None
    source = None

    if kp is not None and kp[5][0] > 0 and kp[6][0] > 0:
        ls = np.array(kp[5], dtype=np.float32)
        rs = np.array(kp[6], dtype=np.float32)
        span = float(np.linalg.norm(rs - ls))
        mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        source = "YOLO-omuz"
    elif refs and refs["shoulder_left"] is not None and refs["shoulder_right"] is not None:
        ls = refs["shoulder_left"]
        rs = refs["shoulder_right"]
        span = float(np.linalg.norm(rs - ls))
        mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        source = "FaceMesh-omuz(tahmini)"

    if (span is None or mid is None) and refs:
        span = refs["face_width"] * 1.7
        mid = (float(refs["nose"][0]), float(refs["chin"][1] + refs["face_width"] * 0.5))
        source = "FaceMesh-yüz(yedek)"

    if span is None or mid is None:
        print("JEWELRY: kolye için referans bulunamadı, atlandı")
        return output

    if refs:
        face_w = refs["face_width"]
        width = face_w * 1.25
        center_x = float(mid[0])
        top_y = float(refs["chin"][1] + 0.15 * face_w)
    else:
        width = span * 0.80
        center_x = float(mid[0])
        top_y = float(mid[1] - 0.25 * span)

    width = width * item_scale

    target_w = max(20, int(width))
    print("JEWELRY: kolye kaynak=%s span=%.1f target_w=%d top=(%d,%d)" %
          (source, span, target_w, int(center_x), int(top_y)))

    return overlay_with_anchor(output, overlay, (center_x, top_y), target_w,
                               anchor=(0.5, 0.0), angle_deg=0.0, opacity=intensity)


def apply_jewelry_with_yolo(image, t_type, item_path, intensity=1.0):
    output = image.copy()
    print("JEWELRY v3 (anchor/yaw) AKTİF:", t_type, os.path.basename(item_path or ""))

    overlay = cv2.imread(item_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        print("JEWELRY: overlay okunamadı:", item_path)
        return output

    item_scale = _jewelry_item_scale(t_type, item_path)

    refs = None
    try:
        fm = process_landmark_pipeline(output)
        landmarks = fm.get("landmarks") if fm.get("success") else None
        ext = fm.get("extended_landmarks") or {}
        refs = _facemesh_refs(landmarks, ext) if landmarks else None
    except Exception as landmark_error:
        print("JEWELRY: FaceMesh hata, YOLO-only yedeğe geçiliyor:", landmark_error)

    if refs:
        print("JEWELRY: FaceMesh OK face_width=%.1f tilt=%.1f" % (refs["face_width"], refs["tilt"]))
    else:
        print("JEWELRY: FaceMesh referansı yok")

    kp = _yolo_keypoints(output)
    print("JEWELRY: YOLO pose", "OK" if kp is not None else "yok")

    if t_type == "earring":
        return _place_earrings(output, overlay, refs, kp, item_scale, intensity)
    if t_type == "necklace":
        return _place_necklace(output, overlay, refs, kp, item_scale, intensity)
    return output

@transform_bp.route("/", methods=["POST"])
def transform_image():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    image_path = data.get("image_path")
    transforms = data.get("transforms", [])
    print("FULL REQUEST DATA:", data)
    print("TRANSFORMS PAYLOAD:", transforms)

    if not transforms and data.get("transform_type"):
        transforms = [{
            "type": data.get("transform_type"),
            "intensity": data.get("intensity", 0.5)
        }]
   
    if not image_path:
        return error_response("image_path is required.", 400)
    
    # Normalize paths for Windows and remove leading slash if present
    image_path = image_path.lstrip('/\\')
    clean_image_path = image_path.replace("transformed.jpg", "original.jpg")
    
    if os.path.exists(clean_image_path):
        image = cv2.imread(clean_image_path)
    else:
        image = cv2.imread(image_path)

    if image is None:
        return error_response(f"Image could not be read: {image_path}", 400)

    output_image = image.copy()
    results_meta = []

    try:
        for transform in transforms:
            t_type = transform.get("type")
            print("TRANSFORM TYPE:", t_type)
            t_intensity = float(transform.get("intensity", 0.0))
            
            # 1. Yoğunluk Hesabı
            actual_intensity = t_intensity
            if t_type == "face_widening" and actual_intensity > 0:
                actual_intensity = -t_intensity
            
            # Sınırları belirle
            actual_intensity = max(-1.0, min(1.0, actual_intensity))
            t_intensity = float(transform.get("intensity", 0.5))
            color = transform.get("color", "#d96b86")
            t_intensity = max(0.0, min(1.0, t_intensity))

            print("TRANSFORM TYPE:", t_type)
            print("INTENSITY:", t_intensity)

            if t_type in TRANSFORM_MAP and t_type not in {"lip_color", "blush", "eyeshadow"}:
                landmark_result = process_landmark_pipeline(output_image)
                if not landmark_result.get("success"): 
                    continue

                # 2. Efekti Tek Seferde Uygula (Try-Except içinde)
                try:
                    output_image, _, _ = apply_expression(
                        image=output_image,
                        landmarks=landmark_result["landmarks"],
                        expression=TRANSFORM_MAP[t_type],
                        intensity=actual_intensity
                    )
                    results_meta.append(t_type)
                    print(f"APPLIED: {t_type} (Intensity: {actual_intensity})")
                    
                except Exception as expression_error:
                    print(f"Expression transform failed for {t_type}: {expression_error}")
                    # Eğer face_widening başarısız olursa yedek plan:
                    if t_type == "face_widening":
                        try:
                            output_image, _, _ = apply_expression(
                                image=output_image,
                                landmarks=landmark_result["landmarks"],
                                expression="face_slimming",
                                intensity=-t_intensity
                            )
                            results_meta.append(t_type)
                        except:
                            pass
                    else:
                        continue # Hata veren efekti atla, diğerine geç

            elif t_type in ["lipstick", "eyeshadow"]:
                landmark_result = process_landmark_pipeline(output_image)

                if not landmark_result.get("success"):
                    print(f"Landmark detection failed for {t_type}")
                    continue

                output_image = apply_makeup_pipeline(
                    image=output_image,
                    landmarks=landmark_result["landmarks"],
                    makeup_type=t_type,
                    color_hex=color,
                    intensity=t_intensity
                )
                results_meta.append(t_type)
                print("APPLIED:", t_type)

            # Mevcut transform handler'ında, diğer elif'lerin yanına:

            
            elif t_type == "accessories":
                params = transform.get("params", {})
                glasses_name = params.get("glasses", None)
                hat_name = params.get("hat", None)

                glasses_path = os.path.join('static', 'accessories', 'glasses', glasses_name) if glasses_name else None
                hat_path = os.path.join('static', 'accessories', 'hats', hat_name) if hat_name else None

                landmark_result = process_landmark_pipeline(output_image)
                if landmark_result.get("success"):
                    if glasses_path:
                        output_image = place_glasses(
                        output_image,
                        landmark_result["landmarks"],
                        glasses_path
            )
                if hat_path:
                    print(f"HAT PATH: {hat_path}")
                    hat_img = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
                    print(f"hat_img: {hat_img is not None}")
                    if hat_img is not None:
                        output_image = place_hat(
                            output_image,
                            landmark_result["landmarks"],
                            hat_img,
                            hat_name=hat_name
                        )
                results_meta.append("accessories")
                print(f"APPLIED: accessories")


            elif t_type in ["eyebrow_piercing", "septum_piercing", "lip_piercing"]:
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    transform_params = transform.get("params", {})
                    item = transform_params.get("item")

                    if not item:
                        print(f"PIERCING ITEM MISSING for {t_type}")
                        continue

                    item_path = os.path.join(
                        current_app.static_folder,
                        "accessories",
                        "piercings",
                        item
                    )

                    print("PIERCING PATH:", item_path)
                    print("PIERCING EXISTS:", os.path.exists(item_path))

                    output_image = apply_piercing(
                        image=output_image,
                        landmarks=landmark_result["landmarks"],
                        piercing_type=t_type,
                        item_path=item_path
                    )

                    results_meta.append(t_type)
                    print(f"APPLIED: {t_type}")
                else:
                    print(f"Landmark detection failed for {t_type}.")

            elif t_type == "landmarks":
                landmark_result = process_landmark_pipeline(output_image)
                if landmark_result.get("success"):
                    output_image = draw_landmarks_on_image(
                        image=output_image,
                        landmarks=landmark_result["landmarks"]
                    )
                    results_meta.append("landmarks")
                    print("APPLIED: landmarks")
                else:
                    print("Landmark detection failed for landmarks display.")
            elif t_type == "eyebrow_color":
                params = transform.get("params", {})
                color_hex = params.get("color", "#000000")
                landmark_result = process_landmark_pipeline(output_image)
                if landmark_result.get("success"):
                    output_image = apply_eyebrow_color(
                        output_image,
                        landmark_result["landmarks"],
                        color_hex=color_hex,
                        intensity=t_intensity
                    )
                    results_meta.append("eyebrow_color")

           
            elif t_type == "hair_color":
                params = transform.get("params", {})
                color_hex = params.get("color", "#3b1f0a")

                landmark_result = process_landmark_pipeline(output_image)
                if landmark_result.get("success"):
                    output_image = apply_hair_color(
                        output_image,
                        landmark_result["landmarks"],
                        color_hex=color_hex,
                        intensity=t_intensity
                    )
                    results_meta.append("hair_color")
                    print("APPLIED: hair_color")
            # SONRA (yeni):
            elif t_type == "aging":
                landmark_result = process_landmark_pipeline(output_image)
                if landmark_result.get("success"):
                    output_image = apply_aging_effect(
                        image=output_image,
                        intensity=t_intensity,
                        landmarks=landmark_result["landmarks"]  # kırışıklık + saç ağarması
                    )
                    results_meta.append("aging")
                    print(f"APPLIED: aging (Intensity: {t_intensity})")
                else:
                    # Landmark olmadan sadece saç ağarması çalışır, kırışıklık atlanır
                    output_image = apply_aging_effect(
                        image=output_image,
                        intensity=t_intensity,
                        landmarks=None
                    )
                    results_meta.append("aging")
                    print(f"APPLIED: aging without landmarks (Intensity: {t_intensity})")
            elif t_type == "deaging":
                # Kendi dosyan içindeki basit deaging fonksiyonunu çağırır
                output_image = apply_deaging_effect(
                    image=output_image, 
                    intensity=t_intensity
                )
                results_meta.append("deaging")
                print(f"APPLIED: deaging (Intensity: {t_intensity})")
            elif t_type == "beard":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    beard_len = int(3 + (t_intensity * 14))  # intensity arttıkça kıl uzasın

                    output_image = apply_beard_effect(
                        output_image,
                        landmark_result["landmarks"],
                        intensity=t_intensity,
                        hair_len=beard_len,
                        color=(3, 3, 3)
                    )

                    results_meta.append("beard")
                    print(f"APPLIED: beard intensity={t_intensity}, hair_len={beard_len}")
                else:
                    print("Landmark detection failed for beard.")
            elif t_type == "moustache":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    moustache_len = int(10 + (t_intensity * 10))

                    output_image = apply_moustache(
                        output_image,
                        landmark_result["landmarks"],
                        intensity=max(0.45, t_intensity),
                        hair_len=moustache_len,
                        color=(20, 20, 20)
                    )

                    results_meta.append("moustache")
                    print(f"APPLIED: moustache intensity={t_intensity}, hair_len={moustache_len}")
                else:
                    print("Landmark detection failed for moustache.")
            elif t_type == "lip_color":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    output_image = apply_lip_color(
                        output_image,
                        landmark_result["landmarks"],
                        color_hex=color,
                        intensity=t_intensity
                    )
                    results_meta.append("lip_color")
                else:
                    print("Landmark detection failed for lip_color.")

            elif t_type == "blush":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    output_image = apply_blush(
                        output_image,
                        landmark_result["landmarks"],
                        color_hex=color,
                        intensity=t_intensity
                    )
                    results_meta.append("blush")
                else:
                    print("Landmark detection failed for blush.")

            elif t_type == "freckles":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    output_image = apply_makeup_pipeline(
                        image=output_image,
                        landmarks=landmark_result["landmarks"],
                        makeup_type="freckles",
                        color_hex=color,
                        intensity=t_intensity
                    )

                    results_meta.append("freckles")
                    print("APPLIED: freckles")
                else:
                    print("Landmark detection failed for freckles.")


            elif t_type == "eyebrow_slit":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    output_image = apply_makeup_pipeline(
                        image=output_image,
                        landmarks=landmark_result["landmarks"],
                        makeup_type="eyebrow_slit",
                        color_hex=color,
                        intensity=t_intensity
                    )

                    results_meta.append("eyebrow_slit")
                    print("APPLIED: eyebrow_slit")
                else:
                    print("Landmark detection failed for eyebrow_slit.")

            elif t_type == "eyeshadow":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    output_image = apply_eyeshadow(
                        output_image,
                        landmark_result["landmarks"],
                        color_hex=color,
                        intensity=t_intensity
                    )
                    results_meta.append("eyeshadow")
                else:
                    print("Landmark detection failed for eyeshadow.")

            elif t_type == "eye_color":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    output_image = apply_eye_color(
                        output_image,
                        landmark_result["landmarks"],
                        color_hex=color,
                        intensity=t_intensity
                    )
                    results_meta.append("eye_color")
                else:
                    print("Landmark detection failed for eye_color.")

            
            elif t_type == "eyeliner":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    params = transform.get("params", {})
                    wing = params.get("wing", True)

                    makeup_type = (
                        "eyeliner"
                        if wing
                        else "eyeliner_no_wing"
                    )

                    output_image = apply_makeup_pipeline(
                        image=output_image,
                        landmarks=landmark_result["landmarks"],
                        makeup_type=makeup_type,
                        color_hex=color,
                        intensity=t_intensity
                    )

                    results_meta.append("eyeliner")
                    print("APPLIED: eyeliner")
                else:
                    print("Landmark detection failed for eyeliner.")
            
            elif t_type == "eyebrow_slit":
                landmark_result = process_landmark_pipeline(output_image)

                if landmark_result.get("success"):
                    output_image = apply_makeup_pipeline(
                        image=output_image,
                        landmarks=landmark_result["landmarks"],
                        makeup_type="eyebrow_slit",
                        color_hex=color,
                        intensity=t_intensity
                    )

                    results_meta.append("eyebrow_slit")
                    print("APPLIED: eyebrow_slit")
                else:
                    print("Landmark detection failed for eyebrow_slit.")


            elif t_type in ["earring", "necklace"]:
                params = transform.get("params", {})
                item_name = params.get("item", "")

                if not item_name:
                    continue

                item_path = os.path.join(
                    os.getcwd(),
                    "static",
                    "accessories",
                    "earrings" if t_type == "earring" else "necklaces",
                    item_name
                )

                output_image = apply_jewelry_with_yolo(
                    image=output_image,
                    t_type=t_type,
                    item_path=item_path,
                    intensity=t_intensity
                )

                results_meta.append(f"{t_type}:{item_name}")
                print("APPLIED:", t_type)
           
            elif t_type == "makeup_intensity":
                continue
            else:
                print("UNKNOWN TRANSFORM TYPE:", t_type)

        output_path = "static/uploads/transformed.jpg"
        saved = cv2.imwrite(output_path, output_image)

        if not saved:
            return error_response("Output image could not be saved.", 500)

        return success_response(
            "Transforms applied successfully.",
            data={
                "output_path": output_path,
                "applied_transforms": results_meta
            }
        )

    except Exception as e:
        import traceback
        print("TRANSFORM ERROR:", repr(e))
        traceback.print_exc()
        return error_response(f"Transform failed: {str(e)}", 500)