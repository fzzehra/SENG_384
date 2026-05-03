from backend.modules.makeup.makeup import apply_lip_color, apply_blush, apply_eyeshadow, apply_eye_color
import os
import cv2
import numpy as np
from flask import Blueprint, request

from backend.modules.utils.helpers import error_response, success_response
from backend.modules.landmark.landmark import process_landmark_pipeline
from backend.modules.warping import apply_expression
from backend.modules.makeup.makeup import apply_makeup_pipeline
from backend.modules.aging.aging import apply_aging_effect
from backend.modules.hair.hair import apply_hair_color, apply_hair_overlay
from backend.modules.hat_glasses.accessory_applier import apply_accessories

transform_bp = Blueprint("transform", __name__)
print("LOADED TRANSFORM FILE:", __file__)

TRANSFORM_MAP = {
    "smile": "smile",
    "eyebrow": "eyebrow_raise",
<<<<<<< Updated upstream
=======
    "lip_widening": "lip_widen",
    "face_widening": "face_widening",

    "lip_color": "lip_color",
    "blush": "blush",
    "eyeshadow": "eyeshadow",

    # Eski isimlerle uyumluluk için bırakıldı.
    "lip_widen": "lip_widen",
>>>>>>> Stashed changes
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


@transform_bp.route("/", methods=["POST"])
def transform_image():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    image_path = data.get("image_path")
    transforms = data.get("transforms", [])

    if not transforms and data.get("transform_type"):
        transforms = [{
            "type": data.get("transform_type"),
            "intensity": data.get("intensity", 0.5)
        }]

    if not image_path:
        return error_response("image_path is required.", 400)
    
    clean_image_path = image_path.replace("transformed.jpg", "original.jpg")
    
    # Dosya sisteminde original.jpg var mı diye kontrol et (opsiyonel ama güvenli)
    import os
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
<<<<<<< Updated upstream
            t_intensity = float(transform.get("intensity", 0.0))
            
            # 1. Yoğunluk Hesabı
            actual_intensity = t_intensity
            if t_type == "face_widening" and actual_intensity > 0:
                actual_intensity = -t_intensity
            
            # Sınırları belirle
            actual_intensity = max(-1.0, min(1.0, actual_intensity))
=======
            t_intensity = float(transform.get("intensity", 0.5))
            color = transform.get("color", "#d96b86")
            t_intensity = max(0.0, min(1.0, t_intensity))

            print("TRANSFORM TYPE:", t_type)
            print("INTENSITY:", t_intensity)
>>>>>>> Stashed changes

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
                    intensity=t_intensity
                )
                results_meta.append(t_type)
                print("APPLIED:", t_type)

            elif t_type == "aging":
                landmark_result = process_landmark_pipeline(output_image)
                landmarks = None
                if landmark_result.get("success"):
                    landmarks = landmark_result["landmarks"]

                output_image = apply_aging_effect(output_image, t_intensity, landmarks)
                results_meta.append("aging")
                print("APPLIED: aging")
            elif t_type == "accessories":
                params = transform.get("params", {})
                hat_name = params.get("hat", None)
                glasses_name = params.get("glasses", None)
                
                # Dosya yollarını oluştur (static klasörü altında olduğunu varsayıyorum)
                hat_path = os.path.join(os.getcwd(), "static", "accessories", hat_name) if hat_name else None
                glasses_path = os.path.join(os.getcwd(), "static", "accessories", glasses_name) if glasses_name else None

                landmark_result = process_landmark_pipeline(output_image)
                if landmark_result.get("success"):
                    output_image = apply_accessories(
                        image=output_image,
                        landmarks=landmark_result["landmarks"],
                        hat_path=hat_path,
                        glasses_path=glasses_path
                    )
                    results_meta.append("accessories")
                    print(f"APPLIED: accessories (Hat: {hat_name}, Glasses: {glasses_name})")
                else:
                    print("Landmark detection failed for accessories.")
            elif t_type == "deaging":
                output_image = apply_deaging_effect(output_image, t_intensity)
                results_meta.append("deaging")
                print("APPLIED: deaging")

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

<<<<<<< Updated upstream
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

            elif t_type == "hair_overlay":
                params = transform.get("params", {})
                overlay_name = params.get("overlay", "")
                overlay_path = os.path.join(os.getcwd(), "static", "hairstyles", overlay_name)

                landmark_result = process_landmark_pipeline(output_image)
                if landmark_result.get("success"):
                    output_image = apply_hair_overlay(
                        output_image,
                        landmark_result["landmarks"],
                        overlay_path=overlay_path,
                        intensity=t_intensity
                    )
                    results_meta.append("hair_overlay")
                    print("APPLIED: hair_overlay")
=======
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
>>>>>>> Stashed changes

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