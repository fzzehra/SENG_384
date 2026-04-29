import os
import cv2
from flask import Blueprint, request

from backend.modules.utils.helpers import error_response, success_response
from backend.modules.landmark.landmark import process_landmark_pipeline
from backend.modules.warping.warping import apply_expression
from backend.modules.makeup.makeup import apply_makeup_pipeline

transform_bp = Blueprint("transform", __name__)

TRANSFORM_MAP = {
    "smile": "smile",
    "eyebrow": "eyebrow_raise",
    "lip_widen": "lip_widen",
    "slim_face": "face_slimming",
    "lipstick": "lipstick",
    "eyeshadow": "eyeshadow",
}


def create_output_path(image_path, transform_type):
    folder = os.path.dirname(image_path)
    filename = os.path.basename(image_path)

    name, ext = os.path.splitext(filename)

    if not ext:
        ext = ".jpg"

    output_filename = f"{name}_{transform_type}{ext}"
    return os.path.join(folder, output_filename)


def apply_aging_effect(image, intensity=0.5):
    intensity = float(max(0.0, min(1.0, intensity)))
    
    # 1. Doku Keskinleştirme (Kırışıklıklar için detayı artır)
    enhanced = cv2.detailEnhance(image, sigma_s=int(25 * intensity + 5), sigma_r=0.25 * intensity + 0.1)
    
    # 2. Renk ve Kontrast
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Doygunluğu azalt
    a = cv2.addWeighted(a, 1 - (0.7 * intensity), np.full_like(a, 128, dtype=np.uint8), 0.7 * intensity, 0)
    b = cv2.addWeighted(b, 1 - (0.5 * intensity), np.full_like(b, 128, dtype=np.uint8), 0.5 * intensity, 0)
    
    # L kanalında kontrastı artır
    l = cv2.convertScaleAbs(l, alpha=1.2, beta=-int(25 * intensity))
    
    lab = cv2.merge((l, a, b))
    aged = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Saç/Sakal Beyazlatma (Daha agresif)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Beyazlatma yoğunluğu artırıldı
    final_aged = cv2.addWeighted(aged, 1.0, mask_3d, 0.5 * intensity, 0)
    
    return final_aged


def apply_deaging_effect(image, intensity=0.5):
    intensity = float(max(0.0, min(1.0, intensity)))

    smooth = cv2.bilateralFilter(image, 9, 75, 75)
    deaged = cv2.addWeighted(image, 1 - intensity, smooth, intensity, 0)
    return deaged


@transform_bp.route("/", methods=["POST"])
def transform_image():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    image_path = data.get("image_path")
    transform_type = data.get("transform_type")
    intensity = data.get("intensity", 0.5)

    if not image_path:
        return error_response("image_path is required.", 400)

    if transform_type not in TRANSFORM_MAP and transform_type not in {"aging", "deaging"}:
        return error_response("Invalid transform_type.", 400)

    try:
        intensity = float(intensity)
    except (TypeError, ValueError):
        return error_response("intensity must be a number.", 400)

    intensity = max(0.0, min(1.0, intensity))

    image = cv2.imread(image_path)

    if image is None:
        return error_response("Image could not be read.", 400)

    try:
        if transform_type in ["lipstick", "eyeshadow"]:
            landmark_result = process_landmark_pipeline(image)
            if not landmark_result["success"]:
                return error_response(landmark_result["validation"]["reason"], 400)
            
            output_image = apply_makeup_pipeline(
                image=image,
                landmarks=landmark_result["landmarks"],
                makeup_type=transform_type,
                intensity=intensity
            )
            extra_data = {"type": "makeup"}

        elif transform_type in TRANSFORM_MAP:
            landmark_result = process_landmark_pipeline(image)

            if not landmark_result["success"]:
                return error_response(
                    landmark_result["validation"]["reason"],
                    400
                )

            output_image, dst_landmarks, triangles = apply_expression(
                image=image,
                landmarks=landmark_result["landmarks"],
                expression=TRANSFORM_MAP[transform_type],
                intensity=intensity
            )

            extra_data = {
                "num_landmarks": landmark_result["num_landmarks"],
                "num_triangles": len(triangles)
            }

        elif transform_type == "aging":
            output_image = apply_aging_effect(image, intensity)
            extra_data = {}

        else:
            output_image = apply_deaging_effect(image, intensity)
            extra_data = {}

        output_path = "static/uploads/transformed.jpg"

        saved = cv2.imwrite(output_path, output_image)

        if not saved:
            return error_response("Output image could not be saved.", 500)

        return success_response(
            "Transform applied successfully.",
            data={
                "original_image": image_path,
                "output_path": output_path,
                "transform_type": transform_type,
                "intensity": intensity,
                **extra_data
            }
        )

    except Exception as e:
        return error_response(f"Transform failed: {str(e)}", 500)