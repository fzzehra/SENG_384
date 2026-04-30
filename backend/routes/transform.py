import os
import cv2
import numpy as np
from flask import Blueprint, request

from backend.modules.utils.helpers import error_response, success_response
from backend.modules.landmark.landmark import process_landmark_pipeline
from backend.modules.warping import apply_expression
from backend.modules.makeup.makeup import apply_makeup_pipeline

transform_bp = Blueprint("transform", __name__)
print("LOADED TRANSFORM FILE:", __file__)

TRANSFORM_MAP = {
    "smile": "smile",
    "eyebrow": "eyebrow_raise",

    # Controls sayfasındaki Face slider'ı bunu gönderiyor.
    "slim_face": "face_slimming",

    # Eski / alternatif isim desteği
    "face": "face_slimming",
    "face_slimming": "face_slimming",
    "lip_widen": "lip_widen",
    "lip_widening": "lip_widen",
    "face_widening": "face_widening",
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

    enhanced = cv2.detailEnhance(
        image,
        sigma_s=int(25 * intensity + 5),
        sigma_r=0.25 * intensity + 0.1
    )

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    a = cv2.addWeighted(
        a,
        1 - (0.7 * intensity),
        np.full_like(a, 128, dtype=np.uint8),
        0.7 * intensity,
        0
    )

    b = cv2.addWeighted(
        b,
        1 - (0.5 * intensity),
        np.full_like(b, 128, dtype=np.uint8),
        0.5 * intensity,
        0
    )

    l = cv2.convertScaleAbs(l, alpha=1.2, beta=-int(25 * intensity))

    lab = cv2.merge((l, a, b))
    aged = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    final_aged = cv2.addWeighted(aged, 1.0, mask_3d, 0.5 * intensity, 0)

    return final_aged


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

    image = cv2.imread(image_path)

    if image is None:
        return error_response(f"Image could not be read: {image_path}", 400)

    output_image = image.copy()
    results_meta = []

    try:
        for transform in transforms:
            t_type = transform.get("type")
            t_intensity = float(transform.get("intensity", 0.5))
            t_intensity = max(0.0, min(1.0, t_intensity))

            print("TRANSFORM TYPE:", t_type)
            print("INTENSITY:", t_intensity)

            # 1) Warping / expression transformations
            if t_type in TRANSFORM_MAP:
                landmark_result = process_landmark_pipeline(output_image)

                if not landmark_result.get("success"):
                    print(f"Landmark detection failed for {t_type}")
                    continue

                try:
                    output_image, _, _ = apply_expression(
                        image=output_image,
                        landmarks=landmark_result["landmarks"],
                        expression=TRANSFORM_MAP[t_type],
                        intensity=t_intensity
                    )

                    results_meta.append(t_type)
                    print("APPLIED:", t_type)

                except Exception as expression_error:
                    print(f"Expression transform failed for {t_type}: {expression_error}")

                    if t_type == "face_widening":
                        output_image, _, _ = apply_expression(
                            image=output_image,
                            landmarks=landmark_result["landmarks"],
                            expression="face_slimming",
                            intensity=-t_intensity
                        )

                        results_meta.append(t_type)
                        print("APPLIED:", t_type)
                    else:
                        raise expression_error

            # 2) Makeup transformations
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

            # 3) Aging
            elif t_type == "aging":
                output_image = apply_aging_effect(output_image, t_intensity)
                results_meta.append("aging")
                print("APPLIED: aging")

            # 4) De-aging, eski destek için kalsın
            elif t_type == "deaging":
                output_image = apply_deaging_effect(output_image, t_intensity)
                results_meta.append("deaging")
                print("APPLIED: deaging")

            # 5) Landmark display, eski destek için kalsın
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