import os
import shutil
import cv2
import numpy as np

from flask import Blueprint, request, current_app, session
from werkzeug.utils import secure_filename

from backend.modules.db import get_db_connection
from backend.modules.utils.helpers import (
    allowed_file,
    error_response,
    success_response,
    timestamped_filename,
    ensure_dir,
)

from backend.modules.landmark.landmark import process_landmark_pipeline
from backend.modules.warping.warping import apply_expression
from backend.modules.makeup.makeup import apply_makeup_pipeline

upload_bp = Blueprint("upload", __name__)

TRANSFORM_MAP = {
    "smile": "smile",
    "eyebrow": "eyebrow_raise",
    "lip_widen": "lip_widen",
    "slim_face": "face_slimming",
    "lipstick": "lipstick",
    "eyeshadow": "eyeshadow",
}


def apply_aging_effect(image, intensity=0.5):
    intensity = float(max(0.0, min(1.0, intensity)))
    
    # 1. Doku Keskinleştirme (Kırışıklıklar için)
    enhanced = cv2.detailEnhance(image, sigma_s=int(15 * intensity + 5), sigma_r=0.15 * intensity + 0.1)
    
    # 2. Renk ve Kontrast
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.addWeighted(a, 1 - (0.6 * intensity), np.full_like(a, 128, dtype=np.uint8), 0.6 * intensity, 0)
    b = cv2.addWeighted(b, 1 - (0.4 * intensity), np.full_like(b, 128, dtype=np.uint8), 0.4 * intensity, 0)
    l = cv2.convertScaleAbs(l, alpha=1.1, beta=-int(15 * intensity))
    lab = cv2.merge((l, a, b))
    aged = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Beyazlatma simülasyonu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return cv2.addWeighted(aged, 1.0, mask_3d, 0.3 * intensity, 0)


def apply_deaging_effect(image, intensity=0.5):
    intensity = float(max(0.0, min(1.0, intensity)))

    smooth = cv2.bilateralFilter(image, 15, 90, 90)
    deaged = cv2.addWeighted(image, 1 - intensity, smooth, intensity, 0)

    bright = np.full_like(deaged, (12, 12, 12))
    deaged = cv2.addWeighted(deaged, 1.0, bright, 0.25 * intensity, 0)

    return deaged


@upload_bp.route("/", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return error_response("No image file found in request.", 400)

    file = request.files["image"]

    if file.filename == "":
        return error_response("No file selected.", 400)

    if not allowed_file(file.filename):
        return error_response("Unsupported file format. Please upload JPG or PNG.", 400)

    transform_type = request.form.get("transform_type", "smile")
    intensity = request.form.get("intensity", 0.5)

    try:
        intensity = float(intensity)
    except (TypeError, ValueError):
        return error_response("intensity must be a number.", 400)

    intensity = max(0.0, min(1.0, intensity))

    valid_extra_transforms = {"aging", "deaging", "landmarks"}

    if transform_type not in TRANSFORM_MAP and transform_type not in valid_extra_transforms:
        return error_response("Invalid transform_type.", 400)

    filename = secure_filename(file.filename)
    filename = timestamped_filename(filename)

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, filename)

    file.save(file_path)

    original_path = os.path.join(upload_folder, "original.jpg")
    transformed_path = os.path.join(upload_folder, "transformed.jpg")

    shutil.copy(file_path, original_path)

    image = cv2.imread(file_path)

    if image is None:
        return error_response("Image could not be read.", 400)

    try:
        if transform_type == "landmarks":
            landmark_result = process_landmark_pipeline(image)

            if not landmark_result["success"]:
                return error_response(
                    landmark_result["validation"]["reason"],
                    400
                )

            output_image = landmark_result["image_with_landmarks"]

        elif transform_type in ["lipstick", "eyeshadow"]:
            landmark_result = process_landmark_pipeline(image)
            if not landmark_result["success"]:
                return error_response(landmark_result["validation"]["reason"], 400)
            
            output_image = apply_makeup_pipeline(
                image=image,
                landmarks=landmark_result["landmarks"],
                makeup_type=transform_type,
                intensity=intensity
            )

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

        elif transform_type == "aging":
            output_image = apply_aging_effect(image, intensity)

        else:
            output_image = apply_deaging_effect(image, intensity)

        saved = cv2.imwrite(transformed_path, output_image)

        if not saved:
            return error_response("Transformed image could not be saved.", 500)

    except Exception as e:
        return error_response(f"Transform failed: {str(e)}", 500)

    # Save to history if requested and user is logged in
    if request.form.get("save_to_history") == "true" and session.get("user_id"):
        try:
            history_folder = os.path.join(current_app.config["RESULT_FOLDER"], "history")
            ensure_dir(history_folder)
            
            # Create a unique filename for the transformed image
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            hist_transformed_name = f"{name}_{transform_type}_result{ext}"
            hist_transformed_path = os.path.join(history_folder, hist_transformed_name)
            
            # Save the transformed image uniquely
            cv2.imwrite(hist_transformed_path, output_image)
            
            # Store relative paths in DB for easy access
            rel_original_path = file_path # This is already static/uploads/filename
            rel_transformed_path = os.path.join("static/results/history", hist_transformed_name)

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_history (user_id, original_image, transformed_image, transform_type, intensity) VALUES (%s, %s, %s, %s, %s)",
                (session["user_id"], rel_original_path, rel_transformed_path, transform_type, intensity)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"HISTORY SAVE ERROR: {e}")

    return success_response(
        "Image uploaded and transformed successfully.",
        data={
            "filename": filename,
            "file_path": file_path,
            "original_path": original_path,
            "transformed_path": transformed_path,
            "transform_type": transform_type,
            "intensity": intensity
        },
        status_code=201
    )