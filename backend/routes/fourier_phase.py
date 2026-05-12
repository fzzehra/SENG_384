import cv2
from flask import Blueprint, request

from backend.modules.utils.helpers import error_response, success_response
from backend.modules.frequency.fourier_phase import apply_fourier_phase_effect

fourier_phase_bp = Blueprint("fourier_phase", __name__)


@fourier_phase_bp.route("/", methods=["POST"])
def fourier_phase():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    image_path = data.get("image_path", "static/uploads/transformed.jpg")
    intensity = float(data.get("intensity", 0.5))

    image = cv2.imread(image_path)

    if image is None:
        return error_response(f"Image could not be read: {image_path}", 400)

    output = apply_fourier_phase_effect(image, intensity)

    output_path = "static/uploads/fourier_phase.jpg"
    cv2.imwrite(output_path, output)

    return success_response(
        "Fourier phase applied successfully.",
        data={"output_path": output_path}
    )