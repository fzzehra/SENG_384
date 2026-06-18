import cv2
from flask import Blueprint, request

from backend.modules.utils.helpers import error_response, success_response
from backend.modules.frequency.filters import apply_frequency_filter

filters_bp = Blueprint("filters", __name__)


@filters_bp.route("/", methods=["POST"])
def apply_filter():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    image_path = data.get("image_path", "static/uploads/transformed.jpg")
    mode = data.get("mode", "low")
    intensity = float(data.get("intensity", 0.5))

    if mode not in ("low", "high"):
        return error_response("mode must be 'low' or 'high'.", 400)

    image = cv2.imread(image_path)
    if image is None:
        return error_response(f"Image could not be read: {image_path}", 400)

    output = apply_frequency_filter(image, mode=mode, intensity=intensity)

    output_path = f"static/uploads/filter_{mode}pass.jpg"
    cv2.imwrite(output_path, output)

    return success_response(
        f"{mode}-pass filter applied successfully.",
        data={"output_path": output_path, "mode": mode}
    )
