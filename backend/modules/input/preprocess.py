from _future_ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_TARGET_SIZE = (256, 256)
MIN_WIDTH = 128
MIN_HEIGHT = 128


class InputModuleError(Exception):
    pass


class InvalidImageError(InputModuleError):
    pass


class FaceNotDetectedError(InputModuleError):
    pass


@dataclass
class FaceDetectionResult:
    x: int
    y: int
    w: int
    h: int


def load_image(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise InvalidImageError("Empty file data received.")

    np_buffer = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise InvalidImageError("Image could not be decoded.")

    return image


def validate_image(
    filename: str,
    image: np.ndarray,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
) -> bool:
    if not filename:
        raise InvalidImageError("Filename is missing.")

    lower_name = filename.lower()
    if not any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise InvalidImageError("Unsupported file format. Use JPG or PNG.")

    if image is None or not isinstance(image, np.ndarray):
        raise InvalidImageError("Invalid image object.")

    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        raise InvalidImageError(
            f"Image resolution too small. Minimum: {min_width}x{min_height}"
        )

    return True


def _detect_face_mediapipe(image: np.ndarray) -> Optional[FaceDetectionResult]:
    mp_face_detection = mp.solutions.face_detection

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as detector:
        results = detector.process(rgb_image)

    if not results.detections:
        return None

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    x = max(0, int(bbox.xmin * width))
    y = max(0, int(bbox.ymin * height))
    w = int(bbox.width * width)
    h = int(bbox.height * height)

    w = min(w, width - x)
    h = min(h, height - y)

    if w <= 0 or h <= 0:
        return None

    return FaceDetectionResult(x=x, y=y, w=w, h=h)


def _detect_face_haar(image: np.ndarray) -> Optional[FaceDetectionResult]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    return FaceDetectionResult(x=int(x), y=int(y), w=int(w), h=int(h))


def detect_face(image: np.ndarray) -> FaceDetectionResult:
    result = _detect_face_mediapipe(image)

    if result is None:
        result = _detect_face_haar(image)

    if result is None:
        raise FaceNotDetectedError("No detectable human face found.")

    return result


def crop_face(
    image: np.ndarray,
    face_box: FaceDetectionResult,
    margin_ratio: float = 0.15,
) -> np.ndarray:
    height, width = image.shape[:2]

    mx = int(face_box.w * margin_ratio)
    my = int(face_box.h * margin_ratio)

    x1 = max(0, face_box.x - mx)
    y1 = max(0, face_box.y - my)
    x2 = min(width, face_box.x + face_box.w + mx)
    y2 = min(height, face_box.y + face_box.h + my)

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        raise InputModuleError("Face crop failed.")

    return cropped


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def process_for_pipeline(
    file_bytes: bytes,
    filename: str,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Dict[str, np.ndarray]:
    original_image = load_image(file_bytes)
    validate_image(filename, original_image)

    face_box = detect_face(original_image)
    cropped_face = crop_face(original_image, face_box)
    resized_face = resize_image(cropped_face, target_size)
    normalized_face = normalize_image(resized_face)
    grayscale_face = to_grayscale(resized_face)

    return {
        "original_image": original_image,
        "cropped_face": cropped_face,
        "resized_face": resized_face,
        "normalized_face": normalized_face,
        "grayscale_face": grayscale_face,
        "face_box": np.array([face_box.x, face_box.y, face_box.w, face_box.h]),
    }
