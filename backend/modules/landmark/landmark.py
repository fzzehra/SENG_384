import os
from typing import List, Tuple, Dict, Any

import cv2
import mediapipe as mp
import numpy as np

LandmarkList = List[Tuple[int, int]]


def detect_landmarks(image: np.ndarray) -> LandmarkList:
    if image is None:
        raise ValueError("Input image is None.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return []

    h, w = image.shape[:2]
    face_landmarks = results.multi_face_landmarks[0]

    landmarks = []

    for lm in face_landmarks.landmark:
        # 🔥 FIX: float → int dönüşümü doğru scale
        x = int(lm.x * w)
        y = int(lm.y * h)

        landmarks.append((x, y))

    return landmarks

def generate_extended_landmarks(
    landmarks: LandmarkList,
    image_shape: Tuple[int, int, int]
) -> Dict[str, Tuple[int, int]]:
    """
    MediaPipe FaceMesh 468 noktasına ek olarak tahmini kafa, kulak,
    boyun ve omuz noktaları üretir.
    """
    if landmarks is None or len(landmarks) < 468:
        return {}

    h, w = image_shape[:2]

    def p(idx):
        x, y = landmarks[idx]
        return int(x), int(y)

    top = p(10)
    chin = p(152)
    left_face = p(234)
    right_face = p(454)
    left_eye = p(33)
    right_eye = p(263)
    left_mouth = p(61)
    right_mouth = p(291)

    face_w = abs(right_face[0] - left_face[0])
    face_h = abs(chin[1] - top[1])
    cx = (left_face[0] + right_face[0]) // 2

    # Saç sınırı için tahmini noktalar
    hair_top_y = max(0, int(top[1] - face_h * 0.48))
    hair_front_y = max(0, int(top[1] - face_h * 0.03))
    hair_side_y = int(top[1] + face_h * 0.22)

    hair_outer_left_x = max(0, int(left_face[0] - face_w * 0.42))
    hair_outer_right_x = min(w - 1, int(right_face[0] + face_w * 0.42))

    hair_inner_left_x = max(0, int(left_face[0] + face_w * 0.12))
    hair_inner_right_x = min(w - 1, int(right_face[0] - face_w * 0.12))

    extended = {
       # kafa / saç sınırı tahmini
        "head_top": (
            cx,
            max(0, int(top[1] - face_h * 0.48))
        ),

        "hairline_left": (
            max(0, int(left_face[0] - face_w * 0.30)),
            int(top[1] + face_h * 0.02)
        ),

        "hairline_right": (
            min(w - 1, int(right_face[0] + face_w * 0.30)),
            int(top[1] + face_h * 0.02)
        ),
                # daha detaylı saç dış sınırı
        "hair_outer_left": (
            hair_outer_left_x,
            hair_side_y
        ),
        "hair_outer_left_top": (
            max(0, int(cx - face_w * 0.50)),
            max(0, int(top[1] - face_h * 0.20))
        ),
        "hair_outer_top": (
            cx,
            hair_top_y
        ),
        "hair_outer_right_top": (
            min(w - 1, int(cx + face_w * 0.50)),
            max(0, int(top[1] - face_h * 0.20))
        ),
        "hair_outer_right": (
            hair_outer_right_x,
            hair_side_y
        ),

        # yüze yakın ön saç çizgisi
        "hair_inner_left": (
            hair_inner_left_x,
            hair_front_y
        ),
        "hair_inner_mid_left": (
            max(0, int(cx - face_w * 0.22)),
            max(0, int(top[1] - face_h * 0.02))
        ),
        "hair_inner_center": (
            cx,
            max(0, int(top[1] - face_h * 0.04))
        ),
        "hair_inner_mid_right": (
            min(w - 1, int(cx + face_w * 0.22)),
            max(0, int(top[1] - face_h * 0.02))
        ),
        "hair_inner_right": (
            hair_inner_right_x,
            hair_front_y
        ),
        # kulak tahmini
        "left_ear": (
            max(0, int(left_face[0] - face_w * 0.16)),
            int((left_eye[1] + left_mouth[1]) / 2)
        ),
        "right_ear": (
            min(w - 1, int(right_face[0] + face_w * 0.16)),
            int((right_eye[1] + right_mouth[1]) / 2)
        ),

        # boyun tahmini
        "neck_left": (
            int(cx - face_w * 0.23),
            min(h - 1, int(chin[1] + face_h * 0.10))
        ),
        "neck_right": (
            int(cx + face_w * 0.23),
            min(h - 1, int(chin[1] + face_h * 0.10))
        ),
        "neck_bottom_left": (
            int(cx - face_w * 0.32),
            min(h - 1, int(chin[1] + face_h * 0.42))
        ),
        "neck_bottom_right": (
            int(cx + face_w * 0.32),
            min(h - 1, int(chin[1] + face_h * 0.42))
        ),

        # omuz tahmini
        "left_shoulder": (
            max(0, int(cx - face_w * 1.08)),
            min(h - 1, int(chin[1] + face_h * 0.58))
        ),
        "right_shoulder": (
            min(w - 1, int(cx + face_w * 1.08)),
            min(h - 1, int(chin[1] + face_h * 0.58))
        ),
    }

    return extended

def draw_landmarks(
    image: np.ndarray,
    landmarks: LandmarkList,
    radius: int = 2
) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None.")

    output = image.copy()
    mp_face_mesh = mp.solutions.face_mesh

    # Tüm yüz ağını (tesselation) ince gri çizgilerle çizmek isterseniz bunu açabilirsiniz:
    # if hasattr(mp_face_mesh, 'FACEMESH_TESSELATION'):
    #     for connection in mp_face_mesh.FACEMESH_TESSELATION:
    #         start_idx, end_idx = connection
    #         if start_idx < len(landmarks) and end_idx < len(landmarks):
    #             cv2.line(output, landmarks[start_idx], landmarks[end_idx], (200, 200, 200), 1)

    # Temel yüz hatlarını (göz, dudak, kaş, yüz ovali) birleştiren çizgiler
    if hasattr(mp_face_mesh, 'FACEMESH_CONTOURS'):
        # Tüm ana hatları (contours), burun (nose) ve gözbebeklerini (irises) birleştirelim
        connections = set(mp_face_mesh.FACEMESH_CONTOURS)
        if hasattr(mp_face_mesh, 'FACEMESH_NOSE'):
            connections.update(mp_face_mesh.FACEMESH_NOSE)
        if hasattr(mp_face_mesh, 'FACEMESH_IRISES'):
            connections.update(mp_face_mesh.FACEMESH_IRISES)
        
        # Önce noktaları birleştiren çizgileri çizelim (Açık Yeşil)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = landmarks[start_idx]
                pt2 = landmarks[end_idx]
                cv2.line(output, pt1, pt2, (0, 255, 0), thickness=max(1, radius - 1))
                
        # Sonra köşe noktalarını çizelim ki belirgin olsun (Kırmızı)
        for connection in connections:
            for idx in connection:
                if idx < len(landmarks):
                    cv2.circle(output, landmarks[idx], max(1, radius - 1), (0, 0, 255), -1)
    else:
        # Eğer eski versiyon kullanılıyorsa sadece noktaları bas
        for x, y in landmarks:
            cv2.circle(output, (x, y), radius, (0, 255, 0), -1)

    extended = generate_extended_landmarks(landmarks, image.shape)

    if extended:
        # boyun
        cv2.line(output, extended["neck_left"], extended["neck_bottom_left"], (180, 0, 255), 2)
        cv2.line(output, extended["neck_right"], extended["neck_bottom_right"], (180, 0, 255), 2)
        cv2.line(output, extended["neck_bottom_left"], extended["neck_bottom_right"], (180, 0, 255), 1)

        # omuz
        cv2.line(output, extended["left_shoulder"], extended["neck_bottom_left"], (255, 0, 180), 2)
        cv2.line(output, extended["neck_bottom_right"], extended["right_shoulder"], (255, 0, 180), 2)

        # kafa / saç sınırı
        # detaylı saç dış sınırı
        hair_outer = [
            extended["hair_outer_left"],
            extended["hair_outer_left_top"],
            extended["hair_outer_top"],
            extended["hair_outer_right_top"],
            extended["hair_outer_right"]
        ]

        for i in range(len(hair_outer) - 1):
            cv2.line(output, hair_outer[i], hair_outer[i + 1], (255, 180, 0), 2)

        # yüze yakın ön saç çizgisi
        hair_inner = [
            extended["hair_inner_left"],
            extended["hair_inner_mid_left"],
            extended["hair_inner_center"],
            extended["hair_inner_mid_right"],
            extended["hair_inner_right"]
        ]

        for i in range(len(hair_inner) - 1):
         cv2.line(output, hair_inner[i], hair_inner[i + 1], (0, 180, 255), 2)

        # kulak
        cv2.circle(output, extended["left_ear"], 5, (255, 120, 0), 2)
        cv2.circle(output, extended["right_ear"], 5, (255, 120, 0), 2)

        # noktaları göster
        for name, point in extended.items():
            cv2.circle(output, point, radius + 2, (255, 0, 255), -1)
    return output


def validate_landmarks(
    landmarks: LandmarkList,
    image_shape: Tuple[int, int, int]
) -> Dict[str, Any]:
    if not landmarks:
        return {
            "is_valid": False,
            "reason": "No landmarks detected.",
            "count": 0,
            "inside_ratio": 0.0
        }

    h, w = image_shape[:2]

    inside_count = 0
    for (x, y) in landmarks:
        if 0 <= x < w and 0 <= y < h:
            inside_count += 1

    inside_ratio = inside_count / len(landmarks)

    if len(landmarks) < 100:
        return {
            "is_valid": False,
            "reason": "Too few landmarks detected.",
            "count": len(landmarks),
            "inside_ratio": inside_ratio
        }

    if inside_ratio < 0.95:
        return {
            "is_valid": False,
            "reason": "Many landmarks are outside image bounds.",
            "count": len(landmarks),
            "inside_ratio": inside_ratio
        }

    return {
        "is_valid": True,
        "reason": "Landmarks successfully detected.",
        "count": len(landmarks),
        "inside_ratio": inside_ratio
    }


def save_image(image: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Image encoding failed.")

    encoded_image.tofile(output_path)


def process_landmark_pipeline(
    image: np.ndarray,
    output_path: str = None
) -> Dict[str, Any]:
    landmarks = detect_landmarks(image)
    validation = validate_landmarks(landmarks, image.shape)

    extended_landmarks = generate_extended_landmarks(landmarks, image.shape)
    landmark_image = draw_landmarks(image, landmarks) if landmarks else image.copy()

    if output_path:
        save_image(landmark_image, output_path)

    return {
        "success": validation["is_valid"],
        "num_landmarks": validation["count"],
        "landmarks": landmarks,
        "extended_landmarks": extended_landmarks,
        "validation": validation,
        "output_path": output_path,
        "image_with_landmarks": landmark_image
    }