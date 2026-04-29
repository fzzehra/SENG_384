import cv2
import numpy as np

def apply_lipstick(image, landmarks, color=(0, 0, 150), intensity=0.5):
    """
    Apply lipstick to the lips.
    color: BGR format
    intensity: 0.0 to 1.0
    """
    pts = np.array(landmarks, dtype=np.int32)
    
    # MediaPipe lip indices (Outer)
    lip_indices = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409
    ]
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    lip_pts = pts[lip_indices]
    cv2.fillPoly(mask, [lip_pts], 255)
    
    # Create colored image
    color_img = np.zeros_like(image)
    color_img[:] = color
    
    # Blending
    mask_blur = cv2.GaussianBlur(mask, (7, 7), 0) / 255.0
    mask_blur = cv2.merge([mask_blur, mask_blur, mask_blur])
    
    # Soft blend
    lipstick = cv2.addWeighted(image, 1.0, color_img, 0.4 * intensity, 0)
    result = image * (1 - mask_blur) + lipstick * mask_blur
    
    return result.astype(np.uint8)

def apply_eyeshadow(image, landmarks, color=(100, 0, 100), intensity=0.5):
    """
    Apply eyeshadow to the upper eyelids.
    """
    pts = np.array(landmarks, dtype=np.int32)
    
    # MediaPipe eyelid indices
    left_eye_top = [226, 247, 30, 29, 27, 28, 56, 190, 243]
    right_eye_top = [463, 414, 286, 258, 257, 259, 260, 467, 446]
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for indices in [left_eye_top, right_eye_top]:
        eye_pts = pts[indices]
        cv2.fillPoly(mask, [eye_pts], 255)
        
    # Dilate mask slightly upwards to cover eyelid area better
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    mask_blur = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
    mask_blur = cv2.merge([mask_blur, mask_blur, mask_blur])
    
    color_img = np.zeros_like(image)
    color_img[:] = color
    
    eyeshadow = cv2.addWeighted(image, 1.0, color_img, 0.3 * intensity, 0)
    result = image * (1 - mask_blur) + eyeshadow * mask_blur
    
    return result.astype(np.uint8)

def apply_makeup_pipeline(image, landmarks, makeup_type, intensity=0.5):
    if makeup_type == "lipstick":
        return apply_lipstick(image, landmarks, intensity=intensity)
    elif makeup_type == "eyeshadow":
        return apply_eyeshadow(image, landmarks, intensity=intensity)
    return image
