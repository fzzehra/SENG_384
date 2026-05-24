#segmentation.py

import numpy as np
import cv2
from PIL import Image

_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "image-segmentation",
            model="jonathandinu/face-parsing",
            device=-1
        )
    return _pipeline


def get_hair_mask(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    pipe = _get_pipeline()
    results = pipe(pil_image)

    h, w = image_bgr.shape[:2]
    hair_mask = np.zeros((h, w), dtype=np.uint8)

    for segment in results:
        if segment["label"].lower() == "hair":
            seg_mask = np.array(segment["mask"])
            if seg_mask.shape[:2] != (h, w):
                seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            hair_mask = np.maximum(hair_mask, seg_mask)

    return hair_mask