# segmentation.py

import numpy as np
import cv2
from PIL import Image

_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        _processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        _model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        _model.eval()
    return _processor, _model


def get_hair_mask(image_bgr: np.ndarray) -> np.ndarray:
    import torch

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    processor, model = _get_model()

    # Saç sınıfının indeksini bul
    hair_idx = None
    for idx, label in model.config.id2label.items():
        if label.lower() == "hair":
            hair_idx = int(idx)
            break

    if hair_idx is None:
        return np.zeros((h, w), dtype=np.uint8)

    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits  # küçük çözünürlükte, tüm sınıflar

    # Sınıf seçimini küçük boyutta yap, sadece saç maskesini büyüt
    predicted = logits.argmax(dim=1)[0].numpy()
    small_mask = (predicted == hair_idx).astype(np.float32)

    big_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    hair_mask = (big_mask > 0.5).astype(np.uint8) * 255

    return hair_mask