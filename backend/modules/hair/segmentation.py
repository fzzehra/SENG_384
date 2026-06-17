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

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    orig_w, orig_h = pil_image.size

    processor, model = _get_model()

    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits back to original image size
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    predicted = logits.argmax(dim=1).squeeze(0).numpy()

    # Find hair label index from model config
    hair_idx = None
    for idx, label in model.config.id2label.items():
        if label.lower() == "hair":
            hair_idx = int(idx)
            break

    if hair_idx is None:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    hair_mask = ((predicted == hair_idx).astype(np.uint8)) * 255

    h, w = image_bgr.shape[:2]
    if hair_mask.shape != (h, w):
        hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return hair_mask
