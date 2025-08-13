# src/preprocessing.py
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def to_numpy(img: Image.Image, size: Optional[Tuple[int,int]] = None) -> np.ndarray:
    """PIL -> np.float32 RGB [0..1] (opz. resize)"""
    if size:
        img = img.resize(size)
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return arr

def default_transform() -> transforms.Compose:
    """Fallback nel caso non si passi weights.transforms()"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

def prepare_input(image: Image.Image, weights=None) -> torch.Tensor:
    """Applica la transform dei pesi se disponibile, altrimenti default."""
    tfm = weights.transforms() if weights is not None else default_transform()
    x = tfm(image).unsqueeze(0)
    x.requires_grad_(True)
    return x
