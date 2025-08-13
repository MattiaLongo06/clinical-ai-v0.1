# src/inference.py
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import models

def load_imagenet_resnet18(device: str = "cpu"):
    """Carica ResNet18 con pesi ImageNet e ritorna (model, weights)."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights).to(device).eval()
    return model, weights

def toy_prob_from_brightness(pil_img: Image.Image) -> Tuple[float, float, str]:
    """Stima DEMO: più l’immagine è scura, più 'pneumonia' (toy)."""
    gray = pil_img.convert("L")
    mean_pix = np.array(gray).mean() / 255.0
    p_pneumonia = float(1.0 - mean_pix)
    p_normal = 1.0 - p_pneumonia
    label = "Pneumonia (toy)" if p_pneumonia >= 0.5 else "Normal (toy)"
    return p_normal, p_pneumonia, label
