# src/explain.py
from typing import Tuple
import numpy as np
import torch
import cv2
from PIL import Image
from torchcam.methods import GradCAM

def gradcam_heatmap(model, input_tensor: torch.Tensor, target_layer: str = "layer4") -> np.ndarray:
    """
    Esegue forward, estrae Grad-CAM per la classe top-1.
    Ritorna HxW float32 normalizzato [0..1] nella risoluzione del tensor (prima del resize esterno).
    """
    with GradCAM(model, target_layer=target_layer) as cam_extractor:
        scores = model(input_tensor)
        class_idx = int(scores.argmax(dim=1).item())
        cams = cam_extractor(class_idx, scores)
    cam = cams[0].squeeze().detach().cpu().numpy().astype("float32")
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam  # [H,W] in [0,1]

def make_overlay(rgb_float01: np.ndarray, cam_float01: np.ndarray, alpha_overlay: float) -> Image.Image:
    """Colorizza la CAM e la fonde sull’immagine RGB [0..1] -> PIL.Image"""
    cam_uint8 = (np.clip(cam_float01, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1 - alpha_overlay) * rgb_float01 + alpha_overlay * heatmap_rgb
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))
