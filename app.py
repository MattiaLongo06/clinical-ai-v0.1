import io
import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision import models
from torchcam.methods import GradCAM
import cv2

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Clinical Imaging v0.1", layout="centered")
st.title("Clinical Imaging v0.1 — Grad-CAM Demo")
st.markdown(":red[**NOT FOR CLINICAL USE**]")

# Sidebar controls
alpha = st.sidebar.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a chest image (PNG/JPG)",
    type=["png", "jpg", "jpeg"]
)

# -----------------------------
# Utils
# -----------------------------
@st.cache_resource
def load_model():
    """Load pretrained ResNet18 (ImageNet) for demo; CPU-only."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.eval()
    return model, weights

def to_numpy(img: Image.Image, size=None) -> np.ndarray:
    """PIL -> float32 numpy RGB in [0,1]. Optionally resize."""
    if size:
        img = img.resize(size)
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return arr

def make_overlay(rgb_float01: np.ndarray, cam_float01: np.ndarray, alpha_overlay: float) -> Image.Image:
    """Blend RGB image (HxWx3, 0..1) with heatmap (HxW, 0..1) using JET colormap."""
    cam_uint8 = (cam_float01 * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1 - alpha_overlay) * rgb_float01 + alpha_overlay * heatmap_rgb
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))

def toy_prob_from_brightness(pil_img: Image.Image) -> float:
    """Demo-only: darker images -> higher 'pneumonia' probability."""
    gray = pil_img.convert("L")
    mean_pix = np.array(gray).mean() / 255.0
    return float(1.0 - mean_pix)

# -----------------------------
# Main app
# -----------------------------
if uploaded_file is None:
    st.info("Carica un'immagine PNG/JPG per iniziare.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Load model & transform
    model, weights = load_model()
    transform = weights.transforms()

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # [1,3,224,224]
    input_tensor.requires_grad_(True)             # needed for CAM

    # Grad-CAM: register hooks BEFORE forward and compute with grads ON
    with GradCAM(model, target_layer="layer4") as cam_extractor:
        scores = model(input_tensor)              # [1, 1000], requires grad
        class_idx = int(scores.argmax(dim=1).item())
        cams = cam_extractor(class_idx, scores)   # list of CAMs for target_layer

    cam = cams[0].squeeze().detach().cpu().numpy()  # HxW

    # Normalize CAM to 0..1 and resize to original image size
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_resized = cv2.resize(cam, image.size, interpolation=cv2.INTER_LINEAR)

    # Create overlay
    base_np = to_numpy(image, size=image.size)
    cam_img = make_overlay(base_np, cam_resized, alpha_overlay=alpha)

    # Demo prediction (toy)
    p_pneumonia = toy_prob_from_brightness(image)
    p_normal = 1.0 - p_pneumonia
    label = "Pneumonia (toy)" if p_pneumonia >= 0.5 else "Normal (toy)"

    st.subheader("Prediction (DEMO)")
    st.write(f"**Label:** {label}")
    st.write(f"**Prob. Normal (toy):** {p_normal:.2f}  |  **Prob. Pneumonia (toy):** {p_pneumonia:.2f}")

    st.subheader("Grad-CAM")
    st.image(cam_img, caption=f"Grad-CAM overlay (alpha={alpha:.2f})", use_container_width=True)

    # Download PNG
    buf = io.BytesIO()
    cam_img.save(buf, format="PNG")
    st.download_button(
        "Download report (PNG)",
        data=buf.getvalue(),
        file_name="result_cam.png",
        mime="image/png"
    )

st.caption("Stack: Streamlit • PyTorch • torchvision • torchcam • OpenCV")
