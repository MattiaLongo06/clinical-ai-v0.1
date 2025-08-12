# app.py — Clinical Imaging v0.1 (medical UI)

import os
os.environ["STREAMLIT_CONFIG_DIR"] = "/app/.streamlit"
import io
import numpy as np
from PIL import Image

import streamlit as st
import torch
from torchvision import models
from torchcam.methods import GradCAM
import cv2

from src.io_dicom import load_dicom_to_pil   # DICOM -> PIL
from src.storage import save_patient_row     # autosave CSV


# -----------------------------
# Page config + light, clean UI
# -----------------------------
st.set_page_config(page_title="Clinical Imaging v0.1 — Demo", layout="wide")

st.markdown("""
<style>
:root {
  --brand:#2363A9; --ink:#1D2430; --muted:#6C7585; --line:#E7E9EE;
  --bg:#FAFAFB; --card:#FFFFFF; --warn-bg:#FFF7ED; --warn-bd:#FFD9B0;
}
html, body, .stApp {
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
main .block-container { padding-top: 1.2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: 0.2px; color: var(--ink); }
.disclaimer { background: var(--warn-bg); border:1px solid var(--warn-bd);
  padding:10px 12px; border-radius:10px; font-size:0.95rem; }
.topbar {
  display:flex; align-items:center; justify-content:space-between;
  background: var(--card); border:1px solid var(--line);
  padding:10px 16px; border-radius:14px; margin-bottom: 14px;
}
.topbar .brand { display:flex; gap:12px; align-items:center; }
.topbar .brand .dot { width:10px; height:10px; border-radius:50%; background:var(--brand); }
.topbar .brand h1 { font-size:1.25rem; margin:0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; }
.card + .card { margin-top:12px; }
.stButton>button, .stDownloadButton>button {
  background: var(--brand); color:#fff; border:0; border-radius:10px;
  padding: 0.55rem 0.95rem; font-weight:600;
}
.stButton>button:hover, .stDownloadButton>button:hover { filter:brightness(0.97); }
.stSlider [data-baseweb="slider"]>div>div { background: var(--brand) !important; }
.stSlider [data-baseweb="slider"]>div>div>div { background: var(--brand) !important; }
section[data-testid="stSidebar"] { background: var(--card); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 12px; }
hr { border: none; border-top:1px solid var(--line); margin: 0.8rem 0 1rem; }
.small-muted { color: var(--muted); font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="topbar">'
    '  <div class="brand"><div class="dot"></div><h1>Clinical Imaging — Grad-CAM Demo</h1></div>'
    '  <div class="small-muted">Prototype • CPU-only</div>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="disclaimer"><b>NOT FOR CLINICAL USE</b> — Educational demo. No diagnostic claims.</div>', unsafe_allow_html=True)
st.caption("Stack: Streamlit · PyTorch · torchvision · torchcam · OpenCV · pydicom")


# -----------------------------
# Sidebar: patient + controls
# -----------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.sidebar:
    st.subheader("Patient (demo)")
    age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", ["", "Male", "Female", "Other"], index=0)
    symptoms = st.multiselect("Symptoms (demo)", ["Fever", "Cough", "Dyspnea", "Chest pain"])
    auto_save = st.checkbox("Save data automatically", value=True)

    st.markdown("---")
    st.subheader("Visualization")
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)
    view_mode = st.radio("View", ["Overlay", "Original only", "Side-by-side"], horizontal=False, index=0)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        clear_clicked = st.button("Clear")

if clear_clicked:
    st.session_state.uploader_key += 1
    st.experimental_rerun()


# -----------------------------
# File uploader
# -----------------------------
left, right = st.columns([1.2, 1.0])

with left:
    uploaded_file = st.file_uploader(
        "Upload image (PNG/JPG/DICOM)",
        type=["png", "jpg", "jpeg", "dcm"],
        key=f"uploader_{st.session_state.uploader_key}",
        help="Chest X-ray preferred for a meaningful overlay"
    )


# -----------------------------
# Utils
# -----------------------------
@st.cache_resource
def load_model():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.eval()
    return model, weights

def to_numpy(img: Image.Image, size=None) -> np.ndarray:
    if size:
        img = img.resize(size)
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return arr

def make_overlay(rgb_float01: np.ndarray, cam_float01: np.ndarray, alpha_overlay: float) -> Image.Image:
    cam_uint8 = (np.clip(cam_float01, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1 - alpha_overlay) * rgb_float01 + alpha_overlay * heatmap_rgb
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))

def toy_prob_from_brightness(pil_img: Image.Image) -> float:
    gray = pil_img.convert("L")
    mean_pix = np.array(gray).mean() / 255.0
    return float(1.0 - mean_pix)


# -----------------------------
# Main
# -----------------------------
if uploaded_file is None:
    with left:
        st.info("Upload an image to begin (ideally a chest X-ray).")
else:
    try:
        if uploaded_file.name.lower().endswith(".dcm"):
            image = load_dicom_to_pil(uploaded_file)
        else:
            image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        with left:
            st.error(f"Image loading error: {e}")
        st.stop()

    # Auto-save patient data
    if auto_save:
        try:
            csv_path = save_patient_row(age, sex, symptoms, uploaded_file.name)
            with right:
                st.success("Patient data saved in data/patient_data.csv")
        except Exception as e:
            with right:
                st.warning(f"Could not save patient data: {e}")

    # Model + preprocessing
    model, weights = load_model()
    transform = weights.transforms()
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad_(True)

    # Grad-CAM
    cam_img = None
    try:
        with GradCAM(model, target_layer="layer4") as cam_extractor:
            scores = model(input_tensor)
            class_idx = int(scores.argmax(dim=1).item())
            cams = cam_extractor(class_idx, scores)

        cam = cams[0].squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_resized = cv2.resize(cam, image.size, interpolation=cv2.INTER_LINEAR)

        base_np = to_numpy(image, size=image.size)
        cam_img = make_overlay(base_np, cam_resized, alpha_overlay=alpha)

    except Exception as e:
        with right:
            st.warning(f"Grad-CAM not available for this image: {e}")

    # Prediction (toy)
    p_pneumonia = toy_prob_from_brightness(image)
    p_normal = 1.0 - p_pneumonia
    label = "Pneumonia (toy)" if p_pneumonia >= 0.5 else "Normal (toy)"

    # Display results
    with left:
        if view_mode == "Overlay" and cam_img is not None:
            st.image(cam_img, caption=f"Grad-CAM overlay (alpha={alpha:.2f})", use_container_width=True)
        elif view_mode == "Original only" or cam_img is None:
            st.image(image, caption="Original image", use_container_width=True)
        else:
            col1, col2 = st.columns(2, gap="small")
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(cam_img if cam_img is not None else image,
                         caption=f"Overlay (alpha={alpha:.2f})", use_container_width=True)

    with right:
        st.subheader("Prediction (DEMO)")
        st.write(f"**Label:** {label}")
        st.write(f"**Prob. Normal (toy):** {p_normal:.2f}  |  **Prob. Pneumonia (toy):** {p_pneumonia:.2f}")
        st.markdown('<span class="small-muted">This is a visual explanation demo on a generic CNN (ResNet18 pre-trained on ImageNet).</span>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Export")
        report_img = cam_img if cam_img is not None else image
        buf = io.BytesIO()
        report_img.save(buf, format="PNG")
        st.download_button(
            "Download report (PNG)",
            data=buf.getvalue(),
            file_name="result_cam.png",
            mime="image/png"
        )
