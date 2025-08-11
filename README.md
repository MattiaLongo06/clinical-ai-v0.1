# Clinical Imaging v0.1 — Grad-CAM Demo (NOT FOR CLINICAL USE)

**What:** Upload a chest image (PNG/JPG/DICOM) → get a toy prediction + a Grad-CAM heatmap showing where the model “looked”.  
**Why (for recruiters):** Demonstrates handling of medical imaging formats (DICOM), AI model inference, and visual explainability in a clean, reproducible app.

---

## 🚀 Quick start

**Online demo:** _TBD (Hugging Face Spaces link)_  
**GitHub repo:** [Clinical AI v0.1](<repo-link>)

### Run locally
```bash
conda create -n clinical-ai-v0_1 python=3.11 -y
conda activate clinical-ai-v0_1
pip install -r requirements.txt
streamlit run app.py

Features
DICOM support — load .dcm files via pydicom with basic windowing.

Pretrained CNN inference (ResNet18, ImageNet weights, CPU-friendly).

Grad-CAM explainability with adjustable opacity slider.

Optional patient metadata (age, sex, symptoms) saved to CSV.

Export overlay as PNG for sharing or documentation.

Clean, medical-style UI with custom Streamlit theme.

Screenshots
Grad-CAM Overlay on Chest DICOM (Demo)

<p align="center"> <img src="assets/screenshotOverlay.png" alt="Grad-CAM overlay on chest DICOM" width="800"> </p>
Patient Metadata & Controls (Sidebar)

<p align="center"> <img src="assets/screenshotSidebar.png" alt="Sidebar with patient metadata and visualization controls" width="800"> </p>
Limitations
Toy prediction only (brightness-based) — not for clinical use.

No diagnostic claims.

Educational and portfolio purposes only.

Tech stack
Frontend/UI: Streamlit (custom CSS theme)

Model: PyTorch + torchvision (ResNet18)

Explainability: torchcam (Grad-CAM)

Medical imaging: pydicom

Image processing: Pillow, OpenCV

Data storage: CSV via pandas

Roadmap
v0.2: Fine-tuned chest X-ray CNN, Docker deploy.

v0.3: Segmentation (U-Net) + FHIR API integration.

v0.4: MLflow tracking, calibration, and external validation.

Author
Developed by Mattia Lorenzo Longo as part of a long-term Clinical AI Engineer career plan.

