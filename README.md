# Clinical Imaging v0.1 — Grad-CAM Demo (NOT FOR CLINICAL USE)

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-brightgreen)
![Model](https://img.shields.io/badge/model-ResNet18-orange)
[![Live Demo](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow)]## Screenshots

**Upload DICOM & UI**
<p align="center">
  <img src="assets/screenshotOverlay.png" alt="Clinical Imaging — upload DICOM screen" width="800">
</p>

**Grad‑CAM overlay (demo)**
<p align="center">
  <img src="assets/screenshotSidebar.png" alt="Clinical Imaging — Grad-CAM overlay on chest image" width="800">
</p>


**What:**  
Upload a chest image (PNG, JPG, DICOM) → see a toy prediction + a Grad-CAM heatmap overlay showing where the model “looked”.

**Why:**  
Demonstrates ability to:
- Handle **DICOM medical images** via `pydicom`
- Run **PyTorch CNN inference** on CPU
- Generate **explainable AI visualizations** with Grad-CAM
- Build a **clean, reproducible UI** for healthcare using Streamlit
- Save patient demo metadata (CSV/SQLite-ready)
- Package and deploy on Hugging Face Spaces (no install required)

⚠ **Disclaimer:** Educational demo only — not a medical device. No diagnostic use.

---

## 🚀 Quick start

### **1. Online demo**
👉 [**Click here to try**](<link-spaces>)

### **2. Local**
```bash
git clone <https://github.com/MattiaLongo06/clinical-ai-v0.1>
cd clinical-ai-v0.1
conda create -n clinical-ai-v0_1 python=3.11 -y
conda activate clinical-ai-v0_1
pip install -r requirements.txt
streamlit run app.py
