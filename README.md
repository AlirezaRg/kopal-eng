# Seismic Inpainting — MDA GAN GUI

A desktop GUI for 3D seismic data reconstruction using the pretrained **MDA GAN** model.  
Load a seismic volume, draw a damaged region, and let the model reconstruct it — with a 4-panel comparison and before/after analysis.

---

## Features

- Load `.npy` or `.segy` seismic volumes
- Draw a missing/damaged region interactively with the mouse
- Two reconstruction methods:
  - **Deterministic** — linear interpolation
  - **Stochastic** — MDA GAN (deep learning, pretrained)
- 4-panel display: Original / Damaged / Reconstructed / Error Map
- Before/After comparison window with MSE and PSNR metrics
- Trace-by-trace comparison plot

---

## Project Structure

```
kopal/
├── MDA_GAN/
│   └── MDA_GAN-main/
│       ├── data/               # Seismic datasets (.npy)
│       │   ├── F3_salt.npy
│       │   ├── kerry.npy
│       │   ├── Parihaka.npy
│       │   └── ...
│       ├── weights/
│       │   └── MDA_GAN.pt      # Pretrained model weights
│       └── TRAIN_CODE/         # Original training code
└── SHELL/
    ├── GUIversion2.py                  # Main GUI application
    ├── seismic_inpainting_pipeline.py  # CLI pipeline (no GUI)
    └── تغییرات_پروژه.txt               # Change log (Persian)
```

---

## Requirements

- Python **3.11** (PyTorch does not support Python 3.13+)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/your-username/kopal.git
cd kopal
pip install -r requirements.txt
```

Download pretrained weights from:
```
https://drive.google.com/drive/folders/11oZC1uwdpCui1tVbAgjwMlNQYSLpozmW
```
Place `MDA_GAN.pt` inside `MDA_GAN/MDA_GAN-main/weights/`.

---

## Usage

### GUI (recommended)

```bash
python SHELL/GUIversion2.py
```

1. Click **Load NPY File** to load a seismic volume
2. Draw a rectangle on the **ORIGINAL** panel to mark the damaged region
3. Select **Stochastic (MDA GAN)** from the Method dropdown
4. Click **Run Inpainting**
5. Click **Show Before / After** to compare results

### CLI Pipeline

```bash
python SHELL/seismic_inpainting_pipeline.py
```

Edit the config section at the top of the file to set input path and damage region:

```python
INPUT_FILE          = None          # None = auto-pick first .npy in data/
DAMAGE_INLINE_START = None          # None = centre 10% of inlines
DAMAGE_INLINE_END   = None
DISPLAY_CROSSLINE   = None          # None = middle crossline
```

---

## Model

**MDA GAN** — Multi-Dimensional Adversarial GAN for 3D seismic interpolation.

> Dou, Yimin, et al.
> "MDA GAN: Adversarial-Learning-based 3-D Seismic Data Interpolation and Reconstruction for Complex Missing."
> arXiv:2204.03197 (2022)

Original repository: [https://github.com/douyimin/MDA_GAN](https://github.com/douyimin/MDA_GAN)

---

## Notes

- Inference runs on **CPU** by default — takes 1–5 minutes depending on mask size
- The GUI automatically crops to the bounding box of the mask for faster processing
- For GPU acceleration, a CUDA-capable NVIDIA GPU is required
