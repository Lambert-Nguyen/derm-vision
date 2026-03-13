# DermVision 🔬

**SJSU CMPE 258 Deep Learning Project**
Skin Disease Classification System Using Deep Learning: A Multi-Class Approach

---

## Project Overview

DermVision is an end-to-end deep learning pipeline for automated skin lesion
classification using the **ISIC 2019** dataset.  The system classifies
dermoscopic images into **8 diagnostic categories** using an
**EfficientNet-B3** transfer learning backbone, weighted cross-entropy loss to
address severe class imbalance, and a Gradio web interface for interactive
inference.

| Attribute | Value |
|-----------|-------|
| Dataset | ISIC 2019 |
| Total images | 25,331 |
| Classes | 8 |
| Primary backbone | EfficientNet-B3 |
| Framework | PyTorch |
| Input resolution | 300 × 300 px |

---

## Dataset — ISIC 2019

Download: <https://challenge.isic-archive.com/data/#2019>

### 8 Diagnostic Classes

| Code | Full Name |
|------|-----------|
| **MEL** | Melanoma |
| **NV** | Melanocytic nevus |
| **BCC** | Basal cell carcinoma |
| **AKIEC** | Actinic keratosis / Intraepithelial carcinoma |
| **BKL** | Benign keratosis-like lesion |
| **DF** | Dermatofibroma |
| **VASC** | Vascular lesion |
| **SCC** | Squamous cell carcinoma |

---

## Repository Structure

```
derm-vision/
├── app/
│   └── app.py                  # Gradio web-app stub
├── configs/
│   └── config.yaml             # Hyperparameters & class names
├── data/
│   ├── raw/                    # Original ISIC 2019 images (git-ignored)
│   ├── processed/              # Resized / pre-cached images (git-ignored)
│   └── splits/                 # train/val/test CSV manifests
├── notebooks/
│   ├── 01_eda.ipynb            # Class distribution, sample images, metadata
│   └── 02_preprocessing.ipynb # Augmentation pipeline demo
├── outputs/
│   ├── checkpoints/            # Saved model weights (git-ignored)
│   └── results/                # Evaluation plots & metrics
├── src/
│   ├── dataset.py              # PyTorch Dataset class
│   ├── evaluate.py             # Balanced accuracy, F1, confusion matrix
│   ├── gradcam.py              # Grad-CAM visualisation
│   ├── train.py                # Training loop + W&B logging
│   ├── transforms.py           # Train / val augmentation pipelines
│   └── models/
│       ├── custom_cnn.py       # Baseline 4-layer CNN
│       ├── efficientnet.py     # EfficientNet-B3 transfer learning
│       └── ensemble.py         # Weighted-averaging ensemble stub
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Team Roles

| Member | Role |
|--------|------|
| **Lam** | Data pipeline (`dataset.py`, `transforms.py`, notebooks) |
| **James** | Model development (`models/`, `train.py`, `evaluate.py`) |
| **Vi** | Deployment (`app/app.py`, `gradcam.py`, CI/CD) |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Lambert-Nguyen/derm-vision.git
cd derm-vision
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

1. Go to <https://challenge.isic-archive.com/data/#2019>
2. Download **ISIC_2019_Training_Input.zip** and
   **ISIC_2019_Training_GroundTruth.csv** and
   **ISIC_2019_Training_Metadata.csv**
3. Extract into `data/raw/`

```
data/raw/
├── ISIC_2019_Training_Input/     # ~25 k .jpg images
├── ISIC_2019_Training_GroundTruth.csv
└── ISIC_2019_Training_Metadata.csv
```

### 5. Prepare train / val / test splits

```bash
python src/dataset.py   # (or run notebooks/01_eda.ipynb to inspect first)
```

Place the generated CSVs in `data/splits/`.

### 6. Train the model

```bash
# With W&B logging
python src/train.py --config configs/config.yaml

# Without W&B
python src/train.py --config configs/config.yaml --no_wandb
```

### 7. Evaluate

```bash
python src/evaluate.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --split test
```

### 8. Grad-CAM visualisation

```bash
python src/gradcam.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --image data/raw/ISIC_2019_Training_Input/ISIC_0024306.jpg
```

### 9. Launch the web app

```bash
python app/app.py --checkpoint outputs/checkpoints/best_model.pt
```

Then open <http://localhost:7860> in your browser.

---

## Key Hyperparameters (defaults in `configs/config.yaml`)

| Parameter | Value |
|-----------|-------|
| `image_size` | 300 |
| `batch_size` | 32 |
| `learning_rate` | 1e-4 |
| `epochs` | 30 |
| `weight_decay` | 1e-4 |
| `num_classes` | 8 |
| `scheduler` | CosineAnnealingWarmRestarts |

---

## License

This project is developed for academic purposes at San José State University
(CMPE 258).  See [LICENSE](LICENSE) for details.
