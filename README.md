# 🦺 PPE Helmet Detection

<div align="center">

<img src="assets/thd_logo.png" height="120" alt="TH Deggendorf"/>

**TH Deggendorf · MSS-M-2 · Machine Learning & Deep Learning · SS26**

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-FF6B6B?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-A78BFA?style=for-the-badge)
![TH Deggendorf](https://img.shields.io/badge/TH_Deggendorf-SS26-003366?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**Real-time safety helmet detection system using deep learning and computer vision.**  
Detects PPE compliance violations on construction sites via live camera feed.

[Results](#results) · [Dataset](#dataset) · [Training](#training) · [Demo](#demo) · [Installation](#installation)

</div>

---

## Overview

This project implements an end-to-end **Personal Protective Equipment (PPE) detection pipeline** that automatically identifies whether workers on construction sites are wearing safety helmets in real time.

Built as a **Machine Learning & Deep Learning Case Study** at **TH Deggendorf (SS26)**, the system processes live CCTV feeds at ~145 FPS and raises instant alerts on compliance violations — eliminating the need for manual safety inspections.

```
Live Camera / Video  →  YOLOv8s  →  Helmet ✅ / No_Helmet ❌  →  Violation Alert
```

> 📍 **Institution:** Technische Hochschule Deggendorf  
> 📚 **Course:** Case Study Machine Learning & Deep Learning  
> 👥 **Group:** MSS-M-2 · Summer Semester 2026

---

## Demo

> HelmGuard desktop application — real-time helmet compliance monitoring via webcam.

![Demo](assets/demo.png)

*HelmGuard detects violations in real time, displaying live metrics and instant alerts.*

---

## Results

| Metric | Target | **Achieved** |
|--------|--------|-------------|
| mAP@0.5 | ≥ 85% | **84.3%** |
| mAP@0.5:0.95 | — | **47.2%** |
| Precision | ≥ 90% | **88.6%** |
| Recall | ≥ 85% | **82.2%** |
| Inference Speed | ≥ 25 FPS | **~145 FPS** (6.9ms/img) |

### Per-Class Performance

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|--------------|
| ✅ Helmet | 192 | 543 | 92.2% | 91.3% | 93.6% | 56.4% |
| ❌ No_Helmet | 33 | 100 | 85.0% | 73.0% | 75.0% | 37.9% |
| **All** | **203** | **643** | **88.6%** | **82.2%** | **84.3%** | **47.2%** |

### Training Curves

![Training Results](assets/results.png)

*Loss curves (box, cls, dfl) and metric progression over 25 epochs on Tesla T4 GPU.*

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

*506 correct Helmet detections · 76 correct No_Helmet detections · 37 Helmet false positives from background.*

### Detection Examples

![Detection Examples](assets/predictions.jpeg)

*Model predictions on test set — bounding boxes with confidence scores across diverse construction environments.*

---

## Dataset

- **Source:** Kaggle — [Helmet Detection Dataset](https://www.kaggle.com/datasets/alirezakiaipoor/helmet)
- **Collected:** 4,877 images
- **Manually annotated:** 1,012 images (via Roboflow)
- **Classes:** `Helmet`, `No_Helmet`
- **Split:** 70% Train / 20% Validation / 10% Test
- **Train set:** 4,925 images (after augmentation) · 0 backgrounds · 0 corrupt
- **Val set:** 203 images · 643 instances
- **Annotation format:** YOLO bounding boxes (Roboflow export)

---

## Data Augmentation

Augmentation was applied via **Roboflow** before export, expanding the dataset ~5× and improving generalization to real-world CCTV conditions:

| Augmentation | Parameters | Purpose |
|-------------|-----------|---------|
| Horizontal Flip | p=0.5 | Camera orientation variance |
| Brightness | ±25% | Indoor / outdoor lighting |
| Grayscale | p=0.15 | CCTV / monochrome feeds |

Additionally, **Ultralytics built-in augmentation** was applied during training:

| Augmentation | Value |
|-------------|-------|
| HSV Hue | 0.015 |
| HSV Saturation | 0.7 |
| HSV Value | 0.4 |
| Horizontal Flip | 0.5 |
| Mosaic | 1.0 (disabled last 10 epochs) |
| Random Erasing | 0.4 |
| Auto Augment | RandAugment |
| Albumentations | Blur, MedianBlur, ToGray, CLAHE |

---

## Training

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(
    data='data.yaml',
    epochs=25,
    imgsz=800,
    batch=16,
    optimizer='auto',   # AdamW selected automatically
    name='ppe-helmet-v1'
)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv8s (11.1M parameters, 28.6 GFLOPs) |
| Pretrained weights | COCO (transferred 349/355 layers) |
| Epochs | 25 |
| Image size | 800 × 800 |
| Batch size | 16 |
| Optimizer | AdamW (lr=0.001667, momentum=0.9) |
| GPU | Tesla T4 (14.9 GB) |
| Training time | ~1 hour 4 minutes |
| Best epoch | 15 (mAP@0.5: 84.3%) |

### Training Progress (selected epochs)

| Epoch | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| 1 | 71.5% | 78.8% | 66.1% |
| 5 | 73.2% | 80.7% | 66.2% |
| 10 | 81.6% | 78.8% | 79.4% |
| 15 | **84.3%** | **88.6%** | **82.2%** ← best |
| 20 | 84.2% | 88.0% | 80.9% |
| 25 | 82.2% | 87.7% | 79.1% |

---

## Features

- 🎯 **Real-time detection** at ~145 FPS on consumer GPU
- 🪖 **Two-class classification** — `Helmet` vs `No_Helmet`
- 📷 **Works on existing CCTV feeds** — no extra hardware required
- 🔔 **Instant violation alerts** for safety supervisors
- 🖥️ **HelmGuard desktop app** — clean professional GUI (PyQt5)
- 🔄 **Multi-stage augmentation** (Roboflow + Ultralytics + Albumentations)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8s (Ultralytics 8.2.103) |
| Desktop GUI | PyQt5 |
| Vision Library | OpenCV |
| Dataset Management | Roboflow |
| Augmentation | Roboflow + Albumentations |
| Training | Google Colab (Tesla T4) |
| Language | Python 3.12 |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ppe-helmet-detection.git
cd ppe-helmet-detection
pip install ultralytics opencv-python PyQt5 numpy
```

---

## Usage

**Run HelmGuard desktop app:**
```bash
python app.py
```

**Run on webcam (script):**
```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.predict(source=0, show=True, conf=0.5)
```

**Run on image:**
```python
results = model.predict(source='worker.jpg', conf=0.5)
results[0].show()
```

---

## Project Structure

```
ppe-helmet-detection/
├── assets/
│   ├── thd_logo.png          # TH Deggendorf logo
│   ├── confusion_matrix.png  # Confusion matrix
│   ├── results.png           # Training curves
│   ├── predictions.jpeg      # Detection examples
│   └── demo.png              # App screenshot
├── data/                     # Dataset (Roboflow export)
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/                     # Training outputs & weights
├── app.py                    # HelmGuard desktop application
├── data.yaml                 # Dataset config
└── README.md
```

---

## Roadmap

- [x] Dataset collection (4,877 images)
- [x] Manual annotation (1,012 images via Roboflow)
- [x] Augmentation pipeline (Roboflow + Albumentations)
- [x] Model training — YOLOv8s (25 epochs, Tesla T4)
- [x] Evaluation & metrics (mAP 84.3%, ~145 FPS)
- [x] HelmGuard desktop application (PyQt5)
- [ ] Live demo video
- [ ] Final presentation (July 2026)

---

## References

- Wang et al. (2021). *Fast PPE Detection for Real Construction Sites Using Deep Learning.* Sensors, 21(10), 3478.
- Nath et al. (2020). *Deep learning for site safety: Real-time detection of PPE.* Automation in Construction, 112.
- Otgonbold et al. (2022). *SHEL5K: An Extended Dataset for Safety Helmet Detection.* Sensors, 22(6).
- Kumar et al. (2024). *PPE Detection using YOLOv8.* Cogent Engineering, 11(1).

---

<div align="center">

<img src="assets/thd_logo.png" height="40" alt="TH Deggendorf"/>

*Made with ❤️ for safer workplaces through AI*  
**TH Deggendorf · MSS-M-2 · SS26**

</div>
