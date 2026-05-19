# 🦺 PPE Helmet Detection

<div align="center">

<img src="assets/thd_logo.png" height="120" alt="TH Deggendorf"/>

**TH Deggendorf · MSS-M-2 · Machine Learning & Deep Learning · SS26**

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B6B?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-A78BFA?style=for-the-badge)
![TH Deggendorf](https://img.shields.io/badge/TH_Deggendorf-SS26-003366?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**Real-time safety helmet detection system using deep learning and computer vision.**  
Detects PPE compliance violations on construction sites via live camera feed.

[Results](#results) · [Dataset](#dataset) · [Demo](#demo) · [Installation](#installation)

</div>

---

## Overview

This project implements an end-to-end **Personal Protective Equipment (PPE) detection pipeline** that automatically identifies whether workers on construction sites are wearing safety helmets in real time.

Built as a **Machine Learning & Deep Learning Case Study** at **TH Deggendorf (SS26)**, the system processes live CCTV feeds at ~75 FPS and raises instant alerts on compliance violations — eliminating the need for manual safety inspections.

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
| mAP@0.5 | ≥ 85% | **84.4%** |
| Precision | ≥ 90% | **88.5%** |
| Recall | ≥ 85% | **82.1%** |
| Inference Speed | ≥ 25 FPS | **~75 FPS** |

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| ✅ Helmet | 92.2% | 91.3% | 93.6% |
| ❌ No_Helmet | 85.0% | 73.0% | 75.0% |

### Training Curves

![Training Results](assets/results.png)

*Loss curves and metric progression over 25 epochs. Model trained on Tesla T4 GPU (~1 hour).*

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

*506 correct Helmet detections · 76 correct No_Helmet detections · Low false positive rate.*

### Detection Examples

![Detection Examples](assets/predictions.jpeg)

*Model predictions on test set images — bounding boxes with confidence scores.*

---

## Dataset

- **Source:** Kaggle — [Helmet Detection Dataset](https://www.kaggle.com/datasets/alirezakiaipoor/helmet)
- **Collected:** 4,877 images
- **Manually annotated:** 1,012 images (via Roboflow)
- **Classes:** `Helmet`, `No_Helmet`
- **Augmented size:** ~5,200 images (Horizontal Flip, Brightness, Grayscale)
- **Split:** 70% Train / 20% Validation / 10% Test
- **Annotation format:** YOLO bounding boxes

---

## Features

- 🎯 **Real-time detection** at ~75 FPS on consumer GPU
- 🪖 **Two-class classification** — `Helmet` vs `No_Helmet`
- 📷 **Works on existing CCTV feeds** — no extra hardware required
- 🔔 **Instant violation alerts** for safety supervisors
- 🖥️ **HelmGuard desktop app** — clean professional GUI (PyQt5)
- 🔄 **Augmentation pipeline** for robust generalization

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8s (Ultralytics) |
| Desktop GUI | PyQt5 |
| Vision Library | OpenCV |
| Dataset Management | Roboflow |
| Training Environment | Google Colab (Tesla T4) |
| Language | Python 3.10+ |

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
├── train.py                  # Training script
├── data.yaml                 # Dataset config
└── README.md
```

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
    name='ppe-helmet-v1'
)
```

**Training config:** 25 epochs · 800×800 · batch 16 · AdamW · Tesla T4 · ~1 hour

---

## Roadmap

- [x] Dataset collection (4,877 images)
- [x] Manual annotation (1,012 images via Roboflow)
- [x] Augmentation pipeline
- [x] Model training — YOLOv8s (25 epochs)
- [x] Evaluation & metrics
- [x] HelmGuard desktop application
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
