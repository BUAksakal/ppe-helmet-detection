# 🦺 PPE Helmet Detection

<div align="center">

<img src="assets/thd_logo.png" height="120" alt="TH Deggendorf"/>

**TH Deggendorf · MSS-M-2 · Machine Learning & Deep Learning · SS26**

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-FF6B6B?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-A78BFA?style=for-the-badge)
![TH Deggendorf](https://img.shields.io/badge/TH_Deggendorf-SS26-003366?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**Real-time safety helmet detection system using deep learning and computer vision.**  
Detects PPE compliance violations on construction sites via live camera feed.

[Demo](#demo) · [Dataset](#dataset) · [Results](#results) · [Installation](#installation)

</div>

---

## Overview

This project implements an end-to-end **Personal Protective Equipment (PPE) detection pipeline** that automatically identifies whether workers on construction sites are wearing safety helmets in real time.

Built as a **Machine Learning & Deep Learning Case Study** at **TH Deggendorf (SS26)**, the system processes live CCTV feeds at 25+ FPS and raises alerts on compliance violations — eliminating the need for manual safety inspections.

```
Input: Live camera / video  →  YOLOv11  →  helmet ✅ / no_helmet ❌  →  Alert
```

> 📍 **Institution:** Technische Hochschule Deggendorf  
> 📚 **Course:** Case Study Machine Learning & Deep Learning  
> 👥 **Group:** MSS-M-2 · Summer Semester 2026

---

## Features

- 🎯 **Real-time detection** at 25+ FPS on consumer GPU
- 🪖 **Two-class classification** — `helmet` vs `no_helmet`
- 📷 **Works on existing CCTV feeds** — no extra hardware required
- 🔔 **Instant violation alerts** for safety supervisors
- 📊 **mAP > 85%** on held-out test set (target)
- 🔄 **Augmentation pipeline** for robust generalization (grayscale, flip, brightness)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv11 (Ultralytics) |
| Vision Library | OpenCV |
| Dataset Management | Roboflow |
| Training | Google Colab / GPU |
| Language | Python 3.10+ |

---

## Dataset

- **Source:** Kaggle — [Helmet Detection Dataset](https://www.kaggle.com/datasets/alirezakiaipoor/helmet)
- **Images:** 1,012 manually annotated (from 4,877 collected)
- **Classes:** `helmet`, `no_helmet`
- **Augmented size:** ~5,200 images (Flip, Brightness, Grayscale)
- **Split:** 70% Train / 20% Validation / 10% Test
- **Format:** YOLO bounding box annotations via Roboflow

---

## Results

| Metric | Target | Achieved |
|--------|--------|----------|
| mAP@0.5 | ≥ 85% | TBD |
| Precision | ≥ 90% | TBD |
| Recall | ≥ 85% | TBD |
| Inference Speed | ≥ 25 FPS | TBD |

> Results will be updated after training is complete.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ppe-helmet-detection.git
cd ppe-helmet-detection
pip install ultralytics opencv-python
```

---

## Usage

**Run on webcam:**
```python
from ultralytics import YOLO

model = YOLO('runs/train/weights/best.pt')
model.predict(source=0, show=True, conf=0.5)
```

**Run on video file:**
```python
model.predict(source='construction_site.mp4', show=True, conf=0.5)
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
│   └── thd_logo.png        # TH Deggendorf logo
├── data/                   # Dataset (Roboflow export)
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/                   # Training outputs & weights
├── train.py                # Training script
├── detect.py               # Inference script
├── data.yaml               # Dataset config
└── README.md
```

---

## Training

```python
from ultralytics import YOLO

model = YOLO('yolo11s.pt')
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='ppe-helmet-v1'
)
```

---

## Roadmap

- [x] Dataset collection & annotation (1,012 images)
- [x] Augmentation pipeline (Roboflow)
- [ ] Model training (YOLOv11s)
- [ ] Evaluation & metrics
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
