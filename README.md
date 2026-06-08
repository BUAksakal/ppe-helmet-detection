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

[Results](#results) · [Dataset](#dataset) · [Training](#training) · [Architecture Comparison](#architecture-comparison) · [Demo](#demo) · [Installation](#installation)

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


<img width="640" height="416" alt="demo" src="https://github.com/user-attachments/assets/47aacd50-89f1-4fb2-861c-adc648bdc201" />


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
<img width="2400" height="1200" alt="result" src="https://github.com/user-attachments/assets/c3abd226-6a96-46c3-9bbc-91c4e511f47e" />

*Loss curves (box, cls, dfl) and metric progression over 25 epochs on Tesla T4 GPU.*

### Confusion Matrix

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e9273859-9f3a-4f6f-ba1a-27da05434093" />

*506 correct Helmet detections · 76 correct No_Helmet detections · 37 Helmet false positives from background.*

### Detection Examples

<img width="1920" height="1920" alt="result2" src="https://github.com/user-attachments/assets/10253b48-250a-4026-9b44-43f4ae5dc016" />

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

## Architecture Comparison

To scientifically justify our model selection, we trained a **Faster R-CNN (ResNet-50 FPN)** benchmark on the identical dataset and compared it against YOLOv8s under controlled, reproducible conditions.

### Dataset Rebalancing

The original test split (~3%) was too small for statistically reliable evaluation. We rebalanced to an **80 / 10 / 10** split, yielding 117 test images evaluated identically for both models. A fixed `random.seed(42)` ensures reproducibility.

### Faster R-CNN Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-50 FPN |
| Pretrained weights | COCO (fine-tuned) |
| Epochs | 10 |
| Optimizer | SGD (lr=0.005, momentum=0.9, weight_decay=0.0005) |
| GPU | Tesla T4 (Google Colab) |

> **Format conversion:** YOLO-format annotations were converted on-the-fly to COCO format via a custom `PPEYoloDataset` class — no separate preprocessing pipeline. Both models trained on the exact same raw images.

### Comparative Results

| Metric | YOLOv8s (One-Stage) | Faster R-CNN (Two-Stage) |
|--------|--------------------|-----------------------|
| mAP@0.5 | **92.80%** | 88.73% |
| Inference Speed | **103 FPS** | 6.42 FPS |
| Inference Latency | **~9.7 ms** | ~155.8 ms |
| Training Epochs | 25 | 10 |
| Real-Time Suitability | ✅ Excellent | ❌ Limited |

### Why the Difference?

**YOLOv8s (One-Stage):** Detects objects in a single forward pass — bounding boxes and class labels predicted simultaneously. 9.7 ms latency makes it ideal for live edge deployments.

**Faster R-CNN (Two-Stage):** First generates region proposals via an RPN, then classifies each region separately. Mathematically more thorough, but the two-pass overhead results in 155.8 ms latency — **16× slower** than YOLO.

### Conclusion

> 🏆 YOLOv8s is **16× faster** and **4.07 percentage points more accurate** — the clear choice for real-time construction site CCTV monitoring. Faster R-CNN independently validated our dataset quality (both architectures converged on the same labels), but its latency is incompatible with live safety monitoring requirements.

The full training notebook is available in [`faster_rcnn/fasterrcnn_training.ipynb`](faster_rcnn/fasterrcnn_training.ipynb).

---

## Features

- 🎯 **Real-time detection** at ~145 FPS on consumer GPU
- 🪖 **Two-class classification** — `Helmet` vs `No_Helmet`
- 📷 **Works on existing CCTV feeds** — no extra hardware required
- 🔔 **Instant violation alerts** for safety supervisors
- 🖥️ **HelmGuard desktop app** — clean professional GUI (PyQt5)
- 🔄 **Multi-stage augmentation** (Roboflow + Ultralytics + Albumentations)
- 🔬 **Architecture comparison study** — YOLOv8s vs. Faster R-CNN

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8s (Ultralytics 8.2.103) |
| Benchmark Model | Faster R-CNN ResNet-50 FPN (torchvision) |
| Desktop GUI | PyQt5 |
| Vision Library | OpenCV |
| Dataset Management | Roboflow |
| Augmentation | Roboflow + Albumentations |
| Training | Google Colab (Tesla T4) |
| Language | Python 3.12 |

---

## Installation

```bash
git clone https://github.com/BUAksakal/ppe-helmet-detection.git
cd ppe-helmet-detection
pip install ultralytics opencv-python PyQt5 numpy
```

---

## Usage

**Run HelmGuard desktop app:**
```bash
python yolov8/app.py
```

**Run on webcam (script):**
```python
from ultralytics import YOLO

model = YOLO('yolov8/best.pt')
model.predict(source=0, show=True, conf=0.5)
```

**Run on image:**
```python
results = model.predict(source='worker.jpg', conf=0.5)
results[0].show()
```

**Run Faster R-CNN training:**
```bash
# Open in Google Colab
faster_rcnn/fasterrcnn_training.ipynb
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
├── yolov8/
│   ├── app.py                # HelmGuard desktop application
│   ├── best.pt               # Trained YOLOv8s weights
│   └── requirements.txt      # Dependencies
├── faster_rcnn/
│   └── fasterrcnn_training.ipynb  # Faster R-CNN training & evaluation
├── .gitignore
├── LICENSE
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
- [x] Architecture comparison — YOLOv8s vs. Faster R-CNN
- [x] Dataset rebalancing (80/10/10 split)
- [ ] Live demo video
- [ ] Final presentation (July 2026)

---

## References

- Wang et al. (2021). *Fast PPE Detection for Real Construction Sites Using Deep Learning.* Sensors, 21(10), 3478.
- Nath et al. (2020). *Deep learning for site safety: Real-time detection of PPE.* Automation in Construction, 112.
- Otgonbold et al. (2022). *SHEL5K: An Extended Dataset for Safety Helmet Detection.* Sensors, 22(6).
- Kumar et al. (2024). *PPE Detection using YOLOv8.* Cogent Engineering, 11(1).
- Ren et al. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS.

---

<div align="center">

<img src="assets/thd_logo.png" height="40" alt="TH Deggendorf"/>

*Made with ❤️ for safer workplaces through AI*  
**TH Deggendorf · MSS-M-2 · SS26**

</div>
