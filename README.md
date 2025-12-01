# Hot Spot Detection in Thermal Imaging of Electrical Equipment

**Description:** This repository contains code and resources for detecting hot spots in thermal images of electrical equipment using machine learning techniques, especially convolutional neural networks (CNNs). The goal is to improve the reliability and efficiency of thermal inspections in industrial settings.

## Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Download and Preprocessing](#dataset-download-and-preprocessing)
  - [Downloading the Dataset](#downloading-the-dataset)
  - [Image Preprocessing](#image-preprocessing)
- [Automatic Annotation Systems](#automatic-annotation-systems)
  - [Binary Hotspot Classification Annotation](#binary-hotspot-classification-annotation)
  - [Hotspot Detection Annotation](#hotspot-detection-annotation-bounding-boxes)
- [Model Training](#model-training)
  - [Equipment Classification Fine-tuning](#equipment-classification-fine-tuning)
  - [Hotspot Binary Classification](#hotspot-binary-classification)
  - [Training Examples](#training-examples)
  - [Inference and Prediction](#inference-and-prediction)
- [Interactive Notebooks](#interactive-notebooks)
- [Usage](#usage)

## Introduction

Thermal imaging is a crucial technique for monitoring the condition of electrical equipment. Hot spots in thermal images can indicate potential failures or inefficiencies. This project aims to develop a machine learning model that can automatically detect these hot spots, thereby enhancing the maintenance process.

In this repository, we provide:
- A complete pipeline for downloading and preprocessing thermal images
- Three preprocessing strategies optimized for different tasks:
  - **Equipment type classification**: Identify the type of electrical equipment
  - **Binary hotspot classification**: Determine if an image contains hotspots
  - **Hotspot detection**: Localize hotspots with bounding boxes
- **Automatic annotation systems** using HSV color space analysis for:
  - Binary hotspot classification (hotspot/no_hotspot)
  - Hotspot detection with bounding boxes (YOLO format)
- Fine-tuning of state-of-the-art CNN models:
  - **Equipment Classification**: ResNet-50, Inception-v3
  - **Hotspot Classification**: EfficientNet-B0, DenseNet-121
  - **Hotspot Detection**: YOLOv11, YOLOv8
- **Focal Loss** implementation for handling class imbalance
- Training, evaluation, and inference scripts
- Interactive Jupyter notebooks for training and inference (6 notebooks total)
- Comprehensive logging and metrics tracking

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/santiago-v-2013/Hotspot_detection_on_electrical_equipment.git
cd your-repo
pip install -r requirements.txt
chmod +x bin/*.sh

# 2. Download thermal images dataset
./bin/download.sh

# 3. Auto-annotate hotspots for binary classification
./bin/annotate_hotspots.sh

# 4. Train hotspot classifier with Focal Loss
./bin/train_hotspot_classification.sh

# 5. View results
cat models/hotspot_classification/efficientnet_b0/test_metrics.json

# 6. Run inference in Jupyter
jupyter notebook notebooks/hotspot_classification_inference.ipynb
```

**Or train equipment classifier:**
```bash
./bin/preprocess_equipment.sh
./bin/train_equipment_classification.sh resnet50 50 32
```

## Key Features

### 🎯 Multi-Task Computer Vision System
- **Equipment Classification**: 5 equipment types (Circuit Breakers, Disconnectors, Power Transformers, Surge Arresters, Wave Traps)
- **Binary Hotspot Classification**: Detect presence/absence of thermal anomalies
- **Hotspot Detection**: Localize hotspots with bounding boxes in YOLO format

### 🤖 State-of-the-Art Models
- **ResNet-50** & **Inception-v3** for equipment classification
- **EfficientNet-B0** & **DenseNet-121** for hotspot classification
- **YOLOv11** & **YOLOv8** for hotspot detection with bounding boxes
- Pre-trained on ImageNet (classification) and COCO (detection)
- Support for both feature extraction and full fine-tuning

### 🔄 Automatic Annotation
- **HSV-based color analysis** for automatic hotspot detection
- **Binary classification annotation**: 895 images → 648 hotspot, 247 no_hotspot
- **Bounding box annotation**: 1,540 hotspots detected across 607 images
- **YOLO format export** ready for object detection training

### ⚖️ Advanced Training Techniques
- **Focal Loss** implementation for handling class imbalance (γ=2.0)
- **Automatic class weighting** based on dataset distribution
- **Multiple LR schedulers**: Step, Cosine Annealing, ReduceLROnPlateau
- **Early stopping** with configurable patience
- **Comprehensive data augmentation**: Flips, rotations, color jitter, affine transforms

### 📊 Extensive Metrics & Visualization
- Accuracy, Precision, Recall, F1-Score per class
- Confusion matrices with visualizations
- Training history plots (loss and accuracy curves)
- JSON export of all metrics for easy analysis
- Per-equipment performance breakdown

### 📓 Interactive Notebooks
- **6 Jupyter notebooks** for training and inference
  - Equipment classification: training + inference
  - Hotspot classification: training + inference
  - Hotspot detection: training + inference
- Side-by-side model comparison (ResNet vs Inception, EfficientNet vs DenseNet, YOLOv11 vs YOLOv8)
- Real-time training progress visualization
- Interactive hyperparameter experimentation

### 🛠️ Production-Ready Pipeline
- Automated bash scripts for all major operations
- Centralized logging system with detailed tracking
- GPU memory cleanup after training/inference
- Checkpoint management with best model selection
- Batch prediction support with JSON output

## Repository Structure

```
.
├── bin/                    # Bash scripts for automation
│   ├── download.sh         # Download images from dataset
│   ├── preprocess_equipment.sh
│   ├── preprocess_hotspot_classification.sh
│   ├── preprocess_hotspot_detection.sh
│   ├── annotate_hotspots.sh
│   ├── annotate_hotspot_detection.sh
│   ├── train_equipment_classification.sh
│   ├── train_hotspot_classification.sh
│   └── train_hotspot_detection.sh
├── data/                   # Directory for storing the dataset
│   ├── raw/                # Raw downloaded images organized by equipment type
│   │   ├── Circuit_Breakers/
│   │   ├── Disconnectors/
│   │   ├── Power_Transformers/
│   │   ├── Surge_Arresters/
│   │   └── Wave_Traps/
│   ├── processed/          # Preprocessed images for different tasks
│   │   ├── equipment_classification/
│   │   ├── hotspot_classification/
│   │   └── hotspot_detection/
│   └── urls.txt            # File containing image URLs
├── models/                 # Saved model checkpoints
│   ├── equipment_classification/
│   ├── hotspot_classification/
│   └── hotspot_detection/
├── src/                    # Source code directory
│   ├── Image_download/     # Image downloading module
│   │   ├── downloader.py   # ImageDownloader class
│   │   └── utils.py        # Utility functions
│   ├── preprocessing/      # Image preprocessing module
│   │   ├── base.py         # Base preprocessor class
│   │   ├── equipment_classifier.py
│   │   ├── hotspot_classifier.py
│   │   └── hotspot_detector.py
│   ├── finetuning/         # Model fine-tuning module
│   │   ├── dataset.py      # Equipment dataset and data loaders
│   │   ├── models.py       # ResNet-50 and Inception-v3 models
│   │   ├── hotspot_dataset.py  # Binary hotspot dataset
│   │   ├── hotspot_models.py   # EfficientNet-B0 and DenseNet-121 models
│   │   └── trainer.py      # Training loop, metrics, Focal Loss
│   ├── detection/          # Object detection module
│   │   ├── models.py       # YOLOv11 and YOLOv8 models
│   │   ├── dataset.py      # YOLO dataset loader
│   │   └── trainer.py      # Detection training interface
│   └── logger_cfg/         # Logging configuration module
│       └── setup.py        # Centralized logging setup
├── scripts/                # Executable scripts
│   ├── download_images.py
│   ├── preprocess_equipment_classification.py
│   ├── preprocess_hotspot_classification.py
│   ├── preprocess_hotspot_detection.py
│   ├── annotate_hotspots.py
│   ├── annotate_hotspot_detection.py
│   ├── prepare_yolo_dataset.py
│   ├── train_equipment_classification.py
│   ├── train_hotspot_classification.py
│   ├── train_hotspot_detection.py
│   ├── predict_equipment_classification.py
│   └── predict_hotspot_classification.py
├── notebooks/              # Jupyter notebooks
│   ├── equipment_classification_finetuning.ipynb
│   ├── equipment_classification_inference.ipynb
│   ├── hotspot_classification_finetuning.ipynb
│   ├── hotspot_classification_inference.ipynb
│   ├── hotspot_detection_training.ipynb
│   └── hotspot_detection_inference.ipynb
├── logs/                   # Directory for log files
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

### Option 1: Using Conda (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create conda environment:**
   ```bash
   conda create -n vision_env python=3.10 -y
   conda activate vision_env
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   # For CUDA 12.8
   conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
   
   # Or check https://pytorch.org/get-started/locally/ for your CUDA version
   ```

4. **Install other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Make scripts executable:**
   ```bash
   chmod +x bin/*.sh
   ```

### Option 2: Using pip

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make scripts executable:**
   ```bash
   chmod +x bin/*.sh
   ```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True
CUDA Version: 12.8
```

## Dataset Download and Preprocessing

### Downloading the Dataset

The dataset consists of thermal images of electrical equipment from the "Infrared Thermal Image Dataset of High Voltage Electrical Power Equipment under Different Operating Conditions" repository.

**Equipment Classes:**
- Circuit Breakers
- Disconnectors
- Power Transformers
- Surge Arresters
- Wave Traps

**Download the dataset:**

```bash
./bin/download.sh
```

The script will:
- Read URLs from `data/urls.txt`
- Download images organized by equipment type
- Save to `data/raw/`
- Log all operations to `logs/app.log`

**Run as background process (recommended for large datasets):**

```bash
nohup ./bin/download.sh &
tail -f logs/app.log  # Monitor progress
```

### Image Preprocessing

We provide three preprocessing pipelines optimized for different tasks:

#### 1. Equipment Type Classification

Prepares images for classifying equipment types:
- Grayscale conversion
- Histogram equalization
- Resizing to 224x224 pixels
- Normalization

```bash
./bin/preprocess_equipment.sh
```

Or with custom parameters:
```bash
python scripts/preprocess_equipment_classification.py \
    --input_dir data/raw \
    --output_dir data/processed/equipment_classification \
    --target_size 224 224 \
    --normalize
```

#### 2. Binary Hotspot Classification

Enhances thermal information for hotspot presence detection:
- Multi-channel color space conversion (HSV, LAB, YCrCb, or RGB)
- CLAHE for hot region enhancement
- Resizing to 224x224 pixels

```bash
./bin/preprocess_hotspot_classification.sh
```

Or with custom parameters:
```bash
python scripts/preprocess_hotspot_classification.py \
    --input_dir data/raw \
    --output_dir data/processed/hotspot_classification \
    --target_size 224 224 \
    --color_mode hsv
```

#### 3. Hotspot Detection with Bounding Boxes

Optimized for object detection models:
- RGB color information
- Edge enhancement (Canny)
- Thermal gradient computation (Sobel)
- Resizing to 416x416 pixels
- Aspect ratio preservation

```bash
./bin/preprocess_hotspot_detection.sh
```

Or with custom parameters:
```bash
python scripts/preprocess_hotspot_detection.py \
    --input_dir data/raw \
    --output_dir data/processed/hotspot_detection \
    --target_size 416 416
```

**Run preprocessing as background:**
```bash
nohup ./bin/preprocess_equipment.sh &
tail -f logs/app.log
```

## Model Training

### Equipment Classification Fine-tuning

Fine-tune pre-trained CNN models for classifying electrical equipment into 5 categories.

#### Available Models

**1. ResNet-50 (Residual Network)**
- 50 layers with residual connections
- Input size: 224×224 pixels
- ~25M parameters
- Fast training, excellent for transfer learning

**2. Inception-v3 (GoogleNet)**
- Inception modules with multi-scale feature extraction
- Input size: 299×299 pixels
- ~27M parameters
- Robust to different object scales

#### Basic Training

**Train ResNet-50:**
```bash
./bin/train_equipment_classification.sh resnet50 50 32
```

**Train Inception-v3:**
```bash
./bin/train_equipment_classification.sh inception_v3 50 32
```

**Train both models:**
```bash
./bin/train_equipment_classification.sh both 50 32
```

Parameters: `[model] [epochs] [batch_size]`

#### Advanced Training Options

```bash
python scripts/train_equipment_classification.py \
    --model resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --pretrained \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --early-stopping 10 \
    --scheduler step \
    --val-split 0.2 \
    --test-split 0.1
```

**Key Options:**
- `--model`: Model type (`resnet50`, `inception_v3`, or `both`)
- `--pretrained`: Use ImageNet pre-trained weights (recommended)
- `--freeze-backbone`: Only train classifier head (feature extraction mode)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: L2 regularization (default: 1e-4)
- `--scheduler`: Learning rate scheduler (`step`, `cosine`, `plateau`, `none`)
- `--early-stopping`: Patience for early stopping (default: 10 epochs)
- `--val-split`: Validation split fraction (default: 0.2)
- `--test-split`: Test split fraction (default: 0.1)

#### Training Features

- **Data Augmentation**: Random crops, flips, rotations, color jitter, affine transformations
- **Early Stopping**: Automatic stopping when validation accuracy plateaus
- **Class Weights**: Handles imbalanced datasets
- **Checkpointing**: Automatically saves best model based on validation accuracy
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Visualizations**: Training curves and confusion matrices
- **Logging**: Detailed logs saved to `logs/equipment_classification_training.log`

#### Training Output

After training, you'll find in `models/equipment_classification/<model>/`:
- `best_model.pth` - Best model checkpoint
- `best_metrics.json` - Validation metrics
- `test_metrics.json` - Test set evaluation
- `training_history.png` - Loss and accuracy curves
- `confusion_matrix.png` - Confusion matrix visualization

### Training Examples

**1. Quick training (feature extraction):**
```bash
python scripts/train_equipment_classification.py \
    --model resnet50 \
    --freeze-backbone \
    --epochs 20 \
    --lr 0.001
```

**2. Full fine-tuning:**
```bash
python scripts/train_equipment_classification.py \
    --model resnet50 \
    --epochs 50 \
    --lr 0.0001
```

**3. Training with cosine annealing:**
```bash
python scripts/train_equipment_classification.py \
    --model inception_v3 \
    --epochs 100 \
    --scheduler cosine \
    --batch-size 64
```

**4. Two-phase training (recommended):**

Phase 1 - Train classifier head only:
```bash
python scripts/train_equipment_classification.py \
    --model resnet50 \
    --freeze-backbone \
    --epochs 20 \
    --lr 0.001 \
    --checkpoint-dir models/equipment_classification_phase1
```

Phase 2 - Fine-tune entire network:
```bash
python scripts/train_equipment_classification.py \
    --model resnet50 \
    --epochs 30 \
    --lr 0.0001 \
    --checkpoint-dir models/equipment_classification_phase2
```

**5. Background training:**
```bash
nohup ./bin/train_equipment_classification.sh resnet50 50 32 > logs/training.log 2>&1 &
tail -f logs/equipment_classification_training.log
```

### Inference and Prediction

**Predict class for a single image:**
```bash
python scripts/predict_equipment_classification.py \
    --checkpoint models/equipment_classification/resnet50/best_model.pth \
    --image path/to/image.jpg \
    --top-k 3
```

**Predict for multiple images:**
```bash
python scripts/predict_equipment_classification.py \
    --checkpoint models/equipment_classification/resnet50/best_model.pth \
    --image-dir data/raw/Circuit_Breakers \
    --output predictions.json
```

**Prediction Output:**
- Top-K predictions with probabilities
- Confidence scores
- Optional JSON output for batch processing

---

## Automatic Annotation Systems

The project includes automatic annotation systems that use HSV color space analysis to detect and annotate hotspots in thermal images.

### Binary Hotspot Classification Annotation

Automatically classifies images as `hotspot` or `no_hotspot` based on the presence and size of hot regions.

**Run annotation:**
```bash
./bin/annotate_hotspots.sh
```

**Or with custom parameters:**
```bash
python scripts/annotate_hotspots.py \
    --input-dir data/raw \
    --output-dir data/processed/hotspot_classification \
    --min-area-ratio 0.01 \
    --intensity-threshold 150
```

**Detection Method:**
- HSV color space analysis (red: 0-10°/170-180°, orange: 10-25°)
- Minimum hot area ratio: 1% of image (adjustable)
- Intensity threshold: 150/255 (adjustable)
- Saturation threshold: 50/255

**Output Structure:**
```
data/processed/hotspot_classification/
├── annotations.json              # Complete annotation metadata
├── annotation_examples.png       # Visual examples (6 samples)
├── Circuit_Breakers/
│   ├── hotspot/                 # Images with detected hotspots
│   └── no_hotspot/              # Images without hotspots
├── Disconnectors/
├── Power_Transformers/
├── Surge_Arresters/
└── Wave_Traps/
```

**Annotation Results:**
- Total images: 895
- With hotspots: 648 (72.4%)
- Without hotspots: 247 (27.6%)

### Hotspot Detection Annotation (Bounding Boxes)

Automatically generates bounding box annotations in YOLO format for training object detection models.

**Run annotation:**
```bash
./bin/annotate_hotspot_detection.sh
```

**Or with custom parameters:**
```bash
python scripts/annotate_hotspot_detection.py \
    --data-dir data/raw \
    --output-dir data/processed/hotspot_detection \
    --min-area-ratio 0.005 \
    --min-intensity 150
```

**Detection Method:**
- HSV color space for red/orange detection
- Contour analysis to find individual hotspot regions
- Bounding box generation for each detected region
- IoU-based merging of overlapping boxes (threshold: 0.3)
- Size filtering: 0.5% to 50% of image area

**Output Structure:**
```
data/processed/hotspot_detection/
├── annotations.json              # Complete annotation metadata
├── annotation_examples.png       # Visual examples with bounding boxes
├── Circuit_Breakers/
│   ├── FLIR2296.jpg             # Original image
│   ├── FLIR2296.txt             # YOLO format annotation
│   └── ...
├── Disconnectors/
├── Power_Transformers/
├── Surge_Arresters/
└── Wave_Traps/
```

**YOLO Format:**
Each `.txt` file contains one line per detection:
```
class_id x_center y_center width height
```
All coordinates are normalized to [0, 1].

**Annotation Results:**
- Total images: 895
- Images with hotspots: 607 (67.8%)
- Total hotspots detected: 1,540
- Average hotspots per image: 2.54

---

## Hotspot Binary Classification

Train models to classify whether thermal images contain hotspots or not.

### Available Models

**1. EfficientNet-B0**
- Compound scaling method
- Input size: 224×224 pixels
- ~5.3M parameters
- Efficient and accurate

**2. DenseNet-121**
- Dense connections between layers
- Input size: 224×224 pixels
- ~8.0M parameters
- Strong feature reuse

### Training with Focal Loss

The training includes **Focal Loss** to handle class imbalance effectively.

**Basic Training:**
```bash
./bin/train_hotspot_classification.sh
```

**Or train specific model:**
```bash
python scripts/train_hotspot_classification.py \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 32
```

**Advanced Training:**
```bash
python scripts/train_hotspot_classification.py \
    --model densenet121 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --focal-loss \
    --focal-gamma 2.0 \
    --scheduler plateau \
    --early-stopping 10
```

**Key Features:**
- **Focal Loss**: Automatically applied with gamma=2.0
- **Class Weights**: Calculated automatically based on class distribution
- **Data Augmentation**: Random flips, rotations, color jitter
- **LR Schedulers**: Step, Cosine, ReduceLROnPlateau
- **Early Stopping**: Monitors validation loss
- **Metrics Export**: JSON files for train, validation, and test metrics

**Training Output:**
```
models/hotspot_classification/<model>/
├── best_model.pth              # Best model checkpoint
├── best_metrics.json           # Validation metrics
├── test_metrics.json           # Test set evaluation
├── training_history.png        # Loss and accuracy curves
└── confusion_matrix.png        # Confusion matrix
```

### Inference

**Predict single image:**
```bash
python scripts/predict_hotspot_classification.py \
    --checkpoint models/hotspot_classification/efficientnet_b0/best_model.pth \
    --image path/to/image.jpg
```

**Batch prediction:**
```bash
python scripts/predict_hotspot_classification.py \
    --checkpoint models/hotspot_classification/efficientnet_b0/best_model.pth \
    --image-dir data/test_images \
    --output predictions.json
```

### Interactive Notebooks

**Training Notebook:**
```bash
jupyter notebook notebooks/hotspot_classification_finetuning.ipynb
```

**Inference Notebook:**
```bash
jupyter notebook notebooks/hotspot_classification_inference.ipynb
```

The inference notebook allows you to:
- Load both EfficientNet-B0 and DenseNet-121 models
- Compare predictions from both models
- Visualize results with confidence scores
- Analyze per-equipment performance

---

## Hotspot Detection with Object Detection

Train YOLOv11 and YOLOv8 models to detect and localize hotspots with bounding boxes.

### Available Models

**1. YOLOv11 (2024)**
- Latest YOLO architecture from Ultralytics
- Input size: 416×416 pixels
- ~9.4M parameters (s variant)
- State-of-the-art performance with improved speed

**2. YOLOv8 (2023)**
- Previous generation YOLO
- Input size: 416×416 pixels
- ~11.2M parameters (s variant)
- Proven reliability for comparison

### Dataset Preparation

Convert annotated images to YOLO format:
```bash
python scripts/prepare_yolo_dataset.py
```

This creates:
- `data/yolo_detection/train/` - 422 images (70%)
- `data/yolo_detection/val/` - 121 images (20%)
- `data/yolo_detection/test/` - 62 images (10%)
- `data/yolo_detection/data.yaml` - Dataset configuration

### Training

**Train both models sequentially:**
```bash
./bin/train_hotspot_detection.sh 50 32 32
```

Parameters: `[epochs] [batch_size_yolov11] [batch_size_yolov8]`

**Train specific model:**
```bash
# YOLOv11
python scripts/train_hotspot_detection.py \
    --model yolov11 \
    --size s \
    --epochs 50 \
    --batch-size 32 \
    --imgsz 416

# YOLOv8
python scripts/train_hotspot_detection.py \
    --model yolov8 \
    --size s \
    --epochs 50 \
    --batch-size 32 \
    --imgsz 416
```

**Advanced options:**
```bash
python scripts/train_hotspot_detection.py \
    --model yolov11 \
    --size m \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --lr0 0.01 \
    --patience 15 \
    --workers 8
```

**Key Options:**
- `--model`: Model architecture (`yolov11`, `yolov8`)
- `--size`: Model size (`n`, `s`, `m`, `l`, `x`)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--imgsz`: Image size (default: 416)
- `--lr0`: Initial learning rate (default: 0.01)
- `--patience`: Early stopping patience (default: 10)
- `--device`: GPU device (default: 0)

### Training Features

- **Automatic metrics**: mAP@50, mAP@50-95, Precision, Recall
- **Real-time validation**: Evaluated every epoch
- **Early stopping**: Based on validation mAP@50
- **Checkpointing**: Saves best model and periodic checkpoints
- **Data augmentation**: Mosaic, mixup, flips, rotations, color jitter
- **Mixed precision**: FP16 training for faster computation
- **Comprehensive logs**: Training curves, confusion matrix, F1-curve

### Training Output

After training, you'll find in `models/hotspot_detection/<model>/`:
- `weights/best.pt` - Best model checkpoint (highest mAP@50)
- `weights/last.pt` - Last epoch checkpoint
- `results.csv` - Training metrics per epoch
- `results.png` - Training curves visualization
- `confusion_matrix.png` - Confusion matrix
- `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png`

### Inference

**Predict with trained model:**
```bash
python scripts/train_hotspot_detection.py \
    --model yolov11 \
    --predict \
    --weights models/hotspot_detection/yolov11s/weights/best.pt \
    --source data/yolo_detection/test/images
```

**Interactive notebooks:**
```bash
jupyter notebook notebooks/hotspot_detection_training.ipynb
jupyter notebook notebooks/hotspot_detection_inference.ipynb
```

The notebooks allow you to:
- Train both YOLOv11 and YOLOv8 interactively
- Compare model performance side-by-side
- Visualize detections on test images
- Analyze metrics (mAP, Precision, Recall)
- Export results to CSV and plots

### Expected Performance

**YOLOv8s** (tested):
- mAP@50: 0.937 (93.7%)
- mAP@50-95: 0.815 (81.5%)
- Precision: 0.911
- Recall: 0.897
- Training time: ~3s/epoch (RTX 4070)

**YOLOv11s** (expected similar or better):
- mAP@50: ~0.94+
- Faster inference
- Better generalization

---

## Interactive Notebooks

The project includes several Jupyter notebooks for interactive model training and inference.

### Equipment Classification

**Training Notebook:**
```bash
jupyter notebook notebooks/equipment_classification_finetuning.ipynb
```

Features:
- Interactive training of ResNet-50 and Inception-v3
- Real-time training progress visualization
- Hyperparameter experimentation
- Results analysis and model comparison

**Inference Notebook:**
```bash
jupyter notebook notebooks/equipment_classification_inference.ipynb
```

Features:
- Load both trained models (ResNet-50, Inception-v3)
- Side-by-side prediction comparison
- Confidence score visualization
- Test set evaluation

### Hotspot Classification

**Training Notebook:**
```bash
jupyter notebook notebooks/hotspot_classification_finetuning.ipynb
```

Features:
- Train EfficientNet-B0 and DenseNet-121 with Focal Loss
- Interactive dataset visualization
- Class imbalance analysis
- Training monitoring and debugging

**Inference Notebook:**
```bash
jupyter notebook notebooks/hotspot_classification_inference.ipynb
```

Features:
- Compare EfficientNet-B0 and DenseNet-121 predictions
- Analyze per-equipment performance
- Visualize prediction confidence
- Load and display test metrics

### Hotspot Detection

**Training Notebook:**
```bash
jupyter notebook notebooks/hotspot_detection_training.ipynb
```

Features:
- Train YOLOv11 and YOLOv8 sequentially
- GPU detection and configuration
- Interactive hyperparameter tuning
- Compare training metrics between models
- Plot mAP, Precision, Recall curves

**Inference Notebook:**
```bash
jupyter notebook notebooks/hotspot_detection_inference.ipynb
```

Features:
- Load both YOLOv11 and YOLOv8 models
- Side-by-side detection comparison
- Visualize bounding boxes with confidence scores
- Evaluate on full test set
- Export results to CSV and plots
- Custom image inference

---

## Usage

### Complete Workflow Example

**Equipment Classification Pipeline:**
```bash
# 1. Download dataset
./bin/download.sh

# 2. Preprocess images
./bin/preprocess_equipment.sh

# 3. Train model
./bin/train_equipment_classification.sh resnet50 50 32

# 4. Make predictions
python scripts/predict_equipment_classification.py \
    --checkpoint models/equipment_classification/resnet50/best_model.pth \
    --image-dir data/test_images \
    --output results.json

# 5. View results
cat models/equipment_classification/resnet50/test_metrics.json
```

**Hotspot Classification Pipeline:**
```bash
# 1. Download dataset (if not done)
./bin/download.sh

# 2. Auto-annotate hotspots
./bin/annotate_hotspots.sh

# 3. Train models with Focal Loss
./bin/train_hotspot_classification.sh

# 4. Make predictions
python scripts/predict_hotspot_classification.py \
    --checkpoint models/hotspot_classification/efficientnet_b0/best_model.pth \
    --image-dir data/test_images \
    --output hotspot_predictions.json

# 5. View results
cat models/hotspot_classification/efficientnet_b0/test_metrics.json
```

**Hotspot Detection Pipeline:**
```bash
# 1. Download dataset (if not done)
./bin/download.sh

# 2. Auto-annotate with bounding boxes
./bin/annotate_hotspot_detection.sh

# 3. Prepare YOLO dataset
python scripts/prepare_yolo_dataset.py

# 4. Train both YOLOv11 and YOLOv8
./bin/train_hotspot_detection.sh 50 32 32

# 5. View results
cat models/hotspot_detection/yolov11s/results.csv
cat models/hotspot_detection/yolov8s/results.csv

# 6. Run inference notebook
jupyter notebook notebooks/hotspot_detection_inference.ipynb
```

### Jupyter Notebook

For interactive training and analysis, use the provided notebooks:

**Equipment Classification:**
```bash
jupyter notebook notebooks/equipment_classification_finetuning.ipynb
jupyter notebook notebooks/equipment_classification_inference.ipynb
```

**Hotspot Classification:**
```bash
jupyter notebook notebooks/hotspot_classification_finetuning.ipynb
jupyter notebook notebooks/hotspot_classification_inference.ipynb
```

The notebooks include:
- Data loading and visualization
- Model creation and configuration
- Training loop with progress tracking
- Results visualization and comparison
- Inference examples with multiple models

### Monitoring and Logs

**View equipment classification training logs:**
```bash
tail -f logs/equipment_classification_training.log
```

**View hotspot classification training logs:**
```bash
tail -f logs/hotspot_classification_training.log
```

**View hotspot detection training logs:**
```bash
tail -f logs/app.log  # YOLO training logs
```

**Search for specific metrics:**
```bash
grep "Val Acc" logs/equipment_classification_training.log
grep "Test Accuracy" logs/equipment_classification_training.log
grep "Focal Loss" logs/hotspot_classification_training.log
grep "mAP" logs/app.log
```

**View model metrics:**
```bash
# Equipment classification
cat models/equipment_classification/resnet50/best_metrics.json | python -m json.tool
cat models/equipment_classification/resnet50/test_metrics.json | python -m json.tool

# Hotspot classification
cat models/hotspot_classification/efficientnet_b0/test_metrics.json | python -m json.tool
cat models/hotspot_classification/densenet121/test_metrics.json | python -m json.tool
```

**Compare models:**
```bash
# Equipment classification
echo "=== ResNet-50 ===" && cat models/equipment_classification/resnet50/test_metrics.json | grep accuracy
echo "=== Inception-v3 ===" && cat models/equipment_classification/inception_v3/test_metrics.json | grep accuracy

# Hotspot classification
echo "=== EfficientNet-B0 ===" && cat models/hotspot_classification/efficientnet_b0/test_metrics.json | grep accuracy
echo "=== DenseNet-121 ===" && cat models/hotspot_classification/densenet121/test_metrics.json | grep accuracy

# Hotspot detection
echo "=== YOLOv11s ===" && tail -1 models/hotspot_detection/yolov11s/results.csv
echo "=== YOLOv8s ===" && tail -1 models/hotspot_detection/yolov8s/results.csv
```

### Troubleshooting

**Out of Memory (OOM):**
```bash
python scripts/train_equipment_classification.py \
    --model resnet50 \
    --batch-size 16 \
    --num-workers 2
```

**Overfitting:**
```bash
python scripts/train_equipment_classification.py \
    --model resnet50 \
    --weight-decay 1e-3 \
    --early-stopping 5 \
    --freeze-backbone
```

**Underfitting:**
```bash
python scripts/train_equipment_classification.py \
    --model resnet50 \
    --epochs 100 \
    --lr 0.0001
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

This project uses thermal images from the "Infrared Thermal Image Dataset of High Voltage Electrical Power Equipment under Different Operating Conditions" repository.

## References

- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Deep Residual Learning for Image Recognition
- [Inception-v3 Paper](https://arxiv.org/abs/1512.00567) - Rethinking the Inception Architecture for Computer Vision
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- [DenseNet Paper](https://arxiv.org/abs/1608.06993) - Densely Connected Convolutional Networks
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002) - Focal Loss for Dense Object Detection
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/) - YOLOv8 Official Documentation
- [YOLOv11 Release](https://docs.ultralytics.com/) - Ultralytics YOLO11 Latest Release
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [YOLO Format Documentation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
