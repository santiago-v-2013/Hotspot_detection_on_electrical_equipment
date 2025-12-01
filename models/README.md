# Models Directory

This directory contains trained models organized by task type.

## Structure

```
models/
├── equipment_classification/    # Equipment type classification models
│   ├── resnet50/
│   │   ├── best_model.pth
│   │   ├── best_metrics.json
│   │   ├── test_metrics.json
│   │   ├── training_history.png
│   │   └── confusion_matrix.png
│   └── inception_v3/
│       └── ...
├── hotspot_classification/      # Binary hotspot classification models
│   ├── efficientnet_b0/
│   │   ├── best_model.pth
│   │   ├── best_metrics.json
│   │   ├── test_metrics.json
│   │   ├── training_history.png
│   │   └── confusion_matrix.png
│   └── densenet121/
│       └── ...
├── hotspot_detection/          # Hotspot detection with bounding boxes
│   ├── yolov11s/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   ├── results.csv
│   │   ├── results.png
│   │   ├── confusion_matrix.png
│   │   ├── F1_curve.png
│   │   ├── PR_curve.png
│   │   ├── P_curve.png
│   │   └── R_curve.png
│   └── yolov8s/
│       └── ...
└── pretrained/                 # Pre-trained weights cache
    ├── yolov11s.pt
    └── yolov8s.pt
```

## Task Types

### 1. Equipment Classification
**Directory**: `equipment_classification/`

Models for classifying electrical equipment into 5 categories:
- Circuit Breakers
- Disconnectors
- Power Transformers
- Surge Arresters
- Wave Traps

**Available Models**:
- `resnet50/` - ResNet-50 fine-tuned model
- `inception_v3/` - Inception-v3 fine-tuned model

**Train**:
```bash
./bin/train_equipment_classification.sh resnet50 50 32
```

**Predict**:
```bash
python scripts/predict_equipment_classification.py \
    --checkpoint models/equipment_classification/resnet50/best_model.pth \
    --image path/to/image.jpg
```

### 2. Hotspot Classification
**Directory**: `hotspot_classification/`

Models for binary classification of hotspot presence:
- Hotspot: Images containing thermal anomalies
- No Hotspot: Images without thermal anomalies

**Available Models**:
- `efficientnet_b0/` - EfficientNet-B0 with Focal Loss
- `densenet121/` - DenseNet-121 with Focal Loss

**Train**:
```bash
./bin/train_hotspot_classification.sh 50 32
```

**Predict**:
```bash
python scripts/predict_hotspot_classification.py \
    --checkpoint models/hotspot_classification/efficientnet_b0/best_model.pth \
    --image path/to/image.jpg
```

### 3. Hotspot Detection
**Directory**: `hotspot_detection/`

Models for localizing hotspots with bounding boxes using YOLO architecture.

**Available Models**:
- `yolov11s/` - YOLOv11-small (2024, latest)
- `yolov8s/` - YOLOv8-small (2023, proven)

**Dataset**:
- 422 training images
- 121 validation images
- 62 test images
- 1,540 hotspot annotations

**Train**:
```bash
# Train both models
./bin/train_hotspot_detection.sh 50 32 32

# Or train individually
python scripts/train_hotspot_detection.py --model yolov11 --size s --epochs 50
python scripts/train_hotspot_detection.py --model yolov8 --size s --epochs 50
```

**Predict**:
```bash
python scripts/train_hotspot_detection.py \
    --model yolov11 \
    --predict \
    --weights models/hotspot_detection/yolov11s/weights/best.pt \
    --source path/to/images
```

**Performance (YOLOv8s)**:
- mAP@50: 0.937 (93.7%)
- mAP@50-95: 0.815 (81.5%)
- Precision: 0.911
- Recall: 0.897

## Model Files

### Classification Models (Equipment & Hotspot)

Each classification model directory contains:
- `best_model.pth` - PyTorch model checkpoint (best validation accuracy)
- `best_metrics.json` - Validation metrics (accuracy, precision, recall, F1)
- `test_metrics.json` - Test set evaluation results
- `training_history.png` - Training and validation curves
- `confusion_matrix.png` - Confusion matrix visualization

### Detection Models (YOLO)

Each detection model directory contains:
- `weights/best.pt` - Best model checkpoint (highest mAP@50)
- `weights/last.pt` - Last epoch checkpoint
- `results.csv` - Training metrics per epoch
- `results.png` - Training curves (loss, mAP, precision, recall)
- `confusion_matrix.png` - Confusion matrix
- `F1_curve.png` - F1-Confidence curve
- `PR_curve.png` - Precision-Recall curve
- `P_curve.png` - Precision-Confidence curve
- `R_curve.png` - Recall-Confidence curve

## Usage

### Loading a Model

```python
import torch
from src.finetuning.models import get_model

# Load checkpoint
checkpoint = torch.load('models/equipment_classification/resnet50/best_model.pth')

# Get model info
class_names = checkpoint['class_names']
num_classes = len(class_names)

# Create and load model
model = get_model('resnet50', num_classes, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
```

### Loading a YOLO Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/hotspot_detection/yolov11s/weights/best.pt')

# Run inference
results = model.predict(
    source='path/to/image.jpg',
    conf=0.25,
    iou=0.45,
    save=True
)

# Access predictions
for result in results:
    boxes = result.boxes  # Bounding boxes
    confs = boxes.conf    # Confidence scores
    classes = boxes.cls   # Class labels
```

### Viewing Metrics

**Classification Models:**
```bash
# View validation metrics
cat models/equipment_classification/resnet50/best_metrics.json | python -m json.tool
cat models/hotspot_classification/efficientnet_b0/test_metrics.json | python -m json.tool

# Compare classification models
echo "=== ResNet-50 ===" && cat models/equipment_classification/resnet50/test_metrics.json | grep accuracy
echo "=== EfficientNet-B0 ===" && cat models/hotspot_classification/efficientnet_b0/test_metrics.json | grep accuracy
```

**Detection Models:**
```bash
# View training results
cat models/hotspot_detection/yolov11s/results.csv
cat models/hotspot_detection/yolov8s/results.csv

# Compare detection models (last epoch)
echo "=== YOLOv11s ===" && tail -1 models/hotspot_detection/yolov11s/results.csv
echo "=== YOLOv8s ===" && tail -1 models/hotspot_detection/yolov8s/results.csv

# View specific metrics
grep "mAP50" models/hotspot_detection/yolov11s/results.csv
```

## Interactive Notebooks

**Equipment Classification:**
```bash
jupyter notebook notebooks/equipment_classification_inference.ipynb
```

**Hotspot Classification:**
```bash
jupyter notebook notebooks/hotspot_classification_inference.ipynb
```

**Hotspot Detection:**
```bash
jupyter notebook notebooks/hotspot_detection_inference.ipynb
```

The notebooks provide:
- Side-by-side model comparison
- Interactive visualization of predictions
- Confidence score analysis
- Test set evaluation
- Export results to CSV and plots

## Notes

- Models are automatically saved during training to the appropriate subdirectory
- Classification: Best model based on validation accuracy
- Detection: Best model based on mAP@50
- All paths in the codebase use `models/<task_type>/` structure
- Git ignores `*.pth` and `*.pt` files to avoid committing large model files
- Pre-trained weights are cached in `models/pretrained/`
