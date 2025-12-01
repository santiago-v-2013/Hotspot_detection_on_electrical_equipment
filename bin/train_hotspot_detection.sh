#!/bin/bash
#
# Train hotspot detection models (YOLOv8 and YOLOv11)
#
# Usage:
#   ./bin/train_hotspot_detection.sh [epochs] [batch_size_v11] [batch_size_v8]
#
# Arguments:
#   epochs           - Number of epochs (default: 50)
#   batch_size_v11   - Batch size for YOLOv11 (default: 32)
#   batch_size_v8    - Batch size for YOLOv8 (default: 32)
#
# Examples:
#   ./bin/train_hotspot_detection.sh 50 32 32
#   ./bin/train_hotspot_detection.sh 100 16 24

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

# Default values
EPOCHS=${1:-50}
BATCH_SIZE_V11=${2:-32}
BATCH_SIZE_V8=${3:-32}

echo "=========================================="
echo "Hotspot Detection Training - Both Models"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "YOLOv11 batch size: $BATCH_SIZE_V11"
echo "YOLOv8 batch size: $BATCH_SIZE_V8"
echo "=========================================="
echo ""

# Train YOLOv11
echo "=========================================="
echo "STEP 1/2: Training YOLOv11s"
echo "=========================================="
python scripts/train_hotspot_detection.py \
    --model yolov11 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE_V11" \
    --pretrained \
    --early-stopping 10

echo ""
echo "YOLOv11s training completed!"
echo ""
sleep 2

# Train YOLOv8
echo "=========================================="
echo "STEP 2/2: Training YOLOv8s"
echo "=========================================="
python scripts/train_hotspot_detection.py \
    --model yolov8 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE_V8" \
    --pretrained \
    --early-stopping 10

echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo "YOLOv11s results: models/hotspot_detection/yolov11s/"
echo "YOLOv8s results: models/hotspot_detection/yolov8s/"
echo "=========================================="
