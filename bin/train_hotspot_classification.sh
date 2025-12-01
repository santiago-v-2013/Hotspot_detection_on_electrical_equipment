#!/bin/bash
#
# Train hotspot classification models (both EfficientNet-B0 and DenseNet-121)
# Usage: ./bin/train_hotspot_classification.sh [EPOCHS] [BATCH_SIZE]
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
EPOCHS="${1:-50}"
BATCH_SIZE="${2:-32}"

echo "=========================================="
echo "Hotspot Classification Training - BOTH MODELS"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Train both models
python scripts/train_hotspot_classification.py \
    --model both \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --pretrained \
    --early-stopping 10 \
    --scheduler step \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --dropout 0.3 \
    --val-split 0.2 \
    --test-split 0.1 \
    --num-workers 4 \
    --seed 42

echo ""
echo "=========================================="
echo "Training completed for both models!"
echo "=========================================="
echo "EfficientNet-B0 models saved in: models/hotspot_classification/efficientnet_b0/"
echo "DenseNet-121 models saved in: models/hotspot_classification/densenet121/"
echo "Training logs available in: logs/"
echo "=========================================="
