#!/bin/bash
# This script runs preprocessing for hotspot object detection with bounding boxes.

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

echo "🎯 Starting hotspot object detection preprocessing..."

# Define paths relative to the project root
INPUT_DIR="data/raw"
OUTPUT_DIR="data/processed/hotspot_detection"
TARGET_SIZE="416 416"

# Run the Python script with the specified arguments
python3 scripts/preprocess_hotspot_detection.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_size $TARGET_SIZE

echo "✅ Hotspot detection preprocessing finished. Check 'logs/app.log' for details."
