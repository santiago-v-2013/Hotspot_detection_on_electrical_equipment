#!/bin/bash
# This script runs preprocessing for binary hotspot classification.

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

echo "🔧 Starting hotspot classification preprocessing..."

# Define paths relative to the project root
INPUT_DIR="data/raw"
OUTPUT_DIR="data/processed/hotspot_classification"
TARGET_SIZE="224 224"
COLOR_MODE="hsv"

# Run the Python script with the specified arguments
python3 scripts/preprocess_hotspot_classification.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_size $TARGET_SIZE \
    --color_mode "$COLOR_MODE"

echo "✅ Hotspot classification preprocessing finished. Check 'logs/app.log' for details."
