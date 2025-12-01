#!/bin/bash
# This script runs preprocessing for equipment type classification.

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

echo "🔧 Starting equipment type classification preprocessing..."

# Define paths relative to the project root
INPUT_DIR="data/raw"
OUTPUT_DIR="data/processed/equipment_classification"
TARGET_SIZE="224 224"

# Run the Python script with the specified arguments
python3 scripts/preprocess_equipment_classification.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_size $TARGET_SIZE \
    --normalize

echo "✅ Equipment classification preprocessing finished. Check 'logs/app.log' for details."
