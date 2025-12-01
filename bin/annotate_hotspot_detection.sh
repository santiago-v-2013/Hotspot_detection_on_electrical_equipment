#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

# Run hotspot detection annotation
python scripts/annotate_hotspot_detection.py \
    --data-dir data/raw \
    --output-dir data/processed/hotspot_detection \
    --min-area-ratio 0.005 \
    --max-area-ratio 0.5 \
    --min-intensity 150 \
    --format yolo \
    --debug

echo "Hotspot detection annotation completed!"
