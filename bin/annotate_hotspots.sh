#!/bin/bash
#
# Automatic hotspot annotation for thermal images
# Usage: ./bin/annotate_hotspots.sh [OPTIONS]
#

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Automatic Hotspot Annotation"
echo "=========================================="
echo "Input: data/raw/"
echo "Output: data/processed/hotspot_classification/"
echo "=========================================="
echo ""

# Run the annotation script
cd "$PROJECT_ROOT"
python scripts/annotate_hotspots.py \
    --input-dir data/raw \
    --output-dir data/processed/hotspot_classification \
    --min-area-ratio 0.01 \
    --max-area-ratio 0.95 \
    --intensity-threshold 150 \
    --saturation-threshold 50 \
    --debug-samples 10

echo ""
echo "=========================================="
echo "Annotation completed!"
echo "=========================================="
echo "Classified images saved in: data/processed/hotspot_classification/"
echo "Annotations file: data/processed/hotspot_classification/annotations.json"
echo "Debug samples (10 random): data/processed/hotspot_classification/*/debug/"
echo "=========================================="
