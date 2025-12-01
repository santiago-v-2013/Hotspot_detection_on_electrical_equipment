#!/bin/bash
# This script ensures the log directory exists and executes the download process.

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

echo "🚀 Starting the image download script..."

# Define paths relative to the project root
URL_FILE="data/urls.txt"
OUTPUT_DIR="data/raw/"
LOG_DIR="logs/"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Run the Python script with the specified arguments
python3 scripts/download_images.py \
    --url_file "$URL_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "✅ Process finished. Check the '$LOG_DIR' directory for output."