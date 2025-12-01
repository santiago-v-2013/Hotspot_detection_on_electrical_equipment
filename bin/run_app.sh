#!/bin/bash
#
# Run Flask web application for thermal imaging hotspot detection
#

set -e

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision_env

echo "=========================================="
echo "Starting Flask Web Application"
echo "=========================================="
echo ""
echo "📍 Access the app at: http://localhost:5000"
echo "📱 Or from network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Run Flask app
python app/app.py
