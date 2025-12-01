#!/bin/bash
#
# Run Flask web application in production mode with Gunicorn
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

# Install gunicorn if not present
pip install gunicorn --quiet

echo "=========================================="
echo "Starting Flask App (Production Mode)"
echo "=========================================="
echo ""
echo "Workers: 4"
echo "Threads per worker: 2"
echo "Worker class: gthread (supports concurrent requests)"
echo "Bind: 0.0.0.0:5000"
echo ""
echo "📍 Access: http://localhost:5000"
echo "📱 Network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

# Run with Gunicorn - gthread worker class for concurrent request handling
cd app
gunicorn \
    -w 4 \
    --threads 2 \
    --worker-class gthread \
    -b 0.0.0.0:5000 \
    --timeout 120 \
    --access-logfile - \
    --log-level info \
    app:app
