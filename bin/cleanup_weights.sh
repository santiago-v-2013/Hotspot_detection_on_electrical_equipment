#!/bin/bash
#
# Move any YOLO weights from project root to models/pretrained/
# This script cleans up weights that YOLO downloads to the current directory
#

set -e

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Create pretrained directory if it doesn't exist
mkdir -p models/pretrained

# Counter for moved files
MOVED=0

echo "=========================================="
echo "Cleaning up YOLO weights in project root"
echo "=========================================="
echo ""

# Find and move any .pt files in root
for file in *.pt; do
    if [ -f "$file" ]; then
        echo "Moving $file to models/pretrained/"
        mv "$file" models/pretrained/
        ((MOVED++))
    fi
done

if [ $MOVED -eq 0 ]; then
    echo "✅ No .pt files found in project root"
else
    echo ""
    echo "=========================================="
    echo "✅ Moved $MOVED file(s) to models/pretrained/"
    echo "=========================================="
fi

echo ""
echo "Current weights in models/pretrained/:"
ls -lh models/pretrained/*.pt 2>/dev/null || echo "  (none)"
