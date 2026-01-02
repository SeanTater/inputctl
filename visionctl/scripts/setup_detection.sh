#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== VisionCtl Detection Setup ==="

# Install base dependencies (opencv, numpy, pillow)
echo "Installing template matching dependencies..."
uv sync

echo ""
echo "=== Template Matching Ready ==="
echo ""
echo "Test template matching:"
echo "  visionctl screenshot /tmp/test.png"
echo "  $SCRIPT_DIR/.venv/bin/python $SCRIPT_DIR/template_match.py /tmp/test.png /path/to/reference.png"
echo ""
echo "Optional: Install ML models"
echo "  uv sync --extra ml        # YOLOE (~800MB)"
echo "  uv sync --extra grounding # Grounding DINO (~800MB)"
