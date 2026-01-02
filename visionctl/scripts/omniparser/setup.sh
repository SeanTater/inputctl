#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== OmniParser V2 Setup ==="

# Clone OmniParser if not present
if [ ! -d "OmniParser" ]; then
    echo "Cloning OmniParser..."
    git clone --depth 1 https://github.com/microsoft/OmniParser.git
fi

cd OmniParser

# Create isolated venv
if [ ! -d ".venv" ]; then
    echo "Creating Python 3.12 venv..."
    uv venv .venv --python 3.12
fi

echo "Installing dependencies..."
uv pip install --python .venv/bin/python \
    torch torchvision \
    transformers \
    pillow \
    ultralytics \
    easyocr \
    gradio \
    einops \
    timm \
    accelerate

# Download model weights using Python
if [ ! -d "weights/icon_detect" ]; then
    echo "Downloading OmniParser V2 weights..."
    mkdir -p weights

    .venv/bin/python -c "
from huggingface_hub import hf_hub_download, snapshot_download
import os

# Download icon detection model files
for f in ['train_args.yaml', 'model.pt', 'model.yaml']:
    print(f'Downloading icon_detect/{f}...')
    hf_hub_download('microsoft/OmniParser-v2.0', f'icon_detect/{f}', local_dir='weights')

# Download icon caption model files
for f in ['config.json', 'generation_config.json', 'model.safetensors']:
    print(f'Downloading icon_caption/{f}...')
    hf_hub_download('microsoft/OmniParser-v2.0', f'icon_caption/{f}', local_dir='weights')

# Rename icon_caption to icon_caption_florence
import shutil
if os.path.exists('weights/icon_caption') and not os.path.exists('weights/icon_caption_florence'):
    shutil.move('weights/icon_caption', 'weights/icon_caption_florence')
print('Downloads complete!')
"
fi

echo ""
echo "=== Setup Complete ==="
echo "Test with:"
echo "  $SCRIPT_DIR/omniparser.py /tmp/test.png"
