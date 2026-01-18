#!/bin/bash
set -e  # Exit on error

echo "======================================================================"
echo "VISIONCTL - COMPLETE REBUILD AND DEMO"
echo "======================================================================"

# Check if running on KDE Plasma
if [ -z "$KDE_SESSION_VERSION" ]; then
    echo ""
    echo "WARNING: KDE_SESSION_VERSION not set"
    echo "This library requires KDE Plasma 6.0+ with KWin compositor."
    echo ""
    echo "Continuing anyway - demo will show error if KWin is not available..."
    echo ""
fi

# Check if we're in the workspace root or visionctl directory
if [ -d "visionctl" ]; then
    # We're in workspace root
    WORKSPACE_ROOT="."
    VISIONCTL_DIR="./visionctl"
elif [ -f "Cargo.toml" ] && grep -q "name = \"visionctl\"" Cargo.toml 2>/dev/null; then
    # We're in visionctl directory
    WORKSPACE_ROOT=".."
    VISIONCTL_DIR="."
else
    echo "ERROR: Run this script from workspace root or visionctl directory"
    exit 1
fi

cd "$WORKSPACE_ROOT"

# Step 1: Clean up old venv if it exists
if [ -d ".venv" ]; then
    echo ""
    echo "[1/5] Removing old virtual environment..."
    sudo rm -rf .venv
else
    echo ""
    echo "[1/5] No existing virtual environment found, skipping cleanup..."
fi

# Step 2: Create fresh venv
echo ""
echo "[2/5] Creating fresh virtual environment..."
uv venv

# Step 3: Build visionctl wheel
echo ""
echo "[3/5] Building visionctl Python wheel (this may take ~30 seconds)..."
cd visionctl
PATH="$HOME/.cargo/bin:$PATH" uv tool run maturin build --release
cd ..

# Step 4: Install wheel
echo ""
echo "[4/5] Installing wheel into venv..."
WHEEL=$(ls -t target/wheels/visionctl-*.whl | head -1)
echo "Installing: $WHEEL"
uv pip install --force-reinstall "$WHEEL"

# Step 5: Check LLM backend
echo ""
echo "[5/5] Checking LLM backend availability..."
BACKEND=${VISIONCTL_BACKEND:-ollama}
URL=${VISIONCTL_URL:-http://localhost:11434}

echo "  Backend: $BACKEND"
echo "  URL: $URL"

if [ "$BACKEND" = "ollama" ]; then
    # Try to check if Ollama is running
    if command -v curl &> /dev/null; then
        if curl -s "$URL/api/tags" > /dev/null 2>&1; then
            echo "  Status: ✓ Ollama is running"
        else
            echo "  Status: ✗ Ollama doesn't seem to be running"
            echo ""
            echo "To start Ollama with a vision model:"
            echo "  ollama run llava"
            echo ""
            echo "Continuing anyway - demo will show error if needed..."
        fi
    fi
fi

# Run demo
echo ""
echo "======================================================================"
echo "RUNNING DEMO"
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Capture your current screen using KWin DBus"
echo "  2. Send it to your LLM and ask questions about it"
echo ""
sleep 2

cd visionctl
../.venv/bin/python demo_vision.py
