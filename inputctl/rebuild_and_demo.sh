#!/bin/bash
set -e  # Exit on error

echo "======================================================================"
echo "COMPLETE REBUILD AND DEMO"
echo "======================================================================"

# Step 1: Clean up old venv
echo ""
echo "[1/5] Removing old virtual environment..."
sudo rm -rf .venv

# Step 2: Create fresh venv
echo ""
echo "[2/5] Creating fresh virtual environment..."
uv venv

# Step 3: Build wheel
echo ""
echo "[3/5] Building Python wheel (this takes ~10 seconds)..."
PATH="$HOME/.cargo/bin:$PATH" uv tool run maturin build --release

# Step 4: Install wheel
echo ""
echo "[4/5] Installing wheel into venv..."
WHEEL=$(ls -t target/wheels/inputctl-*.whl | head -1)
echo "Installing: $WHEEL"
uv pip install --force-reinstall "$WHEEL"

# Step 5: Run demo
echo ""
echo "[5/5] Running demo with EXTREME noise settings..."
echo "      (±20 pixel deviation, 240ms control points)"
echo ""
echo "======================================================================"
echo "If you see LARGE smooth curves, the new code is working!"
echo "If you see small jittery movements, something is still wrong."
echo "======================================================================"
echo ""
sleep 2

# Run with sudo for uinput access
echo "Running demo (needs sudo for /dev/uinput access)..."
sudo .venv/bin/python demo_hold_state.py
