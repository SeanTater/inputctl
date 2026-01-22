# inputctl

A Rust workspace for Linux GUI automation, from low-level input primitives to LLM-driven agents.

## Workspace Structure

```
inputctl/                 Low-level keyboard/mouse control via uinput
inputctl-capture/         Screen capture primitives (Wayland portal + KWin cursor)
inputctl-vision/          LLM-based GUI automation and autonomous agent
inputctl-reflex/          ONNX inference for game AI with trained models
reflex_train/             Python training pipeline for Reflex models
```

### Tier Architecture

```
                    ┌─────────────────────────┐
                    │      reflex_train       │  Python training
                    │   (PyTorch → ONNX)      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │    inputctl-reflex      │  Fast game AI inference
                    │    (ONNX runtime)       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │    inputctl-vision      │  LLM agent + tools
                    │   (GPT-4V, Qwen3, etc)  │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐       ┌───────▼───────┐               │
│   inputctl    │       │inputctl-capture│              │
│ (keyboard,    │       │ (screenshots,  │              │
│  mouse)       │       │  cursor pos)   │              │
└───────────────┘       └────────────────┘              │
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     Linux (uinput,      │
                    │   portal, KWin DBus)    │
                    └─────────────────────────┘
```

## Quick Start

### Task Shortcuts (Just)

Install just once:

```bash
cargo install just
```

```bash
# Record a run
just record

# Label new runs (no overwrite)
just label

# Train (IQL/AWR on by default)
just train

# Train with common knobs
just train-custom EPOCHS=50 BATCH_SIZE=32 LR=1e-4
```


### System Dependencies

Ubuntu/Debian:
```bash
sudo apt-get install -y \
  xdg-desktop-portal
```

Arch:
```bash
sudo pacman -S --needed xdg-desktop-portal
```

### Build

```bash
# Build all crates
cargo build --release

# Build Python packages (for inputctl and inputctl-vision)
uv tool install maturin
cd inputctl && maturin develop --uv && cd ..
cd inputctl-vision && maturin develop --uv && cd ..
```

### Usage

**Low-level input (Python):**
```python
from inputctl import InputCtl
ctl = InputCtl()
ctl.type_text("Hello!")
ctl.move_mouse_smooth(100, 50, 1.0, "ease-in-out")
ctl.click("left")
```

**LLM-driven automation (CLI):**
```bash
export VISIONCTL_BACKEND=ollama
export VISIONCTL_URL=http://localhost:11434
export VISIONCTL_MODEL=llava

# Ask about the screen
inputctl-vision "What windows are open?"

# Run autonomous agent
inputctl-vision agent "Open Firefox and navigate to example.com"
```

**Record training data:**
```bash
inputctl-record --output dataset --fps 10 --preset ultrafast
```

## Components

| Crate | Description |
|-------|-------------|
| [inputctl](inputctl/README.md) | Keyboard/mouse control via Linux uinput |
| [inputctl-capture](inputctl-capture/README.md) | Screen capture via Wayland portal, cursor via KWin |
| [inputctl-vision](inputctl-vision/README.md) | LLM integration, tool-calling agent, web debugger |
| [inputctl-reflex](inputctl-reflex/README.md) | ONNX inference for game AI |
| [reflex_train](reflex_train/README.md) | Python training pipeline (separate venv) |

## Development

```bash
# Run all tests
cargo test --workspace

# Run inputctl integration tests (requires sudo for /dev/uinput)
sudo cargo test -p inputctl -- --ignored

# Check all crates build
cargo build --workspace --release
```

See individual README files for crate-specific documentation.

## License

MIT
