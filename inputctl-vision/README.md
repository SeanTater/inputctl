# inputctl-vision

LLM-based GUI automation toolkit for Linux - combining vision analysis with autonomous agents.

## Features

- **Screen capture** via xdg-desktop-portal (Wayland native)
- **LLM vision analysis** - query local or remote LLMs about screen content
- **Autonomous agent** - LLM-driven tool-calling agent for GUI automation
- **Web debugger** - real-time visualization of agent state
- **Multiple LLM backends**: Ollama, vLLM, OpenAI, Anthropic
- **Python bindings** and Rust API
- **Template matching** - available but rarely used in practice (prefer LLM vision)

## Requirements

- **Linux** with xdg-desktop-portal (most Wayland desktops)
- **KDE Plasma** for cursor position tracking (optional)
- **uinput access** for input automation (see inputctl README)
- LLM backend running (Ollama recommended for local usage)

**Install Ollama (recommended):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llava  # Download and run vision model
```

## Quick Start

### CLI Usage

```bash
# Configure environment
export VISIONCTL_BACKEND=ollama
export VISIONCTL_URL=http://localhost:11434
export VISIONCTL_MODEL=llava

# Ask about the screen
inputctl-vision "What windows are open?"

# Run autonomous agent
inputctl-vision agent "Open Firefox and navigate to example.com"

# Run with web debugger
inputctl-vision agent --debug "Open Settings"
# Visit http://localhost:3000 to see agent state
```

### Python

```python
from inputctl_vision import VisionCtl

# Configure with LLM backend
ctl = VisionCtl(
    backend="ollama",
    url="http://localhost:11434",
    model="llava"
)

# Ask LLM what's on screen
answer = ctl.ask("Where is the terminal window?")
print(answer)

# Take actions
ctl.move_to(500, 300, smooth=True)  # Normalized 0-1000 coords
ctl.click("left")
ctl.type_text("echo 'hello from inputctl-vision'")
```

### Rust Agent

```rust
use inputctl_vision::{Agent, LlmConfig};

let config = LlmConfig::Ollama {
    url: "http://localhost:11434".to_string(),
    model: "llava".to_string(),
};

let agent = Agent::new(config)?
    .with_max_iterations(Some(20));

let result = agent.run("Open the file manager and navigate to Downloads")?;
println!("Success: {}, Message: {}", result.success, result.message);
```

## Architecture

### Core Components

- `VisionCtl` - Main controller for screenshots, coordinate conversion, and LLM queries
- `Agent` - Autonomous agent loop with tool execution
- `LlmClient` - HTTP client for LLM APIs with tool-calling support
- `StateStore` / `DebugServer` - Agent state tracking and web debugger

### Agent Tool System

The agent has access to these tools (defined in `llm/tools.rs`):

**Movement & Click:**
- `move_to(x, y)` - Move cursor (0-1000 normalized coordinates)
- `click(button)` - Click mouse button
- `double_click(button)` - Double click
- `mouse_down(button)` / `mouse_up(button)` - Hold/release

**Keyboard:**
- `type_text(text)` - Type text
- `key_press(key)` - Press key (enter, escape, tab, etc.)
- `key_down(key)` / `key_up(key)` - Hold/release key

**Vision (optional, can be disabled):**
- `point_at(description)` - Ask pointing model for coordinates
- `ask_screen(question)` - Ask question about current screen
- `find_template(template)` - Template matching (rarely used)

**Control:**
- `task_complete(success, message)` - Signal task completion
- `stuck()` - Signal agent needs help
- `scroll(direction, amount)` - Scroll

**Viewport:**
- `set_viewport(x, y, w, h)` - Focus on region
- `clear_viewport()` - Return to full screen

### Coordinate System

The agent uses normalized coordinates (0-1000):
- (0, 0) = top-left corner
- (1000, 1000) = bottom-right corner
- Automatically converted to screen pixels

Screenshots include a red crosshair (+) marking current cursor position.

## CLI Reference

```bash
# Simple query
inputctl-vision "What's on my screen?"

# Run agent
inputctl-vision agent "Open Firefox"
inputctl-vision agent --max-iterations 50 "Complete this form"
inputctl-vision agent --debug "Open Settings"  # With web debugger

# Screenshot commands
inputctl-vision screenshot output.png
inputctl-vision screenshot --with-grid output.png

# Window management
inputctl-vision window list

# Recording (delegates to inputctl-capture)
inputctl-vision record --output dataset --fps 10

# Configuration
inputctl-vision setup  # Interactive configuration wizard
inputctl-vision install-desktop-file  # For KDE permissions
```

### Environment Variables

- `VISIONCTL_BACKEND` - LLM backend: ollama, vllm, openai [default: ollama]
- `VISIONCTL_URL` - Backend URL [default: http://localhost:11434]
- `VISIONCTL_MODEL` - Model name [default: llava]
- `VISIONCTL_API_KEY` - API key (required for OpenAI)

## LLM Backend Configuration

### Ollama (Local)

```bash
export VISIONCTL_BACKEND=ollama
export VISIONCTL_URL=http://localhost:11434
export VISIONCTL_MODEL=llava
```

### vLLM

```bash
export VISIONCTL_BACKEND=vllm
export VISIONCTL_URL=http://localhost:8000
export VISIONCTL_MODEL=your-vision-model
```

### OpenAI

```bash
export VISIONCTL_BACKEND=openai
export VISIONCTL_URL=https://api.openai.com
export VISIONCTL_MODEL=gpt-4-vision-preview
export VISIONCTL_API_KEY=your-api-key
```

## Web Debugger

Run with `--debug` to enable the web debugger:

```bash
inputctl-vision agent --debug "Your task here"
```

Then visit http://localhost:3000 to see:
- Current screenshot with cursor position
- Message history
- Tool calls and results
- Agent state (running/paused/completed)

You can pause the agent and inject messages for debugging.

## Python API

```python
from inputctl_vision import VisionCtl

# Construction
ctl = VisionCtl(backend, url, model, api_key=None)
ctl = VisionCtl.new_headless()  # Without LLM

# Screenshots
png_bytes = VisionCtl.screenshot()  # Static
png_bytes = ctl.screenshot_with_cursor()  # With cursor marker

# LLM interaction
answer = ctl.ask(question)

# Actions (0-1000 normalized coordinates)
ctl.move_to(500, 300, smooth=True)
ctl.click("left")
ctl.type_text("hello")
ctl.key_press("enter")

# Tool-calling API
tools = ctl.get_tool_definitions()
result = ctl.execute_tool(name, params)
```

## Development

```bash
# Build
cargo build -p inputctl-vision

# Build Python wheel
cd inputctl-vision && maturin develop --uv

# Run CLI
cargo run -p inputctl-vision --bin inputctl-vision -- "What's on screen?"

# Run agent with logging
RUST_LOG=debug cargo run -p inputctl-vision --bin inputctl-vision -- agent "Open Firefox"
```

## Template Matching

Template matching is available via `detection/` module but is rarely used in practice. LLM vision is generally more flexible and robust. The template matching code remains for edge cases where exact pixel matching is needed.

## Persistent Configuration

The CLI supports a config file at `~/.config/visionctl/config.toml`:

```toml
[llm]
backend = "ollama"
base_url = "http://localhost:11434"
model = "llava"
temperature = 0.0

[pointing]  # Optional dedicated pointing model
backend = "ollama"
base_url = "http://localhost:11434"
model = "qwen3-vl"
temperature = 0.0

[cursor]
smooth_fps = 60
```

Run `inputctl-vision setup` for interactive configuration.

## License

MIT
