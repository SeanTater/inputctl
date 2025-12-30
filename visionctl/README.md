# visionctl

LLM-based GUI automation toolkit for KDE Plasma - combining vision analysis with input control.

## Features

- **Screen capture** using KWin DBus interface (fast, native)
- **Grid overlay system** for spatial reasoning (A1, B2, C3 coordinates)
- **LLM vision analysis** - query local or remote LLMs about screen content
- **Input automation** - mouse movement, clicking, and typing via inputctl
- **Dual usage patterns**:
  - **Script-driven**: Your code controls flow, LLM provides perception
  - **LLM-driven**: LLM controls automation via tool-calling API
- **Multiple LLM backends**: Ollama, vLLM, OpenAI, Anthropic
- **Python bindings** and Rust API
- Native KWin integration (no external tools)

## Requirements

**System:**
- **KDE Plasma 6.0+** with KWin compositor
- **uinput access** for input automation (see inputctl README)
- LLM backend running (Ollama recommended for local usage)

**Note:** This library uses KWin's DBus ScreenShot2 interface, which requires KDE Plasma 6.0 or newer. For other Wayland compositors, you'll need a different screenshot solution.

**Install Ollama (recommended for script-driven pattern):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llava  # Download and run vision model
```

**For LLM-driven pattern with Claude:**
```bash
export ANTHROPIC_API_KEY="your-key-here"
pip install anthropic
```

## KDE Screenshot Authorization

KWin requires applications to be authorized before they can capture screenshots. This is done via desktop files.

### Quick Setup (Recommended)

Use the built-in installer:
```bash
# Build visionctl CLI
cargo build --release -p visionctl --bin visionctl

# Install desktop file (grants screenshot permission)
./target/release/visionctl install-desktop-file
```

This automatically:
- Detects the executable path
- Creates the desktop file with proper permissions
- Updates the desktop database
- Works from any installation location

### Manual Setup

Alternatively, create a desktop file manually at `~/.local/share/applications/your-app.desktop`:

```ini
[Desktop Entry]
Name=Your Application Name
Exec=/path/to/your/application
Type=Application
Terminal=true
Categories=Development;
X-KDE-DBUS-Restricted-Interfaces=org.kde.KWin.ScreenShot2
```

The critical line is `X-KDE-DBUS-Restricted-Interfaces=org.kde.KWin.ScreenShot2` which grants screenshot permission.

After creating the file, update the desktop database:
```bash
update-desktop-database ~/.local/share/applications/
```

### Testing Authorization

Run the screenshot test example to verify authorization works:
```bash
cargo run --example screenshot_test --release
# Saves to /tmp/visionctl_test_screenshot.png with grid overlay
```

## Quick Start

### Script-Driven Pattern (Python)

Your script maintains control, using LLM for perception:

```python
from visionctl import VisionCtl

# Configure with LLM backend
ctl = VisionCtl(
    backend="ollama",
    url="http://localhost:11434",
    model="llava"
)

# Take screenshot with grid overlay
screenshot = ctl.screenshot_with_grid()

# Ask LLM what's on screen
answer = ctl.ask("Where is the terminal window? Use grid coordinates.")
print(f"LLM says: {answer}")  # "The terminal is at cell B3"

# Your script decides what to do
ctl.click_at_grid("B3")
ctl.type_text("echo 'hello from visionctl'")
```

### LLM-Driven Pattern (Python)

LLM controls the automation using tools:

```python
import anthropic
from visionctl import VisionCtl

# Create headless controller (no LLM backend needed)
ctl = VisionCtl.new_headless()

# Get tool definitions for Claude
tools = ctl.get_tool_definitions()

# Let Claude control the automation
client = anthropic.Anthropic(api_key="...")
messages = [{"role": "user", "content": "Open a terminal and run 'date'"}]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    tools=tools,
    messages=messages
)

# Execute tools as Claude requests them
for block in response.content:
    if block.type == "tool_use":
        result = ctl.execute_tool(block.name, block.input)
        # Add result back to conversation...
```

### Rust

```rust
use visionctl::{VisionCtl, LlmConfig};

let config = LlmConfig::Ollama {
    url: "http://localhost:11434".to_string(),
    model: "llava".to_string(),
};

let ctl = VisionCtl::new(config)?;

// Take screenshot with grid
let png = ctl.screenshot_with_grid()?;

// Ask LLM for guidance
let answer = ctl.ask("Where should I click to open settings?")?;
println!("{}", answer);

// Automate based on LLM response
ctl.click_at_grid("C5")?;
```

## Architecture

### Two Usage Patterns

Both patterns use the same core primitives - no code duplication:

**Script-Driven:**
- Script maintains control flow
- LLM used for perception/understanding
- Script makes decisions and executes actions
- Good for: Deterministic workflows with vision assistance

**LLM-Driven:**
- LLM controls the automation flow
- Uses tool-calling API (Anthropic/OpenAI compatible)
- LLM decides which actions to take
- Good for: Flexible, autonomous tasks

### Grid Coordinate System

Visionctl overlays a letter-number grid on screenshots (like spreadsheet cells):

```
  A    B    C    D
1 +----+----+----+
  |    |    |    |
2 +----+----+----+
  |    | X  |    |  <- Cursor at B2
3 +----+----+----+
```

- Default cell size: 100px
- Coordinates: A1, B2, C3, etc.
- LLM can easily reference locations
- Convert between grid cells and pixel coordinates

## API Reference

### Python

```python
from visionctl import VisionCtl

# Construction
ctl = VisionCtl(backend, url, model, api_key=None)  # With LLM
ctl = VisionCtl.new_headless()  # Without LLM (for LLM-driven pattern)

# Primitives
png_bytes = VisionCtl.screenshot()  # Static, no grid
png_bytes = ctl.screenshot_with_grid(grid=True)  # With grid overlay

# LLM interaction (script-driven)
answer = ctl.ask(question)  # Query LLM about screen

# Actions
ctl.click_at_grid("B3")  # Click at grid cell
ctl.move_to_grid("C5", smooth=True)  # Move mouse to grid cell
ctl.type_text("hello world")  # Type text
ctl.key_press("a")  # Press key (single char only for now)

# Tool-calling API (LLM-driven)
tools = ctl.get_tool_definitions()  # Get tool schemas
result = ctl.execute_tool(name, params)  # Execute tool
```

### Rust

```rust
use visionctl::{VisionCtl, LlmConfig, GridConfig};

// Construction
let config = LlmConfig::Ollama { url, model };
let ctl = VisionCtl::new(config)?;
let ctl = VisionCtl::new_headless();  // Without LLM

// Primitives
let png = VisionCtl::screenshot()?;  // Static, no grid
let png = ctl.screenshot_with_grid()?;  // With grid overlay
let cursor = ctl.find_cursor()?;  // Get cursor position (TODO)

// LLM interaction
let answer = ctl.ask(question)?;

// Actions
ctl.click_at_grid("B3")?;
ctl.move_to_grid("C5", true)?;  // true = smooth movement
ctl.type_text("hello")?;
ctl.key_press("a")?;

// Tool-calling API
let tools = ctl.get_tool_definitions();
let result = ctl.execute_tool(name, params)?;
```

## CLI Tool

The `visionctl` binary provides a command-line interface for screenshot authorization and LLM queries.

### Installation

```bash
# Build the CLI
cargo build --release -p visionctl --bin visionctl

# Optionally install to system
cargo install --path visionctl
```

### Commands

**Install Desktop File (for KDE screenshot permission):**
```bash
visionctl install-desktop-file
```
Automatically creates the desktop file with proper permissions for KDE screenshot authorization.

**Query LLM about screen:**
```bash
# Set environment variables
export VISIONCTL_BACKEND=ollama
export VISIONCTL_URL=http://localhost:11434
export VISIONCTL_MODEL=llava

# Ask a question
visionctl "What's on my screen?"

# Output is JSON
{
  "question": "What's on my screen?",
  "answer": "I see a terminal window with code..."
}
```

**Help:**
```bash
visionctl --help
```

### Environment Variables

- `VISIONCTL_BACKEND` - LLM backend (ollama, vllm, openai) [default: ollama]
- `VISIONCTL_URL` - Backend URL [default: http://localhost:11434]
- `VISIONCTL_MODEL` - Model name [default: llava]
- `VISIONCTL_API_KEY` - API key (required for openai backend)

## Examples

### Testing Examples
- `examples/screenshot_test.rs` - Simple screenshot capture test (no LLM required)
- Captures screenshot with grid overlay and saves to `/tmp/visionctl_test_screenshot.png`
- Useful for verifying KDE authorization and grid overlay functionality

### Script-Driven Examples
- `examples/script_driven.rs` - Rust example
- `examples/script_driven.py` - Python example
- Shows script maintaining control, querying LLM for perception

### LLM-Driven Examples
- `examples/llm_driven.py` - Python with Anthropic Claude
- Shows LLM controlling automation via tool-calling API

### Legacy Examples
- `demo_vision.py` - Original vision query demo
- `examples/basic.rs` - Original Rust example

## Tool Definitions

For LLM-driven pattern, visionctl provides these tools:

- `screenshot` - Capture screen with optional grid overlay
- `click_at_grid` - Click at grid cell (e.g., "B3")
- `move_to_grid` - Move mouse to grid cell
- `type_text` - Type text using keyboard
- `key_press` - Press a key
- `find_cursor` - Get current cursor position (TODO)

All tools use Anthropic/OpenAI compatible schema format.

## Development

### Build

```bash
# Build Rust library
cargo build --release -p visionctl

# Build Python wheel
maturin build --release

# Install for development
maturin develop --uv

# Run examples
cargo run --example script_driven
python examples/script_driven.py
python examples/llm_driven.py  # Requires ANTHROPIC_API_KEY
```

### Testing

```bash
# Rust tests
cargo test -p visionctl

# Manual testing (requires KDE)
cargo run --example script_driven
```

## LLM Backend Configuration

### Ollama (Local)

```python
ctl = VisionCtl(
    backend="ollama",
    url="http://localhost:11434",
    model="llava"
)
```

### vLLM

```python
ctl = VisionCtl(
    backend="vllm",
    url="http://localhost:8000",
    model="your-vision-model",
    api_key="optional-key"
)
```

### OpenAI-compatible

```python
ctl = VisionCtl(
    backend="openai",
    url="https://api.openai.com",
    model="gpt-4-vision-preview",
    api_key="your-api-key"
)
```

## Limitations

- **KWin-only**: Requires KDE Plasma 6.0+ with KWin compositor
- **Cursor finding**: Not yet implemented (stub exists)
- **Key press**: Currently only supports single characters, not special keys (enter, escape, etc.)
- **Input automation**: Requires uinput access (see inputctl documentation)

## Related Projects

- [inputctl](../inputctl) - Low-level Linux input automation library (used by visionctl)
- [kdotool](https://github.com/jinliu/kdotool) - Inspiration for KWin cursor finding approach

## License

MIT
