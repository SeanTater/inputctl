#!/usr/bin/env python3
"""
Demo script for visionctl - screen capture and LLM vision analysis

Requirements:
- KDE Plasma 6.0+ with KWin compositor
- Ollama running with a vision model (e.g., llava)
  Install: curl -fsSL https://ollama.com/install.sh | sh
  Run: ollama run llava

Configure with environment variables:
  VISIONCTL_BACKEND=ollama (default)
  VISIONCTL_URL=http://localhost:11434 (default)
  VISIONCTL_MODEL=llava (default)
"""

import os
import sys

try:
    from visionctl import VisionCtl
except ImportError:
    print("ERROR: visionctl not installed")
    print("Run: cd visionctl && maturin develop --uv")
    sys.exit(1)

def main():
    print("=" * 70)
    print("VISIONCTL DEMO - Screen Capture + LLM Vision Analysis")
    print("=" * 70)

    # Get config from environment or use defaults
    backend = os.getenv("VISIONCTL_BACKEND", "ollama")
    url = os.getenv("VISIONCTL_URL", "http://localhost:11434")
    model = os.getenv("VISIONCTL_MODEL", "llava")
    api_key = os.getenv("VISIONCTL_API_KEY")

    print(f"\nConfiguration:")
    print(f"  Backend: {backend}")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    if api_key:
        print(f"  API Key: {'*' * 20}")
    print()

    # Test 1: Take a screenshot
    print("[1/3] Testing screenshot capture...")
    try:
        screenshot_bytes = VisionCtl.screenshot()
        print(f"✓ Screenshot captured: {len(screenshot_bytes):,} bytes (PNG)")
    except Exception as e:
        error_str = str(e)
        print(f"✗ Screenshot failed: {e}")

        if "NoAuthorized" in error_str or "not authorized" in error_str.lower():
            print("\nKWin authorization error. Try one of these:")
            print("  1. Allow the screenshot in the KDE permission dialog (if one appeared)")
            print("  2. Add this script to KWin's trusted applications")
            print("  3. Check KDE System Settings > Privacy > Screen Capture permissions")
        elif "DBus connection failed" in error_str or "KDE running" in error_str:
            print("\nThis requires KDE Plasma 6.0+ with KWin compositor.")
            print("Current environment:", os.getenv("XDG_CURRENT_DESKTOP", "unknown"))
        else:
            print("\nMake sure you're running KDE Plasma 6.0+ with KWin compositor.")
        sys.exit(1)

    # Test 2: Create VisionCtl instance
    print("\n[2/3] Connecting to LLM backend...")
    try:
        ctl = VisionCtl(
            backend=backend,
            url=url,
            model=model,
            api_key=api_key
        )
        print(f"✓ Connected to {backend}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        if backend == "ollama":
            print("\nMake sure Ollama is running:")
            print("  ollama run llava")
        sys.exit(1)

    # Test 3: Ask questions about the screen
    print("\n[3/3] Querying LLM about current screen...")
    print("-" * 70)

    questions = [
        "What's on my screen? Give a brief description.",
        "What colors are prominent on the screen?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("Thinking...", end="", flush=True)

        try:
            answer = ctl.ask(question)
            print(f"\r{'Answer:':<12} {answer}")
        except Exception as e:
            print(f"\r✗ Error: {e}")
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                print("\nMake sure your LLM backend is running:")
                if backend == "ollama":
                    print("  ollama run llava")
                elif backend == "vllm":
                    print(f"  Check that vLLM is running at {url}")
            sys.exit(1)

    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nYou can now use visionctl in your own scripts:")
    print("  from visionctl import VisionCtl")
    print(f"  ctl = VisionCtl('{backend}', '{url}', '{model}')")
    print("  answer = ctl.ask('What do you see?')")
    print()

if __name__ == "__main__":
    main()
