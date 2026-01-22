#!/usr/bin/env python3
"""
Script-Driven Pattern Example (Python)

In this pattern, the script controls the flow and uses the LLM
occasionally for perception (understanding what's on screen).

Usage:
    # Build and install visionctl
    cd visionctl
    maturin develop --uv

    # Run example
    python examples/script_driven.py

Requirements:
    - KDE Plasma 6.0+ with KWin
    - Ollama running with llava model
    - /dev/uinput access
"""

import sys

try:
    from visionctl import VisionCtl
except ImportError:
    print("ERROR: visionctl not installed")
    print("Run: cd visionctl && maturin develop --uv")
    sys.exit(1)


def main():
    print("=== Script-Driven GUI Automation Example (Python) ===\n")

    # Configure LLM backend (Ollama with llava model)
    ctl = VisionCtl(
        backend="ollama",
        url="http://localhost:11434",
        model="llava"
    )

    print("Step 1: Taking screenshot...")
    screenshot = ctl.screenshot()
    print(f"✓ Screenshot captured: {len(screenshot):,} bytes\n")

    print("Step 2: Asking LLM to identify what's on screen...")
    answer = ctl.ask("What application windows are visible on the screen? List them briefly.")
    print(f"LLM Response: {answer}\n")

    print("Step 3: Script decides next action based on LLM's response...")
    print("(In a real scenario, you'd parse the LLM response and take action)\n")

    # Example: Ask LLM where a specific element is
    print("Step 4: Asking LLM where to click...")
    location = ctl.ask(
        "Where on the screen is a terminal or console window? "
        "Answer with approximate x,y coordinates from 0 to 1000, or say 'none' if you don't see one."
    )
    print(f"LLM says the terminal is at: {location}\n")

    # Script decides what to do based on LLM's answer
    if "none" not in location.lower():
        print("Step 5: Script executing action based on LLM guidance...")
        # Extract cell identifier (simple parsing)
        coords = location.strip().split()[0].strip("(),")

        print(f"Would click at: {coords} (skipping actual click for safety)")
        # Uncomment to actually click:
        # x_str, y_str = coords.split(",")
        # ctl.click_at(float(x_str), float(y_str))
        # print(f"✓ Clicked at {coords}")
    else:
        print("Step 5: No terminal found, skipping click")

    print("\n=== Example Complete ===")
    print("\nKey Points:")
    print("- Script maintains control flow")
    print("- LLM used for perception/understanding")
    print("- Script makes decisions based on LLM responses")
    print("- LLM returns approximate 0..1000 coordinates for locations")

    print("\nTry uncommenting the click line to actually interact with the screen!")


if __name__ == "__main__":
    main()
