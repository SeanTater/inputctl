#!/usr/bin/env python3
"""
Mouse demo - moves the mouse in a square pattern and clicks.

Setup:
    uv tool install maturin
    uv sync
    maturin develop --uv

Usage:
    sudo uv run python examples/mouse_demo.py

Requires /dev/uinput access (typically needs root).
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Mouse movement and click demo")
    parser.add_argument(
        "--distance", type=int, default=100, help="Distance to move in pixels (default: 100)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Delay between movements in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--clicks", type=int, default=1, help="Number of clicks at each corner (default: 1)"
    )
    args = parser.parse_args()

    try:
        import inputctl
    except ImportError:
        print("Error: inputctl module not found.")
        print("Build it first with: maturin develop --uv")
        return 1

    print("Creating virtual input device...")
    print("(This takes ~1 second for kernel initialization)")

    try:
        yd = inputctl.InputCtl()
    except OSError as e:
        print(f"Error: {e}")
        print("\nMake sure you have access to /dev/uinput.")
        print("Try running with sudo, or add yourself to the 'input' group.")
        return 1

    print("Device ready!")
    print(f"Moving mouse in a {args.distance}x{args.distance} square pattern...")
    print("Press Ctrl+C to stop\n")

    try:
        # Move right
        print("Moving right...")
        yd.move_mouse(args.distance, 0)
        for _ in range(args.clicks):
            yd.click("left")
        time.sleep(args.delay)

        # Move down
        print("Moving down...")
        yd.move_mouse(0, args.distance)
        for _ in range(args.clicks):
            yd.click("left")
        time.sleep(args.delay)

        # Move left
        print("Moving left...")
        yd.move_mouse(-args.distance, 0)
        for _ in range(args.clicks):
            yd.click("left")
        time.sleep(args.delay)

        # Move up (back to start)
        print("Moving up...")
        yd.move_mouse(0, -args.distance)
        for _ in range(args.clicks):
            yd.click("left")

        print("\nDone!")

    except KeyboardInterrupt:
        print("\nStopped by user")

    return 0


if __name__ == "__main__":
    exit(main())
