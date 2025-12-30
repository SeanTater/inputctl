#!/usr/bin/env python3
"""
Demo of smooth mouse movement with low-frequency noise.

This demo shows the difference between instant movement and smooth movement
with natural-looking low-frequency noise that mimics human hand tremor.

Run with: sudo python demo_smooth_movement.py
"""

import time
from inputctl import InputCtl

def main():
    print("Initializing input device (this takes ~1 second)...")
    ctl = InputCtl()
    print("Ready!\n")
    print("=" * 70)
    print("SMOOTH MOUSE MOVEMENT DEMO - Low-Frequency Noise")
    print("=" * 70)

    # Give user time to position their mouse/focus
    print("\nStarting in 3 seconds - watch your mouse cursor closely!")
    print("Notice the smooth, natural wavering path instead of a straight line.")
    time.sleep(3)

    print("\n1. Instant movement (traditional):")
    print("   Moving 200px right instantly (straight line, no realism)...")
    ctl.move_mouse(200, 0)
    time.sleep(1.5)

    print("\n2. Perfectly smooth (no noise):")
    print("   Moving 200px down over 1.5 seconds with noise=0.0...")
    print("   → Perfectly straight line, no variation")
    ctl.move_mouse_smooth(0, 200, 1.5, "linear", noise=0.0)
    time.sleep(1.5)

    print("\n3. Subtle noise (default ±2px):")
    print("   Moving 200px left over 2 seconds...")
    print("   → Barely noticeable natural wavering!")
    ctl.move_mouse_smooth(-200, 0, 0.3, "ease-in-out", noise=2.0)
    time.sleep(1.5)

    print("\n4. Moderate noise (±5px):")
    print("   Moving diagonally (-200, -200) over 2.5 seconds...")
    print("   → More obvious smooth curved path!")
    ctl.move_mouse_smooth(-200, -200, 0.5, "ease-in-out", noise=5.0)
    time.sleep(1)

    ctl.click("right")

    print("\n" + "=" * 70)
    print("✓ Demo complete! The mouse should be back near starting position.")
    print("=" * 70)
    print("\nKey features demonstrated:")
    print("  • noise=0.0: Perfectly smooth, no variation")
    print("  • noise=2.0: Subtle, barely-noticeable wavering (default)")
    print("  • noise=5.0: More obvious natural curves")
    print("  • Low-frequency variation (~200ms control points, not jittery)")
    print("  • Always hits exact target position")
    print("\nUse noise=2.0 for realistic human-like movements!")

if __name__ == "__main__":
    main()
