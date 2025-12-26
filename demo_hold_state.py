#!/usr/bin/env python3
"""
Demo of key/button hold state tracking.

This demo shows the new hold state management features including:
- Holding keys down (modifiers like SHIFT, CTRL, ALT)
- Holding mouse buttons down
- Querying what's currently held
- Automatic cleanup on drop
- The motivating use case: SHIFT + click + drag for rectangular selection

Run with: sudo python demo_hold_state.py
"""

import time
from inputctl import InputCtl

def main():
    print("Initializing input device (this takes ~1 second)...")
    ctl = InputCtl()
    print("Ready!\n")
    print("=" * 70)
    print("HOLD STATE TRACKING DEMO")
    print("=" * 70)

    # Give user time to position their mouse/focus
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    print("\n1. Testing key hold state tracking:")
    print("   Pressing SHIFT down...")
    ctl.key_down("shift")
    print(f"   Is SHIFT held? {ctl.is_key_held('shift')}")
    print(f"   Held keys: {ctl.get_held_keys()}")
    time.sleep(1)

    print("   Releasing SHIFT...")
    ctl.key_up("shift")
    print(f"   Is SHIFT held? {ctl.is_key_held('shift')}")
    print(f"   Held keys: {ctl.get_held_keys()}")
    time.sleep(1)

    print("\n2. Testing mouse button hold state:")
    print("   Pressing LEFT mouse button down...")
    ctl.mouse_down("left")
    print(f"   Is LEFT held? {ctl.is_mouse_button_held('left')}")
    print(f"   Held buttons: {ctl.get_held_buttons()}")
    time.sleep(1)

    print("   Releasing LEFT button...")
    ctl.mouse_up("left")
    print(f"   Is LEFT held? {ctl.is_mouse_button_held('left')}")
    print(f"   Held buttons: {ctl.get_held_buttons()}")
    time.sleep(1)

    print("\n3. Multiple keys/buttons held simultaneously:")
    print("   Holding CTRL, SHIFT, and LEFT button...")
    ctl.key_down("ctrl")
    ctl.key_down("shift")
    ctl.mouse_down("left")
    print(f"   Held keys: {ctl.get_held_keys()}")
    print(f"   Held buttons: {ctl.get_held_buttons()}")
    time.sleep(1)

    print("   Releasing all with release_all()...")
    ctl.release_all()
    print(f"   Held keys: {ctl.get_held_keys()}")
    print(f"   Held buttons: {ctl.get_held_buttons()}")
    time.sleep(1)

    print("\n4. SHIFT + click + drag workflow (rectangular selection):")
    print("   This demonstrates the motivating use case!")
    print("   Holding SHIFT...")
    ctl.key_down("shift")

    print("   Pressing LEFT mouse button...")
    ctl.mouse_down("left")

    print("   Dragging mouse 200px right and 150px down...")
    print("   (In a file manager, this would select a rectangular region)")
    ctl.move_mouse_smooth(200, 150, 1.0, "ease-in-out", noise=2.0)

    print("   Releasing mouse button...")
    ctl.mouse_up("left")

    print("   Releasing SHIFT...")
    ctl.key_up("shift")

    print(f"   Final state - Held keys: {ctl.get_held_keys()}")
    print(f"   Final state - Held buttons: {ctl.get_held_buttons()}")
    time.sleep(1)

    print("\n5. Testing idempotent operations:")
    print("   Pressing 'A' down twice (should not error)...")
    ctl.key_down("a")
    ctl.key_down("a")
    print(f"   Held keys: {ctl.get_held_keys()} (should only show once)")

    print("   Releasing 'A' twice (should not error)...")
    ctl.key_up("a")
    ctl.key_up("a")
    print(f"   Held keys: {ctl.get_held_keys()} (should be empty)")
    time.sleep(1)

    print("\n6. Testing automatic cleanup on drop:")
    print("   Creating new InputCtl, holding keys, then deleting it...")
    temp_ctl = InputCtl()
    temp_ctl.key_down("alt")
    temp_ctl.mouse_down("right")
    print(f"   Before drop - Held keys: {temp_ctl.get_held_keys()}")
    print(f"   Before drop - Held buttons: {temp_ctl.get_held_buttons()}")
    del temp_ctl
    print("   ✓ InputCtl dropped - all keys/buttons automatically released!")
    time.sleep(1)

    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)
    print("\nKey features demonstrated:")
    print("  • key_down() / key_up() for holding keys")
    print("  • mouse_down() / mouse_up() for holding mouse buttons")
    print("  • is_key_held() / is_mouse_button_held() for checking state")
    print("  • get_held_keys() / get_held_buttons() for listing held inputs")
    print("  • release_all() for manual cleanup")
    print("  • Automatic cleanup when InputCtl is dropped")
    print("  • Idempotent operations (duplicate down/up calls are safe)")
    print("  • SHIFT + drag workflow for rectangular selection")
    print("\nThis enables complex input sequences like modifier + click + drag!")

if __name__ == "__main__":
    main()
