"""
Tests for the inputctl Python bindings.

Unit tests run without uinput access.
Integration tests require uinput access and are marked with @pytest.mark.integration.

Run unit tests: pytest python_tests/ -v -m "not integration"
Run all tests (needs sudo): sudo $(which pytest) python_tests/ -v
"""

import pytest


class TestImport:
    """Test that the module can be imported."""

    def test_import_module(self):
        import inputctl
        assert hasattr(inputctl, "InputCtl")

    def test_inputctl_class_exists(self):
        from inputctl import InputCtl
        assert InputCtl is not None


@pytest.mark.integration
class TestDeviceCreation:
    """Integration tests requiring /dev/uinput access."""

    def test_create_device(self):
        from inputctl import InputCtl
        yd = InputCtl()
        assert yd is not None

    def test_type_text(self):
        from inputctl import InputCtl
        yd = InputCtl()
        # Should not raise
        yd.type_text("hello")

    def test_type_text_with_delays(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.type_text("hi", key_delay_ms=10, key_hold_ms=5)

    def test_click_left(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.click("left")

    def test_click_right(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.click("right")

    def test_click_middle(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.click("middle")

    def test_click_invalid_button(self):
        from inputctl import InputCtl
        yd = InputCtl()
        with pytest.raises(ValueError):
            yd.click("invalid")

    def test_move_mouse(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.move_mouse(10, 20)

    def test_move_mouse_negative(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.move_mouse(-10, -20)

    def test_scroll_up(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.scroll(3)

    def test_scroll_down(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.scroll(-3)

    def test_key_press(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.key_press("enter")

    def test_key_down_up(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.key_down("shift")
        yd.key_up("shift")

    def test_key_press_single_char(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.key_press("a")
        yd.key_press("1")

    def test_key_press_function_key(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.key_press("f1")
        yd.key_press("f12")

    def test_key_press_invalid(self):
        from inputctl import InputCtl
        yd = InputCtl()
        with pytest.raises(ValueError):
            yd.key_press("invalidkey")

    def test_move_mouse_smooth_linear(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.move_mouse_smooth(100, 50, 0.5, "linear")

    def test_move_mouse_smooth_ease_in_out(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.move_mouse_smooth(100, 50, 1.0, "ease-in-out")

    def test_move_mouse_smooth_default_curve(self):
        from inputctl import InputCtl
        yd = InputCtl()
        # Default should be linear
        yd.move_mouse_smooth(50, 25, 0.3)

    def test_move_mouse_smooth_negative(self):
        from inputctl import InputCtl
        yd = InputCtl()
        yd.move_mouse_smooth(-100, -50, 0.5, "linear")

    def test_move_mouse_smooth_invalid_curve(self):
        from inputctl import InputCtl
        yd = InputCtl()
        with pytest.raises(ValueError):
            yd.move_mouse_smooth(100, 50, 0.5, "invalid")

    def test_move_mouse_smooth_alternate_curve_name(self):
        from inputctl import InputCtl
        yd = InputCtl()
        # Test alternative naming formats
        yd.move_mouse_smooth(50, 25, 0.3, "ease_in_out")
        yd.move_mouse_smooth(50, 25, 0.3, "easeinout")

    def test_hold_state_tracking(self):
        from inputctl import InputCtl
        ctl = InputCtl()
        assert not ctl.is_key_held("a"), "key should not be held initially"

        ctl.key_down("a")
        assert ctl.is_key_held("a"), "key should be held after key_down"

        ctl.key_up("a")
        assert not ctl.is_key_held("a"), "key should not be held after key_up"

    def test_mouse_button_hold_state(self):
        from inputctl import InputCtl
        ctl = InputCtl()
        assert not ctl.is_mouse_button_held("left"), "button should not be held initially"

        ctl.mouse_down("left")
        assert ctl.is_mouse_button_held("left"), "button should be held after mouse_down"

        ctl.mouse_up("left")
        assert not ctl.is_mouse_button_held("left"), "button should not be held after mouse_up"

    def test_shift_click_drag_workflow(self):
        """Test the motivating use case: SHIFT + click + drag"""
        from inputctl import InputCtl
        ctl = InputCtl()

        ctl.key_down("shift")
        ctl.mouse_down("left")

        assert ctl.is_key_held("shift"), "shift should be held"
        assert ctl.is_mouse_button_held("left"), "left button should be held"

        ctl.move_mouse(100, 100)

        ctl.mouse_up("left")
        ctl.key_up("shift")

        assert not ctl.is_key_held("shift"), "shift should not be held after release"
        assert not ctl.is_mouse_button_held("left"), "left button should not be held after release"

    def test_release_all(self):
        from inputctl import InputCtl
        ctl = InputCtl()
        ctl.key_down("ctrl")
        ctl.key_down("shift")
        ctl.mouse_down("left")

        held_keys = ctl.get_held_keys()
        assert len(held_keys) == 2, f"should have 2 held keys, got {len(held_keys)}"

        held_buttons = ctl.get_held_buttons()
        assert len(held_buttons) == 1, f"should have 1 held button, got {len(held_buttons)}"

        ctl.release_all()

        assert len(ctl.get_held_keys()) == 0, "should have no held keys after release_all"
        assert len(ctl.get_held_buttons()) == 0, "should have no held buttons after release_all"

    def test_get_held_keys(self):
        from inputctl import InputCtl
        ctl = InputCtl()

        assert ctl.get_held_keys() == [], "should start with no held keys"

        ctl.key_down("a")
        ctl.key_down("b")
        ctl.key_down("shift")

        held = ctl.get_held_keys()
        assert len(held) == 3, f"should have 3 held keys, got {len(held)}"
        # Keys are returned in evdev format like "KEY_A"
        assert any("A" in k for k in held), "should contain KEY_A"
        assert any("B" in k for k in held), "should contain KEY_B"
        assert any("SHIFT" in k for k in held), "should contain KEY_*SHIFT"

    def test_get_held_buttons(self):
        from inputctl import InputCtl
        ctl = InputCtl()

        assert ctl.get_held_buttons() == [], "should start with no held buttons"

        ctl.mouse_down("left")
        ctl.mouse_down("right")

        held = ctl.get_held_buttons()
        assert len(held) == 2, f"should have 2 held buttons, got {len(held)}"
        assert "left" in held, "should contain 'left'"
        assert "right" in held, "should contain 'right'"

    def test_idempotent_key_operations(self):
        from inputctl import InputCtl
        ctl = InputCtl()

        # Duplicate key_down should not error
        ctl.key_down("a")
        ctl.key_down("a")
        assert len(ctl.get_held_keys()) == 1, "should only track key once"

        # Duplicate key_up should not error
        ctl.key_up("a")
        ctl.key_up("a")
        assert not ctl.is_key_held("a"), "key should not be held"

    def test_mouse_down_up_methods(self):
        from inputctl import InputCtl
        ctl = InputCtl()

        # Test all button types
        for button in ["left", "right", "middle", "side", "extra"]:
            ctl.mouse_down(button)
            assert ctl.is_mouse_button_held(button), f"{button} should be held"
            ctl.mouse_up(button)
            assert not ctl.is_mouse_button_held(button), f"{button} should not be held"

    def test_auto_cleanup_on_drop(self):
        """Test that keys are released when object is deleted"""
        from inputctl import InputCtl
        ctl = InputCtl()
        ctl.key_down("shift")
        ctl.mouse_down("left")
        del ctl
        # Manual verification: no stuck keys in system
        # We can't easily verify this in a test, but at least ensure no crash
