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
