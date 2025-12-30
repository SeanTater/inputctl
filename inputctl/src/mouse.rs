use evdev::Key;

/// Mouse buttons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Side,
    Extra,
}

impl MouseButton {
    /// Convert to evdev Key code
    pub fn to_key(self) -> Key {
        match self {
            MouseButton::Left => Key::BTN_LEFT,
            MouseButton::Right => Key::BTN_RIGHT,
            MouseButton::Middle => Key::BTN_MIDDLE,
            MouseButton::Side => Key::BTN_SIDE,
            MouseButton::Extra => Key::BTN_EXTRA,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn button_to_key_mapping() {
        assert_eq!(MouseButton::Left.to_key(), Key::BTN_LEFT);
        assert_eq!(MouseButton::Right.to_key(), Key::BTN_RIGHT);
        assert_eq!(MouseButton::Middle.to_key(), Key::BTN_MIDDLE);
        assert_eq!(MouseButton::Side.to_key(), Key::BTN_SIDE);
        assert_eq!(MouseButton::Extra.to_key(), Key::BTN_EXTRA);
    }

    #[test]
    fn buttons_are_distinct() {
        let buttons = [
            MouseButton::Left,
            MouseButton::Right,
            MouseButton::Middle,
            MouseButton::Side,
            MouseButton::Extra,
        ];
        for (i, a) in buttons.iter().enumerate() {
            for (j, b) in buttons.iter().enumerate() {
                if i != j {
                    assert_ne!(a.to_key(), b.to_key());
                }
            }
        }
    }
}
