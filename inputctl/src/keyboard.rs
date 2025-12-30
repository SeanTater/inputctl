use evdev::Key;

/// Flag indicating the key requires shift modifier
const FLAG_UPPERCASE: i32 = 0x4000_0000;

/// Maps ASCII characters to (keycode, needs_shift)
/// Returns None for characters that can't be typed
pub fn ascii_to_key(c: char) -> Option<(Key, bool)> {
    if !c.is_ascii() || c as usize >= 128 {
        return None;
    }

    let keycode = ASCII_TO_KEYCODE[c as usize];
    if keycode < 0 {
        return None;
    }

    let needs_shift = (keycode & FLAG_UPPERCASE) != 0;
    let code = (keycode & 0xFFFF) as u16;

    Some((Key::new(code), needs_shift))
}

/// ASCII to keycode mapping table (US keyboard layout)
/// Negative values mean the character cannot be typed
/// High bit set means shift is required
#[rustfmt::skip]
static ASCII_TO_KEYCODE: [i32; 128] = [
    // 0x00 - 0x0F (control characters)
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, Key::KEY_TAB.code() as i32, Key::KEY_ENTER.code() as i32, -1, -1, -1, -1, -1,

    // 0x10 - 0x1F (control characters)
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,

    // 0x20 - 0x2F: space ! " # $ % & ' ( ) * + , - . /
    Key::KEY_SPACE.code() as i32,
    Key::KEY_1.code() as i32 | FLAG_UPPERCASE,           // !
    Key::KEY_APOSTROPHE.code() as i32 | FLAG_UPPERCASE,  // "
    Key::KEY_3.code() as i32 | FLAG_UPPERCASE,           // #
    Key::KEY_4.code() as i32 | FLAG_UPPERCASE,           // $
    Key::KEY_5.code() as i32 | FLAG_UPPERCASE,           // %
    Key::KEY_7.code() as i32 | FLAG_UPPERCASE,           // &
    Key::KEY_APOSTROPHE.code() as i32,                   // '
    Key::KEY_9.code() as i32 | FLAG_UPPERCASE,           // (
    Key::KEY_0.code() as i32 | FLAG_UPPERCASE,           // )
    Key::KEY_8.code() as i32 | FLAG_UPPERCASE,           // *
    Key::KEY_EQUAL.code() as i32 | FLAG_UPPERCASE,       // +
    Key::KEY_COMMA.code() as i32,                        // ,
    Key::KEY_MINUS.code() as i32,                        // -
    Key::KEY_DOT.code() as i32,                          // .
    Key::KEY_SLASH.code() as i32,                        // /

    // 0x30 - 0x3F: 0-9 : ; < = > ?
    Key::KEY_0.code() as i32,
    Key::KEY_1.code() as i32,
    Key::KEY_2.code() as i32,
    Key::KEY_3.code() as i32,
    Key::KEY_4.code() as i32,
    Key::KEY_5.code() as i32,
    Key::KEY_6.code() as i32,
    Key::KEY_7.code() as i32,
    Key::KEY_8.code() as i32,
    Key::KEY_9.code() as i32,
    Key::KEY_SEMICOLON.code() as i32 | FLAG_UPPERCASE,   // :
    Key::KEY_SEMICOLON.code() as i32,                    // ;
    Key::KEY_COMMA.code() as i32 | FLAG_UPPERCASE,       // <
    Key::KEY_EQUAL.code() as i32,                        // =
    Key::KEY_DOT.code() as i32 | FLAG_UPPERCASE,         // >
    Key::KEY_SLASH.code() as i32 | FLAG_UPPERCASE,       // ?

    // 0x40 - 0x4F: @ A-O (uppercase)
    Key::KEY_2.code() as i32 | FLAG_UPPERCASE,           // @
    Key::KEY_A.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_B.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_C.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_D.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_E.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_F.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_G.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_H.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_I.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_J.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_K.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_L.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_M.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_N.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_O.code() as i32 | FLAG_UPPERCASE,

    // 0x50 - 0x5F: P-Z [ \ ] ^ _
    Key::KEY_P.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_Q.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_R.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_S.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_T.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_U.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_V.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_W.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_X.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_Y.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_Z.code() as i32 | FLAG_UPPERCASE,
    Key::KEY_LEFTBRACE.code() as i32,                    // [
    Key::KEY_BACKSLASH.code() as i32,                    // \
    Key::KEY_RIGHTBRACE.code() as i32,                   // ]
    Key::KEY_6.code() as i32 | FLAG_UPPERCASE,           // ^
    Key::KEY_MINUS.code() as i32 | FLAG_UPPERCASE,       // _

    // 0x60 - 0x6F: ` a-o (lowercase)
    Key::KEY_GRAVE.code() as i32,                        // `
    Key::KEY_A.code() as i32,
    Key::KEY_B.code() as i32,
    Key::KEY_C.code() as i32,
    Key::KEY_D.code() as i32,
    Key::KEY_E.code() as i32,
    Key::KEY_F.code() as i32,
    Key::KEY_G.code() as i32,
    Key::KEY_H.code() as i32,
    Key::KEY_I.code() as i32,
    Key::KEY_J.code() as i32,
    Key::KEY_K.code() as i32,
    Key::KEY_L.code() as i32,
    Key::KEY_M.code() as i32,
    Key::KEY_N.code() as i32,
    Key::KEY_O.code() as i32,

    // 0x70 - 0x7F: p-z { | } ~ DEL
    Key::KEY_P.code() as i32,
    Key::KEY_Q.code() as i32,
    Key::KEY_R.code() as i32,
    Key::KEY_S.code() as i32,
    Key::KEY_T.code() as i32,
    Key::KEY_U.code() as i32,
    Key::KEY_V.code() as i32,
    Key::KEY_W.code() as i32,
    Key::KEY_X.code() as i32,
    Key::KEY_Y.code() as i32,
    Key::KEY_Z.code() as i32,
    Key::KEY_LEFTBRACE.code() as i32 | FLAG_UPPERCASE,   // {
    Key::KEY_BACKSLASH.code() as i32 | FLAG_UPPERCASE,   // |
    Key::KEY_RIGHTBRACE.code() as i32 | FLAG_UPPERCASE,  // }
    Key::KEY_GRAVE.code() as i32 | FLAG_UPPERCASE,       // ~
    -1,                                                   // DEL
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercase_letters_no_shift() {
        for c in 'a'..='z' {
            let result = ascii_to_key(c);
            assert!(result.is_some(), "should map '{}'", c);
            let (_, needs_shift) = result.unwrap();
            assert!(!needs_shift, "'{}' should not need shift", c);
        }
    }

    #[test]
    fn uppercase_letters_need_shift() {
        for c in 'A'..='Z' {
            let result = ascii_to_key(c);
            assert!(result.is_some(), "should map '{}'", c);
            let (_, needs_shift) = result.unwrap();
            assert!(needs_shift, "'{}' should need shift", c);
        }
    }

    #[test]
    fn digits_no_shift() {
        for c in '0'..='9' {
            let result = ascii_to_key(c);
            assert!(result.is_some(), "should map '{}'", c);
            let (_, needs_shift) = result.unwrap();
            assert!(!needs_shift, "'{}' should not need shift", c);
        }
    }

    #[test]
    fn special_chars_shift_status() {
        // Characters that need shift
        let shifted = "!@#$%^&*()_+{}|:\"<>?~";
        for c in shifted.chars() {
            let result = ascii_to_key(c);
            assert!(result.is_some(), "should map '{}'", c);
            let (_, needs_shift) = result.unwrap();
            assert!(needs_shift, "'{}' should need shift", c);
        }

        // Characters that don't need shift
        let unshifted = "`-=[]\\;',./";
        for c in unshifted.chars() {
            let result = ascii_to_key(c);
            assert!(result.is_some(), "should map '{}'", c);
            let (_, needs_shift) = result.unwrap();
            assert!(!needs_shift, "'{}' should not need shift", c);
        }
    }

    #[test]
    fn control_chars_return_none() {
        // Most control characters can't be typed
        for i in 0..9u8 {
            assert!(ascii_to_key(i as char).is_none());
        }
        // But tab and enter can
        assert!(ascii_to_key('\t').is_some());
        assert!(ascii_to_key('\n').is_some());
    }

    #[test]
    fn non_ascii_returns_none() {
        assert!(ascii_to_key('é').is_none());
        assert!(ascii_to_key('日').is_none());
        assert!(ascii_to_key('🎉').is_none());
    }
}
