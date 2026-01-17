import torch

# Define the vocabulary of keys we care about.
# We can track ALL keys, or a subset.
# For SuperTux and general gaming, we want a broad coverage.
# We effectively create a "vocab" of keys.

# Standard Gaming Keys (Subset)
# This list can be expanded to full 104 keys if desired.
TRACKED_KEYS = [
    "KEY_ESC",
    "KEY_1",
    "KEY_2",
    "KEY_3",
    "KEY_4",
    "KEY_5",
    "KEY_6",
    "KEY_7",
    "KEY_8",
    "KEY_9",
    "KEY_0",
    "KEY_MINUS",
    "KEY_EQUAL",
    "KEY_BACKSPACE",
    "KEY_TAB",
    "KEY_Q",
    "KEY_W",
    "KEY_E",
    "KEY_R",
    "KEY_T",
    "KEY_Y",
    "KEY_U",
    "KEY_I",
    "KEY_O",
    "KEY_P",
    "KEY_LEFTBRACE",
    "KEY_RIGHTBRACE",
    "KEY_ENTER",
    "KEY_LEFTCTRL",
    "KEY_A",
    "KEY_S",
    "KEY_D",
    "KEY_F",
    "KEY_G",
    "KEY_H",
    "KEY_J",
    "KEY_K",
    "KEY_L",
    "KEY_SEMICOLON",
    "KEY_APOSTROPHE",
    "KEY_GRAVE",
    "KEY_LEFTSHIFT",
    "KEY_BACKSLASH",
    "KEY_Z",
    "KEY_X",
    "KEY_C",
    "KEY_V",
    "KEY_B",
    "KEY_N",
    "KEY_M",
    "KEY_COMMA",
    "KEY_DOT",
    "KEY_SLASH",
    "KEY_RIGHTSHIFT",
    "KEY_KPASTERISK",
    "KEY_LEFTALT",
    "KEY_SPACE",
    "KEY_CAPSLOCK",
    "KEY_F1",
    "KEY_F2",
    "KEY_F3",
    "KEY_F4",
    "KEY_F5",
    "KEY_F6",
    "KEY_F7",
    "KEY_F8",
    "KEY_F9",
    "KEY_F10",
    "KEY_NUMLOCK",
    "KEY_SCROLLLOCK",
    "KEY_KP7",
    "KEY_KP8",
    "KEY_KP9",
    "KEY_KPMINUS",
    "KEY_KP4",
    "KEY_KP5",
    "KEY_KP6",
    "KEY_KPPLUS",
    "KEY_KP1",
    "KEY_KP2",
    "KEY_KP3",
    "KEY_KP0",
    "KEY_KPDOT",
    "KEY_F11",
    "KEY_F12",
    "KEY_RIGHTCTRL",
    "KEY_KPSLASH",
    "KEY_SYSRQ",
    "KEY_RIGHTALT",
    "KEY_HOME",
    "KEY_UP",
    "KEY_PAGEUP",
    "KEY_LEFT",
    "KEY_RIGHT",
    "KEY_END",
    "KEY_DOWN",
    "KEY_PAGEDOWN",
    "KEY_INSERT",
    "KEY_DELETE",
]

KEY_TO_IDX = {k: i for i, k in enumerate(TRACKED_KEYS)}
IDX_TO_KEY = {i: k for i, k in enumerate(TRACKED_KEYS)}
NUM_KEYS = len(TRACKED_KEYS)


def get_key_index(key_name: str) -> int:
    """Map string key name (e.g. 'KEY_SPACE') to index."""
    # Handle "Key(KEY_SPACE)" format from inputctl if present
    if "Key(" in key_name:
        key_name = key_name.replace("Key(", "").replace(")", "")
    return KEY_TO_IDX.get(key_name, -1)


def keys_to_vector(active_keys: set) -> torch.Tensor:
    """Convert set of active key strings to multi-hot vector."""
    vec = torch.zeros(NUM_KEYS)
    for k in active_keys:
        idx = get_key_index(k)
        if idx >= 0:
            vec[idx] = 1.0
    return vec
