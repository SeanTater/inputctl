INTENTS = [
    "LOOT",
    "EVADE",
    "ATTACK",
    "CLIMB",
    "LEAP",
    "RUN",
    "WAIT",
]

INTENT_TO_IDX = {intent: i for i, intent in enumerate(INTENTS)}


def intent_to_vector(intent: str):
    if intent not in INTENT_TO_IDX:
        raise ValueError(f"Unknown intent: {intent}")
    vec = [0.0] * len(INTENTS)
    vec[INTENT_TO_IDX[intent]] = 1.0
    return vec


def infer_intent_from_keys(active_keys: set) -> str:
    # Priority order: explicit interactions first, then movement.
    if "KEY_E" in active_keys or "KEY_ENTER" in active_keys:
        return "LOOT"
    if "KEY_SPACE" in active_keys:
        return "LEAP"
    if "KEY_UP" in active_keys:
        return "CLIMB"
    if (
        "KEY_LEFTCTRL" in active_keys
        or "KEY_RIGHTCTRL" in active_keys
        or "KEY_LEFTALT" in active_keys
        or "KEY_RIGHTALT" in active_keys
        or "KEY_X" in active_keys
        or "KEY_C" in active_keys
    ):
        return "ATTACK"
    if "KEY_DOWN" in active_keys:
        return "EVADE"
    if "KEY_LEFT" in active_keys or "KEY_RIGHT" in active_keys:
        return "RUN"
    return "WAIT"


def infer_intent_from_key_window(key_window: list[set]) -> str:
    if not key_window:
        return "WAIT"

    union_keys = set().union(*key_window)
    if not union_keys:
        return "WAIT"

    return infer_intent_from_keys(union_keys)


def infer_intent_from_signals(key_window: list[set], sprite_window: list[dict]) -> str:
    union_keys = set().union(*key_window) if key_window else set()
    has_loot = any(w.get("loot") for w in sprite_window) if sprite_window else False
    has_enemy = any(w.get("enemy") for w in sprite_window) if sprite_window else False

    if (
        "KEY_LEFTCTRL" in union_keys
        or "KEY_RIGHTCTRL" in union_keys
        or "KEY_LEFTALT" in union_keys
        or "KEY_RIGHTALT" in union_keys
        or "KEY_X" in union_keys
        or "KEY_C" in union_keys
    ):
        return "ATTACK"
    if "KEY_SPACE" in union_keys:
        return "LEAP"
    if "KEY_UP" in union_keys:
        return "CLIMB"
    if has_enemy or "KEY_DOWN" in union_keys:
        return "EVADE"
    if has_loot:
        return "LOOT"
    if "KEY_LEFT" in union_keys or "KEY_RIGHT" in union_keys:
        return "RUN"
    return "WAIT"
