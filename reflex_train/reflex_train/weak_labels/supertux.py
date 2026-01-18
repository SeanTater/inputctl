import os
import cv2
import math
from reflex_train.data.intent import infer_intent_from_keys


ATTACKED_KEYWORDS = (
    "stomp",
    "stomped",
    "flat",
    "flatten",
    "melting",
    "dead",
    "die",
    "squash",
    "squished",
    "hurt",
    "hit",
    "kill",
    "killed",
    "crush",
    "burn",
    "explode",
    "boom",
)


class SuperTuxIntentLabeler:
    def __init__(
        self,
        base_dir: str = "/usr/share/games/supertux2/images",
        intent_horizon: int = 10,
        sprite_scale: float = 0.5,
        sprite_threshold: float = 0.85,
        proximity_px: float = 96.0,
    ):
        self.base_dir = base_dir
        self.intent_horizon = intent_horizon
        self.sprite_scale = sprite_scale
        self.sprite_threshold = sprite_threshold
        self.proximity_px = proximity_px
        self._cache = {}

        self.tux_templates = self._load_templates(self._find_tux_sprites())
        self.enemy_templates, self.attacked_enemy_templates = self._load_enemy_templates()
        self.loot_templates = self._load_templates(self._find_loot_sprites())

    def label_intents(self, video_path, frame_indices, key_set_by_frame):
        sprite_hits = self._scan_video(video_path)
        intents = []
        for i in range(len(frame_indices)):
            end = min(i + self.intent_horizon, len(frame_indices) - 1)
            key_window = key_set_by_frame[i:end + 1]
            union_keys = set().union(*key_window) if key_window else set()
            window_hits = sprite_hits[i:end + 1]
            has_enemy = any(h["enemy_near"] for h in window_hits)
            has_enemy_attacked = any(h["enemy_attacked_near"] for h in window_hits)
            has_loot = any(h["loot_near"] for h in window_hits)

            if (
                "KEY_LEFTCTRL" in union_keys
                or "KEY_RIGHTCTRL" in union_keys
                or "KEY_LEFTALT" in union_keys
                or "KEY_RIGHTALT" in union_keys
                or "KEY_X" in union_keys
                or "KEY_C" in union_keys
            ):
                intents.append("ATTACK")
            elif "KEY_SPACE" in union_keys:
                intents.append("LEAP")
            elif "KEY_UP" in union_keys:
                intents.append("CLIMB")
            elif has_enemy_attacked:
                intents.append("ATTACK")
            elif has_enemy or "KEY_DOWN" in union_keys:
                intents.append("EVADE")
            elif has_loot:
                intents.append("LOOT")
            elif "KEY_LEFT" in union_keys or "KEY_RIGHT" in union_keys:
                intents.append("RUN")
            elif union_keys:
                intents.append(infer_intent_from_keys(union_keys))
            else:
                intents.append("WAIT")

        return intents

    def _find_tux_sprites(self):
        tux_dir = os.path.join(self.base_dir, "creatures", "tux")
        return self._collect_pngs(tux_dir)

    def _find_enemy_sprites(self):
        creatures_dir = os.path.join(self.base_dir, "creatures")
        enemy_paths = []
        for name in os.listdir(creatures_dir):
            if name in {"tux", "penny"}:
                continue
            path = os.path.join(creatures_dir, name)
            if not os.path.isdir(path):
                continue
            enemy_paths.extend(self._collect_pngs(path))
        return enemy_paths

    def _find_loot_sprites(self):
        loot_dir = os.path.join(self.base_dir, "objects", "coin")
        return self._collect_pngs(loot_dir)

    def _collect_pngs(self, root):
        pngs = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".png"):
                    pngs.append(os.path.join(dirpath, name))
        return pngs

    def _load_templates(self, paths):
        templates = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if self.sprite_scale != 1.0:
                width = max(1, int(img.shape[1] * self.sprite_scale))
                height = max(1, int(img.shape[0] * self.sprite_scale))
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            templates.append((img, path))
        return templates

    def _load_enemy_templates(self):
        enemy_paths = self._find_enemy_sprites()
        enemy_templates = []
        attacked_templates = []
        for path in enemy_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if self.sprite_scale != 1.0:
                width = max(1, int(img.shape[1] * self.sprite_scale))
                height = max(1, int(img.shape[0] * self.sprite_scale))
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            entry = (img, path)
            if self._is_attacked_sprite(path):
                attacked_templates.append(entry)
            enemy_templates.append(entry)
        return enemy_templates, attacked_templates

    def _is_attacked_sprite(self, path):
        lower = os.path.basename(path).lower()
        return any(keyword in lower for keyword in ATTACKED_KEYWORDS)

    def _scan_video(self, video_path):
        if video_path in self._cache:
            return self._cache[video_path]

        cap = cv2.VideoCapture(video_path)
        hits = []
        threshold = self.sprite_threshold
        proximity = self.proximity_px * self.sprite_scale

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.sprite_scale != 1.0:
                width = max(1, int(gray.shape[1] * self.sprite_scale))
                height = max(1, int(gray.shape[0] * self.sprite_scale))
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

            tux_pos = self._best_match(gray, self.tux_templates, threshold)
            enemy_positions = self._all_matches(gray, self.enemy_templates, threshold)
            attacked_positions = self._all_matches(gray, self.attacked_enemy_templates, threshold)
            loot_positions = self._all_matches(gray, self.loot_templates, threshold)

            enemy_near = self._is_near(tux_pos, enemy_positions, proximity)
            enemy_attacked_near = self._is_near(tux_pos, attacked_positions, proximity)
            loot_near = self._is_near(tux_pos, loot_positions, proximity)

            hits.append({
                "enemy_near": enemy_near,
                "enemy_attacked_near": enemy_attacked_near,
                "loot_near": loot_near,
            })

        cap.release()
        self._cache[video_path] = hits
        return hits

    def _best_match(self, frame, templates, threshold):
        best_pos = None
        best_score = threshold
        for tmpl, _path in templates:
            if tmpl.shape[0] > frame.shape[0] or tmpl.shape[1] > frame.shape[1]:
                continue
            res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= best_score:
                best_score = max_val
                best_pos = (max_loc[0] + tmpl.shape[1] / 2, max_loc[1] + tmpl.shape[0] / 2)
        return best_pos

    def _all_matches(self, frame, templates, threshold):
        positions = []
        for tmpl, _path in templates:
            if tmpl.shape[0] > frame.shape[0] or tmpl.shape[1] > frame.shape[1]:
                continue
            res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                positions.append((max_loc[0] + tmpl.shape[1] / 2, max_loc[1] + tmpl.shape[0] / 2))
        return positions

    def _is_near(self, tux_pos, positions, proximity):
        if tux_pos is None or not positions:
            return False
        for pos in positions:
            if self._distance(tux_pos, pos) <= proximity:
                return True
        return False

    def _distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
