"""SuperTux intent labeling using GPU-accelerated template matching."""

from __future__ import annotations

import os

from reflex_train.data.intent import infer_intent_from_keys

from .gpu_matching import GPUTemplateMatcher, GPUVideoScanner

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
        self._cache: dict[str, list[dict]] = {}

        self._matcher = GPUTemplateMatcher()
        self._scanner = GPUVideoScanner(
            matcher=self._matcher,
            sprite_scale=sprite_scale,
        )

        # Load templates (GPU if available, CPU fallback)
        self.tux_templates = self._matcher.load_templates(
            self._find_tux_sprites(), scale=sprite_scale
        )
        enemy_paths = self._find_enemy_sprites()
        self.enemy_templates = self._matcher.load_templates(enemy_paths, scale=sprite_scale)
        attacked_paths = [p for p in enemy_paths if self._is_attacked_sprite(p)]
        self.attacked_enemy_templates = self._matcher.load_templates(
            attacked_paths, scale=sprite_scale
        )
        self.loot_templates = self._matcher.load_templates(
            self._find_loot_sprites(), scale=sprite_scale
        )

    def label_intents(self, video_path, frame_indices, key_set_by_frame):
        sprite_hits = self._scan_video(video_path)
        intents = []
        for i in range(len(frame_indices)):
            end = min(i + self.intent_horizon, len(frame_indices) - 1)
            key_window = key_set_by_frame[i : end + 1]
            union_keys = set().union(*key_window) if key_window else set()
            window_hits = sprite_hits[i : end + 1]
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

    def _find_tux_sprites(self) -> list[str]:
        tux_dir = os.path.join(self.base_dir, "creatures", "tux")
        return self._collect_pngs(tux_dir)

    def _find_enemy_sprites(self) -> list[str]:
        creatures_dir = os.path.join(self.base_dir, "creatures")
        enemy_paths = []
        if not os.path.isdir(creatures_dir):
            return enemy_paths
        for name in os.listdir(creatures_dir):
            if name in {"tux", "penny"}:
                continue
            path = os.path.join(creatures_dir, name)
            if not os.path.isdir(path):
                continue
            enemy_paths.extend(self._collect_pngs(path))
        return enemy_paths

    def _find_loot_sprites(self) -> list[str]:
        loot_dir = os.path.join(self.base_dir, "objects", "coin")
        return self._collect_pngs(loot_dir)

    def _collect_pngs(self, root: str) -> list[str]:
        pngs = []
        if not os.path.isdir(root):
            return pngs
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".png"):
                    pngs.append(os.path.join(dirpath, name))
        return pngs

    def _is_attacked_sprite(self, path: str) -> bool:
        lower = os.path.basename(path).lower()
        return any(keyword in lower for keyword in ATTACKED_KEYWORDS)

    def _scan_video(self, video_path: str) -> list[dict]:
        if video_path in self._cache:
            return self._cache[video_path]

        hits = self._scanner.scan_video(
            video_path,
            self.tux_templates,
            self.enemy_templates,
            self.attacked_enemy_templates,
            self.loot_templates,
            self.sprite_threshold,
            self.proximity_px,
        )
        self._cache[video_path] = hits
        return hits
