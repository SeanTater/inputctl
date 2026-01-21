"""Event detection for death/attack in SuperTux gameplay videos.

Win events are sourced from manual key presses in inputs logs.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from typing import Iterator

import polars as pl

from .gpu_matching import GPUTemplateMatcher, GPUVideoScanner

# Keywords indicating an enemy has been attacked/killed
ATTACKED_KEYWORDS = (
    "stomp",
    "flat",
    "melt",
    "dead",
    "squash",
    "squish",
    "hurt",
    "hit",
    "kill",
    "burn",
    "explode",
    "boom",
    "shatter",
)


@dataclass
class Event:
    """A detected event in a video."""

    frame_idx: int
    event: str  # "DEATH", "WIN", or "ATTACK"
    confidence: float


class EventDetector:
    """Detects death/attack events in SuperTux gameplay videos.

    Death: Template matches gameover sprites (tux falling/dying)
    Attack: Attacked/squashed enemy sprites detected
    """

    def __init__(
        self,
        base_dir: str = "/usr/share/games/supertux2/images",
        death_threshold: float = 0.75,
        attack_threshold: float = 0.75,
        win_key: str = "KEY_BACKSLASH",
        win_key_min_presses: int = 1,
        win_key_window_s: float = 2.0,
        win_key_cooldown_s: float = 30.0,
        frame_stride: int = 1,
        blank_frame_mean_threshold: float | None = 40.0,
        blank_frame_std_threshold: float | None = 10.0,
        attack_min_gap: int = 1,  # one per frame
        death_min_gap: int = 30 * 5,  # five seconds
    ):
        self.base_dir = base_dir
        self.death_threshold = death_threshold
        self.attack_threshold = attack_threshold
        self.win_key = win_key
        self.win_key_min_presses = max(1, win_key_min_presses)
        self.win_key_window_s = max(0.01, win_key_window_s)
        self.win_key_cooldown_s = max(0.0, win_key_cooldown_s)
        self.frame_stride = frame_stride
        self.attack_min_gap = attack_min_gap
        self.death_min_gap = death_min_gap

        # GPU matching setup
        self._matcher = GPUTemplateMatcher(max_templates_per_batch=128)
        self._scanner = GPUVideoScanner(
            matcher=self._matcher,
            blank_frame_mean_threshold=blank_frame_mean_threshold,
            blank_frame_std_threshold=blank_frame_std_threshold,
        )

        # Load templates (scale=1.0)
        self.death_templates = self._matcher.load_templates(
            self._find_death_sprites(), scale=1.0
        )
        self.attacked_templates = self._matcher.load_templates(
            self._find_attacked_sprites(), scale=1.0
        )

        if not self.death_templates:
            print(f"Warning: No death templates found in {base_dir}")
        if not self.attacked_templates:
            print(f"Warning: No attacked enemy templates found in {base_dir}")

    def _find_death_sprites(self) -> list[str]:
        """Find gameover sprite paths."""
        tux_dir = os.path.join(self.base_dir, "creatures", "tux")
        death_paths = []
        if not os.path.isdir(tux_dir):
            return death_paths
        for root, _, files in os.walk(tux_dir):
            for fname in files:
                if "gameover" in fname.lower() and fname.endswith(".png"):
                    death_paths.append(os.path.join(root, fname))
        return death_paths

    def _find_attacked_sprites(self) -> list[str]:
        """Find sprites of attacked/squashed enemies."""
        creatures_dir = os.path.join(self.base_dir, "creatures")
        attacked_paths = []
        if not os.path.isdir(creatures_dir):
            return attacked_paths
        for root, _, files in os.walk(creatures_dir):
            # Skip tux/penny directories
            rel_root = os.path.relpath(root, creatures_dir)
            top_dir = rel_root.split(os.sep, 1)[0]
            if top_dir in {"tux", "penny"}:
                continue
            for fname in files:
                if not fname.lower().endswith(".png"):
                    continue
                lower = fname.lower()
                if any(kw in lower for kw in ATTACKED_KEYWORDS):
                    attacked_paths.append(os.path.join(root, fname))
        return attacked_paths


    def detect_events(
        self, video_path: str, inputs_log: str | None = None, show_progress: bool = True
    ) -> list[Event] | None:
        """Scan a video and detect all death/attack events, plus win key markers."""
        try:
            frame_results = self._scanner.detect_events(
                video_path,
                self.death_templates,
                self.attacked_templates,
                self.death_threshold,
                self.attack_threshold,
                frame_stride=self.frame_stride,
                show_progress=show_progress,
            )
        except Exception as e:
            print(f"Warning: Cannot process video {video_path}: {e}")
            return None

        if not frame_results:
            return []

        key_press_events = []
        if inputs_log:
            key_press_events = self._load_win_key_presses(inputs_log)

        # Convert per-frame results to events
        events = []
        in_death = False
        in_attack = False

        for frame_idx, result in enumerate(frame_results):
            death_conf = result["death_conf"]
            attack_conf = result["attack_conf"]

            # Death detection with debouncing
            if death_conf >= self.death_threshold:
                if not in_death:
                    events.append(Event(frame_idx, "DEATH", death_conf))
                    in_death = True
            else:
                in_death = False

            # Attack detection with debouncing
            if attack_conf >= self.attack_threshold:
                if not in_attack:
                    events.append(Event(frame_idx, "ATTACK", attack_conf))
                    in_attack = True
            else:
                in_attack = False

        if key_press_events:
            events.extend(self._win_events_from_key_presses(key_press_events))
        elif inputs_log:
            print(f"Warning: no win key presses found in {inputs_log}")
        elif show_progress:
            print("Warning: inputs.parquet missing; win key labels disabled")

        # Sort by frame index and deduplicate nearby events
        events.sort(key=lambda e: e.frame_idx)
        return self._deduplicate_events(events)

    def _load_win_key_presses(self, inputs_log: str) -> list[dict]:
        if not os.path.exists(inputs_log):
            print(f"Warning: inputs log missing for win keys: {inputs_log}")
            return []

        df_inputs = pl.read_parquet(inputs_log)
        if df_inputs.is_empty():
            return []
        key_variants = [self.win_key, f"Key({self.win_key})"]
        df_keys = (
            df_inputs.filter(pl.col("event_type") == "key")
            .filter(pl.col("key_name").is_in(key_variants))
            .sort("timestamp")
        )
        if df_keys.is_empty():
            return []

        key_events = df_keys.select(["timestamp", "state"]).to_dicts()
        frame_indices, frame_timestamps = self._load_frame_times_for_inputs(inputs_log)
        if not frame_indices:
            return []

        press_events = []
        for evt in key_events:
            state = evt.get("state")
            if state in {"down", "repeat"}:
                press_events.append(evt)

        if not press_events:
            return []

        press_times = [evt["timestamp"] for evt in press_events]
        press_frames = self._map_timestamps_to_frames(press_times, frame_indices, frame_timestamps)
        return [
            {"timestamp": press_events[i]["timestamp"], "frame_idx": press_frames[i]}
            for i in range(len(press_frames))
        ]

    def _load_frame_times_for_inputs(self, inputs_log: str) -> tuple[list[int], list[int]]:
        run_dir = os.path.dirname(inputs_log)
        frames_log = os.path.join(run_dir, "frames.parquet")
        if not os.path.exists(frames_log):
            return [], []

        df_frames = pl.read_parquet(frames_log)
        frame_indices = df_frames["frame_idx"].to_list()
        frame_timestamps = df_frames["timestamp"].to_list()
        return frame_indices, frame_timestamps

    @staticmethod
    def _map_timestamps_to_frames(
        press_times: list[int], frame_indices: list[int], frame_timestamps: list[int]
    ) -> list[int]:
        press_frames = []
        for ts in press_times:
            pos = bisect.bisect_right(frame_timestamps, ts) - 1
            if pos < 0:
                pos = 0
            if pos < len(frame_indices):
                press_frames.append(frame_indices[pos])
        return press_frames

    def _win_events_from_key_presses(self, press_events: list[dict]) -> list[Event]:
        if not press_events:
            return []
        press_events = sorted(press_events, key=lambda e: e["timestamp"])
        events: list[Event] = []
        cooldown_ms = int(self.win_key_cooldown_s * 1000)
        window_ms = max(1, int(self.win_key_window_s * 1000))
        last_win_ts = None
        i = 0
        while i < len(press_events):
            start_ts = press_events[i]["timestamp"]
            window_end = start_ts + window_ms
            j = i
            count = 0
            first_frame = press_events[i]["frame_idx"]
            while j < len(press_events) and press_events[j]["timestamp"] <= window_end:
                count += 1
                j += 1
            if count >= self.win_key_min_presses:
                if last_win_ts is None or start_ts - last_win_ts >= cooldown_ms:
                    events.append(Event(first_frame, "WIN", 1.0))
                    last_win_ts = start_ts
                i = j
            else:
                i += 1
        return events

    def _deduplicate_events(self, events: list[Event]) -> list[Event]:
        """Remove duplicate events that are too close together."""
        if not events:
            return events

        deduped = [events[0]]
        win_min_gap = int(self.win_key_cooldown_s * 30)
        for event in events[1:]:
            prev = deduped[-1]
            if event.event == "ATTACK":
                min_gap = self.attack_min_gap
            elif event.event == "WIN":
                min_gap = win_min_gap
            else:
                min_gap = self.death_min_gap
            if (
                event.event == prev.event
                and (event.frame_idx - prev.frame_idx) < min_gap
            ):
                continue
            deduped.append(event)
        return deduped

    def iter_events(self, video_path: str) -> Iterator[Event]:
        """Generator version for memory efficiency on long videos."""
        events = self.detect_events(video_path)
        if events is None:
            return
        yield from events
