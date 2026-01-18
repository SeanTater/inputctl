"""Event detection for death/win in SuperTux gameplay videos."""

import base64
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class Event:
    """A detected event in a video."""

    frame_idx: int
    event: str  # "DEATH" or "WIN"
    confidence: float


class EventDetector:
    """Detects death and win events in SuperTux gameplay videos.

    Death: Template matches gameover sprites (tux falling/dying)
    Win: Goal tiles + sparkle particles near Tux
    """

    def __init__(
        self,
        base_dir: str = "/usr/share/games/supertux2/images",
        sprite_scale: float = 0.5,
        death_threshold: float = 0.75,
        win_proximity_px: float = 96.0,
        sparkle_threshold: float = 0.8,
        win_min_frames: int = 3,
        check_every_n: int = 3,  # Skip frames for speed (death anim lasts 30+ frames)
        win_check_every_n: int = 30,
        win_llm_gate: bool = False,
        win_llm_sample_stride: int = 30,
        win_llm_prompt: str = (
            "Is this the SuperTux level-complete or win screen? Reply YES or NO."
        ),
        win_llm_timeout_s: float = 30.0,
        win_llm_model: str = "qwen3-vl:4b",
        win_llm_url: str = "http://localhost:11434/api/generate",
    ):
        self.base_dir = base_dir
        self.sprite_scale = sprite_scale
        self.death_threshold = death_threshold
        self.win_proximity_px = win_proximity_px
        self.sparkle_threshold = sparkle_threshold
        self.win_min_frames = win_min_frames
        self.check_every_n = check_every_n
        self.win_check_every_n = max(1, win_check_every_n)
        self.win_llm_gate = win_llm_gate
        self.win_llm_sample_stride = max(1, win_llm_sample_stride)
        self.win_llm_prompt = win_llm_prompt
        self.win_llm_timeout_s = win_llm_timeout_s
        self.win_llm_model = win_llm_model
        self.win_llm_url = win_llm_url

        self.death_templates = self._load_death_templates()
        self.tux_templates = self._load_templates(self._find_tux_sprites())
        self.sparkle_templates = self._load_templates(self._find_sparkle_sprites())
        self.sparkle_templates = self._filter_sparkle_templates(self.sparkle_templates)
        if not self.death_templates:
            print(f"Warning: No death templates found in {base_dir}")
        if not self.sparkle_templates:
            print(f"Warning: No sparkle templates found in {base_dir}")

    def _load_death_templates(self) -> list[tuple[np.ndarray, str]]:
        """Load gameover sprite templates."""
        templates = []
        # Look for gameover sprites in all tux variants
        tux_dir = os.path.join(self.base_dir, "creatures", "tux")

        for root, _, files in os.walk(tux_dir):
            for fname in files:
                if "gameover" in fname.lower() and fname.endswith(".png"):
                    path = os.path.join(root, fname)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    if self.sprite_scale != 1.0:
                        w = max(1, int(img.shape[1] * self.sprite_scale))
                        h = max(1, int(img.shape[0] * self.sprite_scale))
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    templates.append((img, path))

        return templates

    def _load_templates(self, paths: list[str]) -> list[tuple[np.ndarray, str]]:
        templates = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if self.sprite_scale != 1.0:
                w = max(1, int(img.shape[1] * self.sprite_scale))
                h = max(1, int(img.shape[0] * self.sprite_scale))
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            templates.append((img, path))
        return templates

    def _filter_sparkle_templates(
        self, templates: list[tuple[np.ndarray, str]]
    ) -> list[tuple[np.ndarray, str]]:
        filtered = [t for t in templates if "light" in os.path.basename(t[1]).lower()]
        return filtered or templates

    def _find_tux_sprites(self) -> list[str]:
        tux_dir = os.path.join(self.base_dir, "creatures", "tux")
        return self._collect_pngs(tux_dir)

    def _find_sparkle_sprites(self) -> list[str]:
        particles_dir = os.path.join(self.base_dir, "particles")
        return self._collect_pngs(particles_dir, prefix="sparkle")

    def _collect_pngs(self, root: str, prefix: Optional[str] = None) -> list[str]:
        pngs: list[str] = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(".png"):
                    continue
                if prefix and not name.lower().startswith(prefix):
                    continue
                pngs.append(os.path.join(dirpath, name))
        return pngs

    def detect_events(
        self, video_path: str, show_progress: bool = True
    ) -> list[Event] | None:
        """Scan a video and detect all death/win events.

        Returns events sorted by frame index, or None if video is corrupted.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        events = []
        frame_idx = 0
        win_streak = 0
        win_streak_start = None
        last_win_conf = 0.0
        in_death = False  # debounce consecutive death frames

        pbar = tqdm(
            total=total_frames, desc="Scanning", unit="f", disable=not show_progress
        )

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Always check win/death on the same grayscale frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Only do expensive template match every N frames
            if frame_idx % self.check_every_n == 0:
                scaled_gray = gray
                if self.sprite_scale != 1.0:
                    w = max(1, int(gray.shape[1] * self.sprite_scale))
                    h = max(1, int(gray.shape[0] * self.sprite_scale))
                    scaled_gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

                death_conf = self._check_death(scaled_gray)
                if death_conf >= self.death_threshold:
                    if not in_death:
                        events.append(Event(frame_idx, "DEATH", death_conf))
                        in_death = True
                else:
                    in_death = False

                if frame_idx % self.win_check_every_n == 0:
                    win_conf = self._check_win(scaled_gray)
                    if win_conf > 0:
                        if win_streak == 0:
                            win_streak_start = frame_idx
                        win_streak += 1
                        last_win_conf = win_conf
                    else:
                        if (
                            win_streak >= self.win_min_frames
                            and win_streak_start is not None
                        ):
                            events.append(Event(win_streak_start, "WIN", last_win_conf))
                        win_streak = 0
                        win_streak_start = None

            frame_idx += 1
            pbar.update(1)

        pbar.close()

        # Handle win streak at end of video
        if win_streak >= self.win_min_frames and win_streak_start is not None:
            events.append(Event(win_streak_start, "WIN", last_win_conf))

        cap.release()

        if self.win_llm_gate:
            events = self._gate_win_events(video_path, events)

        # Sort by frame index and deduplicate nearby events
        events.sort(key=lambda e: e.frame_idx)
        return self._deduplicate_events(events)

    def _gate_win_events(self, video_path: str, events: list[Event]) -> list[Event]:
        if not events:
            return events
        if not self._ollama_available():
            print("Warning: Ollama not available; skipping win gating")
            return events

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video for gating: {video_path}")
            return events

        gated: list[Event] = []
        for event in events:
            if event.event != "WIN":
                gated.append(event)
                continue
            if self._confirm_win_event(cap, event.frame_idx):
                gated.append(event)

        cap.release()
        return gated

    def _confirm_win_event(self, cap: cv2.VideoCapture, frame_idx: int) -> bool:
        for offset in range(0, self.win_min_frames, self.win_llm_sample_stride):
            sample_idx = frame_idx + offset
            frame = self._read_frame(cap, sample_idx)
            if frame is None:
                continue
            if self._ollama_is_win(frame):
                return True
        return False

    def _read_frame(
        self, cap: cv2.VideoCapture, frame_idx: int
    ) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame

    def _ollama_is_win(self, frame: np.ndarray) -> bool:
        jpg_bytes = self._encode_jpeg(frame)
        if not jpg_bytes:
            return False

        payload = {
            "model": self.win_llm_model,
            "prompt": self.win_llm_prompt,
            "images": [base64.b64encode(jpg_bytes).decode("utf-8")],
            "stream": False,
        }
        try:
            response = self._ollama_generate(payload)
        except Exception as exc:
            print(f"Warning: Ollama request failed: {exc}")
            return False

        text = response.get("response", "")
        return self._is_yes(text)

    def _encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return buffer.tobytes()

    def _check_win(self, gray: np.ndarray) -> float:
        if not self.tux_templates or not self.sparkle_templates:
            return 0.0

        tux_pos = self._best_match(gray, self.tux_templates, self.sparkle_threshold)
        if tux_pos is None:
            return 0.0

        proximity = self.win_proximity_px * self.sprite_scale
        sparkle_positions = self._all_matches(
            gray, self.sparkle_templates, self.sparkle_threshold
        )
        if not sparkle_positions:
            return 0.0

        for sparkle_pos in sparkle_positions:
            if self._distance(tux_pos, sparkle_pos) <= proximity:
                return 1.0
        return 0.0

        tux_pos = self._best_match(gray, self.tux_templates, self.goal_threshold)
        if tux_pos is None:
            return 0.0

        proximity = self.win_proximity_px * self.sprite_scale
        goal_pos = self._best_match(gray, self.goal_templates, self.goal_threshold)
        if goal_pos is None or self._distance(tux_pos, goal_pos) > proximity:
            return 0.0

        sparkle_positions = self._all_matches(
            gray, self.sparkle_templates, self.sparkle_threshold
        )
        if not sparkle_positions:
            return 0.0

        for sparkle_pos in sparkle_positions:
            if self._distance(tux_pos, sparkle_pos) <= proximity:
                return 1.0
        return 0.0

    def _best_match(
        self,
        frame: np.ndarray,
        templates: list[tuple[np.ndarray, str]],
        threshold: float,
    ) -> Optional[tuple[float, float]]:
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
                best_pos = (
                    max_loc[0] + tmpl.shape[1] / 2,
                    max_loc[1] + tmpl.shape[0] / 2,
                )
        return best_pos

    def _all_matches(
        self,
        frame: np.ndarray,
        templates: list[tuple[np.ndarray, str]],
        threshold: float,
    ) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        for tmpl, _path in templates:
            if tmpl.shape[0] > frame.shape[0] or tmpl.shape[1] > frame.shape[1]:
                continue
            res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                positions.append(
                    (max_loc[0] + tmpl.shape[1] / 2, max_loc[1] + tmpl.shape[0] / 2)
                )
        return positions

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _ollama_generate(self, payload: dict) -> dict:
        timeout = max(1.0, float(self.win_llm_timeout_s))
        if shutil.which("ollama"):
            return self._ollama_generate_cli(payload, timeout)
        return self._ollama_generate_http(payload, timeout)

    def _ollama_generate_cli(self, payload: dict, timeout: float) -> dict:
        cmd = ["ollama", "generate", "-m", payload["model"], "-p", payload["prompt"]]
        images = payload.get("images") or []
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(base64.b64decode(images[0]))
            tmp_path = tmp.name
        cmd.extend(["--image", tmp_path])
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ollama generate failed")
        return {"response": result.stdout.strip()}

    def _ollama_generate_http(self, payload: dict, timeout: float) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.win_llm_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    def _ollama_available(self) -> bool:
        if shutil.which("ollama"):
            return True
        try:
            req = urllib.request.Request(self.win_llm_url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as _:
                return True
        except urllib.error.URLError:
            return False

    def _is_yes(self, text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        return normalized.startswith("yes")

    def _check_death(self, gray: np.ndarray) -> float:
        """Check if any death template matches in the frame."""
        best_conf = 0.0
        for tmpl, _ in self.death_templates:
            if tmpl.shape[0] > gray.shape[0] or tmpl.shape[1] > gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            _, max_val, _, _ = cv2.minMaxLoc(res)
            best_conf = max(best_conf, max_val)
        return best_conf

    def _deduplicate_events(
        self, events: list[Event], min_gap: int = 30
    ) -> list[Event]:
        """Remove duplicate events that are too close together.

        If two events of the same type are within min_gap frames, keep only the first.
        """
        if not events:
            return events

        deduped = [events[0]]
        for event in events[1:]:
            prev = deduped[-1]
            # If same type and too close, skip
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
            return iter(())
        for event in events:
            yield event
        return iter(())
