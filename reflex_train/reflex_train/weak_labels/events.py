"""Event detection for death/win/attack in SuperTux gameplay videos."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from PIL import Image

from .gpu_matching import GPUTemplateMatcher, GPUVideoScanner

# Keywords indicating an enemy has been attacked/killed
ATTACKED_KEYWORDS = (
    "stomp", "stomped", "flat", "flatten", "melting", "dead", "die",
    "squash", "squished", "hurt", "hit", "kill", "killed", "crush",
    "burn", "explode", "boom",
)


@dataclass
class Event:
    """A detected event in a video."""

    frame_idx: int
    event: str  # "DEATH", "WIN", or "ATTACK"
    confidence: float


class EventDetector:
    """Detects death/win/attack events in SuperTux gameplay videos.

    Death: Template matches gameover sprites (tux falling/dying)
    Win: Sparkle particles near Tux (goal completion)
    Attack: Attacked/squashed enemy sprites detected
    """

    def __init__(
        self,
        base_dir: str = "/usr/share/games/supertux2/images",
        sprite_scale: float = 0.5,
        death_threshold: float = 0.75,
        attack_threshold: float = 0.8,
        win_proximity_px: float = 96.0,
        sparkle_threshold: float = 0.8,
        win_min_frames: int = 3,
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
        self.attack_threshold = attack_threshold
        self.win_proximity_px = win_proximity_px
        self.sparkle_threshold = sparkle_threshold
        self.win_min_frames = win_min_frames
        self.win_llm_gate = win_llm_gate
        self.win_llm_sample_stride = max(1, win_llm_sample_stride)
        self.win_llm_prompt = win_llm_prompt
        self.win_llm_timeout_s = win_llm_timeout_s
        self.win_llm_model = win_llm_model
        self.win_llm_url = win_llm_url

        # GPU matching setup
        self._matcher = GPUTemplateMatcher()
        self._scanner = GPUVideoScanner(
            matcher=self._matcher,
            sprite_scale=sprite_scale,
        )

        # Load templates
        self.death_templates = self._matcher.load_templates(
            self._find_death_sprites(), scale=sprite_scale
        )
        self.tux_templates = self._matcher.load_templates(
            self._find_tux_sprites(), scale=sprite_scale
        )
        self.attacked_templates = self._matcher.load_templates(
            self._find_attacked_sprites(), scale=sprite_scale
        )
        sparkle_templates = self._matcher.load_templates(
            self._find_sparkle_sprites(), scale=sprite_scale
        )
        self.sparkle_templates = self._filter_sparkle_templates(sparkle_templates)

        if not self.death_templates:
            print(f"Warning: No death templates found in {base_dir}")
        if not self.sparkle_templates:
            print(f"Warning: No sparkle templates found in {base_dir}")
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

    def _find_tux_sprites(self) -> list[str]:
        tux_dir = os.path.join(self.base_dir, "creatures", "tux")
        return self._collect_pngs(tux_dir)

    def _find_attacked_sprites(self) -> list[str]:
        """Find sprites of attacked/squashed enemies."""
        creatures_dir = os.path.join(self.base_dir, "creatures")
        attacked_paths = []
        if not os.path.isdir(creatures_dir):
            return attacked_paths
        for root, _, files in os.walk(creatures_dir):
            # Skip tux/penny directories
            if "tux" in root or "penny" in root:
                continue
            for fname in files:
                if not fname.lower().endswith(".png"):
                    continue
                lower = fname.lower()
                if any(kw in lower for kw in ATTACKED_KEYWORDS):
                    attacked_paths.append(os.path.join(root, fname))
        return attacked_paths

    def _find_sparkle_sprites(self) -> list[str]:
        particles_dir = os.path.join(self.base_dir, "particles")
        return self._collect_pngs(particles_dir, prefix="sparkle")

    def _collect_pngs(self, root: str, prefix: str | None = None) -> list[str]:
        pngs: list[str] = []
        if not os.path.isdir(root):
            return pngs
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(".png"):
                    continue
                if prefix and not name.lower().startswith(prefix):
                    continue
                pngs.append(os.path.join(dirpath, name))
        return pngs

    def _filter_sparkle_templates(self, templates: list[tuple]) -> list[tuple]:
        """Filter to prefer 'light' sparkle templates."""
        filtered = [t for t in templates if "light" in os.path.basename(t[1]).lower()]
        return filtered or templates

    def detect_events(
        self, video_path: str, show_progress: bool = True
    ) -> list[Event] | None:
        """Scan a video and detect all death/win/attack events.

        Returns events sorted by frame index, or None if video is corrupted.
        """
        try:
            frame_results = self._scanner.detect_events(
                video_path,
                self.tux_templates,
                self.death_templates,
                self.attacked_templates,
                self.sparkle_templates,
                self.death_threshold,
                self.attack_threshold,
                self.sparkle_threshold,
                self.win_proximity_px,
                show_progress=show_progress,
            )
        except Exception as e:
            print(f"Warning: Cannot process video {video_path}: {e}")
            return None

        if not frame_results:
            return []

        # Convert per-frame results to events
        events = []
        in_death = False
        in_attack = False
        win_streak = 0
        win_streak_start = None
        last_win_conf = 0.0

        for frame_idx, result in enumerate(frame_results):
            death_conf = result["death_conf"]
            attack_conf = result["attack_conf"]
            win_conf = result["win_conf"]

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

            # Win detection with streak requirement
            if win_conf > 0:
                if win_streak == 0:
                    win_streak_start = frame_idx
                win_streak += 1
                last_win_conf = win_conf
            else:
                if win_streak >= self.win_min_frames and win_streak_start is not None:
                    events.append(Event(win_streak_start, "WIN", last_win_conf))
                win_streak = 0
                win_streak_start = None

        # Handle win streak at end of video
        if win_streak >= self.win_min_frames and win_streak_start is not None:
            events.append(Event(win_streak_start, "WIN", last_win_conf))

        if self.win_llm_gate:
            events = self._gate_win_events(video_path, events)

        # Sort by frame index and deduplicate nearby events
        events.sort(key=lambda e: e.frame_idx)
        return self._deduplicate_events(events)

    def _gate_win_events(self, video_path: str, events: list[Event]) -> list[Event]:
        """Use LLM to validate win events."""
        if not events:
            return events
        if not self._ollama_available():
            print("Warning: Ollama not available; skipping win gating")
            return events

        from reflex_train.data.dataset import _get_torchcodec_decoders, _ensure_nchw_frames

        try:
            VideoDecoder, _ = _get_torchcodec_decoders()
            decoder = VideoDecoder(video_path, device="cpu")
        except Exception as e:
            print(f"Warning: Cannot open video for gating: {e}")
            return events

        gated: list[Event] = []
        for event in events:
            if event.event != "WIN":
                gated.append(event)
                continue
            if self._confirm_win_event(decoder, event.frame_idx):
                gated.append(event)

        return gated

    def _confirm_win_event(self, decoder, frame_idx: int) -> bool:
        """Confirm win event using LLM."""
        from reflex_train.data.dataset import _ensure_nchw_frames

        for offset in range(0, self.win_min_frames, self.win_llm_sample_stride):
            sample_idx = frame_idx + offset
            try:
                frame_data = decoder.get_frame_at(sample_idx)
                frame = _ensure_nchw_frames(frame_data.data)
                frame_np = frame[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
                if self._ollama_is_win(frame_np):
                    return True
            except Exception:
                continue
        return False

    def _ollama_is_win(self, frame_np) -> bool:
        """Query Ollama to check if frame shows win screen."""
        img = Image.fromarray(frame_np)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        jpg_bytes = buffer.getvalue()

        payload = {
            "model": self.win_llm_model,
            "prompt": self.win_llm_prompt,
            "images": [base64.b64encode(jpg_bytes).decode("utf-8")],
            "stream": False,
        }
        try:
            response = self._ollama_request(payload)
        except Exception as exc:
            print(f"Warning: Ollama request failed: {exc}")
            return False

        text = response.get("response", "")
        return text.strip().lower().startswith("yes")

    def _ollama_request(self, payload: dict) -> dict:
        """Send request to Ollama HTTP API."""
        timeout = max(1.0, float(self.win_llm_timeout_s))
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
        """Check if Ollama API is reachable."""
        try:
            # Check the /api/tags endpoint (lightweight)
            base_url = self.win_llm_url.rsplit("/", 1)[0]
            req = urllib.request.Request(f"{base_url}/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.0):
                return True
        except Exception:
            return False

    def _deduplicate_events(
        self, events: list[Event], min_gap: int = 30
    ) -> list[Event]:
        """Remove duplicate events that are too close together."""
        if not events:
            return events

        deduped = [events[0]]
        for event in events[1:]:
            prev = deduped[-1]
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
