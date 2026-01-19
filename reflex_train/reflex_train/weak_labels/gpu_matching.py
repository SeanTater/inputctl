"""GPU-accelerated template matching using FFT-based normalized cross-correlation.

Provides TM_CCOEFF_NORMED equivalent matching on CUDA tensors, with CPU fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from torchcodec.decoders import VideoDecoder

# Reuse torchcodec setup from dataset module
from reflex_train.data.dataset import (
    _get_torchcodec_decoders,
    _ensure_nchw_frames,
)


def _rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to grayscale.

    Uses ITU-R BT.601 weights: 0.299*R + 0.587*G + 0.114*B

    Args:
        rgb: (C, H, W) or (B, C, H, W) with C=3

    Returns:
        (H, W) or (B, H, W) float grayscale tensor
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=torch.float32)
    if rgb.ndim == 4:
        return torch.einsum("bchw,c->bhw", rgb.float(), weights)
    return torch.einsum("chw,c->hw", rgb.float(), weights)


class GPUTemplateMatcher:
    """GPU-accelerated template matching using FFT-based NCC.

    Provides TM_CCOEFF_NORMED equivalent matching. Falls back to CPU
    PyTorch if CUDA is unavailable.
    """

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self._template_cache: dict[str, torch.Tensor] = {}

    def load_template(self, path: str, scale: float = 1.0) -> torch.Tensor:
        """Load a template image and convert to grayscale GPU tensor.

        Args:
            path: Path to PNG image
            scale: Resize factor (0.5 = half size)

        Returns:
            (H, W) grayscale tensor on self.device
        """
        cache_key = f"{path}:{scale}"
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        img = Image.open(path).convert("L")  # Grayscale
        if scale != 1.0:
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        tensor = torch.tensor(list(img.getdata()), dtype=self.dtype, device=self.device)
        tensor = tensor.reshape(img.height, img.width)
        self._template_cache[cache_key] = tensor
        return tensor

    def load_templates(
        self, paths: list[str], scale: float = 1.0
    ) -> list[tuple[torch.Tensor, str]]:
        """Load multiple templates.

        Returns:
            List of (tensor, path) tuples
        """
        templates = []
        for path in paths:
            try:
                tensor = self.load_template(path, scale)
                templates.append((tensor, path))
            except Exception:
                continue
        return templates

    def match_template_ncc(
        self,
        frame: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """FFT-based normalized cross-correlation.

        Equivalent to cv2.matchTemplate with TM_CCOEFF_NORMED.

        Args:
            frame: (H, W) grayscale frame
            template: (h, w) grayscale template

        Returns:
            (H-h+1, W-w+1) correlation map with values in [-1, 1]
        """
        H, W = frame.shape
        h, w = template.shape

        if h > H or w > W:
            return torch.zeros(1, 1, device=self.device, dtype=self.dtype)

        # Mean-subtract and normalize template
        t_mean = template.mean()
        t_centered = template - t_mean
        t_var = t_centered.pow(2).sum()

        if t_var < 1e-7:
            # Template is constant, no meaningful correlation
            return torch.zeros(H - h + 1, W - w + 1, device=self.device, dtype=self.dtype)

        t_std = t_var.sqrt()

        # Pad template to frame size for FFT
        t_padded = torch.zeros(H, W, device=self.device, dtype=self.dtype)
        t_padded[:h, :w] = t_centered

        # FFT correlation: ifft(fft(frame) * conj(fft(template)))
        frame_fft = torch.fft.fft2(frame)
        template_fft = torch.fft.fft2(t_padded)
        correlation = torch.fft.ifft2(frame_fft * template_fft.conj()).real

        # Compute local statistics of frame using FFT convolution
        # This is equivalent to sliding window mean/variance
        n_pixels = h * w
        ones = torch.zeros(H, W, device=self.device, dtype=self.dtype)
        ones[:h, :w] = 1.0
        ones_fft = torch.fft.fft2(ones)

        # Local sum via FFT convolution
        local_sum = torch.fft.ifft2(frame_fft * ones_fft.conj()).real
        local_mean = local_sum / n_pixels

        # Local sum of squares
        frame_sq_fft = torch.fft.fft2(frame.pow(2))
        local_sum_sq = torch.fft.ifft2(frame_sq_fft * ones_fft.conj()).real

        # Local variance: E[X^2] - E[X]^2
        local_var = local_sum_sq / n_pixels - local_mean.pow(2)
        local_var = local_var.clamp(min=1e-7)
        local_std = (local_var * n_pixels).sqrt()  # Unnormalized std for NCC formula

        # NCC = sum((I - I_mean) * (T - T_mean)) / (local_std * t_std)
        # correlation already has sum(I * T_centered), need to subtract mean contribution
        ncc = (correlation - local_mean * t_centered.sum()) / (local_std * t_std)

        # Extract valid region
        result = ncc[:H - h + 1, :W - w + 1]

        # Clamp to valid range (numerical precision issues)
        return result.clamp(-1.0, 1.0)

    def best_match(
        self,
        frame: torch.Tensor,
        templates: list[tuple[torch.Tensor, str]],
        threshold: float,
    ) -> tuple[float, float] | None:
        """Find best matching template location above threshold.

        Args:
            frame: (H, W) grayscale frame
            templates: List of (tensor, path) tuples
            threshold: Minimum correlation score

        Returns:
            Center position (x, y) or None if no match above threshold
        """
        best_pos = None
        best_score = threshold

        for tmpl, _path in templates:
            if tmpl.shape[0] > frame.shape[0] or tmpl.shape[1] > frame.shape[1]:
                continue

            ncc = self.match_template_ncc(frame, tmpl)
            if ncc.numel() == 0:
                continue

            max_val = ncc.max().item()
            if max_val >= best_score:
                max_idx = ncc.argmax()
                max_y = (max_idx // ncc.shape[1]).item()
                max_x = (max_idx % ncc.shape[1]).item()
                best_score = max_val
                best_pos = (max_x + tmpl.shape[1] / 2, max_y + tmpl.shape[0] / 2)

        return best_pos

    def all_matches(
        self,
        frame: torch.Tensor,
        templates: list[tuple[torch.Tensor, str]],
        threshold: float,
    ) -> list[tuple[float, float]]:
        """Find all template matches above threshold.

        Args:
            frame: (H, W) grayscale frame
            templates: List of (tensor, path) tuples
            threshold: Minimum correlation score

        Returns:
            List of center positions (x, y)
        """
        positions = []

        for tmpl, _path in templates:
            if tmpl.shape[0] > frame.shape[0] or tmpl.shape[1] > frame.shape[1]:
                continue

            ncc = self.match_template_ncc(frame, tmpl)
            if ncc.numel() == 0:
                continue

            max_val = ncc.max().item()
            if max_val >= threshold:
                max_idx = ncc.argmax()
                max_y = (max_idx // ncc.shape[1]).item()
                max_x = (max_idx % ncc.shape[1]).item()
                positions.append((max_x + tmpl.shape[1] / 2, max_y + tmpl.shape[0] / 2))

        return positions


class GPUVideoScanner:
    """Combines torchcodec GPU decoding with GPU template matching.

    Processes videos entirely on GPU for maximum throughput.
    """

    def __init__(
        self,
        matcher: GPUTemplateMatcher | None = None,
        batch_size: int = 16,
        sprite_scale: float = 0.5,
    ):
        self.matcher = matcher or GPUTemplateMatcher()
        self.batch_size = batch_size
        self.sprite_scale = sprite_scale
        self._decoder_cache: dict[str, VideoDecoder] = {}

    def _get_video_decoder(self, path: str) -> VideoDecoder:
        """Get or create video decoder for path."""
        if path in self._decoder_cache:
            return self._decoder_cache[path]

        VideoDecoder, set_cuda_backend = _get_torchcodec_decoders()
        device = self.matcher.device

        if device.type == "cuda":
            # Try CUDA backends in order
            for backend in ["beta", "ffmpeg"]:
                try:
                    with set_cuda_backend(backend):
                        decoder = VideoDecoder(path, device="cuda")
                        self._decoder_cache[path] = decoder
                        return decoder
                except Exception:
                    continue

        # Fallback to CPU
        decoder = VideoDecoder(path, device="cpu")
        self._decoder_cache[path] = decoder
        return decoder

    def _get_frame_count(self, decoder: VideoDecoder) -> int:
        """Get total frame count from decoder."""
        # torchcodec stores metadata
        try:
            return len(decoder)
        except Exception:
            # Fallback: decode and count (slow)
            return decoder.metadata.num_frames

    def scan_video(
        self,
        video_path: str,
        tux_templates: list[tuple[torch.Tensor, str]],
        enemy_templates: list[tuple[torch.Tensor, str]],
        attacked_enemy_templates: list[tuple[torch.Tensor, str]],
        loot_templates: list[tuple[torch.Tensor, str]],
        sprite_threshold: float,
        proximity_px: float,
        show_progress: bool = True,
    ) -> list[dict]:
        """Scan video and return per-frame sprite hit info.

        Same output format as SuperTuxIntentLabeler._scan_video().

        Returns:
            List of dicts with keys: enemy_near, enemy_attacked_near, loot_near
        """
        decoder = self._get_video_decoder(video_path)
        total_frames = self._get_frame_count(decoder)
        proximity = proximity_px * self.sprite_scale

        hits = []
        pbar = tqdm(total=total_frames, desc="Scanning", unit="f", disable=not show_progress)

        for frame_idx in range(total_frames):
            try:
                frame_data = decoder.get_frame_at(frame_idx)
                frame = frame_data.data
            except Exception:
                # Frame decode failed, use empty hit
                hits.append({
                    "enemy_near": False,
                    "enemy_attacked_near": False,
                    "loot_near": False,
                })
                pbar.update(1)
                continue

            # Ensure NCHW format and move to matcher device
            frame = _ensure_nchw_frames(frame)
            if frame.device != self.matcher.device:
                frame = frame.to(self.matcher.device)

            # Convert to grayscale (squeeze batch dim)
            gray = _rgb_to_gray(frame[0])  # (H, W)

            # Resize if needed
            if self.sprite_scale != 1.0:
                h, w = gray.shape
                new_h = max(1, int(h * self.sprite_scale))
                new_w = max(1, int(w * self.sprite_scale))
                gray = F.interpolate(
                    gray.unsqueeze(0).unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            # Find Tux position
            tux_pos = self.matcher.best_match(gray, tux_templates, sprite_threshold)

            # Find enemies, attacked enemies, and loot
            enemy_positions = self.matcher.all_matches(gray, enemy_templates, sprite_threshold)
            attacked_positions = self.matcher.all_matches(gray, attacked_enemy_templates, sprite_threshold)
            loot_positions = self.matcher.all_matches(gray, loot_templates, sprite_threshold)

            # Check proximity
            enemy_near = self._is_near(tux_pos, enemy_positions, proximity)
            enemy_attacked_near = self._is_near(tux_pos, attacked_positions, proximity)
            loot_near = self._is_near(tux_pos, loot_positions, proximity)

            hits.append({
                "enemy_near": enemy_near,
                "enemy_attacked_near": enemy_attacked_near,
                "loot_near": loot_near,
            })
            pbar.update(1)

        pbar.close()
        return hits

    def detect_events(
        self,
        video_path: str,
        tux_templates: list[tuple[torch.Tensor, str]],
        death_templates: list[tuple[torch.Tensor, str]],
        attacked_templates: list[tuple[torch.Tensor, str]],
        sparkle_templates: list[tuple[torch.Tensor, str]],
        death_threshold: float,
        attack_threshold: float,
        sparkle_threshold: float,
        win_proximity_px: float,
        show_progress: bool = True,
    ) -> list[dict]:
        """Detect death, attack, and win events in video.

        Returns:
            List of dicts with keys: death_conf, attack_conf, win_conf for each frame
        """
        decoder = self._get_video_decoder(video_path)
        total_frames = self._get_frame_count(decoder)
        proximity = win_proximity_px * self.sprite_scale

        results = []
        pbar = tqdm(total=total_frames, desc="Detecting", unit="f", disable=not show_progress)

        for frame_idx in range(total_frames):
            try:
                frame_data = decoder.get_frame_at(frame_idx)
                frame = frame_data.data
            except Exception:
                results.append({"death_conf": 0.0, "attack_conf": 0.0, "win_conf": 0.0})
                pbar.update(1)
                continue

            frame = _ensure_nchw_frames(frame)
            if frame.device != self.matcher.device:
                frame = frame.to(self.matcher.device)

            gray = _rgb_to_gray(frame[0])

            if self.sprite_scale != 1.0:
                h, w = gray.shape
                new_h = max(1, int(h * self.sprite_scale))
                new_w = max(1, int(w * self.sprite_scale))
                gray = F.interpolate(
                    gray.unsqueeze(0).unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            # Check death (tux gameover sprite)
            death_conf = self._max_template_conf(gray, death_templates, death_threshold)

            # Check attack (squashed/killed enemy sprites)
            attack_conf = self._max_template_conf(gray, attacked_templates, attack_threshold)

            # Check win (tux near sparkle)
            win_conf = self._check_win(
                gray, tux_templates, sparkle_templates, sparkle_threshold, proximity
            )

            results.append({
                "death_conf": death_conf,
                "attack_conf": attack_conf,
                "win_conf": win_conf,
            })
            pbar.update(1)

        pbar.close()
        return results

    def _max_template_conf(
        self,
        gray: torch.Tensor,
        templates: list[tuple[torch.Tensor, str]],
        threshold: float,
    ) -> float:
        """Get max template match confidence above threshold."""
        best_conf = 0.0
        for tmpl, _ in templates:
            if tmpl.shape[0] > gray.shape[0] or tmpl.shape[1] > gray.shape[1]:
                continue
            ncc = self.matcher.match_template_ncc(gray, tmpl)
            if ncc.numel() > 0:
                conf = ncc.max().item()
                if conf >= threshold:
                    best_conf = max(best_conf, conf)
        return best_conf

    def _check_win(
        self,
        gray: torch.Tensor,
        tux_templates: list[tuple[torch.Tensor, str]],
        sparkle_templates: list[tuple[torch.Tensor, str]],
        threshold: float,
        proximity: float,
    ) -> float:
        """Check win condition (tux near sparkle)."""
        tux_pos = self.matcher.best_match(gray, tux_templates, threshold)
        if tux_pos is None:
            return 0.0

        sparkle_positions = self.matcher.all_matches(gray, sparkle_templates, threshold)
        if not sparkle_positions:
            return 0.0

        for sparkle_pos in sparkle_positions:
            if self._distance(tux_pos, sparkle_pos) <= proximity:
                return 1.0
        return 0.0

    @staticmethod
    def _is_near(
        tux_pos: tuple[float, float] | None,
        positions: list[tuple[float, float]],
        proximity: float,
    ) -> bool:
        """Check if tux is near any position."""
        if tux_pos is None or not positions:
            return False
        for pos in positions:
            if GPUVideoScanner._distance(tux_pos, pos) <= proximity:
                return True
        return False

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
