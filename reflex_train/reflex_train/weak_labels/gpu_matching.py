"""GPU-accelerated template matching using FFT-based normalized cross-correlation.

Provides TM_CCOEFF_NORMED equivalent matching on CUDA tensors, with CPU fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ContextManager, Iterable, cast

from PIL.Image import Image as PILImage

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
    """GPU-accelerated template matching using conv2d-based NCC.

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
        self._template_stats_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._template_mask_cache: dict[int, torch.Tensor] = {}
        self._ones_kernel_cache: dict[tuple[int, int], torch.Tensor] = {}

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

        img = Image.open(path).convert("RGBA")
        img = cast(PILImage, img)
        alpha = img.getchannel("A")
        gray = img.convert("RGB").convert("L")
        if scale != 1.0:
            new_size = (max(1, int(gray.width * scale)), max(1, int(gray.height * scale)))
            gray = gray.resize(new_size, Image.Resampling.LANCZOS)
            alpha = alpha.resize(new_size, Image.Resampling.LANCZOS)

        gray_tensor = torch.tensor(list(cast(Iterable[int], gray.getdata())), dtype=self.dtype, device=self.device)
        alpha_tensor = torch.tensor(list(cast(Iterable[int], alpha.getdata())), dtype=self.dtype, device=self.device)
        gray_tensor = gray_tensor.reshape(gray.height, gray.width)
        alpha_tensor = alpha_tensor.reshape(gray.height, gray.width) / 255.0
        tensor = gray_tensor * alpha_tensor
        mask = (alpha_tensor > 0.0).to(self.dtype)
        self._template_cache[cache_key] = tensor
        self._template_mask_cache[id(tensor)] = mask
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

    def _get_ones_kernel(self, h: int, w: int) -> torch.Tensor:
        """Get cached ones kernel for local sums."""
        key = (h, w)
        if key not in self._ones_kernel_cache:
            self._ones_kernel_cache[key] = torch.ones(
                1, 1, h, w, device=self.device, dtype=self.dtype
            )
        return self._ones_kernel_cache[key]

    def _get_template_stats(
        self, template: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cached template stats (centered, std, sum, mask)."""
        key = id(template)
        if key not in self._template_stats_cache:
            mask = self._template_mask_cache.get(key)
            if mask is None:
                mask = (template != 0.0).to(self.dtype)
                self._template_mask_cache[key] = mask
            t_sum = (template * mask).sum()
            mask_sum = mask.sum().clamp(min=1.0)
            t_mean = t_sum / mask_sum
            t_centered = (template - t_mean) * mask
            t_var = t_centered.pow(2).sum()
            t_std = t_var.sqrt()
            self._template_stats_cache[key] = (t_centered, t_std, t_sum, mask)
        return self._template_stats_cache[key]

    def match_template_ncc(
        self,
        frame: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """conv2d-based normalized cross-correlation.

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

        t_centered, t_std, t_sum, mask = self._get_template_stats(template)
        if t_std.item() < 1e-7:
            return torch.zeros(H - h + 1, W - w + 1, device=self.device, dtype=self.dtype)

        frame_4d = frame.unsqueeze(0).unsqueeze(0)
        template_kernel = t_centered.unsqueeze(0).unsqueeze(0)

        correlation = F.conv2d(frame_4d, template_kernel)

        mask_kernel = mask.unsqueeze(0).unsqueeze(0)
        local_mask_sum = F.conv2d(frame_4d.new_ones((1, 1, H, W)), mask_kernel)
        local_mask_sum = local_mask_sum.clamp(min=1.0)

        local_sum = F.conv2d(frame_4d, mask_kernel)
        local_sum_sq = F.conv2d(frame_4d * frame_4d, mask_kernel)

        local_mean = local_sum / local_mask_sum
        local_var = local_sum_sq / local_mask_sum - local_mean.pow(2)
        local_var = local_var.clamp(min=1e-7)
        local_std = (local_var * local_mask_sum).sqrt()

        ncc = (correlation - local_mean * t_sum) / (local_std * t_std)
        result = ncc.squeeze(0).squeeze(0)
        return result.clamp(-1.0, 1.0)

    def _pad_templates(
        self, templates: list[tuple[torch.Tensor, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[str]]:
        """Pad templates/masks to common size and stack as conv2d kernels."""
        if not templates:
            return (
                torch.empty(0, 1, 1, 1, device=self.device, dtype=self.dtype),
                torch.empty(0, 1, 1, 1, device=self.device, dtype=self.dtype),
                [],
                [],
            )

        sizes = [(tmpl.shape[0], tmpl.shape[1]) for tmpl, _ in templates]
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)

        kernels: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        paths: list[str] = []
        for tmpl, path in templates:
            t_centered, _t_std, _t_sum, mask = self._get_template_stats(tmpl)
            pad_h = max_h - tmpl.shape[0]
            pad_w = max_w - tmpl.shape[1]
            t_padded = F.pad(t_centered, (0, pad_w, 0, pad_h))
            m_padded = F.pad(mask, (0, pad_w, 0, pad_h))
            kernels.append(t_padded.unsqueeze(0))
            masks.append(m_padded.unsqueeze(0))
            paths.append(path)

        kernel_stack = torch.stack(kernels, dim=0)
        mask_stack = torch.stack(masks, dim=0)
        return kernel_stack, mask_stack, sizes, paths

    def match_templates_ncc_batch(
        self,
        frame: torch.Tensor,
        templates: list[tuple[torch.Tensor, str]],
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[str]]:
        """Batch NCC for templates padded to common size.

        Returns:
            (N, H-h+1, W-w+1) NCC maps, sizes per template, paths.
        """
        H, W = frame.shape
        kernel_stack, mask_stack, sizes, paths = self._pad_templates(templates)
        if kernel_stack.numel() == 0:
            return (
                torch.empty(0, 0, 0, device=self.device, dtype=self.dtype),
                [],
                [],
            )

        max_h = kernel_stack.shape[-2]
        max_w = kernel_stack.shape[-1]
        if max_h > H or max_w > W:
            return (
                torch.empty(0, 0, 0, device=self.device, dtype=self.dtype),
                sizes,
                paths,
            )

        frame_4d = frame.unsqueeze(0).unsqueeze(0)
        correlation = F.conv2d(frame_4d, kernel_stack)

        ones = frame_4d.new_ones((1, 1, H, W))
        local_mask_sum = F.conv2d(ones, mask_stack).clamp(min=1.0)
        local_sum = F.conv2d(frame_4d, mask_stack)
        local_sum_sq = F.conv2d(frame_4d * frame_4d, mask_stack)

        local_mean = local_sum / local_mask_sum
        local_var = local_sum_sq / local_mask_sum - local_mean.pow(2)
        local_var = local_var.clamp(min=1e-7)
        local_std = (local_var * local_mask_sum).sqrt()

        t_sum = kernel_stack.sum(dim=(2, 3), keepdim=True)
        t_std = kernel_stack.pow(2).sum(dim=(2, 3), keepdim=True).sqrt().clamp(min=1e-7)

        ncc = (correlation - local_mean * t_sum) / (local_std * t_std)
        return ncc.squeeze(0).clamp(-1.0, 1.0), sizes, paths



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
        ncc_maps, sizes, _paths = self.match_templates_ncc_batch(frame, templates)
        if ncc_maps.numel() == 0:
            return None

        best_pos = None
        best_score = threshold
        for idx, (h, w) in enumerate(sizes):
            ncc = ncc_maps[idx]
            max_val = ncc.max().item()
            if max_val >= best_score:
                max_idx = ncc.argmax()
                max_y = (max_idx // ncc.shape[1]).item()
                max_x = (max_idx % ncc.shape[1]).item()
                best_score = max_val
                best_pos = (max_x + w / 2, max_y + h / 2)

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
        positions: list[tuple[float, float]] = []
        ncc_maps, sizes, _paths = self.match_templates_ncc_batch(frame, templates)
        if ncc_maps.numel() == 0:
            return positions

        for idx, (h, w) in enumerate(sizes):
            ncc = ncc_maps[idx]
            max_val = ncc.max().item()
            if max_val >= threshold:
                max_idx = ncc.argmax()
                max_y = (max_idx // ncc.shape[1]).item()
                max_x = (max_idx % ncc.shape[1]).item()
                positions.append((max_x + w / 2, max_y + h / 2))

        return positions

    def best_match_cached(
        self,
        frame: torch.Tensor,
        templates: list[tuple[torch.Tensor, str]],
        threshold: float,
        early_exit: float = 0.0,
    ) -> tuple[float, float] | None:
        """Find best match using conv2d NCC.

        Args:
            early_exit: If > 0, return immediately when score exceeds this value.
        """
        ncc_maps, sizes, _paths = self.match_templates_ncc_batch(frame, templates)
        if ncc_maps.numel() == 0:
            return None

        best_pos = None
        best_score = threshold
        for idx, (h, w) in enumerate(sizes):
            ncc = ncc_maps[idx]
            max_val = ncc.max().item()
            if max_val >= best_score:
                max_idx = ncc.argmax()
                max_y = (max_idx // ncc.shape[1]).item()
                max_x = (max_idx % ncc.shape[1]).item()
                best_score = max_val
                best_pos = (max_x + w / 2, max_y + h / 2)
                if early_exit > 0 and best_score >= early_exit:
                    return best_pos

        return best_pos

    def all_matches_cached(
        self,
        frame: torch.Tensor,
        templates: list[tuple[torch.Tensor, str]],
        threshold: float,
    ) -> list[tuple[float, float]]:
        """Find all matches using conv2d NCC."""
        positions: list[tuple[float, float]] = []
        ncc_maps, sizes, _paths = self.match_templates_ncc_batch(frame, templates)
        if ncc_maps.numel() == 0:
            return positions

        for idx, (h, w) in enumerate(sizes):
            ncc = ncc_maps[idx]
            max_val = ncc.max().item()
            if max_val >= threshold:
                max_idx = ncc.argmax()
                max_y = (max_idx // ncc.shape[1]).item()
                max_x = (max_idx % ncc.shape[1]).item()
                positions.append((max_x + w / 2, max_y + h / 2))

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
        blank_frame_mean_threshold: float | None = None,
        blank_frame_std_threshold: float | None = None,
    ):
        self.matcher = matcher or GPUTemplateMatcher()
        self.batch_size = batch_size
        self.sprite_scale = sprite_scale
        self.blank_frame_mean_threshold = blank_frame_mean_threshold
        self.blank_frame_std_threshold = blank_frame_std_threshold
        self._decoder_cache: dict[str, VideoDecoder] = {}

    def _get_video_decoder(self, path: str) -> VideoDecoder:
        """Get or create video decoder for path."""
        if path in self._decoder_cache:
            return self._decoder_cache[path]

        VideoDecoder, set_cuda_backend = _get_torchcodec_decoders()
        device = self.matcher.device

        decoder_cls = cast(Callable[..., VideoDecoder], VideoDecoder)
        if device.type == "cuda":
            set_cuda_backend = cast(Callable[[str], ContextManager[None]], set_cuda_backend)
            for backend in ["beta", "ffmpeg"]:
                try:
                    with set_cuda_backend(backend):
                        decoder = decoder_cls(path, device="cuda")
                        self._decoder_cache[path] = decoder
                        return decoder
                except Exception:
                    continue

        # Fallback to CPU
        decoder = decoder_cls(path, device="cpu")
        self._decoder_cache[path] = decoder
        return decoder

    def _get_frame_count(self, decoder: VideoDecoder) -> int:
        """Get total frame count from decoder."""
        # torchcodec stores metadata
        try:
            return len(decoder)
        except Exception:
            # Fallback: decode and count (slow)
            num_frames = cast(int | None, decoder.metadata.num_frames)
            if num_frames is None:
                return 0
            return int(num_frames)

    def _is_near(
        self,
        origin: tuple[float, float] | None,
        targets: list[tuple[float, float]],
        proximity: float,
    ) -> bool:
        if origin is None:
            return False
        ox, oy = origin
        for tx, ty in targets:
            if abs(tx - ox) <= proximity and abs(ty - oy) <= proximity:
                return True
        return False

    def scan_video(
        self,
        video_path: str,
        tux_templates: list[tuple[torch.Tensor, str]],
        enemy_templates: list[tuple[torch.Tensor, str]],
        attacked_enemy_templates: list[tuple[torch.Tensor, str]],
        loot_templates: list[tuple[torch.Tensor, str]],
        sprite_threshold: float,
        proximity_px: float,
        frame_stride: int = 1,
        show_progress: bool = True,
    ) -> list[dict]:
        """Scan video and return per-frame sprite hit info.

        Same output format as the weak-label sprite scan.

        Returns:
            List of dicts with keys: enemy_near, enemy_attacked_near, loot_near
        """
        decoder = self._get_video_decoder(video_path)
        total_frames = self._get_frame_count(decoder)
        proximity = proximity_px * self.sprite_scale

        hits: list[dict] = []
        stride = max(1, frame_stride)
        pbar = tqdm(total=total_frames, desc="Scanning", unit="f", disable=not show_progress)

        # Process in batches
        batch_size = self.batch_size
        prev_idx = None
        prev_hit = None
        for batch_start in range(0, total_frames, batch_size * stride):
            batch_end = min(batch_start + batch_size * stride, total_frames)
            indices = list(range(batch_start, batch_end, stride))
            actual_batch_size = len(indices)

            try:
                batch_data = decoder.get_frames_at(indices=indices)
                frames = _ensure_nchw_frames(batch_data.data)
            except Exception:
                for idx in indices:
                    hit = {
                        "enemy_near": False,
                        "enemy_attacked_near": False,
                        "loot_near": False,
                    }
                    if prev_hit is not None and prev_idx is not None:
                        span = max(0, idx - prev_idx)
                        if span:
                            hits.extend([prev_hit] * span)
                            pbar.update(span)
                    prev_idx = idx
                    prev_hit = hit
                continue

            if frames.device != self.matcher.device:
                frames = frames.to(self.matcher.device)

            # Batch grayscale conversion
            grays = _rgb_to_gray(frames)

            # Batch resize if needed
            if self.sprite_scale != 1.0:
                _, h, w = grays.shape
                new_h = max(1, int(h * self.sprite_scale))
                new_w = max(1, int(w * self.sprite_scale))
                grays = F.interpolate(
                    grays.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            # Process each frame in batch
            for i in range(actual_batch_size):
                gray = grays[i]
                idx = indices[i]

                if self._is_blank_frame(gray):
                    hit = {
                        "enemy_near": False,
                        "enemy_attacked_near": False,
                        "loot_near": False,
                    }
                    if prev_hit is not None and prev_idx is not None:
                        span = max(0, idx - prev_idx)
                        if span:
                            hits.extend([prev_hit] * span)
                            pbar.update(span)
                    prev_idx = idx
                    prev_hit = hit
                    continue

                # Find Tux position (early exit - just need to find, not best match)
                tux_pos = self.matcher.best_match_cached(
                    gray,
                    tux_templates,
                    sprite_threshold,
                    early_exit=0.0,
                )

                # Find enemies, attacked enemies, and loot
                enemy_positions = self.matcher.all_matches_cached(gray, enemy_templates, sprite_threshold)
                attacked_positions = self.matcher.all_matches_cached(gray, attacked_enemy_templates, sprite_threshold)
                loot_positions = self.matcher.all_matches_cached(gray, loot_templates, sprite_threshold)

                # Check proximity
                enemy_near = self._is_near(tux_pos, enemy_positions, proximity)
                enemy_attacked_near = self._is_near(tux_pos, attacked_positions, proximity)
                loot_near = self._is_near(tux_pos, loot_positions, proximity)

                hit = {
                    "enemy_near": enemy_near,
                    "enemy_attacked_near": enemy_attacked_near,
                    "loot_near": loot_near,
                }
                if prev_hit is not None and prev_idx is not None:
                    span = max(0, idx - prev_idx)
                    if span:
                        hits.extend([prev_hit] * span)
                        pbar.update(span)
                prev_idx = idx
                prev_hit = hit

        if prev_hit is not None and prev_idx is not None:
            span = max(0, total_frames - prev_idx)
            if span:
                hits.extend([prev_hit] * span)
                pbar.update(span)

        pbar.close()
        return hits

    def detect_events(
        self,
        video_path: str,
        death_templates: list[tuple[torch.Tensor, str]],
        attacked_templates: list[tuple[torch.Tensor, str]],
        death_threshold: float,
        attack_threshold: float,
        frame_stride: int = 1,
        show_progress: bool = True,
    ) -> list[dict]:
        """Detect death and attack events in video.

        Returns:
            List of dicts with keys: death_conf, attack_conf for each frame
        """
        decoder = self._get_video_decoder(video_path)
        total_frames = self._get_frame_count(decoder)

        results: list[dict] = []
        stride = max(1, frame_stride)
        pbar = tqdm(total=total_frames, desc="Detecting", unit="f", disable=not show_progress)

        # Process in batches
        batch_size = self.batch_size
        prev_idx = None
        prev_result = None
        for batch_start in range(0, total_frames, batch_size * stride):
            batch_end = min(batch_start + batch_size * stride, total_frames)
            indices = list(range(batch_start, batch_end, stride))
            actual_batch_size = len(indices)

            try:
                batch_data = decoder.get_frames_at(indices=indices)
                frames = _ensure_nchw_frames(batch_data.data)
            except Exception:
                # Batch decode failed, fall back to per-frame
                for idx in indices:
                    result = {"death_conf": 0.0, "attack_conf": 0.0}
                    if prev_result is not None and prev_idx is not None:
                        span = max(0, idx - prev_idx)
                        if span:
                            results.extend([prev_result] * span)
                            pbar.update(span)
                    prev_idx = idx
                    prev_result = result
                continue

            if frames.device != self.matcher.device:
                frames = frames.to(self.matcher.device)

            # Batch grayscale conversion: (N, C, H, W) -> (N, H, W)
            grays = _rgb_to_gray(frames)

            # Batch resize if needed
            if self.sprite_scale != 1.0:
                _, h, w = grays.shape
                new_h = max(1, int(h * self.sprite_scale))
                new_w = max(1, int(w * self.sprite_scale))
                grays = F.interpolate(
                    grays.unsqueeze(1),  # (N, 1, H, W)
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # (N, new_h, new_w)

            # Process each frame in batch
            for i in range(actual_batch_size):
                gray = grays[i]
                idx = indices[i]

                if self._is_blank_frame(gray):
                    result = {"death_conf": 0.0, "attack_conf": 0.0}
                    if prev_result is not None and prev_idx is not None:
                        span = max(0, idx - prev_idx)
                        if span:
                            results.extend([prev_result] * span)
                            pbar.update(span)
                    prev_idx = idx
                    prev_result = result
                    continue

                # Check death (tux gameover sprite)
                death_conf = self._max_template_conf(gray, death_templates, death_threshold)

                # Check attack (squashed/killed enemy sprites)
                attack_conf = self._max_template_conf(gray, attacked_templates, attack_threshold)

                result = {
                    "death_conf": death_conf,
                    "attack_conf": attack_conf,
                }
                if prev_result is not None and prev_idx is not None:
                    span = max(0, idx - prev_idx)
                    if span:
                        results.extend([prev_result] * span)
                        pbar.update(span)
                prev_idx = idx
                prev_result = result

        if prev_result is not None and prev_idx is not None:
            span = max(0, total_frames - prev_idx)
            if span:
                results.extend([prev_result] * span)
                pbar.update(span)

        pbar.close()
        return results

    def _max_template_conf(
        self,
        gray: torch.Tensor,
        templates: list[tuple[torch.Tensor, str]],
        threshold: float,
    ) -> float:
        """Get max template match confidence using conv2d NCC."""
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

    def _is_blank_frame(self, gray: torch.Tensor) -> bool:
        if self.blank_frame_mean_threshold is None and self.blank_frame_std_threshold is None:
            return False
        mean_val = gray.mean().item()
        std_val = gray.std().item()
        mean_ok = self.blank_frame_mean_threshold is None or mean_val <= self.blank_frame_mean_threshold
        std_ok = self.blank_frame_std_threshold is None or std_val <= self.blank_frame_std_threshold
        return mean_ok and std_ok
