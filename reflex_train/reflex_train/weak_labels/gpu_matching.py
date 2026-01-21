"""GPU-accelerated template matching using cosine similarity.

Uses a simplified matching approach: templates are L2-normalized at load time,
and frame patches are normalized via local sum-of-squares. This gives cosine
similarity with just 2 conv2d calls per batch instead of 4 for full NCC.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _rgb_to_gray(rgb: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert RGB tensor to grayscale.

    Uses ITU-R BT.601 weights: 0.299*R + 0.587*G + 0.114*B

    Args:
        rgb: (C, H, W) or (B, C, H, W) with C=3
        dtype: Output dtype for grayscale tensor

    Returns:
        (H, W) or (B, H, W) float grayscale tensor
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=dtype)
    rgb_f = rgb.to(dtype)
    if rgb.ndim == 4:
        return torch.einsum("bchw,c->bhw", rgb_f, weights)
    return torch.einsum("chw,c->hw", rgb_f, weights)


@dataclass
class TemplateBatch:
    kernels: torch.Tensor  # L2-normalized, zero-padded templates
    masks: torch.Tensor  # Binary masks for valid pixels
    sizes: list[tuple[int, int]]
    paths: list[str]
    max_h: int
    max_w: int


@dataclass
class TemplateBatches:
    batches: list[TemplateBatch]
    sizes: list[tuple[int, int]]
    paths: list[str]


class GPUTemplateMatcher:
    """GPU-accelerated template matching using batched conv2d NCC."""

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        max_templates_per_batch: int | None = 128,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_templates_per_batch = max_templates_per_batch
        self._template_cache: dict[str, torch.Tensor] = {}
        self._template_mask_cache: dict[int, torch.Tensor] = {}

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

    def _build_batch(self, templates: list[tuple[torch.Tensor, str]]) -> TemplateBatch:
        """Pad templates/masks to common size and L2-normalize."""
        if not templates:
            empty = torch.empty(0, 1, 1, 1, device=self.device, dtype=self.dtype)
            return TemplateBatch(
                kernels=empty,
                masks=empty,
                sizes=[],
                paths=[],
                max_h=0,
                max_w=0,
            )

        sizes = [(tmpl.shape[0], tmpl.shape[1]) for tmpl, _ in templates]
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)

        kernels: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        paths: list[str] = []
        for tmpl, path in templates:
            key = id(tmpl)
            mask = self._template_mask_cache.get(key)
            if mask is None:
                mask = (tmpl != 0.0).to(self.dtype)
                self._template_mask_cache[key] = mask
            
            # L2-normalize the template (only over masked pixels)
            t_masked = tmpl * mask
            t_norm = t_masked.pow(2).sum().sqrt().clamp(min=1e-7)
            t_normalized = t_masked / t_norm
            
            pad_h = max_h - tmpl.shape[0]
            pad_w = max_w - tmpl.shape[1]
            t_padded = F.pad(t_normalized, (0, pad_w, 0, pad_h))
            m_padded = F.pad(mask, (0, pad_w, 0, pad_h))
            kernels.append(t_padded.unsqueeze(0))
            masks.append(m_padded.unsqueeze(0))
            paths.append(path)

        kernel_stack = torch.stack(kernels, dim=0)
        mask_stack = torch.stack(masks, dim=0)

        return TemplateBatch(
            kernels=kernel_stack,
            masks=mask_stack,
            sizes=sizes,
            paths=paths,
            max_h=max_h,
            max_w=max_w,
        )

    def build_batches(self, templates: list[tuple[torch.Tensor, str]]) -> TemplateBatches:
        """Build size-grouped template batches with optional chunking."""
        if not templates:
            return TemplateBatches(batches=[], sizes=[], paths=[])

        grouped: dict[tuple[int, int], list[tuple[torch.Tensor, str]]] = {}
        for tmpl, path in templates:
            grouped.setdefault((tmpl.shape[0], tmpl.shape[1]), []).append((tmpl, path))

        max_per_batch = self.max_templates_per_batch
        batches: list[TemplateBatch] = []
        sizes: list[tuple[int, int]] = []
        paths: list[str] = []
        for group in grouped.values():
            if max_per_batch is None or max_per_batch <= 0:
                chunked = [group]
            else:
                chunked = [group[i : i + max_per_batch] for i in range(0, len(group), max_per_batch)]
            for chunk in chunked:
                batch = self._build_batch(chunk)
                if batch.kernels.numel() == 0:
                    continue
                batches.append(batch)
                sizes.extend(batch.sizes)
                paths.extend(batch.paths)

        return TemplateBatches(batches=batches, sizes=sizes, paths=paths)

    def match_batch(self, frame: torch.Tensor, batch: TemplateBatch) -> torch.Tensor:
        """Batch cosine similarity matching.

        Templates are pre-normalized; we compute local L2 norm of frame patches
        and divide correlation by it to get cosine similarity.

        Returns:
            (N, H-h+1, W-w+1) similarity maps in [-1, 1].
        """
        if batch.kernels.numel() == 0:
            return torch.empty(0, 0, 0, device=self.device, dtype=self.dtype)

        H, W = frame.shape
        if batch.max_h > H or batch.max_w > W:
            return torch.empty(0, 0, 0, device=self.device, dtype=self.dtype)

        frame_4d = frame.unsqueeze(0).unsqueeze(0)
        
        # Correlation with L2-normalized templates
        correlation = F.conv2d(frame_4d, batch.kernels)
        
        # Local L2 norm of frame patches (sum of squares under each mask)
        frame_sq = frame_4d * frame_4d
        local_sum_sq = F.conv2d(frame_sq, batch.masks)
        local_norm = local_sum_sq.sqrt().clamp(min=1e-7)
        
        # Cosine similarity = correlation / local_norm (templates already normalized)
        similarity = correlation / local_norm
        
        similarity = similarity.squeeze(0)
        if similarity.dim() == 2:
            similarity = similarity.unsqueeze(0)
        return similarity.clamp(-1.0, 1.0)

    def match_batches(self, frame: torch.Tensor, batches: TemplateBatches) -> torch.Tensor:
        """Match multiple template batches and concat results."""
        if not batches.batches:
            return torch.empty(0, 0, 0, device=self.device, dtype=self.dtype)

        maps: list[torch.Tensor] = []
        max_h = 0
        max_w = 0
        for batch in batches.batches:
            max_h = max(max_h, batch.max_h)
            max_w = max(max_w, batch.max_w)

        H, W = frame.shape
        out_h = max(0, H - max_h + 1)
        out_w = max(0, W - max_w + 1)

        for batch in batches.batches:
            ncc = self.match_batch(frame, batch)
            if ncc.numel() == 0:
                continue
            if ncc.shape[1] != out_h or ncc.shape[2] != out_w:
                ncc = ncc[:, :out_h, :out_w]
            maps.append(ncc)

        if not maps:
            return torch.empty(0, 0, 0, device=self.device, dtype=self.dtype)

        dims = {m.dim() for m in maps}
        if dims != {3}:
            shapes = [tuple(m.shape) for m in maps]
            raise RuntimeError(f"match_batches expects 3D NCC maps; got shapes={shapes}")

        heights = {m.shape[1] for m in maps}
        widths = {m.shape[2] for m in maps}
        if len(heights) != 1 or len(widths) != 1:
            shapes = [tuple(m.shape) for m in maps]
            raise RuntimeError(
                "match_batches expects same spatial size across batches after crop; "
                f"got shapes={shapes}"
            )

        return torch.cat(maps, dim=0)



    def best_match(
        self,
        frame: torch.Tensor,
        batches: TemplateBatches,
        threshold: float,
    ) -> tuple[float, float] | None:
        """Find best matching template location above threshold."""
        ncc_maps = self.match_batches(frame, batches)
        if ncc_maps.numel() == 0:
            return None

        best_pos = None
        best_score = threshold
        for idx, (h, w) in enumerate(batches.sizes):
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
        batches: TemplateBatches,
        threshold: float,
    ) -> list[tuple[float, float]]:
        """Find all template matches above threshold."""
        positions: list[tuple[float, float]] = []
        ncc_maps = self.match_batches(frame, batches)
        if ncc_maps.numel() == 0:
            return positions

        for idx, (h, w) in enumerate(batches.sizes):
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

    # Fixed scale factor for frame resizing (balances speed vs accuracy)
    _FRAME_SCALE: float = 0.5

    def __init__(
        self,
        matcher: GPUTemplateMatcher | None = None,
        batch_size: int = 16,
        blank_frame_mean_threshold: float | None = None,
        blank_frame_std_threshold: float | None = None,
        max_templates_per_batch: int | None = 128,
    ):
        self.matcher = matcher or GPUTemplateMatcher(
            max_templates_per_batch=max_templates_per_batch
        )
        self.batch_size = batch_size
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
        proximity = proximity_px * self._FRAME_SCALE

        tux_batches = self.matcher.build_batches(tux_templates)
        enemy_batches = self.matcher.build_batches(enemy_templates)
        attacked_batches = self.matcher.build_batches(attacked_enemy_templates)
        loot_batches = self.matcher.build_batches(loot_templates)

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
            grays = _rgb_to_gray(frames, dtype=self.matcher.dtype)

            # Resize frames to match template scale
            _, h, w = grays.shape
            new_h = max(1, int(h * self._FRAME_SCALE))
            new_w = max(1, int(w * self._FRAME_SCALE))
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
                tux_pos = self.matcher.best_match(
                    gray,
                    tux_batches,
                    sprite_threshold,
                )

                # Find enemies, attacked enemies, and loot
                enemy_positions = self.matcher.all_matches(gray, enemy_batches, sprite_threshold)
                attacked_positions = self.matcher.all_matches(gray, attacked_batches, sprite_threshold)
                loot_positions = self.matcher.all_matches(gray, loot_batches, sprite_threshold)

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

        death_batches = self.matcher.build_batches(death_templates)
        attacked_batches = self.matcher.build_batches(attacked_templates)

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
            grays = _rgb_to_gray(frames, dtype=self.matcher.dtype)

            # Resize frames to match template scale
            _, h, w = grays.shape
            new_h = max(1, int(h * self._FRAME_SCALE))
            new_w = max(1, int(w * self._FRAME_SCALE))
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
                death_conf = self._max_template_conf(gray, death_batches, death_threshold)

                # Check attack (squashed/killed enemy sprites)
                attack_conf = self._max_template_conf(gray, attacked_batches, attack_threshold)

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
        batches: TemplateBatches,
        threshold: float,
    ) -> float:
        """Get max template match confidence using conv2d NCC."""
        ncc_maps = self.matcher.match_batches(gray, batches)
        if ncc_maps.numel() == 0:
            return 0.0

        best_conf = 0.0
        for ncc in ncc_maps:
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
