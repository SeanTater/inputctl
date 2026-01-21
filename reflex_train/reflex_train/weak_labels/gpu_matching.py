"""GPU-accelerated template matching using cosine similarity.

All templates are padded to a fixed 128x128 size, enabling a single batched
conv2d call for all templates. This is optimized for bootstrapping weak labels,
not for production use.
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

from reflex_train.data.dataset import (
    _get_torchcodec_decoders,
    _ensure_nchw_frames,
)


# Fixed sizes for all templates and frame scaling
_TEMPLATE_SIZE = 96
_FRAME_SCALE = 0.25


def _rgb_to_gray(rgb: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert RGB tensor to grayscale using ITU-R BT.601 weights.
    
    Returns values normalized to [0, 1] to avoid fp16 overflow in downstream ops.
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=torch.float32)
    # Convert to float32 first, normalize, then convert to target dtype
    # This avoids precision issues with uint8 -> fp16 -> divide
    rgb_f = rgb.float() / 255.0
    if rgb.ndim == 4:
        gray = torch.einsum("bchw,c->bhw", rgb_f, weights)
    else:
        gray = torch.einsum("chw,c->hw", rgb_f, weights)
    return gray.to(dtype)


@dataclass
class TemplateBatch:
    """All templates in a single batch, padded to _TEMPLATE_SIZE x _TEMPLATE_SIZE."""
    kernels: torch.Tensor  # (N, 1, 128, 128) L2-normalized templates
    masks: torch.Tensor    # (N, 1, 128, 128) binary masks
    n_templates: int


class GPUTemplateMatcher:
    """GPU-accelerated template matching using cosine similarity."""

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self._template_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _load_template(self, path: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load and preprocess a single template.
        
        Returns:
            (kernel, mask) tuple, both padded to _TEMPLATE_SIZE, or None if too large.
        """
        if path in self._template_cache:
            return self._template_cache[path]

        img = Image.open(path).convert("RGBA")
        img = cast(PILImage, img)
        alpha = img.getchannel("A")
        gray = img.convert("RGB").convert("L")
        
        # Apply frame scale to templates so they match scaled frames
        if _FRAME_SCALE != 1.0:
            new_size = (max(1, int(gray.width * _FRAME_SCALE)), max(1, int(gray.height * _FRAME_SCALE)))
            gray = gray.resize(new_size, Image.Resampling.LANCZOS)
            alpha = alpha.resize(new_size, Image.Resampling.LANCZOS)

        # Skip templates larger than fixed size
        if gray.height > _TEMPLATE_SIZE or gray.width > _TEMPLATE_SIZE:
            return None

        gray_tensor = torch.tensor(list(cast(Iterable[int], gray.getdata())), dtype=torch.float32, device=self.device)
        alpha_tensor = torch.tensor(list(cast(Iterable[int], alpha.getdata())), dtype=torch.float32, device=self.device)
        gray_tensor = gray_tensor.reshape(gray.height, gray.width) / 255.0
        alpha_tensor = alpha_tensor.reshape(gray.height, gray.width) / 255.0
        
        # Binary mask: only pixels with alpha > 0.5 contribute
        mask = (alpha_tensor > 0.5).to(self.dtype)
        
        # Zero out transparent pixels (gray already in [0,1])
        tensor = (gray_tensor * mask).to(self.dtype)
        t_norm = tensor.pow(2).sum().sqrt().clamp(min=1e-7)
        tensor = tensor / t_norm
        
        # Pad to fixed size
        pad_h = _TEMPLATE_SIZE - gray.height
        pad_w = _TEMPLATE_SIZE - gray.width
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        mask = F.pad(mask, (0, pad_w, 0, pad_h))
        
        self._template_cache[path] = (tensor, mask)
        return tensor, mask

    def build_batch(self, paths: list[str]) -> TemplateBatch:
        """Load templates and build a single batch."""
        kernels: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        loaded_paths: list[str] = []
        
        for path in paths:
            try:
                result = self._load_template(path)
                if result is None:
                    print(f"  Skipped (too large): {path}")
                    continue
                kernel, mask = result
                kernels.append(kernel.unsqueeze(0))
                masks.append(mask.unsqueeze(0))
                loaded_paths.append(path)
            except Exception as e:
                print(f"  Failed to load {path}: {e}")
                continue
        
        if not kernels:
            empty = torch.empty(0, 1, _TEMPLATE_SIZE, _TEMPLATE_SIZE, device=self.device, dtype=self.dtype)
            return TemplateBatch(kernels=empty, masks=empty, n_templates=0)
        
        print(f"  Loaded {len(kernels)} templates: {loaded_paths}")
        return TemplateBatch(
            kernels=torch.stack(kernels, dim=0),
            masks=torch.stack(masks, dim=0),
            n_templates=len(kernels),
        )

    def max_similarity(self, frame: torch.Tensor, batch: TemplateBatch, debug: bool = False) -> float:
        """Get maximum similarity score across all templates.
        
        Args:
            frame: (H, W) grayscale frame
            batch: Pre-built template batch
            debug: Print debug info
            
        Returns:
            Maximum cosine similarity in [0, 1], or 0 if no templates.
        """
        if batch.n_templates == 0:
            if debug:
                print(f"  max_similarity: n_templates=0")
            return 0.0

        H, W = frame.shape
        if _TEMPLATE_SIZE > H or _TEMPLATE_SIZE > W:
            if debug:
                print(f"  max_similarity: frame {H}x{W} smaller than template {_TEMPLATE_SIZE}")
            return 0.0

        if debug:
            print(f"  max_similarity: frame {H}x{W}, {batch.n_templates} templates")
            print(f"    frame: min={frame.min().item():.4f} max={frame.max().item():.4f}")
            print(f"    kernels sum: {batch.kernels.abs().sum().item():.4f}")
            print(f"    masks sum: {batch.masks.sum().item():.0f}")

        frame_4d = frame.unsqueeze(0).unsqueeze(0)
        
        # Correlation with L2-normalized templates (2 conv2d calls total)
        correlation = F.conv2d(frame_4d, batch.kernels)
        local_sum_sq = F.conv2d(frame_4d * frame_4d, batch.masks)
        local_norm = local_sum_sq.sqrt().clamp(min=1e-7)
        
        similarity = (correlation / local_norm).clamp(-1.0, 1.0)
        
        if debug:
            print(f"    correlation max: {correlation.max().item():.4f}")
            print(f"    local_norm min/max: {local_norm.min().item():.4f} / {local_norm.max().item():.4f}")
            print(f"    similarity max: {similarity.max().item():.4f}")
        
        return similarity.max().item()


class GPUVideoScanner:
    """Scans videos for events using GPU template matching."""

    def __init__(
        self,
        matcher: GPUTemplateMatcher | None = None,
        batch_size: int = 16,
        blank_frame_mean_threshold: float | None = None,
        blank_frame_std_threshold: float | None = None,
    ):
        self.matcher = matcher or GPUTemplateMatcher()
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

        decoder = decoder_cls(path, device="cpu")
        self._decoder_cache[path] = decoder
        return decoder

    def _get_frame_count(self, decoder: VideoDecoder) -> int:
        """Get total frame count from decoder."""
        try:
            return len(decoder)
        except Exception:
            num_frames = cast(int | None, decoder.metadata.num_frames)
            return int(num_frames) if num_frames else 0

    def _is_blank_frame(self, gray: torch.Tensor) -> bool:
        if self.blank_frame_mean_threshold is None and self.blank_frame_std_threshold is None:
            return False
        mean_val = gray.mean().item()
        std_val = gray.std().item()
        # Thresholds are specified in [0,255] range, but frames are now [0,1]
        mean_ok = self.blank_frame_mean_threshold is None or mean_val <= self.blank_frame_mean_threshold / 255.0
        std_ok = self.blank_frame_std_threshold is None or std_val <= self.blank_frame_std_threshold / 255.0
        return mean_ok and std_ok

    def detect_events(
        self,
        video_path: str,
        death_batch: TemplateBatch,
        attacked_batch: TemplateBatch,
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

            # Batch grayscale + resize
            grays = _rgb_to_gray(frames, dtype=self.matcher.dtype)
            _, h, w = grays.shape
            new_h = max(1, int(h * _FRAME_SCALE))
            new_w = max(1, int(w * _FRAME_SCALE))
            grays = F.interpolate(
                grays.unsqueeze(1),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            for i in range(actual_batch_size):
                gray = grays[i]
                idx = indices[i]
                debug_this = (idx == 0)  # Debug first frame only

                if self._is_blank_frame(gray):
                    result = {"death_conf": 0.0, "attack_conf": 0.0}
                else:
                    if debug_this:
                        print(f"Frame {idx}: gray shape={gray.shape}, min={gray.min().item():.4f}, max={gray.max().item():.4f}")
                    death_conf = self.matcher.max_similarity(gray, death_batch, debug=debug_this)
                    attack_conf = self.matcher.max_similarity(gray, attacked_batch, debug=debug_this)
                    result = {"death_conf": death_conf, "attack_conf": attack_conf}

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
