import ctypes
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from .intent import INTENT_TO_IDX, intent_to_vector
from .keys import keys_to_vector
from .logs import load_key_sets


# Pre-allocated normalization constants (lazily moved to correct device)
_NORM_MEAN: torch.Tensor | None = None
_NORM_STD: torch.Tensor | None = None


def _get_norm_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get normalization tensors on the correct device (cached)."""
    global _NORM_MEAN, _NORM_STD
    if _NORM_MEAN is None or _NORM_MEAN.device != device:
        _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return _NORM_MEAN, _NORM_STD


def batched_gpu_transform(
    frames: torch.Tensor, size: int = 224
) -> torch.Tensor:
    """Fast batched GPU transform: resize + normalize.

    Args:
        frames: (N, C, H, W) uint8 tensor on any device
        size: target size for resize

    Returns:
        (N, C, size, size) float32 normalized tensor
    """
    mean, std = _get_norm_tensors(frames.device)
    x = frames.float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - mean) / std


@dataclass
class SessionMeta:
    """Lazily-loaded metadata for a single recording session."""

    session_dir: str
    video_path: str
    frame_indices: list  # List of frame indices from parquet
    _key_state_by_frame: list | None = None
    _intent_by_frame: list | None = None
    _returns_map: dict | None = None
    _episode_lookup: dict | None = None
    _episode_end_frames: set | None = None

    def _load_keys(self):
        if self._key_state_by_frame is not None:
            return
        frames_log = os.path.join(self.session_dir, "frames.parquet")
        inputs_log = os.path.join(self.session_dir, "inputs.parquet")
        _, key_set_by_frame = load_key_sets(frames_log, inputs_log)
        self._key_state_by_frame = [keys_to_vector(keys) for keys in key_set_by_frame]

    def _load_intents(self, intent_labeler=None):
        if self._intent_by_frame is not None:
            return
        intent_log = os.path.join(self.session_dir, "intent.parquet")
        if os.path.exists(intent_log):
            df_intent = pl.read_parquet(intent_log)
            intent_map = {
                row["frame_idx"]: row["intent"] for row in df_intent.to_dicts()
            }
            self._intent_by_frame = [
                intent_map.get(idx, "WAIT") for idx in self.frame_indices
            ]
        elif intent_labeler is not None:
            self._intent_by_frame = intent_labeler.label_intents(
                self.video_path, self.frame_indices, self.key_set_by_frame
            )
        else:
            self._intent_by_frame = ["WAIT"] * len(self.frame_indices)

    def _load_returns(self):
        if self._returns_map is not None:
            return
        returns_log = os.path.join(self.session_dir, "returns.parquet")
        episodes_log = os.path.join(self.session_dir, "episodes.parquet")

        self._returns_map = {}
        self._episode_lookup = {}
        self._episode_end_frames = set()

        if os.path.exists(returns_log):
            df_returns = pl.read_parquet(returns_log)
            self._returns_map = {
                row["frame_idx"]: row["return"] for row in df_returns.to_dicts()
            }

        if os.path.exists(episodes_log):
            df_episodes = pl.read_parquet(episodes_log)
            episodes_list = df_episodes.to_dicts()
            self._episode_end_frames = {ep["end_frame"] for ep in episodes_list}
            for ep in episodes_list:
                for f in range(ep["start_frame"], ep["end_frame"] + 1):
                    self._episode_lookup[f] = (ep["episode_id"], ep["reward"])

    def get_key_state(self, frame_idx_in_session: int) -> torch.Tensor:
        self._load_keys()
        return self._key_state_by_frame[frame_idx_in_session]

    def get_intent(self, frame_idx_in_session: int, intent_labeler=None) -> str:
        self._load_intents(intent_labeler)
        intent = self._intent_by_frame[frame_idx_in_session]
        return intent if intent is not None else "WAIT"

    def get_return(self, frame_idx: int) -> float:
        self._load_returns()
        return self._returns_map.get(frame_idx, 0.0)

    def get_episode_info(self, frame_idx: int) -> tuple:
        self._load_returns()
        ep_info = self._episode_lookup.get(frame_idx, (0, 0.0))
        episode_id, ep_reward = ep_info
        reward = ep_reward if frame_idx in self._episode_end_frames else 0.0
        done = 1.0 if frame_idx in self._episode_end_frames else 0.0
        return episode_id, reward, done


_TORCHCODEC_IMPORT_ERROR: Exception | None = None
_TORCHCODEC_DECODERS: tuple[object, object] | None = None


def _preload_torchcodec_deps() -> None:
    candidates: list[Path] = []
    try:
        import nvidia  # type: ignore[import-not-found]

        roots = [Path(p).resolve() for p in getattr(nvidia, "__path__", [])]
        for root in roots:
            candidates.extend(
                [
                    root / "cu13" / "lib" / "libnppicc.so.13",
                    root / "npp" / "lib" / "libnppicc.so.13",
                    root / "cu12" / "lib" / "libnppicc.so.12",
                    root / "npp" / "lib" / "libnppicc.so.12",
                ]
            )
    except Exception:
        pass

    for cand in candidates:
        if cand.exists():
            ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
            return


def _get_torchcodec_decoders():
    global _TORCHCODEC_DECODERS, _TORCHCODEC_IMPORT_ERROR
    if _TORCHCODEC_DECODERS is not None:
        return _TORCHCODEC_DECODERS
    if _TORCHCODEC_IMPORT_ERROR is not None:
        raise _TORCHCODEC_IMPORT_ERROR

    try:
        _preload_torchcodec_deps()
        from torchcodec.decoders import VideoDecoder, set_cuda_backend  # type: ignore[import-not-found]

        _TORCHCODEC_DECODERS = (VideoDecoder, set_cuda_backend)
        return _TORCHCODEC_DECODERS
    except Exception as e:
        _TORCHCODEC_IMPORT_ERROR = e
        raise


def _ensure_nchw_frames(frames: torch.Tensor) -> torch.Tensor:
    """Normalize decoder output to (T, C, H, W) with C=3."""
    if frames.ndim == 3:
        # Single frame: CHW or HWC.
        if frames.shape[0] == 3:
            return frames.unsqueeze(0)
        if frames.shape[-1] == 3:
            return frames.permute(2, 0, 1).unsqueeze(0)
        return frames.unsqueeze(0)
    if frames.ndim == 4:
        # Batch of frames: NCHW or NHWC.
        if frames.shape[1] == 3:
            return frames
        if frames.shape[-1] == 3:
            return frames.permute(0, 3, 1, 2)
    return frames


class MultiStreamDataset(Dataset):
    def __init__(
        self,
        run_dirs: List[str],
        transform=None,
        context_frames: int = 3,
        goal_intent: Optional[str] = None,
        action_horizon: int = 0,
        intent_labeler=None,
        load_returns: bool = True,
        chunk_size: int = 500,
    ):
        self.transform = transform
        self.context_frames = context_frames
        self.fixed_goal = None
        self.fixed_intent = None
        if goal_intent is not None:
            self.fixed_intent = goal_intent
            self.fixed_goal = torch.tensor(
                intent_to_vector(goal_intent), dtype=torch.float32
            )
        self.action_horizon = action_horizon
        self.intent_labeler = intent_labeler
        self.load_returns = load_returns
        self.chunk_size = chunk_size

        # Lazy-loaded session metadata (shared across workers via CoW)
        self.sessions: List[SessionMeta] = []
        # Lightweight sample index: (session_idx, frame_idx_in_session, target_idx_in_session, frame_idx)
        self.samples: List[tuple] = []

        # Chunk cache for batch decoding
        self._frame_cache: dict[int, torch.Tensor] = {}
        self._cached_video_path: str | None = None
        self._cached_chunk_start: int = -1
        self._cached_chunk_end: int = -1

        print(f"Indexing {len(run_dirs)} sessions...")
        for d in run_dirs:
            self._index_session(d)

        print(f"Indexed {len(self.samples)} samples across {len(self.sessions)} sessions.")

    def _index_session(self, session_dir: str):
        video_path = os.path.join(session_dir, "recording.mp4")
        frames_log = os.path.join(session_dir, "frames.parquet")

        if not (os.path.exists(video_path) and os.path.exists(frames_log)):
            return

        try:
            # Only load frame indices - no heavy data yet
            df_frames = pl.read_parquet(frames_log)
            frame_indices = df_frames["frame_idx"].to_list()
            if not frame_indices:
                return

            session_idx = len(self.sessions)
            self.sessions.append(
                SessionMeta(
                    session_dir=session_dir,
                    video_path=video_path,
                    frame_indices=frame_indices,
                )
            )

            # Only store lightweight indices
            for i, f_idx in enumerate(frame_indices):
                if i < self.context_frames - 1:
                    continue
                target_i = i + self.action_horizon
                if target_i >= len(frame_indices):
                    break
                # (session_idx, frame_idx_in_session, target_idx_in_session, actual_frame_idx)
                self.samples.append((session_idx, i, target_i, f_idx))
        except Exception as e:
            print(f"Error indexing {session_dir}: {e}")

    def __len__(self):
        return len(self.samples)

    def _get_video_decoder(self, path: str):
        VideoDecoder, _ = _get_torchcodec_decoders()
        if not hasattr(self, "_decoders"):
            self._decoders = {}

        if path not in self._decoders:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                decoder, mode = self._init_cuda_decoder(path)
                if decoder is None:
                    if not hasattr(self, "_cuda_warned"):
                        print("Torchcodec CUDA decode unavailable; falling back to CPU")
                        self._cuda_warned = True
                    self._decoders[path] = VideoDecoder(path, device="cpu")
                else:
                    if not hasattr(self, "_cuda_warned"):
                        print(f"Torchcodec using CUDA decoder ({mode})")
                        self._cuda_warned = True
                    self._decoders[path] = decoder
            else:
                self._decoders[path] = VideoDecoder(path, device=device)
        return self._decoders[path]

    @staticmethod
    def _init_cuda_decoder(path: str):
        VideoDecoder, set_cuda_backend = _get_torchcodec_decoders()
        try:
            with set_cuda_backend("beta"):
                return VideoDecoder(path, device="cuda"), "beta"
        except Exception:
            try:
                with set_cuda_backend("ffmpeg"):
                    return VideoDecoder(path, device="cuda"), "ffmpeg"
            except Exception:
                return None, None

    def _clear_frame_cache(self):
        """Clear the frame chunk cache."""
        self._frame_cache.clear()
        self._cached_video_path = None
        self._cached_chunk_start = -1
        self._cached_chunk_end = -1

    def _decode_and_cache_chunk(self, video_path: str, start_idx: int, end_idx: int):
        """Batch-decode a chunk of frames and cache transformed results on GPU.

        Args:
            video_path: Path to video file
            start_idx: Starting frame index (inclusive)
            end_idx: Ending frame index (exclusive)
        """
        decoder = self._get_video_decoder(video_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Batch decode all frames in chunk
        indices = list(range(start_idx, end_idx))
        try:
            frame_batch = decoder.get_frames_at(indices=indices)
            frames = frame_batch.data
        except Exception as e:
            print(f"Failed to decode chunk {start_idx}-{end_idx}: {e}")
            return

        if frames is None or frames.numel() == 0:
            return

        frames = _ensure_nchw_frames(frames)

        # Batch transform on GPU (already fast: 0.2ms per 4 frames)
        if self.transform:
            frames_transformed = batched_gpu_transform(frames.to(device), size=224)
        else:
            frames_transformed = frames.to(device).float() / 255.0

        # Cache individual frames
        for i, frame_idx in enumerate(indices):
            if i < frames_transformed.shape[0]:
                self._frame_cache[frame_idx] = frames_transformed[i]

        # Update cache metadata
        self._cached_video_path = video_path
        self._cached_chunk_start = start_idx
        self._cached_chunk_end = end_idx

    def __getitem__(self, idx):
        session_idx, frame_i, target_i, frame_idx = self.samples[idx]
        session = self.sessions[session_idx]
        vpath = session.video_path
        num_frames = len(session.frame_indices)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Clear cache if switching videos
        if self._cached_video_path != vpath:
            self._clear_frame_cache()

        # Determine which frames we need
        start_f = max(frame_idx - self.context_frames + 1, 0)
        context_indices = list(range(start_f, start_f + self.context_frames))
        has_next = frame_i < num_frames - 1
        next_idx = frame_idx + 1 if has_next else frame_idx

        # Check if we need to decode a new chunk
        all_indices = context_indices + [next_idx]
        needs_decode = any(
            idx not in self._frame_cache
            for idx in all_indices
        )

        if needs_decode:
            # Decode a chunk starting from the earliest frame we need
            chunk_start = min(all_indices)
            chunk_end = min(chunk_start + self.chunk_size, max(session.frame_indices) + 1)
            self._decode_and_cache_chunk(vpath, chunk_start, chunk_end)

        # Retrieve frames from cache (or create fallback if cache failed)
        frames = []
        for idx in context_indices:
            if idx in self._frame_cache:
                frames.append(self._frame_cache[idx])
            else:
                # Fallback: create zero frame
                frames.append(torch.zeros((3, 224, 224), dtype=torch.float32, device=device))

        if len(frames) < self.context_frames:
            # Pad if needed
            pad_count = self.context_frames - len(frames)
            pad_frame = frames[-1] if frames else torch.zeros((3, 224, 224), dtype=torch.float32, device=device)
            frames.extend([pad_frame] * pad_count)

        # Get next frame
        if next_idx in self._frame_cache:
            next_frame_tensor = self._frame_cache[next_idx]
        else:
            # Fallback: use last context frame
            next_frame_tensor = frames[-1].clone()

        # Stack context frames
        input_stack = torch.stack(frames).reshape(-1, 224, 224)

        # Lazily compute labels from session metadata
        label_keys = session.get_key_state(target_i)
        if label_keys is None:
            label_keys = keys_to_vector(set())

        current_keys = session.get_key_state(frame_i)
        if current_keys is None:
            current_keys = keys_to_vector(set())

        if self.fixed_goal is not None:
            intent = self.fixed_intent
            goal_vec = self.fixed_goal
        else:
            intent = session.get_intent(frame_i, self.intent_labeler)
            goal_vec = torch.tensor(intent_to_vector(intent), dtype=torch.float32)

        label_intent = torch.tensor(INTENT_TO_IDX[intent], dtype=torch.long)
        label_mouse = torch.tensor([0.5, 0.5])

        # RL fields (lazy load)
        if self.load_returns:
            return_value = session.get_return(frame_idx)
            episode_id, reward, done = session.get_episode_info(frame_idx)
        else:
            return_value, reward, done, episode_id = 0.0, 0.0, 0.0, 0

        return {
            "pixels": input_stack,
            "goal": goal_vec,
            "label_keys": label_keys,
            "label_mouse": label_mouse,
            "label_intent": label_intent,
            "next_pixels": next_frame_tensor,
            "current_keys": current_keys,
            # RL fields
            "return": torch.tensor(return_value, dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.float32),
        }


class StreamingDataset(IterableDataset):
    """IterableDataset that groups samples by video for chunk cache locality."""

    def __init__(self, subset, seed: int):
        self.subset = subset
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        base_dataset = self.subset.dataset
        indices = self.subset.indices

        grouped = {}
        for base_idx in indices:
            session_idx, _, _, _ = base_dataset.samples[base_idx]
            video_path = base_dataset.sessions[session_idx].video_path
            grouped.setdefault(video_path, []).append(base_idx)

        video_paths = list(grouped.keys())
        rng.shuffle(video_paths)

        for video_path in video_paths:
            frames = sorted(grouped[video_path])
            for base_idx in frames:
                yield base_dataset[base_idx]
