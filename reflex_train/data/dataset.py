import os
import random
from typing import List, Optional

import polars as pl
import torch
from torch.utils.data import Dataset, IterableDataset
from torchcodec.decoders import VideoDecoder  # type: ignore[import-not-found]


from .intent import INTENT_TO_IDX, intent_to_vector
from .keys import keys_to_vector
from .logs import load_key_sets


class MultiStreamDataset(Dataset):
    def __init__(
        self,
        run_dirs: List[str],
        transform=None,
        context_frames: int = 3,
        goal_intent: Optional[str] = None,
        action_horizon: int = 0,
        intent_labeler=None,
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

        self.samples = []

        print(f"Indexing {len(run_dirs)} sessions...")
        for d in run_dirs:
            self._index_session(d)

        print(f"Indexed {len(self.samples)} samples across {len(run_dirs)} sessions.")

    def _index_session(self, session_dir: str):
        video_path = os.path.join(session_dir, "recording.mp4")
        frames_log = os.path.join(session_dir, "frames.jsonl")
        inputs_log = os.path.join(session_dir, "inputs.jsonl")
        intent_log = os.path.join(session_dir, "intent.jsonl")

        if not (os.path.exists(video_path) and os.path.exists(frames_log)):
            return

        try:
            frame_indices, key_set_by_frame = load_key_sets(frames_log, inputs_log)
            if not frame_indices:
                return

            key_state_by_frame = [keys_to_vector(keys) for keys in key_set_by_frame]

            intent_by_frame = None
            if os.path.exists(intent_log):
                df_intent = pl.read_ndjson(intent_log)
                intent_map = {
                    row["frame_idx"]: row["intent"] for row in df_intent.to_dicts()
                }
                intent_by_frame = [intent_map.get(idx, "WAIT") for idx in frame_indices]
            elif self.intent_labeler is not None:
                intent_by_frame = self.intent_labeler.label_intents(
                    video_path, frame_indices, key_set_by_frame
                )

            for i, f_idx in enumerate(frame_indices):
                if i < self.context_frames - 1:
                    continue

                target_i = i + self.action_horizon
                if target_i >= len(frame_indices):
                    break

                label_keys = key_state_by_frame[target_i]
                if label_keys is None:
                    label_keys = keys_to_vector(set())
                if intent_by_frame is not None:
                    intent = intent_by_frame[i]
                elif self.fixed_goal is not None:
                    intent = self.fixed_intent
                else:
                    intent = "WAIT"

                if intent is None:
                    intent = "WAIT"

                goal_vec = self.fixed_goal
                if goal_vec is None:
                    goal_vec = torch.tensor(
                        intent_to_vector(intent), dtype=torch.float32
                    )

                label_intent = torch.tensor(INTENT_TO_IDX[intent], dtype=torch.long)
                label_mouse = torch.tensor([0.5, 0.5])

                self.samples.append(
                    {
                        "video_path": video_path,
                        "frame_idx": f_idx,
                        "label_keys": label_keys,
                        "label_mouse": label_mouse,
                        "goal": goal_vec,
                        "label_intent": label_intent,
                        "has_next": (i < len(frame_indices) - 1),
                    }
                )
        except Exception as e:
            print(f"Error indexing {session_dir}: {e}")

    def __len__(self):
        return len(self.samples)

    def _get_video_decoder(self, path: str) -> VideoDecoder:
        if not hasattr(self, "_decoders"):
            self._decoders = {}

        if path not in self._decoders:
            self._decoders[path] = VideoDecoder(path, device="cpu")
        return self._decoders[path]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vpath = sample["video_path"]
        current_fidx = sample["frame_idx"]

        decoder = self._get_video_decoder(vpath)

        start_f = max(current_fidx - self.context_frames + 1, 0)
        indices = list(range(start_f, start_f + self.context_frames))

        try:
            frame_batch = decoder.get_frames_at(indices=indices)
            frames = frame_batch.data
        except Exception:
            frames = None

        if frames is None or frames.numel() == 0:
            frames = torch.zeros((self.context_frames, 3, 224, 224), dtype=torch.uint8)

        if frames.shape[0] < self.context_frames:
            pad_count = self.context_frames - frames.shape[0]
            pad_frame = frames[-1:].repeat(pad_count, 1, 1, 1)
            frames = torch.cat([frames, pad_frame], dim=0)

        if sample["has_next"]:
            try:
                next_batch = decoder.get_frames_at(indices=[current_fidx + 1])
                next_frame = next_batch.data
            except Exception:
                next_frame = None
        else:
            next_frame = None

        if next_frame is None or next_frame.numel() == 0:
            next_frame = frames[-1:].clone()

        if next_frame.shape[0] > 1:
            next_frame = next_frame[:1]

        next_frame = next_frame.squeeze(0)

        if self.transform:
            tensor_frames = [self.transform(f) for f in frames]
            input_stack = torch.cat(tensor_frames, dim=0)
        else:
            input_stack = frames.float() / 255.0
            input_stack = input_stack.reshape(
                -1, input_stack.shape[-2], input_stack.shape[-1]
            )

        if self.transform:
            next_frame_tensor = self.transform(next_frame)
        else:
            next_frame_tensor = next_frame.float() / 255.0

        return {
            "pixels": input_stack,
            "goal": sample["goal"],
            "label_keys": sample["label_keys"],
            "label_mouse": sample["label_mouse"],
            "label_intent": sample["label_intent"],
            "next_pixels": next_frame_tensor,
        }


class StreamingDataset(IterableDataset):
    def __init__(self, subset, seed: int):
        self.subset = subset
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        base_dataset = self.subset.dataset
        indices = self.subset.indices

        grouped = {}
        for base_idx in indices:
            sample = base_dataset.samples[base_idx]
            grouped.setdefault(sample["video_path"], []).append(base_idx)

        video_paths = list(grouped.keys())
        rng.shuffle(video_paths)

        for video_path in video_paths:
            frames = sorted(grouped[video_path])
            for base_idx in frames:
                yield base_dataset[base_idx]
