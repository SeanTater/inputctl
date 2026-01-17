import torch
from torch.utils.data import Dataset
import cv2
import polars as pl
import numpy as np
import os
from typing import List, Optional
from .keys import keys_to_vector
from .intent import intent_to_vector, INTENT_TO_IDX
from .logs import load_key_sets

class MultiStreamDataset(Dataset):
    def __init__(self, 
                 run_dirs: List[str], 
                 transform=None, 
                 context_frames: int = 3,
                 goal_intent: Optional[str] = None,
                 action_horizon: int = 0,
                 intent_labeler=None):
        """
        Args:
            run_dirs: List of directory paths containing recording sessions.
            transform: PyTorch transforms for image.
            context_frames: Number of past frames to stack (including current).
            goal_intent: Fixed high-level intent; if None, infer from key states.
            action_horizon: Predict actions this many frames ahead.
            intent_labeler: Optional labeler that provides intent labels per frame.
        """
        self.transform = transform
        self.context_frames = context_frames
        self.fixed_goal = None
        self.fixed_intent = None
        if goal_intent is not None:
            self.fixed_intent = goal_intent
            self.fixed_goal = torch.tensor(intent_to_vector(goal_intent), dtype=torch.float32)
        self.action_horizon = action_horizon
        self.intent_labeler = intent_labeler
        
        self.samples = []
        self.video_handles = {} # Cache video captures? No, not thread safe for DataLoader workers.
                                # actually, we can't share cv2.VideoCapture across threads.
                                # Each worker needs its own. We handle this in __getitem__ carefully or
                                # re-open. Re-opening is slow.
                                # Best practice: Open in `worker_init_fn`?
                                # Or just open/seek/close if seek is fast on SSD?
                                # MP4 seek is slowish. 
                                # Better: Use `decord` or `pims`? 
                                # For strict control, we will open in __init__? 
                                # Note: DataLoader creates copies of dataset. 
                                # We'll rely on opening on demand but caching per-worker.
        
        # Index all data
        print(f"Indexing {len(run_dirs)} sessions...")
        for d in run_dirs:
            self._index_session(d)
            
        print(f"Indexed {len(self.samples)} samples across {len(run_dirs)} sessions.")

    def _index_session(self, session_dir: str):
        video_path = os.path.join(session_dir, "recording.mp4")
        frames_log = os.path.join(session_dir, "frames.jsonl")
        inputs_log = os.path.join(session_dir, "inputs.jsonl")
        intent_log = os.path.join(session_dir, "intent.jsonl")
        mouse_bin = os.path.join(session_dir, "mouse.bin")
        
        if not (os.path.exists(video_path) and os.path.exists(frames_log)):
            return

        # Load Logs
        try:
            frame_indices, key_set_by_frame = load_key_sets(frames_log, inputs_log)
            if not frame_indices:
                return

            key_state_by_frame = [keys_to_vector(keys) for keys in key_set_by_frame]

            intent_by_frame = None
            if os.path.exists(intent_log):
                df_intent = pl.read_ndjson(intent_log)
                intent_map = {row["frame_idx"]: row["intent"] for row in df_intent.to_dicts()}
                intent_by_frame = [intent_map.get(idx, "WAIT") for idx in frame_indices]
            elif self.intent_labeler is not None:
                intent_by_frame = self.intent_labeler.label_intents(
                    video_path,
                    frame_indices,
                    key_set_by_frame
                )

            # We skip the first (context_frames - 1) frames because we can't stack them
            # unless we pad with duplicates. Let's just skip them to be clean.

            for i, f_idx in enumerate(frame_indices):
                if i < self.context_frames - 1:
                    continue

                target_i = i + self.action_horizon
                if target_i >= len(frame_indices):
                    break

                label_keys = key_state_by_frame[target_i]
                if intent_by_frame is not None:
                    intent = intent_by_frame[i]
                elif self.fixed_goal is not None:
                    # Use the fixed intent as the training label if no labeler exists.
                    intent = self.fixed_intent
                else:
                    intent = "WAIT"
                goal_vec = self.fixed_goal
                if goal_vec is None:
                    goal_vec = torch.tensor(intent_to_vector(intent), dtype=torch.float32)
                label_intent = torch.tensor(INTENT_TO_IDX[intent], dtype=torch.long)

                # Mouse Label (Placeholder: 0.5, 0.5 normalized)
                # TODO: Load mouse.bin lookups
                label_mouse = torch.tensor([0.5, 0.5]) 

                self.samples.append({
                    "video_path": video_path,
                    "frame_idx": f_idx,
                    "label_keys": label_keys,
                    "label_mouse": label_mouse,
                    "goal": goal_vec,
                    "label_intent": label_intent,
                    "has_next": (i < len(frame_indices) - 1)
                })
                
        except Exception as e:
            print(f"Error indexing {session_dir}: {e}")

    def __len__(self):
        return len(self.samples)

    def _get_video_cap(self, path):
        """Worker-safe video capture caching"""
        if not hasattr(self, '_caps'):
            self._caps = {}
        
        if path not in self._caps:
            cap = cv2.VideoCapture(path)
            self._caps[path] = cap
        return self._caps[path]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vpath = sample['video_path']
        current_fidx = sample['frame_idx']
        
        cap = self._get_video_cap(vpath)
        
        # Read Context Frames
        # We need [t, t-1, t-2]
        # Since they are sequential, we can seek to t-2 and read 3 frames?
        # Seeking is expensive. Reading sequential is faster.
        # Ideally we seek once to (t - context + 1).
        
        start_f = current_fidx - self.context_frames + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f if start_f >= 0 else 0)
        
        frames = []
        for _ in range(self.context_frames):
            ret, f = cap.read()
            if not ret or f is None:
                # Fallback: black frame
                if frames:
                    f = np.zeros_like(frames[0])
                else:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
                    f = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
            
        # If we failed to read enough (e.g. start of video hiccup), pad
        while len(frames) < self.context_frames:
            frames.append(np.zeros_like(frames[0]))
            
        # Read Next Frame (for Inverse Dynamics)
        # We are already at `current_fidx + 1` after reading loop
        next_frame_img = None
        if sample['has_next']:
            ret, nf = cap.read()
            if ret and nf is not None:
                next_frame_img = cv2.cvtColor(nf, cv2.COLOR_BGR2RGB)
        
        if next_frame_img is None:
            next_frame_img = np.zeros_like(frames[0]) if frames else np.zeros((224, 224, 3), dtype=np.uint8)

        # Transforms
        # Frames is list of H,W,C
        # We transform each? Or transform stack?
        # Standard transform (Resize, ToTensor) expects H,W,C -> C,H,W
        # We want final output: (C*k, H, W)
        
        tensor_frames = []
        for f in frames:
            # TODO: Apply transforms here
            # Manual for now to ensure stacking
            # Assuming resize happens in transform or we do it here
            tf = self.transform(f) if self.transform else torch.from_numpy(f).permute(2,0,1).float()/255.0
            tensor_frames.append(tf)
        
        # Stack channels
        # [C,H,W], [C,H,W] -> [C*3, H, W]
        input_stack = torch.cat(tensor_frames, dim=0)
        
        # Next frame
        next_frame_tensor = self.transform(next_frame_img) if self.transform else torch.from_numpy(next_frame_img).permute(2,0,1).float()/255.0

        return {
            "pixels": input_stack,         # (9, H, W)
            "goal": sample['goal'],
            "label_keys": sample['label_keys'], # (NUM_KEYS,)
            "label_mouse": sample['label_mouse'], # (2,)
            "label_intent": sample['label_intent'],
            "next_pixels": next_frame_tensor # (3, H, W)
        }
