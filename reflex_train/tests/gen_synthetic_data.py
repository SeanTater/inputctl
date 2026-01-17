import cv2
import numpy as np
import os
import json
import time

def create_synthetic_data(data_root="dataset"):
    """Creates a synthetic dataset session for testing."""
    session_dir = os.path.join(data_root, "run_synthetic_01")
    os.makedirs(session_dir, exist_ok=True)
    
    print(f"Generating synthetic data in {session_dir}...")
    
    # 1. Create Video
    # 60 frames (6 seconds at 10fps? or 60 at 10fps for 6s)
    # 224x224
    width, height = 224, 224
    fps = 10
    num_frames = 60
    
    video_path = os.path.join(session_dir, "recording.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Generate bouncing ball video
    cx, cy = 50, 50
    dx, dy = 5, 3
    
    for i in range(num_frames):
        # Update pos
        cx += dx
        cy += dy
        if cx < 10 or cx > width-10: dx = -dx
        if cy < 10 or cy > height-10: dy = -dy
        
        # Draw
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background color change to verify color
        frame[:,:] = [20, 20, 20]
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1) # Green ball
        
        # Add frame number text
        cv2.putText(frame, f"{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print("Video generated.")
    
    # 2. Create frames.jsonl
    # Map frame index to timestamp
    start_ts = int(time.time() * 1000)
    with open(os.path.join(session_dir, "frames.jsonl"), "w") as f:
        for i in range(num_frames):
            record = {
                "frame_idx": i,
                "timestamp": start_ts + (i * 100) # 100ms per frame = 10fps
            }
            f.write(json.dumps(record) + "\n")
            
    # 3. Create inputs.jsonl (Keys)
    # Simulate keys
    # Frame 10: Hold RIGHT
    # Frame 30: Release RIGHT, Hold LEFT
    # Frame 50: Release LEFT
    
    events = []
    
    def add_event(frame_idx, key, state):
        ts = start_ts + (frame_idx * 100)
        events.append({
            "timestamp": ts,
            "event_type": "key",
            "key_name": key,
            "state": state
        })
        
    add_event(10, "KEY_RIGHT", "down")
    add_event(30, "KEY_RIGHT", "up")
    add_event(30, "KEY_LEFT", "down")
    add_event(50, "KEY_LEFT", "up")
    
    with open(os.path.join(session_dir, "inputs.jsonl"), "w") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")
            
    # 4. Create mouse.bin (Optional, skipped for simple test)
    # Just touch the file
    with open(os.path.join(session_dir, "mouse.bin"), "wb") as f:
        pass
        
    print("Synthetic session complete.")

if __name__ == "__main__":
    create_synthetic_data()
