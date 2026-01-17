import torch
from torchvision.transforms import ToTensor, Resize
from reflex_train.data.dataset import MultiStreamDataset
from reflex_train.data.keys import IDX_TO_KEY
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def verify_dataset(data_dir: str):
    print(f"Verifying dataset in {data_dir}...")
    
    # Simple transform
    def transform(img):
        # Resize to standard
        img = cv2.resize(img, (224, 224))
        # HWC -> CHW, Scale
        return torch.from_numpy(img).permute(2,0,1).float()/255.0

    import cv2
    
    dataset = MultiStreamDataset(
        run_dirs=[os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        transform=transform,
        context_frames=3
    )
    
    if len(dataset) == 0:
        print("No samples found.")
        return

    print(f"Dataset size: {len(dataset)}")
    
    # Pick a random sample
    idx = np.random.randint(0, len(dataset))
    sample = dataset[idx]
    
    pixels = sample['pixels'] # 9 channels
    label_keys = sample['label_keys']
    
    # Unpack 9 channels into 3 images
    # Shape: (9, H, W) -> [ (3,H,W), (3,H,W), (3,H,W) ]
    # Order: [t-2, t-1, t] ? Or [t, t-1 t-2]?
    # Dataset implementation appends to list: stack = cat([frame_t-2, frame_t-1, frame_t])
    # So first 3 channels are t-2 (oldest). Last 3 are t (newest).
    
    img_t_minus_2 = pixels[0:3, :, :].permute(1,2,0).numpy()
    img_t_minus_1 = pixels[3:6, :, :].permute(1,2,0).numpy()
    img_t         = pixels[6:9, :, :].permute(1,2,0).numpy()
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_t_minus_2)
    axes[0].set_title(f"t-2")
    axes[1].imshow(img_t_minus_1)
    axes[1].set_title(f"t-1")
    axes[2].imshow(img_t)
    axes[2].set_title(f"t (Current)")
    
    # Decode keys
    active_keys = []
    for i, val in enumerate(label_keys):
        if val > 0.5:
            active_keys.append(IDX_TO_KEY.get(i, f"UNK_{i}"))
            
    print(f"Sample {idx}")
    print(f"Active Keys: {active_keys}")
    
    plt.suptitle(f"Keys: {active_keys}")
    plt.savefig("debug_stack.png")
    print("Saved debug_stack.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_dataset.py <dataset_root_dir>")
        sys.exit(1)
    
    verify_dataset(sys.argv[1])
