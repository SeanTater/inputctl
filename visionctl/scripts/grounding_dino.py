#!/usr/bin/env python3
"""
Grounding DINO wrapper for zero-shot object detection.

Usage:
    python grounding_dino.py <image_path> <prompt> [threshold]

Example:
    python grounding_dino.py /tmp/screenshot.png "minimize button"
    python grounding_dino.py /tmp/screenshot.png "close button. firefox icon." 0.25

Output:
    JSON array of detections: [{"box": [x1,y1,x2,y2], "score": 0.85, "label": "minimize button"}, ...]
"""

import sys
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# Lazy load model (only on first call)
_model = None
_processor = None
_device = None


def get_model():
    global _model, _processor, _device
    if _model is None:
        model_id = "IDEA-Research/grounding-dino-tiny"
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Grounding DINO ({model_id}) on {_device}...", file=sys.stderr)
        _processor = AutoProcessor.from_pretrained(model_id)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(_device)
        print("Model loaded.", file=sys.stderr)
    return _model, _processor, _device


def find_objects(image_path: str, prompt: str, threshold: float = 0.3) -> list[dict]:
    """
    Find objects in image matching the text prompt.

    Args:
        image_path: Path to image file
        prompt: Text description of objects to find (e.g., "minimize button. close button.")
        threshold: Confidence threshold (0.0-1.0)

    Returns:
        List of detections with box coordinates, score, and label
    """
    model, processor, device = get_model()

    image = Image.open(image_path).convert("RGB")

    # Grounding DINO requires lowercase text ending with period
    text = prompt.lower().strip()
    if not text.endswith('.'):
        text += '.'

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]  # (height, width)
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        detections.append({
            "box": [round(x, 1) for x in box.tolist()],  # [x1, y1, x2, y2]
            "score": round(score.item(), 3),
            "label": label
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["score"], reverse=True)

    return detections


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

    detections = find_objects(image_path, prompt, threshold)

    # Print JSON to stdout (for Rust to parse)
    print(json.dumps(detections, indent=2))

    # Also print summary to stderr
    if detections:
        print(f"\nFound {len(detections)} detection(s):", file=sys.stderr)
        for d in detections:
            box = d["box"]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            print(f"  {d['label']}: {d['score']:.1%} at ({cx:.0f}, {cy:.0f})", file=sys.stderr)
    else:
        print(f"\nNo detections found for '{prompt}'", file=sys.stderr)


if __name__ == "__main__":
    main()
