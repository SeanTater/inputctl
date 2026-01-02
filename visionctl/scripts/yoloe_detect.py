#!/usr/bin/env python3
"""
Find objects using YOLOE open-vocabulary detection.

Usage:
    # Text prompt (find by description)
    python yoloe_detect.py <screenshot> --text "button" "icon" "minimize"

    # Image prompt (find similar to reference region)
    python yoloe_detect.py <screenshot> --image <reference.png>

Example:
    python yoloe_detect.py /tmp/screen.png --text "button"
    python yoloe_detect.py /tmp/screen.png --image refs/orb.png

Output:
    JSON array: [{"x": 100, "y": 200, "box": [...], "confidence": 0.85, "label": "button"}, ...]

Requires: pip install ultralytics
Model downloads on first use (~800MB)
"""
import json
import sys
import argparse
from pathlib import Path


def find_by_text(image_path: str, prompts: list[str], threshold: float = 0.25) -> list[dict]:
    """Find objects matching text prompts."""
    from ultralytics import YOLOE

    # Load model (downloads on first use)
    model = YOLOE("yoloe-11l-seg.pt")

    # Set classes to detect
    model.set_classes(prompts, model.get_text_pe(prompts))

    # Run inference
    results = model.predict(image_path, conf=threshold, verbose=False)

    detections = []
    for r in results:
        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = prompts[cls_idx] if cls_idx < len(prompts) else "unknown"

            detections.append({
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "label": label,
                "method": "yoloe_text"
            })

    # Sort by confidence
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def find_by_image(target_path: str, reference_path: str, threshold: float = 0.25) -> list[dict]:
    """Find objects similar to reference image."""
    import numpy as np
    from PIL import Image
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    # Load reference image to get dimensions
    ref_img = Image.open(reference_path)
    ref_w, ref_h = ref_img.size

    # Load model
    model = YOLOE("yoloe-11l-seg.pt")

    # Use entire reference image as the visual prompt
    visual_prompts = dict(
        bboxes=np.array([[0, 0, ref_w, ref_h]]),  # Full image as bbox
        cls=np.array([0])  # Single class
    )

    # Run prediction
    results = model.predict(
        target_path,
        refer_image=reference_path,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=threshold,
        verbose=False
    )

    detections = []
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            detections.append({
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "label": "reference_match",
                "method": "yoloe_visual"
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def main():
    parser = argparse.ArgumentParser(description="YOLOE open-vocabulary detection")
    parser.add_argument("screenshot", help="Path to screenshot")
    parser.add_argument("--text", "-t", nargs="+", help="Text prompts to search for")
    parser.add_argument("--image", "-i", help="Reference image to find similar objects")
    parser.add_argument("--threshold", "-c", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if not Path(args.screenshot).exists():
        print(f"Error: Screenshot not found: {args.screenshot}", file=sys.stderr)
        sys.exit(1)

    if args.text:
        print(f"Searching for: {args.text}", file=sys.stderr)
        results = find_by_text(args.screenshot, args.text, args.threshold)
    elif args.image:
        if not Path(args.image).exists():
            print(f"Error: Reference image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        print(f"Searching for objects similar to: {args.image}", file=sys.stderr)
        results = find_by_image(args.screenshot, args.image, args.threshold)
    else:
        print("Error: Must specify --text or --image", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    print(json.dumps(results, indent=2))

    # Summary
    if results:
        print(f"\nFound {len(results)} detection(s):", file=sys.stderr)
        for r in results[:5]:
            print(f"  {r['label']}: ({r['x']}, {r['y']}) conf={r['confidence']}", file=sys.stderr)
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more", file=sys.stderr)
    else:
        print(f"\nNo detections above threshold {args.threshold}", file=sys.stderr)


if __name__ == "__main__":
    main()
