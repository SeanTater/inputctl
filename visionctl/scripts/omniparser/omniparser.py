#!/usr/bin/env python3
"""
OmniParser V2 wrapper for UI element detection.

Usage:
    python omniparser.py <image_path> [box_threshold] [iou_threshold]

Example:
    python omniparser.py /tmp/screenshot.png
    python omniparser.py /tmp/screenshot.png 0.05 0.1

Output:
    JSON array of UI elements: [{"box": [x1,y1,x2,y2], "label": "button", "text": "Submit"}, ...]

Requires OmniParser to be set up via setup.sh first.
"""

import sys
import os
import json

# Add OmniParser to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OMNIPARSER_DIR = os.path.join(SCRIPT_DIR, "OmniParser")
sys.path.insert(0, OMNIPARSER_DIR)

import torch
from PIL import Image

# Lazy load models
_yolo_model = None
_caption_model_processor = None


def get_models():
    global _yolo_model, _caption_model_processor
    if _yolo_model is None:
        print("Loading OmniParser models...", file=sys.stderr)

        from util.utils import get_yolo_model, get_caption_model_processor

        weights_dir = os.path.join(OMNIPARSER_DIR, "weights")

        _yolo_model = get_yolo_model(
            model_path=os.path.join(weights_dir, "icon_detect/model.pt")
        )

        _caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=os.path.join(weights_dir, "icon_caption_florence")
        )

        print("Models loaded.", file=sys.stderr)

    return _yolo_model, _caption_model_processor


def parse_screenshot(image_path: str, box_threshold: float = 0.05,
                     iou_threshold: float = 0.1) -> list[dict]:
    """
    Parse a screenshot and return detected UI elements.

    Args:
        image_path: Path to screenshot PNG
        box_threshold: Confidence threshold for detection (default: 0.05)
        iou_threshold: IOU threshold for NMS (default: 0.1)

    Returns:
        List of elements with box coordinates (normalized 0-1), label, and text
    """
    from util.utils import check_ocr_box, get_som_labeled_img

    yolo_model, caption_model_processor = get_models()

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Configure drawing (not important for JSON output)
    box_overlay_ratio = width / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # Run OCR
    try:
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=False  # Use EasyOCR by default
        )
        text, ocr_bbox = ocr_bbox_rslt
    except Exception as e:
        print(f"OCR failed: {e}", file=sys.stderr)
        text, ocr_bbox = [], []

    # Run icon detection + captioning
    dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,  # Return normalized coordinates
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=640
    )

    # Convert to our JSON format
    elements = []
    for i, content in enumerate(parsed_content_list):
        # content is a dict with keys like 'type', 'text', 'interactivity'
        # label_coordinates has the bounding boxes
        if i < len(label_coordinates):
            coord = label_coordinates.get(str(i), label_coordinates.get(i, None))
            if coord is not None:
                # Coordinates are in ratio format [x1, y1, x2, y2] where values are 0-1
                # Convert to pixel coordinates
                box = [
                    round(coord[0] * width, 1),
                    round(coord[1] * height, 1),
                    round(coord[2] * width, 1),
                    round(coord[3] * height, 1)
                ]
                elements.append({
                    "box": box,
                    "label": content.get("type", "icon"),
                    "text": content.get("content", content.get("text", "")),
                    "interactivity": content.get("interactivity", None)
                })

    return elements


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    box_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    iou_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    elements = parse_screenshot(image_path, box_threshold, iou_threshold)

    # Print JSON to stdout
    print(json.dumps(elements, indent=2))

    # Summary to stderr
    print(f"\nFound {len(elements)} UI element(s):", file=sys.stderr)
    for e in elements[:10]:  # Show first 10
        box = e["box"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        text = e.get("text", "")[:30]
        print(f"  {e['label']}: '{text}' at ({cx:.0f}, {cy:.0f})", file=sys.stderr)
    if len(elements) > 10:
        print(f"  ... and {len(elements) - 10} more", file=sys.stderr)


if __name__ == "__main__":
    main()
