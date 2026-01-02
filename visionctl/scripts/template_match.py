#!/usr/bin/env python3
"""
Find objects by template matching using SIFT/ORB.

Usage:
    python template_match.py <screenshot> <template> [threshold]

Example:
    python template_match.py /tmp/screen.png refs/orb_of_scouring.png 0.7

Output:
    JSON array: [{"x": 100, "y": 200, "confidence": 0.85}, ...]
"""
import cv2
import json
import sys
import numpy as np


def find_template_sift(screenshot_path: str, template_path: str,
                       threshold: float = 0.7, max_dimension: int = 2000) -> list[dict]:
    """Find template in screenshot using SIFT feature matching."""

    screenshot = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if screenshot is None or template is None:
        return []

    # Scale down large images to avoid OOM
    scale = 1.0
    h, w = screenshot.shape
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        screenshot = cv2.resize(screenshot, None, fx=scale, fy=scale)
        template = cv2.resize(template, None, fx=scale, fy=scale)
        print(f"Scaled images by {scale:.2f} for SIFT", file=sys.stderr)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(screenshot, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return []

    # FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < threshold * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        return []

    # Find homography to get bounding box
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        return []

    # Get template corners and transform
    h, w = template.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, M)

    # Calculate center point and bounding box
    center = transformed.mean(axis=0)[0]
    x_coords = transformed[:, 0, 0]
    y_coords = transformed[:, 0, 1]

    confidence = len(good_matches) / len(matches) if matches else 0

    # Scale back to original coordinates if we resized
    inv_scale = 1.0 / scale

    return [{
        "x": int(center[0] * inv_scale),
        "y": int(center[1] * inv_scale),
        "box": [int(x_coords.min() * inv_scale), int(y_coords.min() * inv_scale),
                int(x_coords.max() * inv_scale), int(y_coords.max() * inv_scale)],
        "confidence": round(confidence, 3),
        "matches": len(good_matches),
        "method": "sift"
    }]


def find_template_multi_scale(screenshot_path: str, template_path: str,
                              threshold: float = 0.8, max_dimension: int = 1920) -> list[dict]:
    """Find template using multi-scale template matching (simpler, faster)."""

    screenshot = cv2.imread(screenshot_path)
    template = cv2.imread(template_path)

    if screenshot is None or template is None:
        return []

    # Scale down large images
    scale = 1.0
    h, w = screenshot.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        screenshot = cv2.resize(screenshot, None, fx=scale, fy=scale)
        template = cv2.resize(template, None, fx=scale, fy=scale)
        print(f"Scaled images by {scale:.2f} for multi-scale", file=sys.stderr)

    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    th, tw = template_gray.shape
    results = []
    inv_scale = 1.0 / scale

    # Multi-scale search (search different template sizes)
    for template_scale in np.linspace(0.5, 1.5, 20):
        resized = cv2.resize(template_gray, None, fx=template_scale, fy=template_scale)
        rh, rw = resized.shape

        if rw > screenshot_gray.shape[1] or rh > screenshot_gray.shape[0]:
            continue

        result = cv2.matchTemplate(screenshot_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            # Scale coordinates back to original image space
            cx = int((max_loc[0] + rw // 2) * inv_scale)
            cy = int((max_loc[1] + rh // 2) * inv_scale)
            x1 = int(max_loc[0] * inv_scale)
            y1 = int(max_loc[1] * inv_scale)
            x2 = int((max_loc[0] + rw) * inv_scale)
            y2 = int((max_loc[1] + rh) * inv_scale)
            results.append({
                "x": cx,
                "y": cy,
                "box": [x1, y1, x2, y2],
                "confidence": round(max_val, 3),
                "template_scale": round(template_scale, 2),
                "method": "multi_scale"
            })

    # Sort by confidence
    results.sort(key=lambda r: r["confidence"], reverse=True)

    # Deduplicate nearby matches (within 20 pixels)
    filtered = []
    for r in results:
        is_duplicate = False
        for existing in filtered:
            if abs(r["x"] - existing["x"]) < 20 and abs(r["y"] - existing["y"]) < 20:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(r)

    return filtered[:5]  # Top 5 matches


def find_all_matches(screenshot_path: str, template_path: str,
                     threshold: float = 0.8, max_dimension: int = 1920) -> list[dict]:
    """Find ALL instances of template in screenshot using NMS."""

    screenshot = cv2.imread(screenshot_path)
    template = cv2.imread(template_path)

    if screenshot is None or template is None:
        return []

    # Scale down large images
    scale = 1.0
    h, w = screenshot.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        screenshot = cv2.resize(screenshot, None, fx=scale, fy=scale)
        template = cv2.resize(template, None, fx=scale, fy=scale)
        print(f"Scaled images by {scale:.2f} for exact match", file=sys.stderr)

    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    th, tw = template_gray.shape
    results = []
    inv_scale = 1.0 / scale

    # Single-scale matching (for exact size matches)
    result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find all locations above threshold
    locations = np.where(result >= threshold)

    for pt in zip(*locations[::-1]):
        x, y = pt
        conf = result[y, x]
        # Scale back to original coordinates
        cx = int((x + tw // 2) * inv_scale)
        cy = int((y + th // 2) * inv_scale)
        results.append({
            "x": cx,
            "y": cy,
            "box": [int(x * inv_scale), int(y * inv_scale),
                    int((x + tw) * inv_scale), int((y + th) * inv_scale)],
            "confidence": round(float(conf), 3),
            "method": "exact"
        })

    # Apply Non-Maximum Suppression
    if results:
        boxes = np.array([[r["box"][0], r["box"][1], r["box"][2], r["box"][3]] for r in results])
        confidences = np.array([r["confidence"] for r in results])

        # NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            score_threshold=threshold,
            nms_threshold=0.3
        )

        if len(indices) > 0:
            indices = indices.flatten()
            results = [results[i] for i in indices]

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    screenshot = sys.argv[1]
    template = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7

    # Try exact matching first (fastest, finds multiple instances)
    print(f"Trying exact match...", file=sys.stderr)
    results = find_all_matches(screenshot, template, threshold)

    if not results:
        # Try multi-scale template matching (handles size variations)
        print(f"Exact failed, trying multi-scale...", file=sys.stderr)
        results = find_template_multi_scale(screenshot, template, threshold)

    if not results:
        # Fall back to SIFT (handles rotation, most memory intensive)
        print(f"Multi-scale failed, trying SIFT...", file=sys.stderr)
        results = find_template_sift(screenshot, template, threshold)

    print(json.dumps(results, indent=2))

    # Summary to stderr
    if results:
        print(f"\nFound {len(results)} match(es):", file=sys.stderr)
        for r in results[:5]:
            print(f"  ({r['x']}, {r['y']}) conf={r['confidence']} method={r['method']}", file=sys.stderr)
    else:
        print(f"\nNo matches found above threshold {threshold}", file=sys.stderr)
