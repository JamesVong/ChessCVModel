"""
Visual QC tool for reviewing auto-generated YOLO labels.
Shows each frame with its bounding box drawn. Prioritizes edge cases.

Controls:
    Space / Enter  = Accept (keep label as-is)
    D              = Delete label (convert to negative/empty)
    Q              = Quit review

Usage:
    python 04_review_labels.py                        # Review all
    python 04_review_labels.py --mode low-confidence  # Review borderline detections
    python 04_review_labels.py --mode no-detection    # Review frames with no detection
    python 04_review_labels.py --ids VIDEO_ID1        # Review specific videos

Requires: pip install opencv-python numpy
"""

import argparse
import csv

import cv2
import numpy as np

from config import DATA_DIR, FRAMES_DIR, LABELS_DIR, VIDEO_IDS


def load_metadata():
    """Load label_metadata.csv and return as list of dicts."""
    path = DATA_DIR / "label_metadata.csv"
    if not path.exists():
        print("No label_metadata.csv found. Run 03_auto_label.py first.")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def read_yolo_label(label_path):
    """Read a YOLO label file. Returns (class_id, x_c, y_c, w, h) or None if empty."""
    text = label_path.read_text().strip()
    if not text:
        return None
    parts = text.split()
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def draw_bbox(image, label, color=(0, 255, 0), thickness=2):
    """Draw YOLO bounding box on image."""
    if label is None:
        return image
    _, x_c, y_c, w, h = label
    img_h, img_w = image.shape[:2]
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def review_frames(frames_to_review: list[dict]):
    """Interactive review loop using cv2.imshow."""
    if not frames_to_review:
        print("No frames to review.")
        return

    accepted, deleted, total = 0, 0, len(frames_to_review)
    print(f"\nReviewing {total} frames. Space=accept, D=delete label, Q=quit\n")

    for i, entry in enumerate(frames_to_review):
        vid_id = entry["video_id"]
        frame_name = entry["frame"]
        confidence = float(entry["confidence"])
        detected = entry["detected"] == "True"

        frame_path = FRAMES_DIR / vid_id / frame_name
        label_path = LABELS_DIR / vid_id / f"{frame_path.stem}.txt"

        if not frame_path.exists():
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        display = image.copy()

        label = read_yolo_label(label_path) if label_path.exists() else None
        if label:
            draw_bbox(display, label)

        status = f"DETECTED (conf={confidence:.3f})" if detected else "NO DETECTION"
        cv2.putText(display, f"[{i+1}/{total}] {vid_id} - {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Space=accept  D=delete  Q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Resize for display if too large
        max_dim = 1200
        h, w = display.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        cv2.imshow("Label Review", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print(f"\nQuit at frame {i+1}/{total}")
            break
        elif key == ord("d"):
            # Delete label -> make it a negative example
            if label_path.exists():
                label_path.write_text("")
            deleted += 1
            print(f"  DELETED {vid_id}/{frame_name}")
        else:
            accepted += 1

    cv2.destroyAllWindows()
    print(f"\nReview complete: {accepted} accepted, {deleted} deleted")


def main():
    parser = argparse.ArgumentParser(description="Review auto-generated labels")
    parser.add_argument("--mode", choices=["all", "low-confidence", "no-detection"],
                        default="low-confidence", help="Which frames to review (default: low-confidence)")
    parser.add_argument("--ids", nargs="+", help="Specific video IDs")
    parser.add_argument("--confidence-threshold", type=float, default=0.55,
                        help="Show detections below this confidence (default: 0.55)")
    args = parser.parse_args()

    metadata = load_metadata()
    if not metadata:
        return

    # Filter by video IDs if specified
    if args.ids:
        ids_set = set(args.ids)
        metadata = [m for m in metadata if m["video_id"] in ids_set]

    if args.mode == "low-confidence":
        # Show detections near the threshold
        to_review = [
            m for m in metadata
            if m["detected"] == "True" and float(m["confidence"]) < args.confidence_threshold
        ]
        to_review.sort(key=lambda m: float(m["confidence"]))
        print(f"Found {len(to_review)} low-confidence detections (< {args.confidence_threshold})")
    elif args.mode == "no-detection":
        to_review = [m for m in metadata if m["detected"] == "False"]
        print(f"Found {len(to_review)} frames with no detection")
    else:
        to_review = metadata
        print(f"Reviewing all {len(to_review)} frames")

    review_frames(to_review)


if __name__ == "__main__":
    main()
