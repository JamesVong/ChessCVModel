"""
Auto-label extracted frames using template-based chessboard detection.
Produces YOLO-format label files (.txt) for each frame.

Semi-supervised: the template detector provides initial labels that
will be reviewed before training.

Usage:
    python 03_auto_label.py
    python 03_auto_label.py --ids VIDEO_ID1 VIDEO_ID2

Requires: pip install opencv-python-headless numpy
"""

import argparse
import csv
import sys

import cv2
import numpy as np

from config import FRAMES_DIR, LABELS_DIR, TEMPLATE_PATH, VIDEO_IDS


# ── Template detector (copied from ChessAtlasBackend) ─────────────
class TemplateBoardDetector:
    def __init__(self, template_image_path):
        template = cv2.imread(str(template_image_path))
        if template is None:
            raise FileNotFoundError(f"Template image not found at {template_image_path}")
        self.template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.template_height, self.template_width = self.template_gray.shape[:2]
        self.COARSE_SCALE_RANGE = np.linspace(0.05, 0.7, 50)
        self.THRESHOLD = 0.4

    def match_template(self, test_gray, template_gray, scale_range):
        found = None
        for scale in scale_range:
            resized = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if resized.shape[0] > test_gray.shape[0] or resized.shape[1] > test_gray.shape[1]:
                continue
            result = cv2.matchTemplate(test_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > self.THRESHOLD and (found is None or max_val > found[0]):
                found = (max_val, max_loc, scale)
        return found

    def detect(self, image):
        """Returns ((top_left, bottom_right), confidence) or (None, 0.0)."""
        ds_factor = 0.5
        frame_gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(frame_gray_full, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        template_small = cv2.resize(self.template_gray, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

        found = self.match_template(frame_small, template_small, self.COARSE_SCALE_RANGE)
        if not found:
            return None, 0.0

        coarse_conf, coarse_max_loc, coarse_best_scale = found

        fine_start = max(coarse_best_scale - 0.01, 0.01)
        fine_end = coarse_best_scale + 0.012
        fine_scale_range = np.arange(fine_start, fine_end, 0.002)
        fine_found = self.match_template(frame_gray_full, self.template_gray, fine_scale_range)

        if fine_found:
            confidence, max_loc, best_scale = fine_found
        else:
            max_loc = (int(coarse_max_loc[0] / ds_factor), int(coarse_max_loc[1] / ds_factor))
            best_scale = coarse_best_scale
            confidence = coarse_conf

        top_left = max_loc
        board_width = int(self.template_width * best_scale)
        board_height = int(self.template_height * best_scale)
        bottom_right = (top_left[0] + board_width, top_left[1] + board_height)

        return (top_left, bottom_right), confidence


def bbox_to_yolo(top_left, bottom_right, img_width, img_height):
    """Convert (top_left, bottom_right) to YOLO format: x_center y_center w h (normalized)."""
    x_center = (top_left[0] + bottom_right[0]) / 2.0 / img_width
    y_center = (top_left[1] + bottom_right[1]) / 2.0 / img_height
    w = (bottom_right[0] - top_left[0]) / img_width
    h = (bottom_right[1] - top_left[1]) / img_height
    return x_center, y_center, w, h


def main():
    parser = argparse.ArgumentParser(description="Auto-label frames with template detection")
    parser.add_argument("--ids", nargs="+", help="Specific video IDs (default: all from config)")
    args = parser.parse_args()

    ids = args.ids if args.ids else VIDEO_IDS

    detector = TemplateBoardDetector(TEMPLATE_PATH)

    metadata_path = LABELS_DIR.parent / "label_metadata.csv"
    metadata_rows = []

    total, detected, missed = 0, 0, 0

    for vid_id in ids:
        frames_dir = FRAMES_DIR / vid_id
        if not frames_dir.exists():
            print(f"  SKIP {vid_id} (no frames directory)")
            continue

        labels_dir = LABELS_DIR / vid_id
        labels_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(frames_dir.glob("*.png"))
        vid_detected = 0

        for frame_path in frames:
            total += 1
            image = cv2.imread(str(frame_path))
            if image is None:
                print(f"  WARN Could not read {frame_path}", file=sys.stderr)
                continue

            img_h, img_w = image.shape[:2]
            bbox, confidence = detector.detect(image)

            label_path = labels_dir / f"{frame_path.stem}.txt"

            if bbox is not None:
                top_left, bottom_right = bbox
                x_c, y_c, w, h = bbox_to_yolo(top_left, bottom_right, img_w, img_h)
                label_path.write_text(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                detected += 1
                vid_detected += 1
            else:
                # Empty label file = negative/background image
                label_path.write_text("")
                missed += 1

            metadata_rows.append({
                "video_id": vid_id,
                "frame": frame_path.name,
                "detected": bbox is not None,
                "confidence": round(confidence, 4),
            })

        rate = vid_detected / len(frames) * 100 if frames else 0
        print(f"  {vid_id}: {vid_detected}/{len(frames)} detected ({rate:.0f}%)")

    # Write metadata CSV for review prioritization
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "frame", "detected", "confidence"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nSummary: {detected} detected, {missed} no-detection, {total} total")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
