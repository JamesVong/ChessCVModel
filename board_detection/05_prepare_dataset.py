"""
Prepare the YOLO dataset by performing a video-level train/val/test split
and organizing files into the Ultralytics directory structure.

Usage:
    python 05_prepare_dataset.py
    python 05_prepare_dataset.py --seed 42 --train-ratio 0.7 --val-ratio 0.15

Generates:
    data/yolo_dataset/
        images/{train,val,test}/
        labels/{train,val,test}/
        dataset.yaml
"""

import argparse
import random
import shutil
import json

import yaml

from config import CLASS_NAMES, FRAMES_DIR, LABELS_DIR, VIDEO_IDS, YOLO_DATASET_DIR


def get_video_frame_counts(video_ids: list[str]) -> dict[str, int]:
    """Count frames per video that have corresponding labels."""
    counts = {}
    for vid_id in video_ids:
        frames_dir = FRAMES_DIR / vid_id
        labels_dir = LABELS_DIR / vid_id
        if not frames_dir.exists() or not labels_dir.exists():
            continue
        # Count frames that have a matching label file
        frames = sorted(frames_dir.glob("*.png"))
        labeled = sum(1 for f in frames if (labels_dir / f"{f.stem}.txt").exists())
        if labeled > 0:
            counts[vid_id] = labeled
    return counts


def split_videos(video_ids: list[str], train_ratio: float, val_ratio: float, seed: int):
    """Video-level split. Returns dict with train/val/test lists of video IDs."""
    rng = random.Random(seed)
    ids = list(video_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))

    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def copy_files(video_ids: list[str], split_name: str):
    """Copy frame images and labels for the given videos into the YOLO split directory."""
    img_dir = YOLO_DATASET_DIR / "images" / split_name
    lbl_dir = YOLO_DATASET_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for vid_id in video_ids:
        frames_dir = FRAMES_DIR / vid_id
        labels_dir = LABELS_DIR / vid_id
        if not frames_dir.exists() or not labels_dir.exists():
            continue

        for frame_path in sorted(frames_dir.glob("*.png")):
            label_path = labels_dir / f"{frame_path.stem}.txt"
            if not label_path.exists():
                continue
            shutil.copy2(frame_path, img_dir / frame_path.name)
            shutil.copy2(label_path, lbl_dir / label_path.name)
            count += 1

    return count


def write_dataset_yaml():
    """Write the Ultralytics dataset.yaml config file."""
    config = {
        "path": str(YOLO_DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    yaml_path = YOLO_DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={test_ratio:.2f}")

    # Count available data
    counts = get_video_frame_counts(VIDEO_IDS)
    available_ids = list(counts.keys())
    print(f"\nFound {len(available_ids)} videos with labeled frames ({sum(counts.values())} total frames)")

    if len(available_ids) < 3:
        print("Need at least 3 videos for train/val/test split. Aborting.")
        return

    # Split
    splits = split_videos(available_ids, args.train_ratio, args.val_ratio, args.seed)

    # Clean and rebuild YOLO dataset directory
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)

    # Copy files
    for split_name, vid_ids in splits.items():
        n_frames = copy_files(vid_ids, split_name)
        vid_frames = {v: counts.get(v, 0) for v in vid_ids}
        print(f"\n{split_name}: {len(vid_ids)} videos, {n_frames} frames")
        for v, c in vid_frames.items():
            print(f"  {v}: {c} frames")

    # Write dataset.yaml
    yaml_path = write_dataset_yaml()
    print(f"\nDataset config: {yaml_path}")

    # Save split metadata
    meta = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "splits": splits,
        "frame_counts": counts,
    }
    meta_path = YOLO_DATASET_DIR / "split_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Split metadata: {meta_path}")


if __name__ == "__main__":
    main()
