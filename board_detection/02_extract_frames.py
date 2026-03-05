"""
Extract frames from downloaded videos using ffmpeg.
Extracts 1 frame every FRAME_INTERVAL_SEC seconds.

Usage:
    python 02_extract_frames.py
    python 02_extract_frames.py --ids VIDEO_ID1 VIDEO_ID2

Requires: ffmpeg on PATH
"""

import argparse
import subprocess
import sys

from config import FRAME_INTERVAL_SEC, FRAMES_DIR, VIDEO_IDS, VIDEOS_DIR


def extract_frames(video_id: str) -> int:
    """Extract frames from a video. Returns number of frames extracted."""
    # Find the video file (could be .mp4, .mkv, .webm)
    video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        print(f"  SKIP {video_id} (video file not found)", file=sys.stderr)
        return 0

    video_path = video_files[0]
    output_dir = FRAMES_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already extracted
    existing = list(output_dir.glob("*.png"))
    if existing:
        print(f"  SKIP {video_id} ({len(existing)} frames already exist)")
        return len(existing)

    output_pattern = str(output_dir / f"{video_id}_frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps=1/{FRAME_INTERVAL_SEC}",
        "-q:v", "2",
        output_pattern,
        "-y",
        "-loglevel", "warning",
    ]

    try:
        subprocess.run(cmd, check=True)
        frames = list(output_dir.glob("*.png"))
        print(f"  OK   {video_id} -> {len(frames)} frames")
        return len(frames)
    except subprocess.CalledProcessError as e:
        print(f"  FAIL {video_id}: {e}", file=sys.stderr)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--ids", nargs="+", help="Specific video IDs (default: all from config)")
    args = parser.parse_args()

    ids = args.ids if args.ids else VIDEO_IDS
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting frames (1 every {FRAME_INTERVAL_SEC}s) from {len(ids)} videos\n")

    total_frames = 0
    for i, vid_id in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {vid_id}")
        total_frames += extract_frames(vid_id)

    print(f"\nDone: {total_frames} total frames extracted")


if __name__ == "__main__":
    main()
