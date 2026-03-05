"""
Download GothamChess videos using yt-dlp.
Skips videos that are already downloaded (idempotent).

Usage:
    python 01_download_videos.py
    python 01_download_videos.py --ids VIDEO_ID1 VIDEO_ID2  # download specific IDs

Requires: pip install yt-dlp
"""

import argparse
import subprocess
import sys

from config import (
    MAX_SLEEP_INTERVAL,
    MAX_VIDEO_HEIGHT,
    SLEEP_INTERVAL,
    VIDEO_IDS,
    VIDEOS_DIR,
)


def download_video(video_id: str) -> bool:
    """Download a single video. Returns True if successful."""
    # Check if already downloaded (any extension)
    existing = list(VIDEOS_DIR.glob(f"{video_id}.*"))
    if existing:
        print(f"  SKIP {video_id} (already exists: {existing[0].name})")
        return True

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={MAX_VIDEO_HEIGHT}]+bestaudio/best[height<={MAX_VIDEO_HEIGHT}]",
        "--merge-output-format", "mp4",
        "-o", str(VIDEOS_DIR / "%(id)s.%(ext)s"),
        "--sleep-interval", str(SLEEP_INTERVAL),
        "--max-sleep-interval", str(MAX_SLEEP_INTERVAL),
        "--no-overwrites",
        url,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"  OK   {video_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAIL {video_id}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download GothamChess videos")
    parser.add_argument("--ids", nargs="+", help="Specific video IDs to download (default: all from config)")
    args = parser.parse_args()

    ids = args.ids if args.ids else VIDEO_IDS
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(ids)} videos to {VIDEOS_DIR}\n")

    success, fail = 0, 0
    for i, vid_id in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {vid_id}")
        if download_video(vid_id):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} downloaded, {fail} failed")


if __name__ == "__main__":
    main()
