"""
One-time helper to find GothamChess video IDs using yt-dlp.
Lists uploads from the channel, filters by duration, and prints candidate IDs.

Usage:
    python find_videos.py [--min-duration 600] [--max-duration 1800] [--limit 50]

Requires: pip install yt-dlp
"""

import argparse
import json
import subprocess
import sys

from config import EXISTING_VIDEO_IDS, GOTHAMCHESS_CHANNEL_URL


def fetch_channel_videos(channel_url: str, limit: int) -> list[dict]:
    """Fetch video metadata from a YouTube channel using yt-dlp --flat-playlist."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", str(limit),
        f"{channel_url}/videos",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    videos = []
    for line in result.stdout.strip().split("\n"):
        if line:
            videos.append(json.loads(line))
    return videos


def filter_videos(
    videos: list[dict],
    min_duration: int,
    max_duration: int,
    exclude_ids: set[str],
) -> list[dict]:
    """Filter videos by duration and exclude already-collected IDs."""
    filtered = []
    for v in videos:
        vid_id = v.get("id", "")
        duration = v.get("duration") or 0
        title = v.get("title", "")

        if vid_id in exclude_ids:
            continue
        if not (min_duration <= duration <= max_duration):
            continue
        # Skip shorts and non-game-analysis content by title heuristics
        title_lower = title.lower()
        if any(skip in title_lower for skip in ["#shorts", "short", "tiktok"]):
            continue

        filtered.append({
            "id": vid_id,
            "title": title,
            "duration_sec": duration,
            "duration_min": round(duration / 60, 1),
        })
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Find GothamChess video IDs")
    parser.add_argument("--min-duration", type=int, default=600, help="Min video duration in seconds (default: 600 = 10min)")
    parser.add_argument("--max-duration", type=int, default=1800, help="Max video duration in seconds (default: 1800 = 30min)")
    parser.add_argument("--limit", type=int, default=100, help="Max number of channel videos to scan (default: 100)")
    args = parser.parse_args()

    print(f"Fetching up to {args.limit} videos from GothamChess channel...")
    videos = fetch_channel_videos(GOTHAMCHESS_CHANNEL_URL, args.limit)
    print(f"Found {len(videos)} total videos")

    existing = set(EXISTING_VIDEO_IDS)
    candidates = filter_videos(videos, args.min_duration, args.max_duration, existing)
    print(f"\nFiltered to {len(candidates)} candidates ({args.min_duration//60}-{args.max_duration//60} min, excluding {len(existing)} existing)\n")

    print(f"{'ID':<15} {'Duration':>8}  Title")
    print("-" * 80)
    for v in candidates:
        print(f"{v['id']:<15} {v['duration_min']:>6.1f}m  {v['title'][:50]}")

    print(f"\n# Copy these IDs to config.py NEW_VIDEO_IDS:")
    print("NEW_VIDEO_IDS = [")
    for v in candidates[:15]:  # Suggest top 15
        print(f'    "{v["id"]}",  # {v["duration_min"]}m - {v["title"][:40]}')
    print("]")


if __name__ == "__main__":
    main()
