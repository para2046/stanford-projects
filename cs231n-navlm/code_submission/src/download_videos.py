"""Stage 1 — download the 8 source videos with yt-dlp.

Saves to <DATA_ROOT>/cities/zurich/videos/<dataset_name>.mp4. Resumable —
skips videos already downloaded.

    python -m src.download_videos                 # all 8
    python -m src.download_videos --only saturday_morning
    python -m src.download_videos --list          # just list, no download

Requires: yt-dlp (`pip install yt-dlp`) and ffmpeg on PATH (for the
video+audio merge).
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# best mp4 video + m4a audio, falling back to best progressive mp4
FORMAT = "bv*[ext=mp4]+ba[ext=m4a]/best[ext=mp4]/best"


def download_one(video_id: str, name: str) -> bool:
    """Download a single video. Returns True if a new file was fetched."""
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.VIDEOS_DIR / f"{name}.mp4"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip  {name:18s} (already downloaded)")
        return False
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"  fetch {name:18s} <- {url}")
    # call yt-dlp as a module — robust whether or not the venv Scripts
    # dir is on PATH
    subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", FORMAT,
         "--merge-output-format", "mp4",
         "--ffmpeg-location", config.FFMPEG_DIR,
         "-o", str(dest), url],
        check=True,
    )
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="dataset name to download (default: all 8)")
    ap.add_argument("--list", action="store_true", help="list videos and exit")
    args = ap.parse_args()

    items = [(vid, name) for vid, name in config.VIDEOS.items()
             if not args.only or name == args.only]
    if not items:
        sys.exit(f"unknown video name: {args.only!r} — "
                 f"choices: {sorted(config.VIDEOS.values())}")

    if args.list:
        for vid, name in items:
            tag = "  (hold-out)" if name == config.HOLDOUT_VIDEO else ""
            print(f"  {name:18s} {vid}{tag}")
        return

    print(f"downloading {len(items)} video(s) -> {config.VIDEOS_DIR}")
    n_new = 0
    for vid, name in items:
        try:
            n_new += download_one(vid, name)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED {name}: {e}", file=sys.stderr)
    print(f"done — {n_new} newly downloaded, "
          f"{len(items) - n_new} already present.")


if __name__ == "__main__":
    main()
