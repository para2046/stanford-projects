"""Stage 2 — video -> frames, with a quality filter.

Per video, three steps:
  1. ffmpeg dense-sample at DENSE_FPS  ->  <name>_dense/   (cache, kept)
  2. quality gate — drop blurry and badly-exposed frames
  3. perceptual-hash dedup — keep a frame only if its pHash differs from
     the last KEPT frame by >= PHASH_THRESHOLD bits
  ->  <DATA_ROOT>/cities/zurich/frames/<name>/frame_NNNNN.jpg

Source videos are discovered recursively under config.VIDEOS_DIR; each
filename maps to a dataset name via config.dataset_name().

    python -m src.extract_frames                 # all videos found
    python -m src.extract_frames --only hidden_streets
    python -m src.extract_frames --list          # list, don't extract

Requires: ffmpeg, and `pip install imagehash pillow opencv-python`.

Method notes (the three filter steps):
  * Blur — variance of the Laplacian. The Laplacian is a 2nd-derivative
    edge operator; a sharp frame has strong edges -> high response
    variance, a blurry one has little -> low variance. Frames below
    BLUR_MIN_VAR are dropped.
  * Exposure — mean grayscale luminance. Below EXPOSURE_DARK the frame
    is too dark (tunnel/night), above EXPOSURE_BRIGHT it is blown out;
    both are dropped.
  * pHash dedup — a 64-bit perceptual hash is robust to small changes;
    Hamming distance between two hashes approximates perceptual
    difference. A long static stretch collapses to one frame.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def dense_sample(video: Path, dense_dir: Path) -> list:
    """ffmpeg dense-sample at config.DENSE_FPS. Cached — reused if present."""
    existing = sorted(dense_dir.glob("*.jpg"))
    if existing:
        print(f"    reusing {len(existing)} cached dense frames")
        return existing
    dense_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [config.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={config.DENSE_FPS}", "-q:v", "3",
         str(dense_dir / "dense_%06d.jpg")],
        check=True,
    )
    return sorted(dense_dir.glob("*.jpg"))


def quality_metrics(path: Path):
    """Return (variance-of-Laplacian, mean-luma). Pure — unit-tested.
    Missing/unreadable file -> (0.0, 0.0)."""
    import cv2
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0.0, 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())


def passes_quality(blur: float, luma: float):
    """Return (ok, reason); reason in {'', 'blur', 'exposure'}. Pure."""
    if blur < config.BLUR_MIN_VAR:
        return False, "blur"
    if luma < config.EXPOSURE_DARK or luma > config.EXPOSURE_BRIGHT:
        return False, "exposure"
    return True, ""


def extract_video(video: Path, name: str) -> dict:
    """Extract one video -> quality-filtered, deduped frames."""
    import imagehash
    from PIL import Image

    out_dir = config.FRAMES_DIR / name
    dense_dir = config.FRAMES_DIR / f"{name}_dense"
    print(f"  {name}  ({video.name})")
    dense = dense_sample(video, dense_dir)
    print(f"    {len(dense)} dense frames @ {config.DENSE_FPS} fps")

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    kept = n_blur = n_exposure = n_dup = 0
    last_hash = None
    for f in tqdm(dense, desc=f"  {name}", unit="frame", leave=False):
        blur, luma = quality_metrics(f)
        ok, reason = passes_quality(blur, luma)
        if not ok:
            n_blur += reason == "blur"
            n_exposure += reason == "exposure"
            continue
        h = imagehash.phash(Image.open(f))
        if last_hash is not None and (h - last_hash) < config.PHASH_THRESHOLD:
            n_dup += 1
            continue
        (out_dir / f"frame_{kept:05d}.jpg").write_bytes(f.read_bytes())
        kept += 1
        last_hash = h

    stats = {"video": name, "source": video.name, "dense": len(dense),
             "kept": kept, "dropped_blur": n_blur,
             "dropped_exposure": n_exposure, "dropped_duplicate": n_dup}
    print(f"    kept {kept}  "
          f"(blur={n_blur} exposure={n_exposure} dup={n_dup})")
    return stats


def discover_videos() -> list:
    """Every *.mp4 under VIDEOS_DIR -> (dataset_name, path)."""
    return [(config.dataset_name(p.name), p)
            for p in sorted(config.VIDEOS_DIR.rglob("*.mp4"))]


def main():
    ap = argparse.ArgumentParser(description="Stage 2 — frame extraction")
    ap.add_argument("--only", help="dataset name (default: all videos found)")
    ap.add_argument("--list", action="store_true", help="list videos and exit")
    args = ap.parse_args()

    videos = discover_videos()
    if args.only:
        videos = [(n, p) for n, p in videos if n == args.only]
    if not videos:
        sys.exit(f"no .mp4 found under {config.VIDEOS_DIR}")

    if args.list:
        for n, p in videos:
            print(f"  {n:18s} {p}")
        return

    print(f"extracting {len(videos)} video(s) -> {config.FRAMES_DIR}\n")
    all_stats = [extract_video(p, n) for n, p in videos]

    total = sum(s["kept"] for s in all_stats)
    config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    report = config.FRAMES_DIR / "extract_report.json"
    report.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\ntotal kept frames: {total}\nper-video report -> {report}")


if __name__ == "__main__":
    main()
