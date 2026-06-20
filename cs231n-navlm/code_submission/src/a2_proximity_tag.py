"""Tag every DINO+VLM-agreed frame by its nearest CANDIDATE_POI landmark,
report per-landmark coverage, and sample a few frame paths per landmark so
we can spot-check the images BEFORE deciding the destination-pool fix.

This is a read-only audit — it writes one diagnostic file
(``landmark_audit.jsonl``) and prints a summary table. Nothing in the
pipeline is changed; nothing is dropped from any existing file.

Input
    data/cities/zurich/gps_recovery_full.jsonl   (the DINO+VLM-agreed
                                                  cohort = tier-1 accepted
                                                  rows from gps_recovery)
    src/poi.py:CANDIDATE_POIS                    (27 hand-curated landmarks
                                                  with canonical GPS)

Output
    data/cities/zurich/landmark_audit.jsonl
        one row per frame:
        { video, frame_id, gps, place_guess (existing OSM name),
          nearest_landmark, nearest_dist_m,
          landmarks_within_50m / _100m / _150m }

    stdout
        - per-landmark counts at 50 / 100 / 150 m
        - per-video × per-landmark cross-tab at 100 m
        - 3 example (video, frame_id) paths per landmark for visual review

    python -m src.landmark_audit
    python -m src.landmark_audit --radius 100 --samples 5
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                        # noqa: E402
from src.poi import CANDIDATE_POIS                   # noqa: E402


# ── geom ────────────────────────────────────────────────────────────
def _haversine_m(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    x = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def _tag_frame(gps, landmarks, radii=(50.0, 100.0, 150.0)):
    """For one frame: nearest landmark + lists of landmarks within each
    radius. ``landmarks`` is [(en, zh, lat, lon, kind), ...]."""
    dists = [(en, zh, kind, _haversine_m(gps, (lat, lon)))
             for (en, zh, lat, lon, kind) in landmarks]
    dists.sort(key=lambda x: x[3])
    nearest_en, nearest_zh, nearest_kind, nearest_d = dists[0]
    within = {r: [d for d in dists if d[3] <= r] for r in radii}
    return {
        "nearest_landmark": nearest_en,
        "nearest_landmark_zh": nearest_zh,
        "nearest_landmark_kind": nearest_kind,
        "nearest_dist_m": round(nearest_d, 1),
        "landmarks_within_50m":  [d[0] for d in within[50.0]],
        "landmarks_within_100m": [d[0] for d in within[100.0]],
        "landmarks_within_150m": [d[0] for d in within[150.0]],
    }


# ── main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "gps_recovery_full.jsonl"))
    ap.add_argument("--output",
                    default=str(config.CITY_DIR / "a2"
                                / "proximity_tag.jsonl"))
    ap.add_argument("--radius", type=float, default=100.0,
                    help="radius (m) for the summary cross-tab "
                         "(default 100)")
    ap.add_argument("--samples", type=int, default=3,
                    help="N example frames to print per landmark "
                         "(default 3)")
    ap.add_argument("--tier", type=int, default=1,
                    help="gps_recovery tier to include: 1 = DINO ∧ VLM "
                         "both agreed (default, the strictest), 2 = "
                         "DINO-only (no VLM signal), 0 = both")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    # ── load frames ──────────────────────────────────────────────────
    # gps_recovery emits two tiers:
    #   tier 1 — DINO + VLM both agreed on the location (the strict
    #            "agreed by two independent signals" subset)
    #   tier 2 — DINO match only, VLM was either not asked or returned
    #            no place name (much larger, but it's a single-signal
    #            location guess)
    # The user wants the strict "DINO+VLM agreed" subset only, so
    # tier=1 is the default. accept_all sets `accepted=true`.
    accepted = []
    tier_counts = {1: 0, 2: 0}
    for line in in_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("accepted"):
            continue
        gps = r.get("gps")
        if not gps or gps[0] is None or gps[1] is None:
            continue
        t = int(r.get("tier", 0))
        tier_counts[t] = tier_counts.get(t, 0) + 1
        if args.tier and t != args.tier:
            continue
        accepted.append(r)
    print(f"[landmark_audit] accepted rows by tier: {tier_counts}")
    print(f"[landmark_audit] filter --tier={args.tier} kept: "
          f"{len(accepted):,} frames "
          f"({'DINO ∧ VLM agreed' if args.tier == 1 else 'DINO only' if args.tier == 2 else 'all'})")
    print(f"[landmark_audit] landmarks (CANDIDATE_POIS): "
          f"{len(CANDIDATE_POIS)}")

    # ── tag each frame ───────────────────────────────────────────────
    rows = []
    for r in accepted:
        tag = _tag_frame(tuple(r["gps"]), CANDIDATE_POIS)
        rows.append({
            "video": r["video"],
            "frame_id": r["frame_id"],
            "gps": r["gps"],
            "place_guess": r.get("place_guess", "") or "",
            **tag,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[landmark_audit] wrote {out_path} ({len(rows):,} rows)")

    # ── summary 1: per-landmark counts at multiple radii ─────────────
    print()
    print("=" * 100)
    print("LANDMARK COVERAGE — frames within radius of each candidate POI")
    print("=" * 100)
    counts_50  = collections.Counter()
    counts_100 = collections.Counter()
    counts_150 = collections.Counter()
    for r in rows:
        for n in r["landmarks_within_50m"]:  counts_50[n]  += 1
        for n in r["landmarks_within_100m"]: counts_100[n] += 1
        for n in r["landmarks_within_150m"]: counts_150[n] += 1

    # use the original CANDIDATE_POIS order so the user can read it
    # top-down rather than bouncing through a sorted list
    print(f"\n{'#':>2}  {'landmark':<22s} {'中文':<14s} "
          f"{'kind':<8s} {'<=50m':>7s} {'<=100m':>8s} {'<=150m':>8s}")
    print("-" * 100)
    for i, (en, zh, _lat, _lon, kind) in enumerate(CANDIDATE_POIS, 1):
        print(f"{i:>2}  {en:<22s} {zh:<14s} {kind:<8s} "
              f"{counts_50[en]:>7d} {counts_100[en]:>8d} "
              f"{counts_150[en]:>8d}")
    print()
    print(f"TOTAL UNIQUE FRAMES tagged with at least one landmark at "
          f"<= 100m: "
          f"{sum(1 for r in rows if r['landmarks_within_100m']):,}")
    print(f"TOTAL UNIQUE FRAMES tagged with at least one landmark at "
          f"<= 150m: "
          f"{sum(1 for r in rows if r['landmarks_within_150m']):,}")
    print(f"TOTAL UNIQUE FRAMES that are NOT within 150 m of any "
          f"candidate POI: "
          f"{sum(1 for r in rows if not r['landmarks_within_150m']):,}  "
          "(walking-but-nowhere-iconic frames)")

    # ── summary 2: per-video × per-landmark cross-tab ────────────────
    print()
    print("=" * 100)
    print(f"PER-VIDEO × PER-LANDMARK COVERAGE (radius = {args.radius:.0f} m)")
    print("=" * 100)
    by_video = collections.defaultdict(collections.Counter)
    for r in rows:
        for n in r[f"landmarks_within_{int(args.radius)}m"]:
            by_video[r["video"]][n] += 1
    videos = sorted(by_video.keys())
    landmarks_in_use = sorted(
        {n for v in videos for n in by_video[v]},
        key=lambda n: -sum(by_video[v][n] for v in videos))
    if landmarks_in_use:
        col_w = max(12, max(len(v) for v in videos) + 1)
        header = "landmark".ljust(22) + "".join(
            v.ljust(col_w) for v in videos) + "  TOTAL"
        print(header)
        print("-" * len(header))
        for n in landmarks_in_use:
            tot = sum(by_video[v][n] for v in videos)
            row = (n.ljust(22)
                   + "".join((str(by_video[v][n]) if by_video[v][n]
                              else ".").ljust(col_w) for v in videos)
                   + f"  {tot}")
            print(row)
    else:
        print("(no frames within radius of any landmark)")

    # ── summary 3: example frames per landmark (for visual check) ────
    print()
    print("=" * 100)
    print(f"EXAMPLE FRAMES per landmark (closest {args.samples}, "
          f"radius = {args.radius:.0f} m) — open these to verify the "
          f"image actually shows the landmark")
    print("=" * 100)
    samples = collections.defaultdict(list)
    for r in rows:
        for n in r[f"landmarks_within_{int(args.radius)}m"]:
            samples[n].append((r["nearest_dist_m"],
                               r["video"], r["frame_id"]))
    for en, _zh, _lat, _lon, _kind in CANDIDATE_POIS:
        items = sorted(samples[en])[:args.samples]
        if not items:
            print(f"\n{en:<22s}  (no frames within {args.radius:.0f} m)")
            continue
        print(f"\n{en:<22s} ({len(samples[en])} candidate frames):")
        for dist, video, fid in items:
            # Frames are stored as PNG/JPG under data/cities/zurich/frames/
            # so we point at the conventional path. Actual file existence
            # depends on which extraction pass produced them.
            print(f"    {dist:>5.1f} m   {video}/frame_{fid}")


if __name__ == "__main__":
    main()
