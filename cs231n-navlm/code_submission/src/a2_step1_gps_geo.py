"""STEP 1 of the per-photo intersection match.

For every video frame DINOv2 accepted (regardless of tier), curate
three lists DERIVED FROM ITS GPS — no VLM signal involved:

  attractions_within_R   — entries from the 21-attraction curated list
                            whose canonical GPS is within R metres
  landmarks_within_R     — OSM POIs within R metres whose `osm_kind`
                            is a "landmark class" (tourism, historic,
                            place_of_worship, townhall, museum, ...)
  pois_within_R          — every OSM POI within R metres of the GPS,
                            regardless of class (includes streets etc.)

Each list is ordered by distance ascending. This is the GPS-side
answer to "what could this photo be a picture of?" — before we look
at what the VLM actually said.

Output: data/cities/zurich/a2/GPS_GEO.jsonl
  one row per accepted frame:
    { video, frame_id, tier, g_dino,
      attractions_within_R: [{name, zh, kind, dist_m}, ...],
      landmarks_within_R:   [{name, osm_kind, kind_label, dist_m}, ...],
      pois_within_R:        [{name, osm_kind, kind_label, dist_m}, ...] }

  python -m src.a2_step1_gps_geo --radius 100
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_attraction_slots import ATTRACTIONS_21  # noqa: E402


# OSM tag classes we consider "landmark-like" (everything you'd point
# to as a tourist destination). Mirrors src/poi_scan.py's L1/L2 tiers.
LANDMARK_OSM_KINDS = {
    "tourism=attraction", "tourism=viewpoint", "tourism=museum",
    "tourism=gallery", "tourism=hotel", "tourism=artwork",
    "tourism=zoo", "tourism=theme_park", "tourism=information",
    "historic=castle", "historic=monument", "historic=memorial",
    "amenity=place_of_worship", "amenity=townhall",
    "amenity=theatre", "amenity=cinema", "amenity=museum",
    "amenity=library", "amenity=marketplace",
    "amenity=university", "amenity=college",
    "railway=station", "man_made=bridge",
    "waterway=river", "natural=water",
    "leisure=park", "leisure=garden", "leisure=stadium",
    "place=square",
}


def _hav(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    x = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gps",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    ap.add_argument("--pois",
                    default=str(config.CITY_DIR / "pois.json"))
    ap.add_argument("--extra-pois",
                    default=str(config.CITY_DIR / "a2" / "extra_pois.json"),
                    help="optional supplementary POI list to merge with "
                         "pois.json — used for landmarks OSM has but our "
                         "src/pois.py extraction filter dropped "
                         "(e.g. Paradeplatz, Rathaus)")
    ap.add_argument("--heading-v2",
                    default=str(config.CITY_DIR / "a2"
                                / "heading_v2.jsonl"),
                    help="optional heading_v2.jsonl to merge in — adds "
                         "heading_v2, heading_v2_decision, "
                         "heading_v2_gap per row")
    ap.add_argument("--radius", type=float, default=100.0,
                    help="proximity radius in metres (default 100)")
    ap.add_argument("--out",
                    default=str(config.CITY_DIR / "a2" / "GPS_GEO.jsonl"))
    args = ap.parse_args()

    # ── load the OSM POI table (1,289 entries) + supplementary ──────
    pois = json.loads(Path(args.pois).read_text(encoding="utf-8"))
    n_main = len(pois)
    extra_path = Path(args.extra_pois)
    if extra_path.exists():
        extras = json.loads(extra_path.read_text(encoding="utf-8"))
        pois.extend(extras)
        print(f"[gps_geo] OSM POIs loaded: {n_main:,} from {Path(args.pois).name}"
              f"  +  {len(extras)} from {extra_path.name}  =  {len(pois):,}")
    else:
        print(f"[gps_geo] OSM POIs loaded: {n_main:,}  "
              f"(no extra_pois.json found)")
    # keep only entries with a usable centroid
    pois = [p for p in pois
            if p.get("lat") is not None and p.get("lon") is not None]

    # ── load optional heading_v2.jsonl lookup ───────────────────────
    heading_v2 = {}
    hv2_path = Path(args.heading_v2)
    if hv2_path.exists():
        for line in hv2_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            heading_v2[(d["video"], d["frame_id"])] = d
        print(f"[gps_geo] heading_v2.jsonl rows loaded: "
              f"{len(heading_v2):,}")
    else:
        print(f"[gps_geo] heading_v2.jsonl not found, skipping merge")
    n_landmark = sum(1 for p in pois
                     if p.get("osm_kind") in LANDMARK_OSM_KINDS)
    print(f"[gps_geo] OSM POIs that are landmark-class: {n_landmark:,}")

    # ── load DINOv2-accepted frames (both tiers) ────────────────────
    frames = []
    for line in Path(args.gps).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("accepted") or not r.get("g_dino"):
            continue
        frames.append(r)
    print(f"[gps_geo] DINOv2-accepted frames: {len(frames):,}")
    tier_counts = collections.Counter(int(r.get("tier", 0))
                                       for r in frames)
    print(f"[gps_geo] by tier: {dict(tier_counts)}  "
          f"(1 = VLM-confirmed, 2 = DINOv2-only)")

    # ── for each frame, compute the three lists ─────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"with_attraction": 0, "with_landmark": 0, "with_poi": 0}
    poi_size_hist = collections.Counter()
    attr_size_hist = collections.Counter()

    with out_path.open("w", encoding="utf-8") as fout:
        for f in frames:
            g = (f["g_dino"][0], f["g_dino"][1])

            # attractions — from the 21-list
            attractions = []
            for en, zh, lat, lon, kind in ATTRACTIONS_21:
                d = _hav(g, (lat, lon))
                if d <= args.radius:
                    attractions.append({"name": en, "zh": zh,
                                         "kind": kind,
                                         "dist_m": round(d, 1)})
            attractions.sort(key=lambda x: x["dist_m"])

            # landmarks + all POIs — from the 1,289-entry OSM table
            landmarks, all_pois = [], []
            for p in pois:
                d = _hav(g, (p["lat"], p["lon"]))
                if d > args.radius:
                    continue
                entry = {
                    "name": p["name"],
                    "osm_kind": p.get("osm_kind", ""),
                    "kind_label": p.get("kind_label", ""),
                    "dist_m": round(d, 1),
                }
                all_pois.append(entry)
                if p.get("osm_kind") in LANDMARK_OSM_KINDS:
                    landmarks.append(entry)
            landmarks.sort(key=lambda x: x["dist_m"])
            all_pois.sort(key=lambda x: x["dist_m"])

            if attractions: counts["with_attraction"] += 1
            if landmarks:   counts["with_landmark"]   += 1
            if all_pois:    counts["with_poi"]        += 1
            poi_size_hist[len(all_pois)] += 1
            attr_size_hist[len(attractions)] += 1

            row = {
                "video": f["video"],
                "frame_id": f["frame_id"],
                "tier": f.get("tier"),
                "g_dino": list(g),
                "attractions_within_R": attractions,
                "landmarks_within_R": landmarks,
                "pois_within_R": all_pois,
            }
            hv2 = heading_v2.get((f["video"], f["frame_id"]))
            if hv2:
                row["heading_v2"] = hv2.get("heading_v2")
                row["heading_v2_decision"] = hv2.get("decision")
                row["heading_v2_gap"] = hv2.get("gap")
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[gps_geo] wrote {out_path}")

    # ── stdout summary ──────────────────────────────────────────────
    n = len(frames)
    print()
    print("=" * 96)
    print(f"COVERAGE — radius = {args.radius:.0f} m")
    print("=" * 96)
    print(f"frames with >=1 attraction (21-list) nearby: "
          f"{counts['with_attraction']:,}  "
          f"({100*counts['with_attraction']/n:.1f} %)")
    print(f"frames with >=1 landmark  (OSM class) nearby: "
          f"{counts['with_landmark']:,}  "
          f"({100*counts['with_landmark']/n:.1f} %)")
    print(f"frames with >=1 OSM POI of any kind nearby: "
          f"{counts['with_poi']:,}  "
          f"({100*counts['with_poi']/n:.1f} %)")

    print()
    print("--- distribution: #attractions per frame ---")
    for k in sorted(attr_size_hist):
        print(f"  {k} attractions : {attr_size_hist[k]:>6d} frames")

    print()
    print("--- distribution: #OSM POIs per frame (any kind) ---")
    bins = [0, 1, 2, 5, 10, 20, 50, 100, 1e9]
    bin_counts = [0] * (len(bins) - 1)
    for sz, c in poi_size_hist.items():
        for i in range(len(bins) - 1):
            if bins[i] <= sz < bins[i + 1]:
                bin_counts[i] += c
                break
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        hi_str = "∞" if hi == 1e9 else f"<{int(hi)}"
        print(f"  {int(lo):>3d}-{hi_str:<3s} : {bin_counts[i]:>6d} frames")


if __name__ == "__main__":
    main()
