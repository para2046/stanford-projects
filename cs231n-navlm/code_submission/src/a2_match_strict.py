"""Match each frame to a famous attraction by:
  1. take DINOv2's GPS,
  2. find the top-N FAMOUS_ATTRACTIONS nearest to that GPS,
  3. require VLM `visible[]` or `guess` to EXACTLY match one of them
     (fold, no polygon-distance slack, no neighborhood-radius gate).

The exact-match step is the strict gate the user asked for — it
rejects lookalike DINOv2 matches because if DINOv2 dropped the frame
into the wrong neighbourhood, the wrong nearest-attraction is what
shows up in step (2), and the VLM (which saw the actual photo) won't
name it.

Output one row per kept frame:
  { video, frame_id, g_dino, matched_attraction, dist_m,
    vlm_named_in: 'visible'|'guess'|'both',
    raw_vlm_guess, raw_vlm_visible }

  python -m src.attraction_match --top-n 3
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.poi import CANDIDATE_POIS                  # noqa: E402


# Famous attractions = the 27 hand-curated candidates from src/poi.py.
# Build (name, lat, lon) tuples.
FAMOUS = [(en, lat, lon, kind)
          for en, _zh, lat, lon, kind in CANDIDATE_POIS]


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Aliases — VLM-text variants that exact-fold-match should accept.
# Strict: only exact equivalences, no "stadthausquai means stadthaus"
# substring fudges. The user wants exact matching.
ALIASES = {
    "Hauptbahnhof":  {"Zürich Hauptbahnhof", "Zurich Main Station",
                      "Main Station", "Zurich HB", "Hauptbahnhof"},
    "Grossmünster":  {"Grossmünster"},
    "Fraumünster":   {"Fraumünster"},
    "St. Peter":     {"St. Peter", "Kirche St. Peter",
                      "St. Peter Church", "St. Peter's Church",
                      "St. Peter Kirche", "St. Peterkirche",
                      "St. Peterhofstatt"},
    "Bellevueplatz": {"Bellevueplatz", "Bellevue"},
    "Opernhaus":     {"Opernhaus", "Opernhaus Zürich",
                      "Zurich Opera House"},
    "Kunsthaus":     {"Kunsthaus", "Kunsthaus Zürich"},
    "Landesmuseum":  {"Landesmuseum", "Landesmuseum Zürich",
                      "Zürich Landesmuseum", "Swiss National Museum"},
    "Limmat river":  {"Limmat", "Limmat river"},
    "Lake Zurich":   {"Lake Zurich", "Zürichsee"},
}

# Build {fold(name) -> canonical_en} for fast exact match.
NAME_TO_CANON = {}
for en, _lat, _lon, _kind in FAMOUS:
    NAME_TO_CANON[fold(en)] = en
    for alias in ALIASES.get(en, set()):
        NAME_TO_CANON[fold(alias)] = en


def _hav(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    x = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def _flat_visible(vis):
    out = []
    for v in (vis or []):
        if isinstance(v, list):
            out.extend(str(x) for x in v)
        else:
            out.append(str(v))
    return out


def _vlm_canon_mentions(scan_row):
    """Return the SET of canonical attraction names the VLM mentioned in
    `visible[]` or `guess`, by exact fold-match against NAME_TO_CANON."""
    if not scan_row:
        return set(), [], ""
    raw_guess = (scan_row.get("guess") or "").strip()
    raw_vis = _flat_visible(scan_row.get("visible"))

    canon_hits = set()
    for s in raw_vis + [raw_guess]:
        # split compound names on '/' and ',' and 'am' / 'at'
        parts = re.split(r"\s*(?:/|,|\bam\b|\bat\b|\bnear\b)\s*", s)
        for p in parts:
            f = fold(p)
            if f in NAME_TO_CANON:
                canon_hits.add(NAME_TO_CANON[f])
    return canon_hits, raw_vis, raw_guess


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gps",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    ap.add_argument("--top-n", type=int, default=3,
                    help="how many nearest attractions to require the "
                         "VLM to match against (default 3)")
    ap.add_argument("--max-dist", type=float, default=300.0,
                    help="ignore attractions farther than this from "
                         "the frame's DINO GPS (m, default 300)")
    ap.add_argument("--out",
                    default=str(config.CITY_DIR / "a2"
                                / "match_strict.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.gps).open(encoding="utf-8") if l.strip()]
    tier1 = [r for r in rows
             if r.get("tier") == 1 and r.get("accepted")
             and r.get("g_dino")]
    print(f"[attr_match] tier-1 accepted frames: {len(tier1):,}")
    print(f"[attr_match] FAMOUS attractions:     {len(FAMOUS)}")
    print(f"[attr_match] top-N to check:         {args.top_n}")
    print(f"[attr_match] max distance (m):       {args.max_dist}")

    scan = {}
    for fn in ["poi_scan.jsonl", "poi_scan_cos0.75.jsonl"]:
        for line in (config.CITY_DIR / fn).open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            scan[(d["video"], d["frame_id"])] = d

    kept = []
    no_vlm_data = 0
    no_canon_nearby = 0
    canon_mismatch = 0
    per_attraction = collections.Counter()
    per_attraction_distance = collections.defaultdict(list)
    drops_by_top_attr = collections.Counter()

    for r in tier1:
        g = r["g_dino"]
        nearest = sorted(
            ((en, _hav((g[0], g[1]), (lat, lon)), lat, lon, kind)
             for en, lat, lon, kind in FAMOUS),
            key=lambda x: x[1])
        nearest = [(en, d, lat, lon, kind) for en, d, lat, lon, kind
                   in nearest if d <= args.max_dist][:args.top_n]
        if not nearest:
            no_canon_nearby += 1
            continue

        sr = scan.get((r["video"], r["frame_id"]))
        if sr is None:
            no_vlm_data += 1
            continue
        vlm_hits, raw_vis, raw_guess = _vlm_canon_mentions(sr)
        if not vlm_hits:
            canon_mismatch += 1
            drops_by_top_attr[nearest[0][0]] += 1
            continue

        # the matched attraction = first of `nearest` that VLM also said
        matched = None
        match_dist = None
        for en, d, _lat, _lon, _kind in nearest:
            if en in vlm_hits:
                matched = en
                match_dist = d
                break
        if matched is None:
            # VLM saw some canonical attraction(s), but NONE are in the
            # top-N nearest by DINO GPS — likely a long-distance sighting
            # (could see Grossmünster from Lindenhof but DINO placed us
            # at Lindenhof, so neither match the other)
            canon_mismatch += 1
            drops_by_top_attr[nearest[0][0]] += 1
            continue

        per_attraction[matched] += 1
        per_attraction_distance[matched].append(match_dist)
        kept.append({
            "video": r["video"], "frame_id": r["frame_id"],
            "g_dino": g, "s_dino": r["s_dino"],
            "matched_attraction": matched,
            "dist_m": round(match_dist, 1),
            "nearest_attractions": [{"name": en, "dist_m": round(d, 1)}
                                     for en, d, _, _, _ in nearest],
            "vlm_visible": raw_vis,
            "vlm_guess": raw_guess,
            "vlm_canon_hits": sorted(vlm_hits),
        })

    print()
    print("=" * 96)
    print("RESULTS")
    print("=" * 96)
    print(f"kept frames (exact match of VLM to nearest canonical): "
          f"{len(kept):,}")
    print(f"dropped — no canonical attr within {args.max_dist:.0f} m: "
          f"{no_canon_nearby:,}")
    print(f"dropped — no VLM scan row:                            "
          f"{no_vlm_data:,}")
    print(f"dropped — VLM didn't name any nearby canonical attr:  "
          f"{canon_mismatch:,}")

    print()
    print(f"--- KEPT frames per matched attraction (n={len(kept)}) ---")
    for en in sorted(per_attraction, key=lambda x: -per_attraction[x]):
        dists = per_attraction_distance[en]
        avg = sum(dists) / len(dists)
        print(f"  {per_attraction[en]:>4d}  {en:<22s}  "
              f"avg dist={avg:>4.0f} m  "
              f"(range {min(dists):.0f}-{max(dists):.0f})")

    print()
    print(f"--- top 15 nearest-attractions that the VLM never confirmed "
          f"(suspect DINO false-positives) ---")
    for en, c in drops_by_top_attr.most_common(15):
        print(f"  {c:>4d}  nearest=\"{en}\" but VLM disagreed")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print()
    print(f"wrote {out_path.name}: {len(kept):,} rows")


if __name__ == "__main__":
    main()
