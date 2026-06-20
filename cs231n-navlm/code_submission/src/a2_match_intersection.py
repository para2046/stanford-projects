"""Per-photo intersection match — LIST_A (GPS-derived) vs LIST_B
(VLM-derived). A photo matches iff the two lists share at least one
attraction.

Conceptual flow (per SV crop, the ground-truth GPS photo):

  STEP 1 — LIST_A from the photo's GPS
    For each SV crop (we have 4,431 with GT GPS), find every
    attraction from the 21-curated list whose canonical GPS is within
    `--radius` metres of the crop's GPS. Order by distance (closest
    first). This is "what attractions is this photo's location near?".

  STEP 2 — LIST_B from VLM scan of frames matched to the photo
    For each video frame DINOv2 matched to this SV crop, pull the
    VLM's `visible[]` and `guess`. Resolve those raw strings against
    the 21-attraction vocabulary (with aliases). Union across all
    matched frames. This is "what attractions did the VLM see when
    looking at images matched to this location?".

  STEP 3 — Intersection
    LIST_A ∩ LIST_B. Non-empty → MATCHED. The matched name(s) are the
    attractions both signals agree on. Empty → either VLM saw
    something we don't have nearby (rare), or the photo's location
    isn't near a curated attraction, or no VLM data exists (no matched
    frames).

Output: data/cities/zurich/a2/match_intersection.jsonl
  one row per SV crop:
    { sv_id, pano_id, gps,
      n_matched_frames,
      list_a_gps:  [{name, dist_m}, ...]      # STEP 1
      list_b_vlm:  {name: matched_frame_count}, # STEP 2
      intersection: [name, ...]               # STEP 3
      matched: bool
      match_status: "matched" | "no_intersection" |
                    "no_vlm_data" | "no_attractions_nearby" |
                    "no_evidence_either_side"
    }

  python -m src.a2_match_intersection --radius 100
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
from src.a2_attraction_slots import (               # noqa: E402
    ATTRACTIONS_21, ALIASES,
)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Build {folded_name -> canonical_en} lookup for VLM-string → attraction
NAME_TO_CANON = {}
for en, *_ in ATTRACTIONS_21:
    NAME_TO_CANON[fold(en)] = en
    for a in ALIASES.get(en, set()):
        NAME_TO_CANON[fold(a)] = en


def _hav(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    x = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def _flat(vis):
    out = []
    for v in (vis or []):
        if isinstance(v, list):
            out.extend(str(x) for x in v)
        else:
            out.append(str(v))
    return out


def _resolve_to_canon(strings):
    """Map raw VLM strings to canonical attractions via fold + alias.
    Splits compound names on `/` `,` `|` `am` `at` `near`."""
    hits = set()
    for s in strings:
        for p in re.split(r"\s*(?:/|,|\||\bam\b|\bat\b|\bnear\b)\s*",
                           s or ""):
            f = fold(p)
            if f and f in NAME_TO_CANON:
                hits.add(NAME_TO_CANON[f])
    return hits


def step1_list_a_from_gps(sv_crops, radius_m):
    """STEP 1 — for each SV crop, the attractions within radius_m of
    its GT GPS. Returns {sv_id: [{name, dist_m}, ...]}, ordered by
    distance ascending."""
    list_a = {}
    for crop in sv_crops:
        gps = (crop["lat"], crop["lon"])
        nearby = []
        for en, _zh, lat, lon, _kind in ATTRACTIONS_21:
            d = _hav(gps, (lat, lon))
            if d <= radius_m:
                nearby.append({"name": en, "dist_m": round(d, 1)})
        nearby.sort(key=lambda x: x["dist_m"])
        list_a[crop["id"]] = nearby
    return list_a


def step2_list_b_from_vlm(sv_crops, frames_by_slot, scan):
    """STEP 2 — for each SV crop, the canonical attractions VLM
    mentioned (in visible[] or guess) across all frames matched to
    that crop. Returns {sv_id: {attraction_name: frame_count}}."""
    list_b = {}
    for crop in sv_crops:
        sv_id = crop["id"]
        matched_frames = frames_by_slot.get(sv_id, [])
        counter = collections.Counter()
        for f in matched_frames:
            sr = scan.get((f["video"], f["frame_id"]))
            if not sr:
                continue
            hits_v = _resolve_to_canon(_flat(sr.get("visible")))
            hits_g = _resolve_to_canon([sr.get("guess") or ""])
            for en in (hits_v | hits_g):
                counter[en] += 1
        list_b[sv_id] = dict(counter)
    return list_b


def step3_intersect(list_a, list_b):
    """STEP 3 — list_a names ∩ list_b names per sv_id."""
    out = {}
    for sv_id in list_a:
        a_names = {x["name"] for x in list_a[sv_id]}
        b_names = set(list_b.get(sv_id, {}))
        out[sv_id] = sorted(a_names & b_names)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--sv-meta",
                    default=str(config.STREETVIEW_DIR / "meta.jsonl"))
    ap.add_argument("--gps",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    ap.add_argument("--radius", type=float, default=100.0,
                    help="STEP 1 radius — attractions within R metres "
                         "of the SV crop's GT GPS count for LIST_A "
                         "(default 100)")
    ap.add_argument("--out",
                    default=str(config.CITY_DIR / "a2"
                                / "match_intersection.jsonl"))
    args = ap.parse_args()

    # ── load inputs ─────────────────────────────────────────────────
    sv_crops = [json.loads(l) for l in
                Path(args.sv_meta).open(encoding="utf-8") if l.strip()]
    print(f"[match] SV crops: {len(sv_crops):,}")

    frames = [json.loads(l) for l in
              Path(args.gps).open(encoding="utf-8") if l.strip()]
    frames = [f for f in frames if f.get("accepted")
              and f.get("top_sv_id")]
    frames_by_slot = collections.defaultdict(list)
    for f in frames:
        frames_by_slot[f["top_sv_id"]].append(f)
    print(f"[match] accepted video frames with SV match: {len(frames):,}")
    print(f"[match] SV slots actually matched by frames: "
          f"{len(frames_by_slot):,}")

    scan = {}
    for fn in ["poi_scan.jsonl", "poi_scan_cos0.75.jsonl"]:
        p = config.CITY_DIR / fn
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            scan[(d["video"], d["frame_id"])] = d
    print(f"[match] VLM scan rows (merged): {len(scan):,}")

    # ── STEP 1 ──────────────────────────────────────────────────────
    print()
    print("STEP 1 — building LIST_A from each SV crop's GT GPS"
          f" (radius = {args.radius:.0f} m)")
    list_a = step1_list_a_from_gps(sv_crops, args.radius)
    nonempty_a = sum(1 for v in list_a.values() if v)
    print(f"  SV crops with ≥1 attraction in LIST_A:  {nonempty_a:,}"
          f"  ({100*nonempty_a/len(sv_crops):.1f} %)")

    # ── STEP 2 ──────────────────────────────────────────────────────
    print()
    print("STEP 2 — building LIST_B from VLM scans of matched frames")
    list_b = step2_list_b_from_vlm(sv_crops, frames_by_slot, scan)
    nonempty_b = sum(1 for v in list_b.values() if v)
    print(f"  SV crops with ≥1 attraction in LIST_B:  {nonempty_b:,}"
          f"  ({100*nonempty_b/len(sv_crops):.1f} %)")

    # ── STEP 3 ──────────────────────────────────────────────────────
    print()
    print("STEP 3 — intersection LIST_A ∩ LIST_B per SV crop")
    intersections = step3_intersect(list_a, list_b)
    n_matched = sum(1 for v in intersections.values() if v)
    print(f"  SV crops with NON-EMPTY intersection (MATCHED): "
          f"{n_matched:,}  ({100*n_matched/len(sv_crops):.1f} %)")

    # ── write ───────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    status_counter = collections.Counter()
    for crop in sv_crops:
        sv_id = crop["id"]
        a, b = list_a[sv_id], list_b.get(sv_id, {})
        inter = intersections[sv_id]
        n_matched_frames = len(frames_by_slot.get(sv_id, []))
        # match status reason
        if inter:
            status = "matched"
        elif not a and not b:
            status = "no_evidence_either_side"
        elif not a:
            status = "no_attractions_nearby"
        elif not b and n_matched_frames == 0:
            status = "no_vlm_data"
        elif not b:
            status = "no_vlm_data"     # had matched frames but no VLM
        else:
            status = "no_intersection"  # both lists exist but disagree
        status_counter[status] += 1
        rows.append({
            "sv_id": sv_id,
            "pano_id": crop["pano_id"],
            "gps": [crop["lat"], crop["lon"]],
            "n_matched_frames": n_matched_frames,
            "list_a_gps": a,
            "list_b_vlm": b,
            "intersection": inter,
            "matched": bool(inter),
            "match_status": status,
        })
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print()
    print(f"wrote {out_path}")

    # ── summary ────────────────────────────────────────────────────
    print()
    print("=" * 96)
    print("RESULT — match_status across all 4,431 SV crops")
    print("=" * 96)
    for s, c in status_counter.most_common():
        print(f"  {c:>5d}  ({100*c/len(sv_crops):>5.1f} %)  {s}")

    print()
    print("=" * 96)
    print("MATCHED CROPS — distribution by intersected attraction")
    print("=" * 96)
    per_attr = collections.Counter()
    for r in rows:
        for n in r["intersection"]:
            per_attr[n] += 1
    for i, (en, *_) in enumerate(ATTRACTIONS_21, 1):
        print(f"  {i:>2}  {en:<22s}  {per_attr[en]:>4d} crops")

    print()
    print(f"sample of 5 MATCHED crops (with their lists):")
    matched_rows = [r for r in rows if r["matched"]][:5]
    for r in matched_rows:
        print(f"  {r['sv_id']}  ({r['n_matched_frames']} matched frames)")
        print(f"      LIST_A (GPS prox): "
              f"{[(a['name'], a['dist_m']) for a in r['list_a_gps']]}")
        print(f"      LIST_B (VLM):      {r['list_b_vlm']}")
        print(f"      INTERSECTION:      {r['intersection']}")

    print()
    print(f"sample of 3 NO_INTERSECTION crops (both lists exist but no overlap):")
    ni = [r for r in rows if r["match_status"] == "no_intersection"][:3]
    for r in ni:
        print(f"  {r['sv_id']}  ({r['n_matched_frames']} matched frames)")
        print(f"      LIST_A (GPS prox): "
              f"{[(a['name'], a['dist_m']) for a in r['list_a_gps']]}")
        print(f"      LIST_B (VLM):      {r['list_b_vlm']}")


if __name__ == "__main__":
    main()
