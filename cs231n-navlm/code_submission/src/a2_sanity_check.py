"""Sanity-check the VLM↔DINOv2 mapping in gps_recovery before we commit
to landmark-based re-annotation.

Three questions, three sections:

  Q1. Are the two signals looking at the same frame?
      Just count: every accepted frame should appear in BOTH the
      DINOv2 cache and (for tier-1) the poi_scan output. If a tier-1
      frame is missing from poi_scan, the agreement was fake.

  Q2. Does the VLM's `guess` actually resolve to an OSM POI?
      Of the 4,891 VLM-scanned frames, how many had `guess` that
      `resolve_poi()` could map to *anything* in our 1,289-POI OSM
      table? The unmatched ones are VLM names we don't know about
      (e.g. "old town", "near the river", restaurant names).

  Q3. Do DINOv2's nearest-POI and VLM's resolved-POI agree?
      For each accepted tier-1 frame:
        - same name?                       (exact agreement)
        - different name, distance ≤ 250m? (neighborhood agreement
                                            — gets accepted by F3)
        - different name, distance > 250m? (disagreement — rejected)
      Spot-check 10 frames per bucket.

  python -m src.vlm_dino_sanity
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gps",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.gps).open(encoding="utf-8") if l.strip()]
    print(f"[vlm_dino_sanity] gps_recovery rows: {len(rows):,}")

    # ── Q1 — are the two signals looking at the same frame? ─────────
    print()
    print("=" * 100)
    print("Q1. ARE DINOv2 AND VLM LOOKING AT THE SAME FRAME?")
    print("=" * 100)
    tier_counts = collections.Counter(int(r.get("tier") or 0) for r in rows)
    print(f"tier-1 rows (VLM scan attached): "
          f"{tier_counts[1]:,}")
    print(f"tier-2 rows (DINO-only, VLM never asked): "
          f"{tier_counts[2]:,}")
    # for tier-1 rows, confirm the poi_scan source row exists
    scan_keys = set()
    for fn in ["poi_scan.jsonl", "poi_scan_cos0.75.jsonl"]:
        p = config.CITY_DIR / fn
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            scan_keys.add((d["video"], d["frame_id"]))
    tier1_missing = sum(
        1 for r in rows if r.get("tier") == 1
        and (r["video"], r["frame_id"]) not in scan_keys)
    print(f"tier-1 rows whose source poi_scan row is MISSING: "
          f"{tier1_missing}")
    print(f"  → if zero, every tier-1 frame had a real VLM scan on the "
          f"exact same image as DINOv2")

    # ── Q2 — does VLM's `guess` resolve to an OSM POI? ───────────────
    print()
    print("=" * 100)
    print("Q2. DOES VLM'S `guess` RESOLVE TO AN OSM POI?")
    print("=" * 100)
    # We compare `place_guess` (set by gps_recovery only if VLM's guess
    # resolved) against the raw VLM guess from poi_scan.
    scan = {}
    for fn in ["poi_scan.jsonl", "poi_scan_cos0.75.jsonl"]:
        for line in (config.CITY_DIR / fn).open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            scan[(d["video"], d["frame_id"])] = d

    raw_guess_present = 0
    guess_resolved = 0
    guess_unresolved = 0
    unresolved_examples = collections.Counter()
    for r in rows:
        if r.get("tier") != 1:
            continue
        sr = scan.get((r["video"], r["frame_id"]))
        if not sr:
            continue
        raw_guess = (sr.get("guess") or "").strip()
        if not raw_guess:
            continue
        raw_guess_present += 1
        resolved = (r.get("place_guess") or "").strip()
        if resolved:
            guess_resolved += 1
        else:
            guess_unresolved += 1
            unresolved_examples[raw_guess] += 1
    print(f"tier-1 rows with non-empty raw VLM guess: "
          f"{raw_guess_present:,}")
    print(f"  → resolved to an OSM POI: "
          f"{guess_resolved:,}  ({100*guess_resolved/max(1,raw_guess_present):.1f} %)")
    print(f"  → UN-resolved (VLM said something we don't know): "
          f"{guess_unresolved:,}  ({100*guess_unresolved/max(1,raw_guess_present):.1f} %)")
    print(f"\ntop 15 UN-resolved guess strings (VLM names not in our "
          f"OSM table):")
    for s, c in unresolved_examples.most_common(15):
        print(f"    {c:>4d}  {s}")

    # ── Q3 — do DINOv2 and VLM agree on the same place? ──────────────
    print()
    print("=" * 100)
    print("Q3. WHEN BOTH RESOLVE, DO THE TWO POI NAMES AGREE?")
    print("=" * 100)
    same_name = []
    neighborhood = []     # different name, within 250 m polygon dist
    disagree = []         # different name, > 250 m
    no_vlm_resolution = []
    for r in rows:
        if r.get("tier") != 1:
            continue
        dn = (r.get("dino_nearest_name") or "").strip()
        vn = (r.get("place_guess") or "").strip()
        if not vn:
            no_vlm_resolution.append(r)
            continue
        if not dn:
            continue
        if dn.lower() == vn.lower():
            same_name.append(r)
        elif r.get("neighborhood_match"):
            neighborhood.append(r)
        else:
            disagree.append(r)
    total = len(same_name) + len(neighborhood) + len(disagree)
    print(f"frames where both resolved:                  {total:,}")
    print(f"  SAME name (DINO.nearest == VLM.guess):     "
          f"{len(same_name):,}  ({100*len(same_name)/max(1,total):.1f} %)")
    print(f"  DIFFERENT name, within 250 m polygon dist: "
          f"{len(neighborhood):,}  ({100*len(neighborhood)/max(1,total):.1f} %)")
    print(f"  DIFFERENT name, >  250 m (disagreement):   "
          f"{len(disagree):,}  ({100*len(disagree)/max(1,total):.1f} %)")
    print(f"VLM guess didn't resolve (excluded above):   "
          f"{len(no_vlm_resolution):,}")

    # spot-check the SAME bucket
    print()
    print("-- 5 examples per bucket — open image_path to verify --")
    def _show(rows, label):
        print(f"\n[{label}] ({len(rows)} total)")
        for r in rows[:5]:
            poi_dist = r.get("poi_dist_m")
            poi_dist_str = (f"{poi_dist:.0f} m"
                            if poi_dist is not None else "n/a")
            sr = scan.get((r["video"], r["frame_id"]), {})
            vis = []
            for v in (sr.get("visible") or []):
                if isinstance(v, list):
                    vis.extend(str(x) for x in v)
                else:
                    vis.append(str(v))
            img = (config.FRAMES_DIR / r["video"]
                   / f"{r['frame_id']}.jpg")
            print(f"  {r['video']}/{r['frame_id']}")
            print(f"      DINO nearest POI    : {r.get('dino_nearest_name','')}"
                  f"  (cos={r.get('s_dino', 0):.3f})")
            print(f"      VLM resolved guess  : {r.get('place_guess','')}"
                  f"  (raw='{sr.get('guess','')}')")
            print(f"      polygon distance    : {poi_dist_str}")
            print(f"      VLM visible[]       : "
                  f"{', '.join(vis[:5])}{' …' if len(vis) > 5 else ''}")
            print(f"      image               : {img}")
    _show(same_name,    "SAME name")
    _show(neighborhood, "DIFFERENT name, within 250 m")
    _show(disagree,     "DIFFERENT name, > 250 m (would be rejected)")


if __name__ == "__main__":
    main()
