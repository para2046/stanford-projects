"""STEP 2 of the per-photo intersection match.

For every video frame the VLM directly scanned (across both
poi_scan.jsonl and poi_scan_cos0.75.jsonl, merged + deduped), curate
three lists DERIVED FROM THE VLM OUTPUT — no GPS involved:

  attractions_from_vlm   — from the 21-attraction curated list, matched
                            against raw VLM strings (fold + alias)
  landmarks_from_vlm     — OSM POIs in the landmark-class subset
                            whose name appears in raw VLM strings
  pois_from_vlm          — every OSM POI (any class, including streets)
                            whose name appears in raw VLM strings

Each entry tracks `source` — whether the name was found in `visible[]`,
`guess`, or both. Compound VLM strings (e.g. "Bahnhofstrasse am
Paradeplatz", "Limmat | Limmatquai") are split on `/`, `,`, `|`,
`am`, `at`, `near` first.

This is the VLM-side answer to "what does this photo contain?" —
mirroring GPS_GEO.jsonl which answered the same question from the
GPS+OSM side. STEP 3 will intersect the two.

Output: data/cities/zurich/a2/VLM_GEO.jsonl
  one row per VLM-scanned frame:
    { video, frame_id,
      raw_vlm_visible: [...]           # original visible[] strings
      raw_vlm_guess: "..."             # original guess
      raw_vlm_confidence: "high|med|low"
      attractions_from_vlm: [{name, zh, kind, source}, ...]
      landmarks_from_vlm:   [{name, osm_kind, kind_label, source}, ...]
      pois_from_vlm:        [{name, osm_kind, kind_label, source}, ...] }

  python -m src.a2_step2_vlm_geo
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_attraction_slots import (               # noqa: E402
    ATTRACTIONS_21, ALIASES,
)
from src.a2_step1_gps_geo import LANDMARK_OSM_KINDS  # noqa: E402


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── attraction vocabulary (21 + aliases) → canonical EN ──────────
ATTRACTION_FOLD_TO_CANON = {}
ATTRACTION_META = {}     # canonical EN → (zh, kind)
for en, zh, _lat, _lon, kind in ATTRACTIONS_21:
    ATTRACTION_FOLD_TO_CANON[fold(en)] = en
    ATTRACTION_META[en] = (zh, kind)
    for alias in ALIASES.get(en, set()):
        ATTRACTION_FOLD_TO_CANON[fold(alias)] = en


# ── OSM POI vocabulary (1,289 + extras) → canonical OSM name ─────
def _load_osm_pois():
    pois = json.loads((config.CITY_DIR / "pois.json").read_text(encoding="utf-8"))
    extra_path = config.CITY_DIR / "a2" / "extra_pois.json"
    if extra_path.exists():
        pois.extend(json.loads(extra_path.read_text(encoding="utf-8")))
    return [p for p in pois
            if p.get("lat") is not None and p.get("lon") is not None]


def _build_osm_lookup(pois):
    """Build {folded_name: poi_dict} for fast exact-fold match."""
    lookup = {}
    for p in pois:
        for name in [p["name"]] + list(p.get("aliases", [])):
            f = fold(name)
            if f and f not in lookup:
                lookup[f] = p
    return lookup


def _flat(vis):
    out = []
    for v in (vis or []):
        if isinstance(v, list):
            out.extend(str(x) for x in v)
        else:
            out.append(str(v))
    return out


def _split_compound(s):
    """Split VLM string on `/`, `,`, `|`, ` am `, ` at `, ` near `
    to break apart compound names."""
    return [p.strip() for p in
            re.split(r"\s*(?:/|,|\||\bam\b|\bat\b|\bnear\b)\s*",
                     s or "")
            if p and p.strip()]


def _match_attractions(raw_visible_strings, raw_guess):
    """Returns [{name, zh, kind, source}] from the 21-list."""
    hits = {}        # canonical_en -> set(sources)
    for s in raw_visible_strings:
        for p in _split_compound(s):
            f = fold(p)
            if f in ATTRACTION_FOLD_TO_CANON:
                en = ATTRACTION_FOLD_TO_CANON[f]
                hits.setdefault(en, set()).add("visible")
    for p in _split_compound(raw_guess):
        f = fold(p)
        if f in ATTRACTION_FOLD_TO_CANON:
            en = ATTRACTION_FOLD_TO_CANON[f]
            hits.setdefault(en, set()).add("guess")
    out = []
    for en, sources in hits.items():
        zh, kind = ATTRACTION_META[en]
        out.append({"name": en, "zh": zh, "kind": kind,
                     "source": "both" if len(sources) == 2
                               else sources.pop()})
    return out


def _match_osm(raw_visible_strings, raw_guess, osm_lookup):
    """Returns ([all_pois], [landmarks_only]) — each entry has
    {name, osm_kind, kind_label, source}."""
    hits = {}        # osm_name -> (poi_dict, set(sources))
    for s in raw_visible_strings:
        for p in _split_compound(s):
            f = fold(p)
            if f in osm_lookup:
                poi = osm_lookup[f]
                entry = hits.setdefault(poi["name"], (poi, set()))
                entry[1].add("visible")
    for p in _split_compound(raw_guess):
        f = fold(p)
        if f in osm_lookup:
            poi = osm_lookup[f]
            entry = hits.setdefault(poi["name"], (poi, set()))
            entry[1].add("guess")
    all_pois, landmarks = [], []
    for osm_name, (poi, sources) in hits.items():
        entry = {
            "name": poi["name"],
            "osm_kind": poi.get("osm_kind", ""),
            "kind_label": poi.get("kind_label", ""),
            "source": "both" if len(sources) == 2 else sources.pop(),
        }
        all_pois.append(entry)
        if poi.get("osm_kind") in LANDMARK_OSM_KINDS:
            landmarks.append(entry)
    return all_pois, landmarks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--scan", action="append", default=None,
                    help="poi_scan jsonl(s) to merge — default: both "
                         "poi_scan.jsonl + poi_scan_cos0.75.jsonl")
    ap.add_argument("--out",
                    default=str(config.CITY_DIR / "a2" / "VLM_GEO.jsonl"))
    args = ap.parse_args()

    sources = args.scan or [
        str(config.CITY_DIR / "poi_scan.jsonl"),
        str(config.CITY_DIR / "poi_scan_cos0.75.jsonl"),
    ]
    scan = {}
    src_counts = []
    for sf in sources:
        if not Path(sf).exists():
            print(f"WARN: {sf} not found, skipping")
            continue
        n = 0
        for line in Path(sf).open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            scan[(d["video"], d["frame_id"])] = d
            n += 1
        src_counts.append((Path(sf).name, n))
    for name, n in src_counts:
        print(f"[vlm_geo] {name}: {n:,} rows")
    print(f"[vlm_geo] unique VLM-scanned frames (deduped): {len(scan):,}")

    pois = _load_osm_pois()
    print(f"[vlm_geo] OSM POIs loaded: {len(pois):,}")
    osm_lookup = _build_osm_lookup(pois)
    print(f"[vlm_geo] OSM POI lookup keys (folded+aliases): "
          f"{len(osm_lookup):,}")
    n_landmark_class = sum(1 for p in pois
                            if p.get("osm_kind") in LANDMARK_OSM_KINDS)
    print(f"[vlm_geo] OSM POIs in landmark class: {n_landmark_class:,}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"with_attraction": 0, "with_landmark": 0, "with_poi": 0}
    n_per_attr = collections.Counter()

    with out_path.open("w", encoding="utf-8") as fout:
        for (video, frame_id), sr in scan.items():
            raw_vis = _flat(sr.get("visible"))
            raw_guess = (sr.get("guess") or "").strip()
            raw_conf = (sr.get("confidence") or "").strip()

            attrs = _match_attractions(raw_vis, raw_guess)
            all_pois, landmarks = _match_osm(raw_vis, raw_guess,
                                              osm_lookup)

            if attrs:     counts["with_attraction"] += 1
            if landmarks: counts["with_landmark"]   += 1
            if all_pois:  counts["with_poi"]        += 1
            for a in attrs:
                n_per_attr[a["name"]] += 1

            row = {
                "video": video, "frame_id": frame_id,
                "raw_vlm_visible": raw_vis,
                "raw_vlm_guess": raw_guess,
                "raw_vlm_confidence": raw_conf,
                "attractions_from_vlm": attrs,
                "landmarks_from_vlm": landmarks,
                "pois_from_vlm": all_pois,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(scan)
    print(f"[vlm_geo] wrote {out_path}")
    print()
    print("=" * 96)
    print("COVERAGE")
    print("=" * 96)
    print(f"frames with >=1 attraction (21-list) named by VLM: "
          f"{counts['with_attraction']:,}  "
          f"({100*counts['with_attraction']/n:.1f} %)")
    print(f"frames with >=1 landmark (OSM landmark class):     "
          f"{counts['with_landmark']:,}  "
          f"({100*counts['with_landmark']/n:.1f} %)")
    print(f"frames with >=1 OSM POI of any kind:               "
          f"{counts['with_poi']:,}  "
          f"({100*counts['with_poi']/n:.1f} %)")

    print()
    print("=" * 96)
    print("PER-ATTRACTION — frames where VLM named it (visible or guess)")
    print("=" * 96)
    print("{:>2}  {:<22s} {:<14s} {:<8s} {:>7s}".format(
        "#","attraction","中文","kind","frames"))
    print("-" * 70)
    for i, (en, zh, _lat, _lon, kind) in enumerate(ATTRACTIONS_21, 1):
        n = n_per_attr[en]
        mark = "✗" if n == 0 else "⚠" if n < 10 else "✓"
        print("{:>2}  {:<22s} {:<14s} {:<8s} {:>7d}  {}".format(
            i, en, zh, kind, n, mark))


if __name__ == "__main__":
    main()
