"""For each of the 21 famous attractions, build the SET of (video,
frame_id, SV slot) tuples that represent it.

A single attraction = many GPS points. Grossmünster has multiple
viewing angles, multiple approaches, multiple SV panos around its
polygon. The 'slot map' for each attraction is the union of all
frames that should be tagged with it.

Evidence sources, in order of strength:

  E1.  VLM `visible[]` mentions the attraction (the strongest signal
       — Gemini Pro literally said it sees the landmark in this image).
  E2.  VLM `guess` resolves to the attraction (medium — the VLM placed
       itself at the attraction).
  E3.  DINOv2 GPS is within R_PROX m of the attraction's canonical GPS
       (weakest — the walker is near, may or may not be looking at it).

Each frame gets a set of (attraction, evidence_set) labels — so a
frame near Grossmünster AND in which VLM saw Grossmünster gets both E1
and E3.

Inputs
  src/poi.py::CANDIDATE_POIS   — for the GPS centroids
  gps_recovery_full.jsonl       — for the per-frame DINO GPS
  poi_scan.jsonl                — for the raw VLM scans
  poi_scan_cos0.75.jsonl        — for the cos>=0.75 expansion scans

Outputs
  data/cities/zurich/landmark_slots.jsonl
    one row per attraction:
    { attraction, en, zh, kind, gps,
      n_frames_total,
      n_frames_by_evidence: { E1, E2, E3, E1+E3, ... },
      sv_slots_used: [top_sv_id, ...],
      frames: [{video, frame_id, evidence: ['E1','E3'],
                dist_m, s_dino, top_sv_id}, ...] }

  stdout
    table of attractions × n_frames × n_slots
    suspect attractions (very low coverage)

  python -m src.landmark_slots --radius 250
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


# The 21 attractions, ordered as in DEV_MANUAL §6.3.
ATTRACTIONS_21 = [
    ("Grossmünster",      "大教堂",         47.37018, 8.54425, "church"),
    ("Fraumünster",       "圣母大教堂",     47.37005, 8.54148, "church"),
    ("St. Peter",         "圣彼得教堂",     47.37154, 8.54126, "church"),
    ("Wasserkirche",      "水教堂",         47.37100, 8.54360, "church"),
    ("Lindenhof",         "林登霍夫山丘",   47.37280, 8.54149, "hill"),
    ("Niederdorfstrasse", "下村街",         47.37318, 8.54417, "street"),
    ("Bahnhofstrasse",    "班霍夫大街",     47.37367, 8.53924, "street"),
    ("Lake Zurich",       "苏黎世湖",       47.36500, 8.54500, "water"),
    ("Limmat river",      "利马特河",       47.37100, 8.54200, "water"),
    ("Landesmuseum",      "瑞士国家博物馆", 47.37926, 8.54021, "museum"),
    ("Kunsthaus",         "苏黎世美术馆",   47.37021, 8.54793, "museum"),
    ("Opernhaus",         "苏黎世歌剧院",   47.36548, 8.54683, "culture"),
    ("Bürkliplatz",       "比尔克利广场",   47.36615, 8.54153, "square"),
    ("Helmhaus",          "赫尔姆豪斯",     47.37088, 8.54363, "civic"),
    ("Hauptbahnhof",      "苏黎世中央车站", 47.37802, 8.54023, "station"),
    ("Münsterhof",        "明斯特霍夫广场", 47.37072, 8.54128, "square"),
    ("Paradeplatz",       "阅兵广场",       47.36953, 8.53866, "square"),
    ("Rathaus",           "市政厅",         47.37160, 8.54280, "civic"),
    ("Münsterbrücke",     "大教堂桥",       47.36970, 8.54200, "bridge"),
    ("Limmatquai",        "利马特河滨道",   47.37200, 8.54330, "street"),
    ("Sechseläutenplatz", "六鸣节广场",     47.36620, 8.54615, "square"),
]


# VLM-text variants that should exact-match each canonical attraction.
ALIASES = {
    "Grossmünster":      {"Grossmünster", "Grossmünsterplatz"},
    "Fraumünster":       {"Fraumünster"},
    "St. Peter":         {"St. Peter", "Kirche St. Peter",
                          "St. Peter Church", "St. Peter's Church",
                          "St. Peter Kirche", "St. Peterkirche",
                          "St. Peterhofstatt"},
    "Wasserkirche":      {"Wasserkirche"},
    "Lindenhof":         {"Lindenhof", "Lindenhofplatz"},
    "Niederdorfstrasse": {"Niederdorfstrasse", "Niederdorf"},
    "Bahnhofstrasse":    {"Bahnhofstrasse"},
    "Lake Zurich":       {"Lake Zurich", "Zürichsee"},
    "Limmat river":      {"Limmat river", "Limmat"},
    "Landesmuseum":      {"Landesmuseum", "Landesmuseum Zürich",
                          "Zürich Landesmuseum",
                          "Swiss National Museum"},
    "Kunsthaus":         {"Kunsthaus", "Kunsthaus Zürich"},
    "Opernhaus":         {"Opernhaus", "Opernhaus Zürich",
                          "Zurich Opera House"},
    "Bürkliplatz":       {"Bürkliplatz", "Bürkliterrasse"},
    "Helmhaus":          {"Helmhaus"},
    "Hauptbahnhof":      {"Hauptbahnhof", "Zürich Hauptbahnhof",
                          "Zurich Main Station", "Main Station",
                          "Zurich HB"},
    "Münsterhof":        {"Münsterhof"},
    "Paradeplatz":       {"Paradeplatz"},
    "Rathaus":           {"Rathaus"},
    "Münsterbrücke":     {"Münsterbrücke"},
    "Limmatquai":        {"Limmatquai"},
    "Sechseläutenplatz": {"Sechseläutenplatz"},
}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# fold(text) -> canonical attraction
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


def _vlm_canon_in(strings):
    """Set of canonical attractions named in any of `strings`. Splits
    compound names on `/` `,` `|` `am` `at` `near` first."""
    hits = set()
    for s in strings:
        for p in re.split(
                r"\s*(?:/|,|\||\bam\b|\bat\b|\bnear\b)\s*", s or ""):
            f = fold(p)
            if f and f in NAME_TO_CANON:
                hits.add(NAME_TO_CANON[f])
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gps",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    ap.add_argument("--radius", type=float, default=250.0,
                    help="E3 proximity radius (m) — frame's DINO GPS "
                         "must be within R of the attraction's canonical "
                         "coordinate (default 250)")
    ap.add_argument("--out",
                    default=str(config.CITY_DIR / "a2"
                                / "attraction_slots.jsonl"))
    args = ap.parse_args()

    # ── load gps_recovery, keep tier-1 accepted (VLM-confirmed) ──────
    rows = [json.loads(l) for l in
            Path(args.gps).open(encoding="utf-8") if l.strip()]
    tier1 = [r for r in rows
             if r.get("tier") == 1 and r.get("accepted")
             and r.get("g_dino")]
    print(f"[slots] VLM-confirmed accepted frames: {len(tier1):,}")

    # ── merge both VLM scan files for the raw text ───────────────────
    scan = {}
    for fn in ["poi_scan.jsonl", "poi_scan_cos0.75.jsonl"]:
        for line in (config.CITY_DIR / fn).open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            scan[(d["video"], d["frame_id"])] = d
    print(f"[slots] VLM scan rows merged: {len(scan):,}")

    # ── per-frame: compute evidence + assign labels ──────────────────
    per_attr = collections.defaultdict(list)        # attr_en -> [row,...]

    for r in tier1:
        g = tuple(r["g_dino"])
        sr = scan.get((r["video"], r["frame_id"])) or {}
        visible = _flat(sr.get("visible"))
        guess = (sr.get("guess") or "").strip()

        # E1 — VLM said it sees the attraction
        e1_attrs = _vlm_canon_in(visible)
        # E2 — VLM placed itself at the attraction
        e2_attrs = _vlm_canon_in([guess])
        # E3 — DINO GPS is within R of the attraction
        e3_attrs = set()
        for en, _zh, lat, lon, _kind in ATTRACTIONS_21:
            if _hav(g, (lat, lon)) <= args.radius:
                e3_attrs.add(en)

        # union — every attraction that has ANY evidence for this frame
        all_attrs = e1_attrs | e2_attrs | e3_attrs
        for en in all_attrs:
            ev = []
            if en in e1_attrs: ev.append("E1_visible")
            if en in e2_attrs: ev.append("E2_guess")
            if en in e3_attrs: ev.append("E3_proximity")
            # distance to the attraction's canonical
            row_en = next(x for x in ATTRACTIONS_21 if x[0] == en)
            d = _hav(g, (row_en[2], row_en[3]))
            per_attr[en].append({
                "video": r["video"],
                "frame_id": r["frame_id"],
                "evidence": ev,
                "dist_m": round(d, 1),
                "s_dino": round(r.get("s_dino", 0.0), 3),
                "top_sv_id": r.get("top_sv_id", ""),
            })

    # ── write the per-attraction slot map ────────────────────────────
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for en, zh, lat, lon, kind in ATTRACTIONS_21:
            frames = per_attr.get(en, [])
            sv_slots = sorted({fr["top_sv_id"] for fr in frames
                                if fr["top_sv_id"]})
            ev_counter = collections.Counter()
            for fr in frames:
                ev_counter["+".join(sorted(fr["evidence"]))] += 1
            row = {
                "attraction": en, "en": en, "zh": zh, "kind": kind,
                "gps": [lat, lon],
                "n_frames_total": len(frames),
                "n_sv_slots": len(sv_slots),
                "n_frames_by_evidence": dict(ev_counter),
                "sv_slots_used": sv_slots,
                "frames": sorted(frames,
                                  key=lambda x: (x["video"],
                                                 x["frame_id"])),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[slots] wrote {out_path.name}")

    # ── stdout summary ───────────────────────────────────────────────
    print()
    print("=" * 100)
    print("PER-ATTRACTION SLOT MAP")
    print("=" * 100)
    print(f"{'#':>2}  {'attraction':<22s} {'中文':<14s} "
          f"{'kind':<8s} {'frames':>7s} {'SV slots':>9s}  "
          f"{'evidence breakdown'}")
    print("-" * 110)
    for i, (en, zh, _lat, _lon, kind) in enumerate(ATTRACTIONS_21, 1):
        frames = per_attr.get(en, [])
        sv_slots = {fr["top_sv_id"] for fr in frames
                    if fr["top_sv_id"]}
        ev_counter = collections.Counter()
        for fr in frames:
            ev_counter["+".join(sorted(fr["evidence"]))] += 1
        ev_str = " ".join(f"{k}={v}" for k, v in
                          sorted(ev_counter.items(),
                                 key=lambda x: -x[1])[:4])
        print(f"{i:>2}  {en:<22s} {zh:<14s} {kind:<8s} "
              f"{len(frames):>7d} {len(sv_slots):>9d}  {ev_str}")

    print()
    suspect = [en for en, *_ in ATTRACTIONS_21
               if len(per_attr.get(en, [])) < 5]
    if suspect:
        print(f"SUSPECT (<5 frames): {', '.join(suspect)}")


if __name__ == "__main__":
    main()
