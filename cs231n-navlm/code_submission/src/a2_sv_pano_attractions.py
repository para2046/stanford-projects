"""Map each Street View pano slot (the ground-truth GPS photos that
DINOv2 matches video frames to) to one or more of the 21 attractions.

The SV panos are the GT-GPS anchor set:
  - 4,431 SV crops total (1,108 pano locations × ~4 compass headings)
  - Each crop has known GPS from Google Street View metadata
  - DINOv2 matches every video frame to its best SV crop
  - → if we map SV crops to attractions, every frame inherits the
    attraction(s) of its matched crop, robustly

For each SV crop, we combine:

  PROXIMITY  — which of the 21 attractions sit within R metres of the
               crop's GPS (geometric)
  VLM        — for each frame that DINOv2 matched to this crop, which
               attractions did the VLM name in `visible[]` and `guess`
               (semantic, but only available for the subset of frames
               where VLM ran)

Output: data/cities/zurich/a2/sv_attractions.jsonl
  one row per SV crop:
    { sv_id, pano_id, compass_angle, gps,
      n_video_frames_matched,
      proximity: [{name, dist_m}, ...],
      vlm_visible: {name: count_of_matched_frames, ...},
      vlm_guess:   {name: count_of_matched_frames, ...},
      consensus:   "Grossmünster",       # best attraction (or null)
      consensus_source: "vlm+proximity"  # which signals concur
    }

  python -m src.a2_sv_pano_attractions --proximity-radius 100
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


def _canon_hits(strings):
    hits = set()
    for s in strings:
        for p in re.split(r"\s*(?:/|,|\||\bam\b|\bat\b|\bnear\b)\s*",
                           s or ""):
            f = fold(p)
            if f and f in NAME_TO_CANON:
                hits.add(NAME_TO_CANON[f])
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--sv-meta",
                    default=str(config.STREETVIEW_DIR / "meta.jsonl"))
    ap.add_argument("--gps",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    ap.add_argument("--proximity-radius", type=float, default=100.0,
                    help="how close (m) an attraction must be to the "
                         "SV crop's GPS to count as 'nearby' (default 100)")
    ap.add_argument("--out",
                    default=str(config.CITY_DIR / "a2"
                                / "sv_attractions.jsonl"))
    args = ap.parse_args()

    # ── load the 4,431 SV crops + their GT GPS ──────────────────────
    sv_crops = []
    for line in Path(args.sv_meta).open(encoding="utf-8"):
        if not line.strip():
            continue
        sv_crops.append(json.loads(line))
    print(f"[sv_attractions] SV crops: {len(sv_crops):,}")
    pano_ids = {c["pano_id"] for c in sv_crops}
    print(f"[sv_attractions] unique panos: {len(pano_ids):,}")

    # ── load all video frames + their matched SV crop ───────────────
    # We use the FULL gps_recovery output (both VLM-confirmed and
    # DINOv2-only) because we want every frame that matched to this
    # crop, not just the VLM-confirmed ones — and to weight the VLM
    # evidence we'll look up scans separately.
    frames = [json.loads(l) for l in
              Path(args.gps).open(encoding="utf-8") if l.strip()]
    # accepted only (visual-similarity threshold passed) — covers both
    # tiers; tier-1 frames had VLM evidence too, tier-2 didn't.
    frames = [f for f in frames if f.get("accepted")
              and f.get("top_sv_id")]
    print(f"[sv_attractions] accepted video frames: {len(frames):,}")

    # ── merge both VLM scan files for visible/guess text ────────────
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

    # ── index frames by SV crop ─────────────────────────────────────
    frames_by_slot = collections.defaultdict(list)
    for f in frames:
        frames_by_slot[f["top_sv_id"]].append(f)

    # ── for each SV crop, compute proximity + VLM mentions ──────────
    out_rows = []
    n_with_any_attr = 0
    consensus_counter = collections.Counter()
    for crop in sv_crops:
        sv_id = crop["id"]
        gps = (crop["lat"], crop["lon"])
        matched = frames_by_slot.get(sv_id, [])

        # PROXIMITY — attractions within R metres
        prox = []
        for en, _zh, lat, lon, _kind in ATTRACTIONS_21:
            d = _hav(gps, (lat, lon))
            if d <= args.proximity_radius:
                prox.append({"name": en, "dist_m": round(d, 1)})
        prox.sort(key=lambda x: x["dist_m"])

        # VLM — count visible/guess mentions across matched frames
        vlm_visible = collections.Counter()
        vlm_guess = collections.Counter()
        for f in matched:
            sr = scan.get((f["video"], f["frame_id"]))
            if not sr:
                continue
            for en in _canon_hits(_flat(sr.get("visible"))):
                vlm_visible[en] += 1
            for en in _canon_hits([sr.get("guess") or ""]):
                vlm_guess[en] += 1

        # CONSENSUS — attraction with strongest combined evidence
        all_attrs = set(vlm_visible) | set(vlm_guess) | {p["name"] for p in prox}
        consensus, consensus_source = None, None
        if all_attrs:
            best, best_score, best_src = None, -1, None
            for en in all_attrs:
                v = vlm_visible.get(en, 0)
                g = vlm_guess.get(en, 0)
                p_hit = any(p["name"] == en for p in prox)
                # score: VLM visible weighted highest, then guess, then prox
                score = 3 * v + 2 * g + (1 if p_hit else 0)
                src = []
                if v: src.append("vlm_visible")
                if g: src.append("vlm_guess")
                if p_hit: src.append("proximity")
                if score > best_score:
                    best, best_score, best_src = en, score, "+".join(src)
            consensus = best
            consensus_source = best_src
            consensus_counter[consensus] += 1
            n_with_any_attr += 1

        out_rows.append({
            "sv_id": sv_id,
            "pano_id": crop["pano_id"],
            "compass_angle": crop.get("compass_angle", crop.get("heading", 0)),
            "gps": [crop["lat"], crop["lon"]],
            "n_video_frames_matched": len(matched),
            "proximity": prox,
            "vlm_visible": dict(vlm_visible),
            "vlm_guess": dict(vlm_guess),
            "consensus": consensus,
            "consensus_source": consensus_source,
        })

    # ── write ──────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sv_attractions] wrote {out_path}")

    # ── stdout summary ──────────────────────────────────────────────
    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"SV crops total:                              {len(out_rows):,}")
    print(f"  ... with >=1 attraction (any evidence):    {n_with_any_attr:,}"
          f"  ({100*n_with_any_attr/len(out_rows):.1f}%)")
    matched_any = sum(1 for r in out_rows if r["n_video_frames_matched"])
    print(f"  ... matched by >=1 video frame:            {matched_any:,}"
          f"  ({100*matched_any/len(out_rows):.1f}%)")

    print()
    print(f"--- consensus attraction per SV crop (top 21) ---")
    for en, *_ in ATTRACTIONS_21:
        n = consensus_counter[en]
        print(f"  {n:>5d}  {en}")
    print()
    print(f"--- consensus source breakdown ---")
    src_counter = collections.Counter(r["consensus_source"] for r in out_rows
                                       if r["consensus_source"])
    for src, c in src_counter.most_common():
        print(f"  {c:>5d}  {src}")


if __name__ == "__main__":
    main()
