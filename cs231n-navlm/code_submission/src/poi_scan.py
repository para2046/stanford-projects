"""POI scan — work out *where* each video frame was taken, then match
the place names against the OSM POI table.

Replaces the v1 closed 27-candidate Gemma scan. This is **open-set and
inference-based**: the VLM is asked to *reason* about the location from
every clue (shop names, architecture, trams, churches, the lake) — it
does not need a visible street sign. It returns, as JSON:
  - `visible` — places it can directly see / read,
  - `guess`   — its single best inference of the street / square / area,
  - `confidence` (high/medium/low) + one-sentence `reasoning`.
`visible` and `guess` names are resolved against the OSM POI table
(`src/pois.py` output) via `resolve_poi()`; each match is tiered
L1 (iconic) / L2 (mid) / L3 (other).

    python -m src.poi_scan --limit 5       # 5-frame trial first
    python -m src.poi_scan --every-n 10    # every 10th extracted frame
    python -m src.poi_scan                 # all extracted frames

Needs GEMINI_API_KEY (.env). Model: `config.GEMINI_SCAN` (Gemini 2.5
Pro, paid API tier). Every API call is logged to logs/gemini_api.jsonl.
Output: data/cities/zurich/poi_scan.jsonl

Tiering is OSM-tag based (poi_tier, below) — edit the TIER_BY_* maps.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                          # noqa: E402
from src.pois import resolve_poi       # noqa: E402
from src.gemini_api import call_gemini  # noqa: E402

# ── tiers, by OSM tag — derived from OpenStreetMap's own categories.
#    Edit these maps to re-tier. A specific 'key=value' wins over the
#    key-only fallback. (OSM gives the *category*; true prominence would
#    need another signal — see DEV_MANUAL §2.3.)
TIER_BY_TAG = {
    # L1 — landmark categories people navigate TO
    "tourism=attraction": 1, "tourism=viewpoint": 1, "tourism=zoo": 1,
    "historic=castle": 1, "historic=monument": 1, "historic=memorial": 1,
    "railway=station": 1, "amenity=place_of_worship": 1,
    "amenity=townhall": 1,
    # L2 — supporting POIs
    "tourism=museum": 2, "tourism=gallery": 2, "tourism=artwork": 2,
    "tourism=hotel": 2, "tourism=theme_park": 2,
    "amenity=theatre": 2, "amenity=cinema": 2, "amenity=marketplace": 2,
    "amenity=library": 2, "amenity=university": 2, "amenity=college": 2,
    "leisure=park": 2, "leisure=garden": 2, "leisure=stadium": 2,
    "man_made=bridge": 2, "waterway=river": 2, "natural=water": 2,
    "place=square": 2,
}
# key-only fallback when the exact key=value is not mapped above
TIER_BY_KEY = {"tourism": 2, "historic": 2, "man_made": 2, "waterway": 2,
               "highway": 2, "amenity": 3, "leisure": 3, "natural": 3,
               "railway": 3, "place": 3}

GEMINI_SYS = ("You are a Zurich local. You work out where street-level "
              "photos were taken by reasoning from visible clues.")
GEMINI_PROMPT = (
    "This is a frame from a walking-tour video in central Zurich, "
    "Switzerland. Work out WHERE it was taken.\n"
    "Reason from every clue: shop / hotel / restaurant names, signs and "
    "street plates, architecture, churches and towers, trams and tram "
    "stops, the river or the lake, cobblestones, how wide the street "
    "is. You do NOT need a visible street sign — make your best "
    "inference from your knowledge of Zurich, even if you are unsure.\n"
    "Reply with ONLY a JSON object, nothing else:\n"
    "{\n"
    '  "visible": ["..."],   place names you can directly see or read '
    "(a sign, an unmistakable landmark); [] if none\n"
    '  "guess": "...",       your single best guess of the street, '
    'square or area this photo is in; "" only if you truly cannot tell\n'
    '  "confidence": "high | medium | low",\n'
    '  "reasoning": "one sentence — the clues that led to your guess"\n'
    "}\n"
    "Use official local German names (Bahnhofstrasse, Hauptbahnhof, "
    "Zürichsee, Grossmünster, Münsterhof). If a place is also widely "
    "known by an English name, write it as 'German | English'."
)


def poi_tier(osm_kind):
    """Tier a POI from its OSM tag 'key=value' (`src/pois.py:osm_kind`):
    1 = iconic landmark, 2 = mid, 3 = other. A specific tag value wins;
    else a key-level fallback; else L3. Pure — unit-tested."""
    if not osm_kind:
        return 3
    if osm_kind in TIER_BY_TAG:
        return TIER_BY_TAG[osm_kind]
    return TIER_BY_KEY.get(osm_kind.split("=", 1)[0], 3)


def _variants(s):
    """'German | English' -> ['German', 'English']. Pure."""
    return [v.strip() for v in str(s).split("|") if v.strip()]


EMPTY_SCAN = {"visible": [], "guess": [], "confidence": "", "reasoning": ""}


def parse_scan(text):
    """Parse the VLM JSON reply into a dict
    `{visible: [[variants],...], guess: [variants], confidence, reasoning}`.
    Robust to ```json fences and stray prose around the object — the
    first balanced-looking `{...}` is taken. Pure — unit-tested."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return dict(EMPTY_SCAN)
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return dict(EMPTY_SCAN)
    if not isinstance(d, dict):
        return dict(EMPTY_SCAN)
    visible = [_variants(x) for x in (d.get("visible") or [])]
    return {
        "visible": [v for v in visible if v],
        "guess": _variants(d.get("guess") or ""),
        "confidence": str(d.get("confidence", "")).strip().lower(),
        "reasoning": str(d.get("reasoning", "")).strip(),
    }


def _resolve(variants, osm_pois, source):
    """Resolve one variant-list against the OSM table. Returns a match
    dict — carrying `source` ('visible' | 'guess') — or None. The first
    variant that resolves wins, so a miss on the English name can still
    hit on the German one. Pure."""
    for v in variants:
        hit = resolve_poi(v, osm_pois)
        if hit:
            kind = hit.get("osm_kind", "")
            return {
                "variants": variants, "matched_name": v, "source": source,
                "osm_name": hit["name"], "osm_kind": kind,
                "kind_label": hit.get("kind_label", ""),
                "tier": poi_tier(kind),
            }
    return None


def match_scan(parsed, osm_pois):
    """Resolve a parsed scan's `visible` places and its `guess` against
    the OSM POI table. Returns (matched, unmatched): matched is
    `[{variants, matched_name, source, osm_name, osm_kind, kind_label,
    tier}]`, unmatched is `[variants, ...]`. Pure — unit-tested."""
    matched, unmatched = [], []
    items = [(v, "visible") for v in parsed.get("visible", [])]
    if parsed.get("guess"):
        items.append((parsed["guess"], "guess"))
    for variants, source in items:
        hit = _resolve(variants, osm_pois, source)
        if hit:
            matched.append(hit)
        else:
            unmatched.append(variants)
    return matched, unmatched


def load_osm_pois():
    """Load the OSM POI table (src/pois.py output). Exits if absent."""
    p = config.CITY_DIR / "pois.json"
    if not p.exists():
        sys.exit(f"OSM POI table not found: {p}\n"
                 f"run `python -m src.pois` first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _downscaled(image_path, max_px=config.POI_SCAN_MAX_PX):
    """Write a downscaled (<= max_px) JPEG copy of the frame to a temp
    file; return its path. The scan only needs recognition, not 4K
    detail — this roughly halves the Gemini image-token cost."""
    import tempfile
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_px, max_px))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, "JPEG", quality=85)
    return tmp.name


def scan_frame(image_path):
    """One open-set Gemini location-inference call → parsed scan dict
    (see `parse_scan`). The frame is downscaled first (cost). The call,
    its 429-aware retries and the API log (`logs/gemini_api.jsonl`) all
    live in `src.gemini_api`."""
    import os
    small = _downscaled(image_path)
    try:
        resp = call_gemini(small, GEMINI_SYS, GEMINI_PROMPT,
                           model=config.GEMINI_SCAN, max_tokens=4096,
                           label=Path(image_path).stem)
        return parse_scan(resp)
    finally:
        try:
            os.unlink(small)
        except OSError:
            pass


def discover_frames(every_n=1):
    """Extracted frames -> [(video, frame_id, path), ...]."""
    out = []
    if not config.FRAMES_DIR.exists():
        return out
    for vdir in sorted(config.FRAMES_DIR.iterdir()):
        if not vdir.is_dir() or vdir.name.endswith("_dense"):
            continue
        for f in sorted(vdir.glob("frame_*.jpg"))[::every_n]:
            out.append((vdir.name, f.stem, f))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="POI scan — Gemini location inference")
    ap.add_argument("--every-n", type=int, default=10,
                    help="scan every Nth extracted frame")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total frames (trial runs)")
    args = ap.parse_args()

    osm_pois = load_osm_pois()
    frames = discover_frames(args.every_n)
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        sys.exit(f"no extracted frames under {config.FRAMES_DIR} — "
                 f"run `python -m src.extract_frames` first")

    out_path = config.CITY_DIR / "poi_scan.jsonl"
    print(f"[poi_scan] {len(frames)} frames · model={config.GEMINI_SCAN} · "
          f"OSM table={len(osm_pois)} POIs -> {out_path}")

    n_l12 = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for video, frame_id, path in tqdm(frames, desc="[poi_scan]",
                                          unit="frame"):
            try:
                parsed = scan_frame(path)
            except Exception as e:
                parsed = dict(EMPTY_SCAN)
                tqdm.write(f"  {frame_id}: {type(e).__name__}: {e}")
            matched, unmatched = match_scan(parsed, osm_pois)
            n_l12 += sum(1 for m in matched if m["tier"] in (1, 2))
            fout.write(json.dumps({
                "video": video, "frame_id": frame_id,
                "guess": " | ".join(parsed["guess"]),
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "visible": parsed["visible"],
                "matched": matched, "unmatched": unmatched,
            }, ensure_ascii=False) + "\n")
            fout.flush()    # write each frame to disk now — crash-safe,
                            # and the file shows live progress
    print(f"[poi_scan] done — {n_l12} L1/L2 POI sightings recorded.")


if __name__ == "__main__":
    main()
