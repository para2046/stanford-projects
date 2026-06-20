"""VLM cross-check for GPS recovery (DEV_MANUAL §2.5, Estimate 2).

Per-frame place-naming → (gps, confidence) — the *VLM* leg of the two
independent GPS estimates that `src.reconcile` combines.

Two entry points:
  geo_check_from_scan(scan_row, pois_map)
      Reuse an existing `poi_scan.jsonl` row. **No new API call** —
      uses the same Gemini Pro location-inference output the POI scan
      already produced.
  geo_check(frame_path, osm_pois)
      Live Gemini call (via `poi_scan.scan_frame`) for a frame that
      isn't in `poi_scan.jsonl`.

Both return the same dict:
    {gps: (lat, lon) | None,
     confidence: "high" | "medium" | "low" | "",
     place_name: str,
     reasoning: str}
`gps` is None when the VLM's location `guess` didn't resolve to a known
OSM POI. The confidence string can be turned into a numeric weight via
`reconcile.conf_num`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config            # noqa: E402, F401  (kept for downstream callers)
from src import poi_scan  # noqa: E402


def _guess_match(scan_row):
    """Return the matched row whose `source == 'guess'`, else None. Pure."""
    return next((m for m in scan_row.get("matched", [])
                 if m.get("source") == "guess"), None)


def geo_check_from_scan(scan_row, pois_map):
    """Pull the VLM location estimate out of one `poi_scan.jsonl` row.

    `pois_map`: `{osm_name: poi_dict}` for O(1) GPS lookup. Pure —
    unit-tested."""
    out = {"gps": None, "confidence": "",
           "place_name": "", "reasoning": scan_row.get("reasoning", "")}
    m = _guess_match(scan_row)
    if not m:
        return out
    p = pois_map.get(m["osm_name"])
    if not p:
        return out
    out["gps"] = (p["lat"], p["lon"])
    out["confidence"] = (scan_row.get("confidence") or "").strip().lower()
    out["place_name"] = m["osm_name"]
    return out


def geo_check(image_path, osm_pois):
    """Live Gemini call for one frame → same dict as
    `geo_check_from_scan`. Uses `poi_scan.scan_frame` (same prompt,
    same model, same backend) so the live and the cached paths are
    interchangeable."""
    parsed = poi_scan.scan_frame(image_path)
    matched, _ = poi_scan.match_scan(parsed, osm_pois)
    pois_map = {p["name"]: p for p in osm_pois}
    fake_row = {
        "guess": " | ".join(parsed["guess"]),
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"],
        "matched": matched,
    }
    return geo_check_from_scan(fake_row, pois_map)
