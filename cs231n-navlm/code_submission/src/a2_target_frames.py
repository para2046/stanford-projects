"""For each of the 21 target attractions, collect the matched frames
that represent it. Builds the per-attraction destination cohort that
re-annotation will use.

A frame represents an attraction iff that attraction's canonical name
appears in any of the frame's matches (gps_name OR vlm_name) in the
GPS_VLM_GEO.jsonl matched cohort. A single frame can represent
multiple attractions (e.g. a panoramic shot from Münsterbrücke that
shows Grossmünster, Fraumünster, AND Münsterhof).

Output: data/cities/zurich/a2/target_attraction_frames.jsonl
  one row per attraction:
    { attraction, zh, kind,
      n_frames, n_unique_panos,
      n_by_match_level: {attraction, landmark, poi},
      frames: [{video, frame_id, match_level, pano_id, s_dino}, ...] }

  python -m src.a2_target_frames
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_attraction_slots import ATTRACTIONS_21  # noqa: E402


CANON = {en for en, *_ in ATTRACTIONS_21}


def frame_attractions(r):
    """Set of 21-list canonical names this matched frame represents."""
    out = set()
    for a in r["list_a_gps"].get("attractions", []):
        if a in CANON: out.add(a)
    for a in r["list_b_vlm"].get("attractions", []):
        if a in CANON: out.add(a)
    for m in r["matches"]:
        for nm in [m["gps_name"], m["vlm_name"]]:
            if nm in CANON: out.add(nm)
    return out


def main():
    # ── load gps_recovery for top_sv_id + s_dino lookup ─────────────
    gps_meta = {}
    for line in (config.CITY_DIR
                 / "gps_recovery_full.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("top_sv_id"):
            pano = (r["top_sv_id"].rsplit("_h", 1)[0]
                    if "_h" in r["top_sv_id"] else r["top_sv_id"])
            gps_meta[(r["video"], r["frame_id"])] = {
                "pano_id": pano,
                "s_dino": r.get("s_dino", 0.0),
            }

    # ── load matched cohort ─────────────────────────────────────────
    matched_rows = []
    for line in (config.CITY_DIR / "a2"
                 / "GPS_VLM_GEO.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("matched"):
            matched_rows.append(r)
    print(f"[target_frames] matched cohort: {len(matched_rows):,} frames")

    # ── per-attraction frame list ───────────────────────────────────
    per_attr = collections.defaultdict(list)
    for r in matched_rows:
        attrs = frame_attractions(r)
        meta = gps_meta.get((r["video"], r["frame_id"]), {})
        for en in attrs:
            per_attr[en].append({
                "video": r["video"],
                "frame_id": r["frame_id"],
                "match_level": r["best_level"],
                "pano_id": meta.get("pano_id", ""),
                "s_dino": round(meta.get("s_dino", 0.0), 3),
            })

    # ── write ───────────────────────────────────────────────────────
    out_path = config.CITY_DIR / "a2" / "target_attraction_frames.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for en, zh, _lat, _lon, kind in ATTRACTIONS_21:
            frames = sorted(per_attr.get(en, []),
                             key=lambda x: (x["video"], x["frame_id"]))
            panos = {fr["pano_id"] for fr in frames if fr["pano_id"]}
            lvl_counts = collections.Counter(fr["match_level"]
                                              for fr in frames)
            row = {
                "attraction": en, "zh": zh, "kind": kind,
                "n_frames": len(frames),
                "n_unique_panos": len(panos),
                "n_by_match_level": dict(lvl_counts),
                "frames": frames,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[target_frames] wrote {out_path}")

    # ── stdout summary ──────────────────────────────────────────────
    print()
    print("=" * 100)
    print("TARGET ATTRACTIONS — frame count per attraction (matched cohort)")
    print("=" * 100)
    print("{:>2}  {:<22s} {:<14s} {:<8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "#", "attraction", "中文", "kind",
        "frames", "panos", "attr-L", "land-L", "poi-L"))
    print("-" * 100)
    tot_f = tot_p = tot_a = tot_l = tot_pp = 0
    for i, (en, zh, _lat, _lon, kind) in enumerate(ATTRACTIONS_21, 1):
        frames = per_attr.get(en, [])
        panos = {fr["pano_id"] for fr in frames if fr["pano_id"]}
        lvl = collections.Counter(fr["match_level"] for fr in frames)
        print("{:>2}  {:<22s} {:<14s} {:<8s} {:>8d} {:>8d} {:>8d} {:>8d} {:>8d}".format(
            i, en, zh, kind, len(frames), len(panos),
            lvl.get("attraction", 0), lvl.get("landmark", 0),
            lvl.get("poi", 0)))
        tot_f += len(frames)
        tot_p += len(panos)
        tot_a += lvl.get("attraction", 0)
        tot_l += lvl.get("landmark", 0)
        tot_pp += lvl.get("poi", 0)
    print("-" * 100)
    print("{:<48s} {:>8d} {:>8s} {:>8d} {:>8d} {:>8d}".format(
        "TOTAL (sums; double-counts cluster frames)",
        tot_f, "—", tot_a, tot_l, tot_pp))

    unique_frames = {(r["video"], r["frame_id"]) for r in matched_rows}
    print()
    print(f"unique matched frames (no double-count): {len(unique_frames):,}")
    print(f"avg attractions per frame (cluster density): "
          f"{tot_f / len(unique_frames):.2f}")


if __name__ == "__main__":
    main()
