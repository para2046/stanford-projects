"""For each tier-1 frame within R metres of a canonical landmark, line
up side-by-side what the VLM ORIGINALLY said vs what the 27-list nearest
neighbour is. Lets us spot-check whether the canonical landmark is the
right rename for the frame, or whether the VLM was looking at something
else entirely.

Joins three jsonl files on (video, frame_id):
  landmark_audit.jsonl       — nearest canonical landmark + radius hits
  gps_recovery_full.jsonl    — tier, accepted, place_guess (resolved OSM)
  poi_scan.jsonl             — raw VLM `guess` text + `visible[]` list

Outputs:
  data/cities/zurich/landmark_vs_vlm.jsonl    — full join, one row/frame
  data/cities/zurich/landmark_vs_vlm.tsv      — flat tab-separated view
                                                 for opening in Excel
                                                 / spot-checking

  stdout
    - summary: how often the canonical landmark == VLM's guess
    - the 4 quadrants:
        BOTH agree     (canonical == VLM guess)
        VLM only       (VLM guess is a non-candidate street; canonical
                        is geographically nearest landmark)
        CANONICAL only (VLM guess is a different landmark than the
                        geographically nearest one)
        NEITHER        (no raw VLM data — gps_recovery 'promoted' frame)
    - 10 sample frames per quadrant with image paths to open

  python -m src.landmark_vs_vlm --radius 100
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
    ap.add_argument("--audit",
                    default=str(config.CITY_DIR / "a2"
                                / "proximity_tag.jsonl"))
    ap.add_argument("--gps",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    ap.add_argument("--scan", action="append", default=None,
                    help="poi_scan jsonl(s) to merge — both the every-10 "
                         "(poi_scan.jsonl) AND the cos>=0.75 expansion "
                         "(poi_scan_cos0.75.jsonl) are loaded by default")
    ap.add_argument("--out-jsonl",
                    default=str(config.CITY_DIR / "a2"
                                / "join_3way.jsonl"))
    ap.add_argument("--out-tsv",
                    default=str(config.CITY_DIR / "a2"
                                / "join_3way.tsv"))
    ap.add_argument("--radius", type=float, default=100.0,
                    help="restrict to frames within R m of a "
                         "canonical landmark (default 100)")
    ap.add_argument("--samples", type=int, default=10,
                    help="N example frames to print per quadrant")
    args = ap.parse_args()

    # ── load the three sources, key by (video, frame_id) ─────────────
    audit = {}
    for line in Path(args.audit).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        audit[(r["video"], r["frame_id"])] = r

    gps = {}
    for line in Path(args.gps).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        gps[(r["video"], r["frame_id"])] = r

    # Merge BOTH VLM scans — the every-10 baseline and the cos>=0.75
    # expansion that re-scanned every DINOv2-confident frame.
    scan_files = args.scan or [
        str(config.CITY_DIR / "poi_scan.jsonl"),
        str(config.CITY_DIR / "poi_scan_cos0.75.jsonl"),
    ]
    scan = {}
    scan_counts = []
    for sf in scan_files:
        if not Path(sf).exists():
            print(f"[landmark_vs_vlm] WARN: {sf} not found, skipping")
            continue
        n = 0
        for line in Path(sf).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            scan[(r["video"], r["frame_id"])] = r
            n += 1
        scan_counts.append((Path(sf).name, n))

    print(f"[landmark_vs_vlm] audit rows:    {len(audit):,}")
    print(f"[landmark_vs_vlm] gps_recovery:  {len(gps):,}")
    for name, n in scan_counts:
        print(f"[landmark_vs_vlm] {name}: {n:,} rows")
    print(f"[landmark_vs_vlm] unique VLM-scanned frames (merged): "
          f"{len(scan):,}")

    # ── flatten visible[] (can be list of strings or list of lists) ─
    def _flat_vis(vis):
        out = []
        for v in (vis or []):
            if isinstance(v, list):
                out.extend(str(x) for x in v)
            else:
                out.append(str(v))
        return out

    # ── walk the audit rows, restrict by radius, build the join ──────
    joined = []
    for key, ar in audit.items():
        if not ar["landmarks_within_100m"] and args.radius == 100.0:
            continue
        # (we keep all in_radius hits, primary = nearest)
        if (ar["nearest_dist_m"] > args.radius
                and not ar["landmarks_within_100m"]):
            continue

        gr = gps.get(key, {})
        sr = scan.get(key, {})
        out = {
            "video": key[0],
            "frame_id": key[1],
            "nearest_canonical": ar["nearest_landmark"],
            "nearest_canonical_zh": ar["nearest_landmark_zh"],
            "nearest_dist_m": ar["nearest_dist_m"],
            "all_canonicals_in_radius":
                ar.get(f"landmarks_within_{int(args.radius)}m", []),
            # VLM side, after OSM resolution (the field we currently use)
            "vlm_resolved_place_guess": gr.get("place_guess", "") or "",
            "tier": gr.get("tier"),
            "accepted": gr.get("accepted"),
            # raw VLM text, pre-resolution
            "vlm_raw_guess": (sr.get("guess") or "").strip(),
            "vlm_visible": _flat_vis(sr.get("visible")),
            "vlm_confidence": (sr.get("confidence") or "").strip(),
            "vlm_reasoning": (sr.get("reasoning") or "").strip()[:200],
            # image path to open
            "image_path": str(config.FRAMES_DIR / key[0]
                               / f"{key[1]}.jpg"),
        }
        joined.append(out)

    print(f"[landmark_vs_vlm] joined rows in radius {args.radius:.0f} m: "
          f"{len(joined):,}")

    # ── classify by quadrant ─────────────────────────────────────────
    def _quadrant(row):
        canon = row["nearest_canonical"]
        canon_in_radius = set(row["all_canonicals_in_radius"])
        vlm_resolved = row["vlm_resolved_place_guess"]
        vlm_raw = row["vlm_raw_guess"]
        vlm_visible = row["vlm_visible"]
        # no VLM data at all → "promoted by gps_recovery, no raw scan"
        if not vlm_raw and not vlm_visible and not vlm_resolved:
            return "no_vlm"
        # VLM mentioned the canonical landmark itself
        vlm_mentions_canon = any(
            canon.lower() in (s or "").lower()
            for s in [vlm_raw, vlm_resolved] + vlm_visible)
        # VLM mentioned ANY of the in-radius landmarks
        vlm_mentions_any_in_radius = any(
            any((c or "").lower() in (s or "").lower()
                for s in [vlm_raw, vlm_resolved] + vlm_visible)
            for c in canon_in_radius)
        if vlm_mentions_canon:
            return "both_agree"
        if vlm_mentions_any_in_radius:
            return "different_landmark_in_radius"
        return "canonical_only"

    quadrants = collections.Counter()
    by_q = collections.defaultdict(list)
    for r in joined:
        q = _quadrant(r)
        r["_quadrant"] = q
        quadrants[q] += 1
        by_q[q].append(r)

    # ── write the joined files ───────────────────────────────────────
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in joined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[landmark_vs_vlm] wrote {out_jsonl}")

    out_tsv = Path(args.out_tsv)
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join([
            "video", "frame_id", "tier", "nearest_canonical",
            "nearest_dist_m", "in_radius",
            "vlm_resolved_place_guess", "vlm_raw_guess",
            "vlm_visible", "quadrant", "image_path",
        ]) + "\n")
        for r in joined:
            f.write("\t".join([
                r["video"], r["frame_id"], str(r["tier"]),
                r["nearest_canonical"],
                f"{r['nearest_dist_m']:.1f}",
                "|".join(r["all_canonicals_in_radius"]),
                r["vlm_resolved_place_guess"],
                r["vlm_raw_guess"],
                "|".join(r["vlm_visible"]),
                r["_quadrant"],
                r["image_path"],
            ]) + "\n")
    print(f"[landmark_vs_vlm] wrote {out_tsv}")

    # ── stdout summary ───────────────────────────────────────────────
    print()
    print("=" * 100)
    print(f"QUADRANTS (n = {len(joined)} frames within "
          f"{args.radius:.0f} m of a canonical landmark)")
    print("=" * 100)
    for q, label in [
        ("both_agree",
         "BOTH agree     — VLM raw/visible/resolved mentions the SAME"
         " canonical landmark we tagged"),
        ("different_landmark_in_radius",
         "DIFFERENT      — VLM mentions a DIFFERENT canonical landmark"
         " that is also in radius"),
        ("canonical_only",
         "CANONICAL only — VLM says a non-candidate name (e.g. street)"
         " but we tagged it with the nearest landmark"),
        ("no_vlm",
         "NO VLM data    — gps_recovery promoted this from a neighbour"
         "; no raw scan row"),
    ]:
        n = quadrants[q]
        pct = 100 * n / max(1, len(joined))
        print(f"  {q:<32s}  {n:>5d}  ({pct:5.1f} %)   {label}")

    # ── per-canonical breakdown of the most "VLM said something
    #    different" cases (the ones to spot-check first) ─────────────
    print()
    print("=" * 100)
    print("PER-CANONICAL — frames where VLM said a DIFFERENT landmark "
          "in radius (suspect rename)")
    print("=" * 100)
    by_canon_diff = collections.Counter()
    for r in by_q["different_landmark_in_radius"]:
        by_canon_diff[r["nearest_canonical"]] += 1
    for canon, n in by_canon_diff.most_common(20):
        print(f"  {canon:<22s} {n:>5d}")

    # ── 10 sample frames per quadrant ───────────────────────────────
    print()
    print("=" * 100)
    print(f"SAMPLE FRAMES ({args.samples} per quadrant) — open the "
          f"image_path to spot-check")
    print("=" * 100)
    for q, label in [
        ("both_agree", "BOTH agree"),
        ("different_landmark_in_radius", "DIFFERENT landmark"),
        ("canonical_only", "CANONICAL only (VLM said street etc.)"),
        ("no_vlm", "NO VLM data"),
    ]:
        rows = by_q[q][:args.samples]
        if not rows:
            continue
        print(f"\n--- {label} (showing {len(rows)} of "
              f"{len(by_q[q])}) ---")
        for r in rows:
            print(f"  {r['video']:<18s} {r['frame_id']:<15s} "
                  f"canonical={r['nearest_canonical']:<22s} "
                  f"(d={r['nearest_dist_m']:.0f} m)")
            print(f"      vlm_resolved : {r['vlm_resolved_place_guess']}")
            print(f"      vlm_raw      : {r['vlm_raw_guess']}")
            if r["vlm_visible"]:
                print(f"      vlm_visible  : "
                      f"{', '.join(r['vlm_visible'][:5])}"
                      f"{' …' if len(r['vlm_visible']) > 5 else ''}")
            print(f"      image_path   : {r['image_path']}")


if __name__ == "__main__":
    main()
