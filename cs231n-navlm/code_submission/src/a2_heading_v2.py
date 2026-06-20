"""Gap-tiered heading decision (replaces the all-4-cosine-weighted
heading in gps_recovery).

For every video frame DINOv2 matched, look at the 4 compass crops at
the top-1 pano. Sort by cosine. Compute the ABSOLUTE gap between
top-1 and top-2:

    gap = sims[0] - sims[1]

Tier rule:
    gap > 0.20            →  decision = "top1"
                              heading = top-1 crop's compass_angle
                              (one direction dominates clearly)

    0.05 < gap ≤ 0.20     →  decision = "top1+top2"
                              heading = cosine-weighted circular mean of
                                        top-1 and top-2 angles only
                              (two adjacent directions both strong)

    gap ≤ 0.05            →  decision = "ambiguous"
                              heading = null  (4-way-tie or near-tie;
                                              the heading number is
                                              meaningless, drop the
                                              frame downstream)

Output: data/cities/zurich/a2/heading_v2.jsonl
  one row per VLM-confirmed accepted frame:
    { video, frame_id, top_sv_id,
      same_pano: [{sv_id, compass_angle, cos}, ...]  (the 4 crops)
      gap, decision,
      heading_v2,         # the new decision
      heading_v1,         # the original all-4-weighted mean (for compare)
    }

Plus stdout: distribution of decisions + per-attraction breakdown.

  python -m src.a2_heading_v2
  python -m src.a2_heading_v2 --hi 0.20 --lo 0.05
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402


def circular_mean(degrees, weights):
    if not degrees:
        return None
    x = sum(w * math.sin(math.radians(d))
            for w, d in zip(weights, degrees))
    y = sum(w * math.cos(math.radians(d))
            for w, d in zip(weights, degrees))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gps",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    ap.add_argument("--sv-meta",
                    default=str(config.STREETVIEW_DIR / "meta.jsonl"))
    ap.add_argument("--frame-cache", default="frames_n1_l0")
    ap.add_argument("--hi", type=float, default=0.20,
                    help="gap threshold: > hi → use top1 only")
    ap.add_argument("--lo", type=float, default=0.05,
                    help="gap threshold: lo < gap ≤ hi → use top1+top2; "
                         "gap ≤ lo → ambiguous (heading = null)")
    ap.add_argument("--out",
                    default=str(config.CITY_DIR / "a2"
                                / "heading_v2.jsonl"))
    args = ap.parse_args()

    # ── load DINOv2 embeddings ──────────────────────────────────────
    cdir = config.CITY_DIR / "dinov2"
    sv_cache = np.load(cdir / "sv_v1.npz", allow_pickle=True)
    fr_cache = np.load(cdir / f"{args.frame_cache}.npz", allow_pickle=True)
    sv_embs = sv_cache["embs"]
    sv_ids = [Path(str(s)).stem for s in sv_cache["paths"]]
    fr_embs = fr_cache["embs"]
    fr_paths = [Path(p) for p in fr_cache["paths"]]
    fr_idx = {(p.parent.name, p.stem): i for i, p in enumerate(fr_paths)}
    print(f"[heading_v2] frame embeddings: {len(fr_embs):,}  "
          f"SV embeddings: {len(sv_embs):,}")

    # ── SV meta + group crops by pano ───────────────────────────────
    sv_meta = {}
    for line in Path(args.sv_meta).open(encoding="utf-8"):
        if not line.strip():
            continue
        m = json.loads(line)
        sv_meta[m["id"]] = m
    pano_to_crop_idx = collections.defaultdict(list)
    for j, sid in enumerate(sv_ids):
        pid = sv_meta.get(sid, {}).get("pano_id", "")
        if pid:
            pano_to_crop_idx[pid].append(j)

    # ── load gps_recovery rows (need top_sv_id + accepted gate) ─────
    gps_rows = []
    for line in Path(args.gps).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("accepted") and r.get("top_sv_id"):
            gps_rows.append(r)
    print(f"[heading_v2] accepted frames: {len(gps_rows):,}")

    # ── decide heading per frame ────────────────────────────────────
    decision_counts = collections.Counter()
    out_rows = []
    for r in gps_rows:
        video, fid = r["video"], r["frame_id"]
        key = (video, fid)
        if key not in fr_idx:
            continue
        top_sv = r["top_sv_id"]
        top_pano = sv_meta.get(top_sv, {}).get("pano_id", "")
        crop_js = pano_to_crop_idx.get(top_pano, [])
        if not crop_js:
            continue

        sims_all = sv_embs @ fr_embs[fr_idx[key]]
        crops = []
        for j in crop_js:
            crops.append({
                "sv_id": sv_ids[j],
                "compass_angle": sv_meta[sv_ids[j]].get(
                    "compass_angle", 0),
                "cos": float(sims_all[j]),
            })
        crops.sort(key=lambda c: -c["cos"])     # by cosine desc
        cos1 = crops[0]["cos"]
        cos2 = crops[1]["cos"] if len(crops) >= 2 else 0.0
        gap = cos1 - cos2

        if gap > args.hi:
            decision = "top1"
            heading_v2 = float(crops[0]["compass_angle"])
        elif gap > args.lo:
            decision = "top1+top2"
            heading_v2 = circular_mean(
                [crops[0]["compass_angle"], crops[1]["compass_angle"]],
                [max(0.0, crops[0]["cos"]), max(0.0, crops[1]["cos"])])
        else:
            decision = "ambiguous"
            heading_v2 = None
        decision_counts[decision] += 1

        out_rows.append({
            "video": video, "frame_id": fid,
            "top_sv_id": top_sv,
            "same_pano": crops,                # sorted by cos desc
            "gap": round(gap, 4),
            "decision": decision,
            "heading_v2": (round(heading_v2, 1)
                            if heading_v2 is not None else None),
            "heading_v1": r.get("heading"),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[heading_v2] wrote {out_path}  ({len(out_rows):,} rows)")

    # ── distribution ───────────────────────────────────────────────
    n = len(out_rows)
    print()
    print("=" * 96)
    print(f"DECISION DISTRIBUTION  (hi={args.hi}, lo={args.lo})")
    print("=" * 96)
    for d in ["top1", "top1+top2", "ambiguous"]:
        c = decision_counts[d]
        print(f"  {d:<14s}  {c:>6,}  ({100*c/max(1,n):.1f} %)")

    # ── gap histogram ───────────────────────────────────────────────
    print()
    print("--- gap distribution (top1 cos − top2 cos) ---")
    bins = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    hist = [0] * (len(bins) - 1)
    for r in out_rows:
        g = r["gap"]
        for i in range(len(bins) - 1):
            if bins[i] <= g < bins[i + 1]:
                hist[i] += 1
                break
    for i, c in enumerate(hist):
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}  : {c:>6,}")

    # ── how often did the new heading differ from the old? ─────────
    diff_buckets = collections.Counter()
    for r in out_rows:
        v1, v2 = r["heading_v1"], r["heading_v2"]
        if v2 is None:
            diff_buckets["v2_null"] += 1
            continue
        if v1 is None:
            diff_buckets["v1_null"] += 1
            continue
        d = abs(((v1 - v2 + 180) % 360) - 180)     # circular diff
        if d < 5:
            diff_buckets["close (<5°)"] += 1
        elif d < 15:
            diff_buckets["small (5-15°)"] += 1
        elif d < 45:
            diff_buckets["medium (15-45°)"] += 1
        else:
            diff_buckets["large (>=45°)"] += 1
    print()
    print("--- v1 vs v2 heading agreement ---")
    for k in ["close (<5°)", "small (5-15°)", "medium (15-45°)",
              "large (>=45°)", "v1_null", "v2_null"]:
        print(f"  {k:<18s}  {diff_buckets[k]:>6,}")


if __name__ == "__main__":
    main()
